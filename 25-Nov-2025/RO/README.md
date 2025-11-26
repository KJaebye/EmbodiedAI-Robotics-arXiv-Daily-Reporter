# Mixture of Horizons in Action Chunking 

**Title (ZH)**: 动作切块中的Horizons混合 

**Authors**: Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.19433)  

**Abstract**: Vision-language-action (VLA) models have shown remarkable capabilities in robotic manipulation, but their performance is sensitive to the $\textbf{action chunk length}$ used during training, termed $\textbf{horizon}$. Our empirical study reveals an inherent trade-off: longer horizons provide stronger global foresight but degrade fine-grained accuracy, while shorter ones sharpen local control yet struggle on long-term tasks, implying fixed choice of single horizons being suboptimal. To mitigate the trade-off, we propose a $\textbf{mixture of horizons (MoH)}$ strategy. MoH rearranges the action chunk into several segments with different horizons, processes them in parallel with a shared action transformer, and fuses outputs with a light linear gate. It has three appealing benefits. 1) MoH exploits long-term foresight and short-term precision jointly within a single model, improving both performance and generalizability to complex tasks. 2) MoH is plug-and-play for full-attention action modules with minimal training or inference overhead. 3) MoH enables dynamic inference with adaptive horizons, which selects stable actions through cross-horizon consensus, achieving 2.5$\times$ higher throughput than baselines while preserving superior performance. Extensive experiments over flow-based policies $\pi_0$, $\pi_{0.5}$, and one-step regression policy $\pi_{\text{reg}}$ demonstrate that MoH yields consistent and significant gains on both simulations and real-world tasks. Notably, under mixed-task setting, $\pi_{0.5}$ with MoH reaches a new state-of-the-art with 99$\%$ average success rate on LIBERO after only $30k$ training iterations. Project page: this https URL 

**Abstract (ZH)**: Vision-语言-行动（VLA）模型在机器人操作中展示了显著的能力，但其性能对训练期间使用的**行动片段长度**（称为**水平面**）高度敏感。我们的实证研究表明，存在一种固有的权衡：较长的水平面提供更强的整体预见性但降低细粒度的准确性，而较短的水平面则提升了局部控制能力但在长期任务上表现不佳，表明固定选择单一水平面是次优的。为缓解这种权衡，我们提出了一种**混合水平面（MoH）**策略。MoH将行动片段重新组织成具有不同水平面的多个段落，使用共享动作变换器并行处理这些段落，并通过轻量线性门融合输出。MoH具有三个令人信服的优势。1）MoH在单一模型中联合利用长期预见性和短期精确性，提高性能和对复杂任务的泛化能力。2）MoH可以无缝集成到具有全关注机制的动作模块中，并具有最小的训练或推理开销。3）MoH能够进行动态推理，通过跨水平面共识选择稳定动作，与基线相比可实现2.5倍的吞吐量增加，同时保持优越的性能。广泛实验表明，MoH在基于流动的策略$\pi_0$、$\pi_{0.5}$以及一阶回归策略$\pi_{\text{reg}}$上均在模拟和真实世界任务中提供了一致且显著的性能提升。在混合任务设置下，$\pi_{0.5}$与MoH结合后，在仅30,000次训练迭代的情况下，于LIBERO数据集上达到了99%的平均成功率，达到了新的最先进水平。项目页面：this https URL。 

---
# Deployment Dynamics and Optimization of Novel Space Antenna Deployable Mechanism 

**Title (ZH)**: 新型空间天线可展开机构的部署动力学与优化 

**Authors**: Mamoon Aamir, Mariyam Sattar, Naveed Ur Rehman Junejo, Aqsa Zafar Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2511.19377)  

**Abstract**: Given the increasing need for large aperture antennas in space missions, the difficulty of fitting such structures into small launch vehicles has prompted the design of deployable antenna systems. The thesis introduces a new Triple Scissors Deployable Truss Mechanism (TSDTM) for space antenna missions. The new mechanism is to be stowed during launch and efficiently deploy in orbit, offering maximum aperture size while taking up minimal launch volume. The thesis covers the entire design process from geometric modeling, kinematic analysis with screw theory and Newtonian approaches, dynamic analysis by eigenvalue and simulation methods, and verification with SolidWorks. In addition, optimization routines were coded based on Support Vector Machines for material choice in LEO environments and machine learning method for geometric setup. The TSDTM presented has enhanced structural dynamics with good comparison between simulation and analytical predictions. The structure optimized proved highly accurate, with a deviation of just 1.94% between machine learning-predicted and simulated natural frequencies, demonstrating the potential of incorporating AI-based methods in space structural design. 

**Abstract (ZH)**: 空间任务中大型天线的需求增加促使设计可展开天线系统。该论文介绍了一种新的三剪刀可展开桁架机构（TSDTM）用于空间天线任务。该新机构在发射时收折并在轨道上高效展开，提供最大天线口径同时占用最小发射体积。该论文涵盖了从几何建模、旋钮理论和牛顿方法的运动分析、特征值和仿真方法的动力学分析，以及使用SolidWorks验证的整个设计过程。此外，基于支持向量机的材料选择优化程序和基于机器学习的几何设置方法被编码。所呈示的TSDTM在结构动力学方面得到了增强，模拟和分析预测之间的对比良好。优化后结构非常准确，机器学习预测的自然频率与仿真值之间的偏差仅为1.94%，展示了在空间结构设计中采用基于AI的方法的潜力。 

---
# Rethinking Intermediate Representation for VLM-based Robot Manipulation 

**Title (ZH)**: 基于VLM的机器人 manipulation 中间表示重思כלול 

**Authors**: Weiliang Tang, Jialin Gao, Jia-Hui Pan, Gang Wang, Li Erran Li, Yunhui Liu, Mingyu Ding, Pheng-Ann Heng, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19315)  

**Abstract**: Vision-Language Model (VLM) is an important component to enable robust robot manipulation. Yet, using it to translate human instructions into an action-resolvable intermediate representation often needs a tradeoff between VLM-comprehensibility and generalizability. Inspired by context-free grammar, we design the Semantic Assembly representation named SEAM, by decomposing the intermediate representation into vocabulary and grammar. Doing so leads us to a concise vocabulary of semantically-rich operations and a VLM-friendly grammar for handling diverse unseen tasks. In addition, we design a new open-vocabulary segmentation paradigm with a retrieval-augmented few-shot learning strategy to localize fine-grained object parts for manipulation, effectively with the shortest inference time over all state-of-the-art parallel works. Also, we formulate new metrics for action-generalizability and VLM-comprehensibility, demonstrating the compelling performance of SEAM over mainstream representations on both aspects. Extensive real-world experiments further manifest its SOTA performance under varying settings and tasks. 

**Abstract (ZH)**: Vision-Language模型(SEAM)在实现稳健机器人操作中的重要性及其表示设计 

---
# SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control 

**Title (ZH)**: SENTINEL：一种端到端的语言-动作模型用于类人全身控制 

**Authors**: Yuxuan Wang, Haobin Jiang, Shiqing Yao, Ziluo Ding, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19236)  

**Abstract**: Existing humanoid control systems often rely on teleoperation or modular generation pipelines that separate language understanding from physical execution. However, the former is entirely human-driven, and the latter lacks tight alignment between language commands and physical behaviors. In this paper, we present SENTINEL, a fully end-to-end language-action model for humanoid whole-body control. We construct a large-scale dataset by tracking human motions in simulation using a pretrained whole body controller, combined with their text annotations. The model directly maps language commands and proprioceptive inputs to low-level actions without any intermediate representation. The model generates action chunks using flow matching, which can be subsequently refined by a residual action head for real-world deployment. Our method exhibits strong semantic understanding and stable execution on humanoid robots in both simulation and real-world deployment, and also supports multi-modal extensions by converting inputs into texts. 

**Abstract (ZH)**: SENTINEL：面向仿人机器人全身控制的端到端语言-动作模型 

---
# Soft pneumatic grippers: Topology optimization, 3D-printing and experimental validation 

**Title (ZH)**: 软气动夹持器：拓扑优化、3D打印及实验验证 

**Authors**: Prabhat Kumar, Chandra Prakash, Josh Pinskier, David Howard, Matthijs Langelaar  

**Link**: [PDF](https://arxiv.org/pdf/2511.19211)  

**Abstract**: This paper presents a systematic topology optimization framework for designing a soft pneumatic gripper (SPG), explicitly considering the design-dependent nature of the actuating load. The load is modeled using Darcy's law with an added drainage term. A 2D soft arm unit is optimized by formulating it as a compliant mechanism design problem using the robust formulation. The problem is posed as a min-max optimization, where the output deformations of blueprint and eroded designs are considered. A volume constraint is imposed on the blueprint part, while a strain-energy constraint is enforced on the eroded part. The MMA is employed to solve the optimization problem and obtain the optimized soft unit. Finite element analysis with the Ogden material model confirms that the optimized 2D unit outperforms a conventional rectangular design under pneumatic loading. The optimized 2D unit is extruded to obtain a 3D module, and ten such units are assembled to create a soft arm. Deformation profiles of the optimized arm are analysed under different pressure loads. Four arms are 3D-printed and integrated with a supporting structure to realize the proposed SPG. The gripping performance of the SPG is demonstrated on objects with different weights, sizes, stiffness, and shapes. 

**Abstract (ZH)**: 一种考虑驱动载荷设计依赖性的软气动夹持器拓扑优化框架 

---
# Reference-Free Sampling-Based Model Predictive Control 

**Title (ZH)**: 无参考自适应采样模型预测控制 

**Authors**: Fabian Schramm, Pierre Fabre, Nicolas Perrin-Gilbert, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2511.19204)  

**Abstract**: We present a sampling-based model predictive control (MPC) framework that enables emergent locomotion without relying on handcrafted gait patterns or predefined contact sequences. Our method discovers diverse motion patterns, ranging from trotting to galloping, robust standing policies, jumping, and handstand balancing, purely through the optimization of high-level objectives. Building on model predictive path integral (MPPI), we propose a dual-space spline parameterization that operates on position and velocity control points. Our approach enables contact-making and contact-breaking strategies that adapt automatically to task requirements, requiring only a limited number of sampled trajectories. This sample efficiency allows us to achieve real-time control on standard CPU hardware, eliminating the need for GPU acceleration typically required by other state-of-the-art MPPI methods. We validate our approach on the Go2 quadrupedal robot, demonstrating various emergent gaits and basic jumping capabilities. In simulation, we further showcase more complex behaviors, such as backflips, dynamic handstand balancing and locomotion on a Humanoid, all without requiring reference tracking or offline pre-training. 

**Abstract (ZH)**: 基于采样的模型预测控制框架：无需手工艺品步态模式或预定义的接触序列的自主运动 emerges from optimization of high-level objectives. 

---
# Efficient Optimization of a Permanent Magnet Array for a Stable 2D Trap 

**Title (ZH)**: 永磁阵列用于稳定2D捕获装置的高效优化 

**Authors**: Ann-Sophia Müller, Moonkwang Jeong, Jiyuan Tian, Meng Zhang, Tian Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19201)  

**Abstract**: Untethered magnetic manipulation of biomedical millirobots has a high potential for minimally invasive surgical applications. However, it is still challenging to exert high actuation forces on the small robots over a large distance. Permanent magnets offer stronger magnetic torques and forces than electromagnetic coils, however, feedback control is more difficult. As proven by Earnshaw's theorem, it is not possible to achieve a stable magnetic trap in 3D by static permanent magnets. Here, we report a stable 2D magnetic force trap by an array of permanent magnets to control a millirobot. The trap is located in an open space with a tunable distance to the magnet array in the range of 20 - 120mm, which is relevant to human anatomical scales. The design is achieved by a novel GPU-accelerated optimization algorithm that uses mean squared error (MSE) and Adam optimizer to efficiently compute the optimal angles for any number of magnets in the array. The algorithm is verified using numerical simulation and physical experiments with an array of two magnets. A millirobot is successfully trapped and controlled to follow a complex trajectory. The algorithm demonstrates high scalability by optimizing the angles for 100 magnets in under three seconds. Moreover, the optimization workflow can be adapted to optimize a permanent magnet array to achieve the desired force vector fields. 

**Abstract (ZH)**: 无缆磁操控的生物医学毫米机器人在微创手术应用中有巨大潜力，但仍然难以在长距离对小型机器人施加高驱动力。永磁体提供的磁力矩和磁力比电磁线圈更强，但是反馈控制更困难。如厄纳什定理所证明，使用静态永磁体无法在三维空间中实现稳定的磁捕获。在此，我们报告了一种使用永磁体阵列实现稳定二维磁力捕获的方法，以控制毫米机器人。捕获装置位于开放空间中，磁体阵列到捕获位置的距离可调范围为20-120毫米，这与人体解剖尺度相关。该设计通过一种新型GPU加速优化算法实现，该算法利用均方误差(MSE)和Adam优化器高效计算阵列中任意数量磁铁的最佳角度。该算法通过数值模拟和物理实验使用两个磁铁的阵列进行了验证。成功地捕获并控制了毫米机器人沿着复杂轨迹运动。该算法通过在不到三秒钟内优化100个磁铁的角度，展示了高可扩展性。此外，优化工作流程可以适应优化永磁体阵列以实现所需的力矢量场。 

---
# Autonomous Docking of Multi-Rotor UAVs on Blimps under the Influence of Wind Gusts 

**Title (ZH)**: 多旋翼无人机在湍流影响下与气球进行自主对接 

**Authors**: Pascal Goldschmid, Aamir Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2511.19135)  

**Abstract**: Multi-rotor UAVs face limited flight time due to battery constraints. Autonomous docking on blimps with onboard battery recharging and data offloading offers a promising solution for extended UAV missions. However, the vulnerability of blimps to wind gusts causes trajectory deviations, requiring precise, obstacle-aware docking strategies. To this end, this work introduces two key novelties: (i) a temporal convolutional network that predicts blimp responses to wind gusts, enabling rapid gust detection and estimation of points where the wind gust effect has subsided; (ii) a model predictive controller (MPC) that leverages these predictions to compute collision-free trajectories for docking, enabled by a novel obstacle avoidance method for close-range manoeuvres near the blimp. Simulation results show our method outperforms a baseline constant-velocity model of the blimp significantly across different scenarios. We further validate the approach in real-world experiments, demonstrating the first autonomous multi-rotor docking control strategy on blimps shown outside simulation. Source code is available here this https URL. 

**Abstract (ZH)**: 多旋翼无人机因电池限制飞行时间有限。通过在系留气球上实现自带电池充电和数据卸载的自主对接，提供了延长无人机任务的有前景解决方案。然而，系留气球对风 gust 的脆弱性会导致轨迹偏移，需要精确的、具有障碍物意识的对接策略。为此，本文介绍了两项关键创新：（i）一个时序卷积网络，预测系留气球对风 gust 的响应，实现快速 gust 检测并估算 gust 效应已消散的点；（ii）一种模型预测控制器（MPC），利用这些预测为对接计算无碰撞轨迹，该方法采用一种新的近距离躲避障碍物方法，以实现在系留气球附近执行机动操作。仿真结果表明，我们的方法在不同场景下显著优于基于系留气球恒定速度模型的基本方法。我们进一步在实际实验中验证了该方法，展示了在系留气球上实现的第一种自主多旋翼对接控制策略，该策略在仿真之外也得到了验证。源代码可以在以下链接获取：this https URL。 

---
# Analysis of Deep-Learning Methods in an ISO/TS 15066-Compliant Human-Robot Safety Framework 

**Title (ZH)**: ISO/TS 15066-合规的人机安全框架中深度学习方法的分析 

**Authors**: David Bricher, Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2511.19094)  

**Abstract**: Over the last years collaborative robots have gained great success in manufacturing applications where human and robot work together in close proximity. However, current ISO/TS-15066-compliant implementations often limit the efficiency of collaborative tasks due to conservative speed restrictions. For this reason, this paper introduces a deep-learning-based human-robot-safety framework (HRSF) that aims at a dynamical adaptation of robot velocities depending on the separation distance between human and robot while respecting maximum biomechanical force and pressure limits. The applicability of the framework was investigated for four different deep learning approaches that can be used for human body extraction: human body recognition, human body segmentation, human pose estimation, and human body part segmentation. Unlike conventional industrial safety systems, the proposed HRSF differentiates individual human body parts from other objects, enabling optimized robot process execution. Experiments demonstrated a quantitative reduction in cycle time of up to 15% compared to conventional safety technology. 

**Abstract (ZH)**: 基于深度学习的人机安全框架：动态调整机器人速度以适应人机协作任务 

---
# Multi-Agent Monocular Dense SLAM With 3D Reconstruction Priors 

**Title (ZH)**: 多智能体单目密集SLAM结合3D重建先验 

**Authors**: Haihang Wu, Yuchen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.19031)  

**Abstract**: Monocular Simultaneous Localization and Mapping (SLAM) aims to estimate a robot's pose while simultaneously reconstructing an unknown 3D scene using a single camera. While existing monocular SLAM systems generate detailed 3D geometry through dense scene representations, they are computationally expensive due to the need for iterative optimization. To address this challenge, MASt3R-SLAM utilizes learned 3D reconstruction priors, enabling more efficient and accurate estimation of both 3D structures and camera poses. However, MASt3R-SLAM is limited to single-agent operation. In this paper, we extend MASt3R-SLAM to introduce the first multi-agent monocular dense SLAM system. Each agent performs local SLAM using a 3D reconstruction prior, and their individual maps are fused into a globally consistent map through a loop-closure-based map fusion mechanism. Our approach improves computational efficiency compared to state-of-the-art methods, while maintaining similar mapping accuracy when evaluated on real-world datasets. 

**Abstract (ZH)**: 单目同时定位与建图（SLAM）旨在使用单个摄像头同时估计机器人姿态并重建未知的3D场景。虽然现有的单目SLAM系统通过密集的场景表示生成详细的3D几何结构，但由于需要迭代优化，计算成本较高。为应对这一挑战，MASt3R-SLAM利用学习到的3D重建先验，能够更高效且准确地估计3D结构和摄像头姿态。然而，MASt3R-SLAM仅支持单-agent操作。在本文中，我们扩展了MASt3R-SLAM，引入了首个基于单目密集SLAM的多机器人系统。每个机器人使用3D重建先验进行局部SLAM，并通过循环闭合机制融合各自的局部地图，形成全局一致的地图。与现有最先进的方法相比，我们的方法在计算效率上得到了改进，同时在真实数据集上进行评估时仍能保持相当的建图精度。 

---
# End-to-end Autonomous Vehicle Following System using Monocular Fisheye Camera 

**Title (ZH)**: 使用单目鱼眼相机的端到端自主车辆跟随系统 

**Authors**: Jiale Zhang, Yeqiang Qian, Tong Qin, Mingyang Jiang, Siyuan Chen, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19011)  

**Abstract**: The increase in vehicle ownership has led to increased traffic congestion, more accidents, and higher carbon emissions. Vehicle platooning is a promising solution to address these issues by improving road capacity and reducing fuel consumption. However, existing platooning systems face challenges such as reliance on lane markings and expensive high-precision sensors, which limits their general applicability. To address these issues, we propose a vehicle following framework that expands its capability from restricted scenarios to general scenario applications using only a camera. This is achieved through our newly proposed end-to-end method, which improves overall driving performance. The method incorporates a semantic mask to address causal confusion in multi-frame data fusion. Additionally, we introduce a dynamic sampling mechanism to precisely track the trajectories of preceding vehicles. Extensive closed-loop validation in real-world vehicle experiments demonstrates the system's ability to follow vehicles in various scenarios, outperforming traditional multi-stage algorithms. This makes it a promising solution for cost-effective autonomous vehicle platooning. A complete real-world vehicle experiment is available at this https URL. 

**Abstract (ZH)**: 车辆所有权的增加导致了交通拥堵加剧、事故增多和碳排放升高。车队编队是一种有望通过提高道路容量和降低燃油消耗来解决这些问题的可行方案。然而，现有的编队系统面临着依赖车道标线和昂贵的高精度传感器等挑战，这限制了其普适性。为了应对这些挑战，我们提出了一种仅使用摄像头的车辆跟随框架，能够将其实用范围从限制定场景扩展到一般场景应用。这一目标通过我们新提出的端到端方法实现，该方法提升了整体驾驶性能。该方法引入了语义掩码以解决多帧数据融合中的因果混淆问题。此外，我们引入了一种动态采样机制，以精确跟踪前车轨迹。在实际车辆实验中的开环验证表明，该系统能够在各种场景下跟随车辆，优于传统的多阶段算法。这使其成为经济实惠的自动驾驶车辆车队编队的有希望的解决方案。完整的实际车辆实验可在以下网址获取：this https URL。 

---
# Compressor-VLA: Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation 

**Title (ZH)**: 压缩器-VLA：指令引导的视觉令牌压缩以实现高效的机器人操作 

**Authors**: Juntao Gao, Feiyang Ye, Jing Zhang, Wenjing Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.18950)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a powerful paradigm in Embodied AI. However, the significant computational overhead of processing redundant visual tokens remains a critical bottleneck for real-time robotic deployment. While standard token pruning techniques can alleviate this, these task-agnostic methods struggle to preserve task-critical visual information. To address this challenge, simultaneously preserving both the holistic context and fine-grained details for precise action, we propose Compressor-VLA, a novel hybrid instruction-conditioned token compression framework designed for efficient, task-oriented compression of visual information in VLA models. The proposed Compressor-VLA framework consists of two token compression modules: a Semantic Task Compressor (STC) that distills holistic, task-relevant context, and a Spatial Refinement Compressor (SRC) that preserves fine-grained spatial details. This compression is dynamically modulated by the natural language instruction, allowing for the adaptive condensation of task-relevant visual information. Experimentally, extensive evaluations demonstrate that Compressor-VLA achieves a competitive success rate on the LIBERO benchmark while reducing FLOPs by 59% and the visual token count by over 3x compared to its baseline. The real-robot deployments on a dual-arm robot platform validate the model's sim-to-real transferability and practical applicability. Moreover, qualitative analyses reveal that our instruction guidance dynamically steers the model's perceptual focus toward task-relevant objects, thereby validating the effectiveness of our approach. 

**Abstract (ZH)**: Vision-Language-Action (VLA) 模型已成为束缚式AI中的强大力量，然而，处理冗余视觉标记的显著计算开销仍然是实时机器人部署的关键瓶颈。虽然标准的标记修剪技术可以缓解这一问题，但这些任务无关的方法难以保留任务关键的视觉信息。为了解决这一挑战，同时保留用于精确动作的完整上下文和精细细节，我们提出了一种名为Compressor-VLA的新型混合指令条件化标记压缩框架，该框架旨在有效地、面向任务地压缩VLA模型中的视觉信息。Compressor-VLA框架由两个标记压缩模块组成：语义任务压缩器（STC）用于提取整体的任务相关上下文，以及空间细化压缩器（SRC）用于保留精细的空间细节。这些压缩由自然语言指令动态调节，允许适应性地凝练任务相关视觉信息。实验结果显示，Compressor-VLA在LIBERO基准测试中的成功率具有竞争力，同时与基线相比，降低FLOPs 59%并减少视觉标记数量超过3倍。实机器人部署在双臂机器人平台上验证了模型的仿真到现实的迁移能力和实际应用性。此外，定性分析表明，我们的指令指导动态地引导模型的知觉焦点转向任务相关对象，从而验证了我们方法的有效性。 

---
# An Efficient Closed-Form Solution to Full Visual-Inertial State Initialization 

**Title (ZH)**: 全视觉-惯性状态初始化的高效闭式解 

**Authors**: Samuel Cerezo, Seong Hun Lee, Javier Civera  

**Link**: [PDF](https://arxiv.org/pdf/2511.18910)  

**Abstract**: In this letter, we present a closed-form initialization method that recovers the full visual-inertial state without nonlinear optimization. Unlike previous approaches that rely on iterative solvers, our formulation yields analytical, easy-to-implement, and numerically stable solutions for reliable start-up. Our method builds on small-rotation and constant-velocity approximations, which keep the formulation compact while preserving the essential coupling between motion and inertial measurements. We further propose an observability-driven, two-stage initialization scheme that balances accuracy with initialization latency. Extensive experiments on the EuRoC dataset validate our assumptions: our method achieves 10-20% lower initialization error than optimization-based approaches, while using 4x shorter initialization windows and reducing computational cost by 5x. 

**Abstract (ZH)**: 在这封信中，我们提出了一种闭式初始化方法，能够在不依赖非线性优化的情况下恢复全视觉-惯性状态。我们的方法不同于依赖迭代求解器的先前方法，能够提供分析式、易于实现且数值稳定的解决方案，以实现可靠的启动。我们的方法基于小旋转和恒定速度近似，这使得方法紧凑的同时保持了运动和惯性测量之间的本质耦合。我们还提出了一种观测性驱动的两阶段初始化方案，能够平衡初始化准确性与初始化延迟。在EuRoC数据集上的大量实验验证了我们的假设：与基于优化的方法相比，该方法将初始化误差降低了10-20%，使用了4倍短的初始化窗口，并将计算成本降低了5倍。 

---
# Accelerating Reinforcement Learning via Error-Related Human Brain Signals 

**Title (ZH)**: 通过错误相关的大脑信号加速强化学习 

**Authors**: Suzie Kim, Hye-Bin Shin, Hyo-Jeong Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18878)  

**Abstract**: In this work, we investigate how implicit neural feed back can accelerate reinforcement learning in complex robotic manipulation settings. While prior electroencephalogram (EEG) guided reinforcement learning studies have primarily focused on navigation or low-dimensional locomotion tasks, we aim to understand whether such neural evaluative signals can improve policy learning in high-dimensional manipulation tasks involving obstacles and precise end-effector control. We integrate error related potentials decoded from offline-trained EEG classifiers into reward shaping and systematically evaluate the impact of human-feedback weighting. Experiments on a 7-DoF manipulator in an obstacle-rich reaching environment show that neural feedback accelerates reinforcement learning and, depending on the human-feedback weighting, can yield task success rates that at times exceed those of sparse-reward baselines. Moreover, when applying the best-performing feedback weighting across all sub jects, we observe consistent acceleration of reinforcement learning relative to the sparse-reward setting. Furthermore, leave-one subject-out evaluations confirm that the proposed framework remains robust despite the intrinsic inter-individual variability in EEG decodability. Our findings demonstrate that EEG-based reinforcement learning can scale beyond locomotion tasks and provide a viable pathway for human-aligned manipulation skill acquisition. 

**Abstract (ZH)**: 本研究探讨了隐式神经反馈如何在复杂的机器人 manipulation 设置中加速强化学习。尽管先前的脑电图（EEG）引导的强化学习研究主要集中在导航或低维度的运动任务上，我们旨在了解此类神经评估信号是否能在涉及障碍物和精确末端执行器控制的高维度 manipulation 任务中改善策略学习。我们将从离线训练的 EEG 分类器解码的错误相关电位整合到奖励塑造中，并系统地评估人类反馈权重的影响。在障碍物丰富的抓取环境中，7 自由度 manipulator 的实验表明，神经反馈可以加速强化学习，并且在不同的人类反馈权重下，有时可以达到甚至超越稀疏奖励基线的任务成功率。此外，当我们使用性能最佳的反馈权重对所有被试进行评估时，我们观察到相对于稀疏奖励设置的强化学习加速是一致的。进一步的逐被试排除验证证实，所提出的框架即使在 EEG 解码的内在个体差异性存在的情况下仍保持稳健。我们的研究结果表明，基于 EEG 的强化学习可以超越运动任务，并为与人类对齐的 manipulation 技能获取提供可行的途径。 

---
# AutoOdom: Learning Auto-regressive Proprioceptive Odometry for Legged Locomotion 

**Title (ZH)**: 自回归本体感受器里程计：用于腿足运动的自回归 proprioceptive 里程计学习 

**Authors**: Changsheng Luo, Yushi Wang, Wenhan Cai, Mingguo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.18857)  

**Abstract**: Accurate proprioceptive odometry is fundamental for legged robot navigation in GPS-denied and visually degraded environments where conventional visual odometry systems fail. Current approaches face critical limitations: analytical filtering methods suffer from modeling uncertainties and cumulative drift, hybrid learning-filtering approaches remain constrained by their analytical components, while pure learning-based methods struggle with simulation-to-reality transfer and demand extensive real-world data collection. This paper introduces AutoOdom, a novel autoregressive proprioceptive odometry system that overcomes these challenges through an innovative two-stage training paradigm. Stage 1 employs large-scale simulation data to learn complex nonlinear dynamics and rapidly changing contact states inherent in legged locomotion, while Stage 2 introduces an autoregressive enhancement mechanism using limited real-world data to effectively bridge the sim-to-real gap. The key innovation lies in our autoregressive training approach, where the model learns from its own predictions to develop resilience against sensor noise and improve robustness in highly dynamic environments. Comprehensive experimental validation on the Booster T1 humanoid robot demonstrates that AutoOdom significantly outperforms state-of-the-art methods across all evaluation metrics, achieving 57.2% improvement in absolute trajectory error, 59.2% improvement in Umeyama-aligned error, and 36.2% improvement in relative pose error compared to the Legolas baseline. Extensive ablation studies provide critical insights into sensor modality selection and temporal modeling, revealing counterintuitive findings about IMU acceleration data and validating our systematic design choices for robust proprioceptive odometry in challenging locomotion scenarios. 

**Abstract (ZH)**: 准确的 proprioceptive 里程计对于 GPS 受限和视觉退化的环境中腿足机器人导航至关重要，这些环境常规的视觉里程计系统会失效。当前方法面临关键限制：基于分析的滤波方法受到建模不确定性的影响并存在累积漂移，混合学习-滤波方法仍然受限于其分析部分，而基于纯学习的方法在模拟到现实的转移方面存在问题，并需要大量实地数据收集。本文提出了 AutoOdom，这是一种新颖的自回归 proprioceptive 里程计系统，通过创新的两阶段训练范式克服了这些挑战。第一阶段利用大规模模拟数据学习腿足运动中固有的复杂非线性动力学和快速变化的接触状态，第二阶段引入自回归增强机制，使用有限的真实世界数据有效弥合模拟到现实的鸿沟。关键创新在于我们自回归的训练方法，模型通过学习其自身的预测来提高对传感器噪声的鲁棒性和在高度动态环境中的适应性。综合实验验证在 Booster T1 人形机器人上表明，AutoOdom 在所有评估指标上显著优于当前最先进的方法，绝对轨迹误差减少 57.2%，Umeyama 对齐误差减少 59.2%，相对姿态误差减少 36.2%，优于 Legolas 基线。广泛的消融研究提供了关于传感器模态选择和时间建模的关键见解，揭示了 IMU 加速数据的非直观发现，并验证了我们在具有挑战性的运动场景中设计鲁棒 proprioceptive 里程计的系统性选择。 

---
# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent 

**Title (ZH)**: MergeVLA：跨技能模型融合 toward 通用的视觉-语言-动作代理 

**Authors**: Yuxia Fu, Zhizhen Zhang, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.18810)  

**Abstract**: Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? With an empirical decomposition of learnable parameters during VLA fine-tuning, we identify two key sources of non-mergeability: (1) Finetuning drives LoRA adapters in the VLM backbone toward divergent, task-specific directions beyond the capacity of existing merging methods to unify. (2) Action experts develop inter-block dependencies through self-attention feedback, causing task information to spread across layers and preventing modular recombination. To address these challenges, we present MergeVLA, a merging-oriented VLA architecture that preserves mergeability by design. MergeVLA introduces sparsely activated LoRA adapters via task masks to retain consistent parameters and reduce irreconcilable conflicts in the VLM. Its action expert replaces self-attention with cross-attention-only blocks to keep specialization localized and composable. When the task is unknown, it uses a test-time task router to adaptively select the appropriate task mask and expert head from the initial observation, enabling unsupervised task inference. Across LIBERO, LIBERO-Plus, RoboTwin, and multi-task experiments on the real SO101 robotic arm, MergeVLA achieves performance comparable to or even exceeding individually finetuned experts, demonstrating robust generalization across tasks, embodiments, and environments. 

**Abstract (ZH)**: 最近的视觉-语言-动作（VLA）模型通过调优数百万个机器人演示重新制定了视觉-语言模型。虽然它们在针对单一实体或任务家族进行微调时表现良好，但将它们扩展到多技能设置仍然具有挑战性：直接将不同任务上训练的VLA专家融合导致成功率极低。这引发了一个基本问题：是什么阻止VLA在一个模型中掌握多种技能？通过在VLA微调过程中对可学习参数的经验分解，我们识别出两种不可融合的关键来源：（1）微调促使VLM主干中的LoRA适配器朝着超出现有融合方法统一能力的任务特定方向发散。（2）动作专家通过自注意力反馈产生跨模块依赖关系，导致任务信息在不同层间蔓延，阻碍了模块化重组。为了解决这些挑战，我们提出了一个以融合为导向的VLA架构——MergeVLA，通过设计保留了融合性。MergeVLA通过任务掩码引入稀疏激活的LoRA适配器以保持一致的参数并减少不可调和的冲突。其动作专家仅使用跨注意力模块代替自注意力模块以保持专一性并使其模块化组合。在任务未知时，它使用测试时任务路由器从初始观察中自适应地选择适当的任务掩码和专家头，从而实现无监督的任务推理。在LIBERO、LIBERO-Plus、RoboTwin以及SO101真实机械臂上的多任务实验中，MergeVLA实现了与甚至超越单独微调专家相当或更好的性能，展示了在任务、实体和环境方面的鲁棒泛化能力。 

---
# SP-VINS: A Hybrid Stereo Visual Inertial Navigation System based on Implicit Environmental Map 

**Title (ZH)**: SP-VINS: 基于隐式环境地图的 hybrid 单目视觉惯性导航系统 

**Authors**: Xueyu Du, Lilian Zhang, Fuan Duan, Xincan Luo, Maosong Wang, Wenqi Wu, JunMao  

**Link**: [PDF](https://arxiv.org/pdf/2511.18756)  

**Abstract**: Filter-based visual inertial navigation system (VINS) has attracted mobile-robot researchers for the good balance between accuracy and efficiency, but its limited mapping quality hampers long-term high-accuracy state estimation. To this end, we first propose a novel filter-based stereo VINS, differing from traditional simultaneous localization and mapping (SLAM) systems based on 3D map, which performs efficient loop closure constraints with implicit environmental map composed of keyframes and 2D keypoints. Secondly, we proposed a hybrid residual filter framework that combines landmark reprojection and ray constraints to construct a unified Jacobian matrix for measurement updates. Finally, considering the degraded environment, we incorporated the camera-IMU extrinsic parameters into visual description to achieve online calibration. Benchmark experiments demonstrate that the proposed SP-VINS achieves high computational efficiency while maintaining long-term high-accuracy localization performance, and is superior to existing state-of-the-art (SOTA) methods. 

**Abstract (ZH)**: 基于滤波的立体视觉惯性导航系统 (VINS) 已经吸引了移动机器人研究人员的关注，由于其在准确性和效率之间良好的平衡，但其有限的建图质量限制了长时间高精度状态估计。为此，我们首先提出了一种新型的基于滤波的立体VINS，不同于传统的基于三维地图的SLAM系统，它通过由关键帧和2D特征点组成的隐式环境地图执行高效的环路闭合约束。其次，我们提出了一种结合地标重新投影和光线约束的混合残差滤波框架，以构建统一的雅可比矩阵来进行测量更新。最后，考虑到退化的环境，我们将相机-IMU外部参数整合到视觉描述中，以实现在线标定。基准实验表明，所提出的SP-VINS在保持长时间高精度定位性能的同时，具有高计算效率，并优于现有的先进技术。 

---
# AIRHILT: A Human-in-the-Loop Testbed for Multimodal Conflict Detection in Aviation 

**Title (ZH)**: AIRHILT: 航空多模态冲突检测的人机交互试验床 

**Authors**: Omar Garib, Jayaprakash D. Kambhampaty, Olivia J. Pinon Fischer, Dimitri N. Mavris  

**Link**: [PDF](https://arxiv.org/pdf/2511.18718)  

**Abstract**: We introduce AIRHILT (Aviation Integrated Reasoning, Human-in-the-Loop Testbed), a modular and lightweight simulation environment designed to evaluate multimodal pilot and air traffic control (ATC) assistance systems for aviation conflict detection. Built on the open-source Godot engine, AIRHILT synchronizes pilot and ATC radio communications, visual scene understanding from camera streams, and ADS-B surveillance data within a unified, scalable platform. The environment supports pilot- and controller-in-the-loop interactions, providing a comprehensive scenario suite covering both terminal area and en route operational conflicts, including communication errors and procedural mistakes. AIRHILT offers standardized JSON-based interfaces that enable researchers to easily integrate, swap, and evaluate automatic speech recognition (ASR), visual detection, decision-making, and text-to-speech (TTS) models. We demonstrate AIRHILT through a reference pipeline incorporating fine-tuned Whisper ASR, YOLO-based visual detection, ADS-B-based conflict logic, and GPT-OSS-20B structured reasoning, and present preliminary results from representative runway-overlap scenarios, where the assistant achieves an average time-to-first-warning of approximately 7.7 s, with average ASR and vision latencies of approximately 5.9 s and 0.4 s, respectively. The AIRHILT environment and scenario suite are openly available, supporting reproducible research on multimodal situational awareness and conflict detection in aviation; code and scenarios are available at this https URL. 

**Abstract (ZH)**: AIRHILT（航空综合推理和有人参与试验台）：一种模块化轻量级模拟环境，用于评估航空冲突检测中的多模式飞行员和空中交通管制辅助系统 

---
# Head Stabilization for Wheeled Bipedal Robots via Force-Estimation-Based Admittance Control 

**Title (ZH)**: 基于力估计的 admittance 控制的轮式 biped 机器人头部稳定技术 

**Authors**: Tianyu Wang, Chunxiang Yan, Xuanhong Liao, Tao Zhang, Ping Wang, Cong Wen, Dingchuan Liu, Haowen Yu, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18712)  

**Abstract**: Wheeled bipedal robots are emerging as flexible platforms for field exploration. However, head instability induced by uneven terrain can degrade the accuracy of onboard sensors or damage fragile payloads. Existing research primarily focuses on stabilizing the mobile platform but overlooks active stabilization of the head in the world frame, resulting in vertical oscillations that undermine overall stability. To address this challenge, we developed a model-based ground force estimation method for our 6-degree-of-freedom wheeled bipedal robot. Leveraging these force estimates, we implemented an admittance control algorithm to enhance terrain adaptability. Simulation experiments validated the real-time performance of the force estimator and the robot's robustness when traversing uneven terrain. 

**Abstract (ZH)**: 轮式两足机器人正在成为场地区域探索的灵活平台。然而，不平地形引起的头部不稳定可能导致机载传感器的精度下降或损坏脆弱的有效载荷。现有研究主要集中在稳定移动平台，而忽视了在世界坐标系中主动稳定头部的问题，导致垂直振荡，从而影响整体稳定性。为解决这一挑战，我们为我们的六自由度轮式两足机器人开发了一种基于模型的地面力估计方法。利用这些力估计，我们实施了一种顺应性控制算法以增强地形适应性。仿真实验验证了力估计器的实时性能以及机器人在穿越不平地形时的鲁棒性。 

---
# Autonomous Surface Selection For Manipulator-Based UV Disinfection In Hospitals Using Foundation Models 

**Title (ZH)**: 基于基础模型的 manipulator-based UV消毒中自主表面选择 

**Authors**: Xueyan Oh, Jonathan Her, Zhixiang Ong, Brandon Koh, Yun Hann Tan, U-Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.18709)  

**Abstract**: Ultraviolet (UV) germicidal radiation is an established non-contact method for surface disinfection in medical environments. Traditional approaches require substantial human intervention to define disinfection areas, complicating automation, while deep learning-based methods often need extensive fine-tuning and large datasets, which can be impractical for large-scale deployment. Additionally, these methods often do not address scene understanding for partial surface disinfection, which is crucial for avoiding unintended UV exposure. We propose a solution that leverages foundation models to simplify surface selection for manipulator-based UV disinfection, reducing human involvement and removing the need for model training. Additionally, we propose a VLM-assisted segmentation refinement to detect and exclude thin and small non-target objects, showing that this reduces mis-segmentation errors. Our approach achieves over 92\% success rate in correctly segmenting target and non-target surfaces, and real-world experiments with a manipulator and simulated UV light demonstrate its practical potential for real-world applications. 

**Abstract (ZH)**: 基于基础模型的紫外(UV)消毒辐射简化表面选择方法：检测和排除薄小非目标物体的VLM辅助分割精炼 

---
# GVD-TG: Topological Graph based on Fast Hierarchical GVD Sampling for Robot Exploration 

**Title (ZH)**: 基于快速分层GVD采样的拓扑图用于机器人探索（GVD-TG） 

**Authors**: Yanbin Li, Canran Xiao, Shenghai Yuan, Peilai Yu, Ziruo Li, Zhiguo Zhang, Wenzheng Chi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18708)  

**Abstract**: Topological maps are more suitable than metric maps for robotic exploration tasks. However, real-time updating of accurate and detail-rich environmental topological maps remains a challenge. This paper presents a topological map updating method based on the Generalized Voronoi Diagram (GVD). First, the newly observed areas are denoised to avoid low-efficiency GVD nodes misleading the topological structure. Subsequently, a multi-granularity hierarchical GVD generation method is designed to control the sampling granularity at both global and local levels. This not only ensures the accuracy of the topological structure but also enhances the ability to capture detail features, reduces the probability of path backtracking, and ensures no overlap between GVDs through the maintenance of a coverage map, thereby improving GVD utilization efficiency. Second, a node clustering method with connectivity constraints and a connectivity method based on a switching mechanism are designed to avoid the generation of unreachable nodes and erroneous nodes caused by obstacle attraction. A special cache structure is used to store all connectivity information, thereby improving exploration efficiency. Finally, to address the issue of frontiers misjudgment caused by obstacles within the scope of GVD units, a frontiers extraction method based on morphological dilation is designed to effectively ensure the reachability of frontiers. On this basis, a lightweight cost function is used to assess and switch to the next viewpoint in real time. This allows the robot to quickly adjust its strategy when signs of path backtracking appear, thereby escaping the predicament and increasing exploration flexibility. And the performance of system for exploration task is verified through comparative tests with SOTA methods. 

**Abstract (ZH)**: 基于广义 Voronoi 图的拓扑地图实时更新方法：提高探索任务中拓扑地图的准确性和效率 

---
# Asynchronous Distributed Multi-Robot Motion Planning Under Imperfect Communication 

**Title (ZH)**: 异步分布式多机器人运动规划研究：通信不完全同步情形 

**Authors**: Ardalan Tajbakhsh, Augustinos Saravanos, James Zhu, Evangelos A. Theodorou, Lorenz T. Biegler, Aaron M. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2511.18703)  

**Abstract**: This paper addresses the challenge of coordinating multi-robot systems under realistic communication delays using distributed optimization. We focus on consensus ADMM as a scalable framework for generating collision-free, dynamically feasible motion plans in both trajectory optimization and receding-horizon control settings. In practice, however, these algorithms are sensitive to penalty tuning or adaptation schemes (e.g. residual balancing and adaptive parameter heuristics) that do not explicitly consider delays. To address this, we introduce a Delay-Aware ADMM (DA-ADMM) variant that adapts penalty parameters based on real-time delay statistics, allowing agents to down-weight stale information and prioritize recent updates during consensus and dual updates. Through extensive simulations in 2D and 3D environments with double-integrator, Dubins-car, and drone dynamics, we show that DA-ADMM significantly improves robustness, success rate, and solution quality compared to fixed-parameter, residual-balancing, and fixed-constraint baselines. Our results highlight that performance degradation is not solely determined by delay length or frequency, but by the optimizer's ability to contextually reason over delayed information. The proposed DA-ADMM achieves consistently better coordination performance across a wide range of delay conditions, offering a principled and efficient mechanism for resilient multi-robot motion planning under imperfect communication. 

**Abstract (ZH)**: 基于通信延迟感知的分布式优化在多机器人系统协调中的应用 

---
# CNN-Based Camera Pose Estimation and Localisation of Scan Images for Aircraft Visual Inspection 

**Title (ZH)**: 基于CNN的相机姿态估计与扫描图像的局部化在航空视觉检测中的应用 

**Authors**: Xueyan Oh, Leonard Loh, Shaohui Foong, Zhong Bao Andy Koh, Kow Leong Ng, Poh Kang Tan, Pei Lin Pearlin Toh, U-Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.18702)  

**Abstract**: General Visual Inspection is a manual inspection process regularly used to detect and localise obvious damage on the exterior of commercial aircraft. There has been increasing demand to perform this process at the boarding gate to minimise the downtime of the aircraft and automating this process is desired to reduce the reliance on human labour. Automating this typically requires estimating a camera's pose with respect to the aircraft for initialisation but most existing localisation methods require infrastructure, which is very challenging in uncontrolled outdoor environments and within the limited turnover time (approximately 2 hours) on an airport tarmac. Additionally, many airlines and airports do not allow contact with the aircraft's surface or using UAVs for inspection between flights, and restrict access to commercial aircraft. Hence, this paper proposes an on-site method that is infrastructure-free and easy to deploy for estimating a pan-tilt-zoom camera's pose and localising scan images. This method initialises using the same pan-tilt-zoom camera used for the inspection task by utilising a Deep Convolutional Neural Network fine-tuned on only synthetic images to predict its own pose. We apply domain randomisation to generate the dataset for fine-tuning the network and modify its loss function by leveraging aircraft geometry to improve accuracy. We also propose a workflow for initialisation, scan path planning, and precise localisation of images captured from a pan-tilt-zoom camera. We evaluate and demonstrate our approach through experiments with real aircraft, achieving root-mean-square camera pose estimation errors of less than 0.24 m and 2 degrees for all real scenes. 

**Abstract (ZH)**: 基于现场的无基础设施全景摄像头姿态估计与扫描图像定位方法 

---
# Stable Multi-Drone GNSS Tracking System for Marine Robots 

**Title (ZH)**: 多无人机稳定GNSS跟踪系统在海洋机器人中的应用 

**Authors**: Shuo Wen, Edwin Meriaux, Mariana Sosa Guzmán, Zhizun Wang, Junming Shi, Gregory Dudek  

**Link**: [PDF](https://arxiv.org/pdf/2511.18694)  

**Abstract**: Accurate localization is essential for marine robotics, yet Global Navigation Satellite System (GNSS) signals are unreliable or unavailable even at a very short distance below the water surface. Traditional alternatives, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM, and acoustic methods, suffer from error accumulation, high computational demands, or infrastructure dependence. In this work, we present a scalable multi-drone GNSS-based tracking system for surface and near-surface marine robots. Our approach combines efficient visual detection, lightweight multi-object tracking, GNSS-based triangulation, and a confidence-weighted Extended Kalman Filter (EKF) to provide stable GNSS estimation in real time. We further introduce a cross-drone tracking ID alignment algorithm that enforces global consistency across views, enabling robust multi-robot tracking with redundant aerial coverage. We validate our system in diversified complex settings to show the scalability and robustness of the proposed algorithm. 

**Abstract (ZH)**: 基于GNSS的可扩展多无人机跟踪系统及其在水面及近水面海洋机器人中的应用 

---
# Online Learning-Enhanced Lie Algebraic MPC for Robust Trajectory Tracking of Autonomous Surface Vehicles 

**Title (ZH)**: 基于在线学习增强的李代数 MPC 方法及其在自主水面车辆轨迹跟踪中的鲁棒性研究 

**Authors**: Yinan Dong, Ziyu Xu, Tsimafei Lazouski, Sangli Teng, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2511.18683)  

**Abstract**: Autonomous surface vehicles (ASVs) are easily influenced by environmental disturbances such as wind and waves, making accurate trajectory tracking a persistent challenge in dynamic marine conditions. In this paper, we propose an efficient controller for trajectory tracking of marine vehicles under unknown disturbances by combining a convex error-state MPC on the Lie group with an online learning module to compensate for these disturbances in real time. This design enables adaptive and robust control while maintaining computational efficiency. Extensive evaluations in numerical simulations, the Virtual RobotX (VRX) simulator, and real-world field experiments demonstrate that our method achieves superior tracking accuracy under various disturbance scenarios compared with existing approaches. 

**Abstract (ZH)**: 自主水面车辆在未知干扰下的轨迹跟踪控制：结合李群凸误差状态MPC与在线学习模块实现适应性和鲁棒性高效控制 

---
# AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient Visual Imitation Learning without Extra Human Annotations 

**Title (ZH)**: AutoFocus-IL：基于VLM的样本高效视觉imitation学习生成标注图方法无需额外的人工标注 

**Authors**: Litian Gong, Fatemeh Bahrani, Yutai Zhou, Amin Banayeeanzade, Jiachen Li, Erdem Biyik  

**Link**: [PDF](https://arxiv.org/pdf/2511.18617)  

**Abstract**: AutoFocus-IL is a simple yet effective method to improve data efficiency and generalization in visual imitation learning by guiding policies to attend to task-relevant features rather than distractors and spurious correlations. Although saliency regularization has emerged as a promising way to achieve this, existing approaches typically require costly supervision such as human gaze data or manual saliency annotations. In contrast, AutoFocus-IL leverages vision-language models (VLMs) to automatically identify and track key objects in demonstrations, generating temporal saliency maps that highlight causal visual signals while suppressing distractors. These maps are then used to regularize behavior cloning policies, yielding stronger alignment between visual attention and task-relevant cues. Experiments in both the CARLA simulator and real-robot manipulation tasks demonstrate that AutoFocus-IL not only outperforms standard behavior cloning but also surpasses state-of-the-art baselines that assume privileged access to human supervision, such as gaze data. Code, datasets, and trained policy videos are available at this https URL. 

**Abstract (ZH)**: AutoFocus-IL是一种简单而有效的方法，通过引导策略关注与任务相关的特征而非干扰项和虚假相关性，以提高视觉模仿学习中的数据效率和泛化能力。尽管显著性正则化已被证明是一种有潜力的方法来实现这一目标，但现有方法通常需要成本高昂的监督，如人类注视数据或手动显著性标注。相比之下，AutoFocus-IL利用视觉语言模型（VLMs）自动识别和跟踪演示中的关键对象，生成能够突出因果视觉信号并抑制干扰项的时空显著性图。然后，这些图用于正则化行为克隆策略，从而在视觉注意力与任务相关线索之间实现更强的对齐。实验结果表明，AutoFocus-IL不仅优于标准的行为克隆方法，而且还超越了假设可以获得人类监督（如注视数据）的先进基线方法。更多代码、数据集和训练策略视频请访问这个网址。 

---
# How to Train Your Latent Control Barrier Function: Smooth Safety Filtering Under Hard-to-Model Constraints 

**Title (ZH)**: 如何训练你的潜在控制障碍函数：在难以建模的约束条件下实现平滑安全过滤 

**Authors**: Kensuke Nakamura, Arun L. Bishop, Steven Man, Aaron M. Johnson, Zachary Manchester, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2511.18606)  

**Abstract**: Latent safety filters extend Hamilton-Jacobi (HJ) reachability to operate on latent state representations and dynamics learned directly from high-dimensional observations, enabling safe visuomotor control under hard-to-model constraints. However, existing methods implement "least-restrictive" filtering that discretely switch between nominal and safety policies, potentially undermining the task performance that makes modern visuomotor policies valuable. While reachability value functions can, in principle, be adapted to be control barrier functions (CBFs) for smooth optimization-based filtering, we theoretically and empirically show that current latent-space learning methods produce fundamentally incompatible value functions. We identify two sources of incompatibility: First, in HJ reachability, failures are encoded via a "margin function" in latent space, whose sign indicates whether or not a latent is in the constraint set. However, representing the margin function as a classifier yields saturated value functions that exhibit discontinuous jumps. We prove that the value function's Lipschitz constant scales linearly with the margin function's Lipschitz constant, revealing that smooth CBFs require smooth margins. Second, reinforcement learning (RL) approximations trained solely on safety policy data yield inaccurate value estimates for nominal policy actions, precisely where CBF filtering needs them. We propose the LatentCBF, which addresses both challenges through gradient penalties that lead to smooth margin functions without additional labeling, and a value-training procedure that mixes data from both nominal and safety policy distributions. Experiments on simulated benchmarks and hardware with a vision-based manipulation policy demonstrate that LatentCBF enables smooth safety filtering while doubling the task-completion rate over prior switching methods. 

**Abstract (ZH)**: 潜在状态安全性滤波器将哈密尔顿-雅可比（HJ）可达性扩展到操作潜在状态表示和直接从高维观察中学习的动力学，使其在难以建模的约束条件下实现安全的视听控制。然而，现有方法实施的是“最不限制”的过滤，这可能会在可能破坏现代视听策略性能的情况下，离散地在名义策略与安全性策略之间切换。虽然可达性价值函数原则上可以调整为控制障碍函数（CBFs）以实现平滑的基于优化的过滤，但我们从理论上和实验上证明了当前的潜在空间学习方法产生了根本不兼容的价值函数。我们识别了两种不兼容的原因：首先，在HJ可达性中，失败是通过潜在空间中的“边缘函数”来编码的，其符号表示该潜在值是否在约束集中。然而，将边缘函数表示为分类器会导致饱和的价值函数，表现出不连续的跳跃。我们证明了价值函数的利普希茨常数与边缘函数的利普希茨常数成线性关系，揭示了光滑的CBFs需要光滑的边缘。其次，仅基于安全性策略数据训练的强化学习（RL）近似会导致对名义策略动作的价值估计不准确，而这些正是CBFs过滤所必需的。我们提出了潜在空间CBF（LatentCBF），它通过梯度惩罚来解决两个挑战，无需额外标签即可获得平滑的边缘函数，并通过混合来自名义策略和安全性策略分布的数据来训练价值函数。在基于视觉的操控策略的模拟基准和硬件实验中，LatentCBF不仅实现了平滑的安全过滤，还在先前切换方法的基础上将任务完成率翻了一番。 

---
# An Analysis of Constraint-Based Multi-Agent Pathfinding Algorithms 

**Title (ZH)**: 基于约束的多智能体路径规划算法分析 

**Authors**: Hannah Lee, James D. Motes, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2511.18604)  

**Abstract**: This study informs the design of future multi-agent pathfinding (MAPF) and multi-robot motion planning (MRMP) algorithms by guiding choices based on constraint classification for constraint-based search algorithms. We categorize constraints as conservative or aggressive and provide insights into their search behavior, focusing specifically on vanilla Conflict-Based Search (CBS) and Conflict-Based Search with Priorities (CBSw/P). Under a hybrid grid-roadmap representation with varying resolution, we observe that aggressive (priority constraint) formulations tend to solve more instances as agent count or resolution increases, whereas conservative (motion constraint) formulations yield stronger solution quality when both succeed. Findings are synthesized in a decision flowchart, aiding users in selecting suitable constraints. Recommendations extend to Multi-Robot Motion Planning (MRMP), emphasizing the importance of considering topological features alongside problem, solution, and representation features. A comprehensive exploration of the study, including raw data and map performance, is available in our public GitHub Repository: this https URL 

**Abstract (ZH)**: 本研究通过基于约束分类指导未来多代理路径规划（MAPF）和多机器人运动规划（MRMP）算法的设计选择，为基于约束的搜索算法提供信息。我们将约束分类为保守型或激进型，并通过对Vanilla Conflict-Based Search (CBS)和Conflict-Based Search with Priorities (CBSw/P)的具体分析，探讨其搜索行为。在混合网格- roadmap表示下，随着代理数量或分辨率的增加，激进型（优先级约束）表述能解决更多的实例，而保守型（运动约束）表述在两者都成功时能提供更强的解的质量。研究成果汇总在决策流程图中，帮助用户选择合适的约束。建议延伸至多机器人运动规划（MRMP），强调在考虑拓扑特征的同时，还要考虑问题、解和表示特征的重要性。完整的研究内容，包括原始数据和地图性能，在我们的公开GitHub仓库中：this https URL 

---
# Object-centric Task Representation and Transfer using Diffused Orientation Fields 

**Title (ZH)**: 以对象为中心的任务表示与传输：扩散方向场方法 

**Authors**: Cem Bilaloglu, Tobias Löw, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2511.18563)  

**Abstract**: Curved objects pose a fundamental challenge for skill transfer in robotics: unlike planar surfaces, they do not admit a global reference frame. As a result, task-relevant directions such as "toward" or "along" the surface vary with position and geometry, making object-centric tasks difficult to transfer across shapes. To address this, we introduce an approach using Diffused Orientation Fields (DOF), a smooth representation of local reference frames, for transfer learning of tasks across curved objects. By expressing manipulation tasks in these smoothly varying local frames, we reduce the problem of transferring tasks across curved objects to establishing sparse keypoint correspondences. DOF is computed online from raw point cloud data using diffusion processes governed by partial differential equations, conditioned on keypoints. We evaluate DOF under geometric, topological, and localization perturbations, and demonstrate successful transfer of tasks requiring continuous physical interaction such as inspection, slicing, and peeling across varied objects. We provide our open-source codes at our website this https URL 

**Abstract (ZH)**: 曲面对象为机器人技能转移提出了基本挑战：与平面表面不同，它们不支持全局参考框架。因此，如“朝向”或“沿着”表面这样的任务相关方向会随位置和几何形状的变化而变化，使得基于对象的任务难以在不同形状之间转移。为了解决这一问题，我们提出了一种使用扩散方向场（DOF）的方法，这是一种局部参考框架的平滑表示，用于在曲面对象间执行任务的迁移学习。通过在这些平滑变化的局部框架中表达操作任务，我们将跨曲面对象转移任务的问题转化为稀疏关键点对应关系的建立。DOF 是通过对点云数据进行受关键点条件限制的偏微分方程调控的扩散过程，在线计算得到。我们对 DOF 进行了几何、拓扑和定位干扰下的评估，并展示了它在不同物体间成功转移要求持续物理交互的任务，如检测、切割和剥离。我们已经在网站 [this https URL] 提供了开源代码。 

---
# Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation 

**Title (ZH)**: Splatblox：考虑通过性的高斯点云生成方法在户外机器人导航中的应用 

**Authors**: Samarth Chopra, Jing Liang, Gershom Seneviratne, Yonghan Lee, Jaehoon Choi, Jianyu An, Stephen Cheng, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2511.18525)  

**Abstract**: We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: this https URL 

**Abstract (ZH)**: 我们提出Splatblox，这是一种适用于茂密植被、不规则障碍物和复杂地形的户外自主导航的实时系统。该方法使用高斯碰撞融合分割的RGB图像和LiDAR点云，构建一个同时编码几何和语义的遍历性感知欧几里得符号距离场（ESDF）。在线更新的此场域能够通过语义推理区分可通行植被（例如，高草）和刚性障碍物（例如，树木），而LiDAR确保了360度的几何覆盖范围，从而支持更长的规划时限。我们在四足机器人上验证了Splatblox，并展示其在轮式平台上应用的可行性。在植被丰富的实地试验中，Splatblox的成功率比最先进的方法高出50%以上，冻车事件减少40%，路径缩短5%，并在某些场景中实现最高13%的更快到达目标时间，同时支持长达100米的远程任务。更多实验视频和细节可在我们的项目页面查看：this https URL。 

---
# SafeFall: Learning Protective Control for Humanoid Robots 

**Title (ZH)**: SafeFall: 学习保护性控制的人形机器人 

**Authors**: Ziyu Meng, Tengyu Liu, Le Ma, Yingying Wu, Ran Song, Wei Zhang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18509)  

**Abstract**: Bipedal locomotion makes humanoid robots inherently prone to falls, causing catastrophic damage to the expensive sensors, actuators, and structural components of full-scale robots. To address this critical barrier to real-world deployment, we present \method, a framework that learns to predict imminent, unavoidable falls and execute protective maneuvers to minimize hardware damage. SafeFall is designed to operate seamlessly alongside existing nominal controller, ensuring no interference during normal operation. It combines two synergistic components: a lightweight, GRU-based fall predictor that continuously monitors the robot's state, and a reinforcement learning policy for damage mitigation. The protective policy remains dormant until the predictor identifies a fall as unavoidable, at which point it activates to take control and execute a damage-minimizing response. This policy is trained with a novel, damage-aware reward function that incorporates the robot's specific structural vulnerabilities, learning to shield critical components like the head and hands while absorbing energy with more robust parts of its body. Validated on a full-scale Unitree G1 humanoid, SafeFall demonstrated significant performance improvements over unprotected falls. It reduced peak contact forces by 68.3\%, peak joint torques by 78.4\%, and eliminated 99.3\% of collisions with vulnerable components. By enabling humanoids to fail safely, SafeFall provides a crucial safety net that allows for more aggressive experiments and accelerates the deployment of these robots in complex, real-world environments. 

**Abstract (ZH)**: SafeFall：一种防止 humanoid 机器人跌倒的框架 

---
# Expanding the Workspace of Electromagnetic Navigation Systems Using Dynamic Feedback for Single- and Multi-agent Control 

**Title (ZH)**: 使用动态反馈实现单agent和multi-agent控制以扩展电磁导航系统的作业空间 

**Authors**: Jasan Zughaibi, Denis von Arx, Maurus Derungs, Florian Heemeyer, Luca A. Antonelli, Quentin Boehler, Michael Muehlebach, Bradley J. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.18486)  

**Abstract**: Electromagnetic navigation systems (eMNS) enable a number of magnetically guided surgical procedures. A challenge in magnetically manipulating surgical tools is that the effective workspace of an eMNS is often severely constrained by power and thermal limits. We show that system-level control design significantly expands this workspace by reducing the currents needed to achieve a desired motion. We identified five key system approaches that enable this expansion: (i) motion-centric torque/force objectives, (ii) energy-optimal current allocation, (iii) real-time pose estimation, (iv) dynamic feedback, and (v) high-bandwidth eMNS components. As a result, we stabilize a 3D inverted pendulum on an eight-coil OctoMag eMNS with significantly lower currents (0.1-0.2 A vs. 8-14 A), by replacing a field-centric field-alignment strategy with a motion-centric torque/force-based approach. We generalize to multi-agent control by simultaneously stabilizing two inverted pendulums within a shared workspace, exploiting magnetic-field nonlinearity and coil redundancy for independent actuation. A structured analysis compares the electromagnetic workspaces of both paradigms and examines current-allocation strategies that map motion objectives to coil currents. Cross-platform evaluation of the clinically oriented Navion eMNS further demonstrates substantial workspace expansion by maintaining stable balancing at distances up to 50 cm from the coils. The results demonstrate that feedback is a practical path to scalable, efficient, and clinically relevant magnetic manipulation. 

**Abstract (ZH)**: 磁导航系统中的电磁导航系统使多种磁引导手术成为可能。磁操纵手术工具的一个挑战是，电磁导航系统的有效工作空间常常由于功率和热限制而受到严重约束。通过减少实现预期运动所需的电流，系统级控制设计显著扩展了这一工作空间。我们确定了五个关键系统方法，使其扩展成为可能：（i）以运动为中心的扭矩/力目标，（ii）能量最优电流分配，（iii）实时姿态估计，（iv）动态反馈，以及（v）高带宽电磁导航系统组件。作为结果，通过用以运动为中心的扭矩/力基于的方法替代磁场为中心的磁场对准策略，我们使用八线圈OctoMag电磁导航系统将3D倒立摆稳定在显著较低的电流（0.1-0.2 A vs. 8-14 A）。我们通过同时在共享工作空间内稳定两个倒立摆在将多agent控制推广中利用了磁场非线性和线圈冗余实现独立操作。结构化分析比较了两种范式的电磁工作空间，并研究了将运动目标映射到线圈电流的电流分配策略。跨平台评估进一步证明了临床导向的Navion电磁导航系统的显著工作空间扩展，即使在距线圈50厘米的距离也能保持稳定的平衡。这些结果表明，反馈是实现可扩展、高效且临床相关的磁操纵的一种实际途径。 

---
# Explicit Bounds on the Hausdorff Distance for Truncated mRPI Sets via Norm-Dependent Contraction Rates 

**Title (ZH)**: 基于范数依赖收缩率的截断mRPI集的Hausdorff距离显式界 

**Authors**: Jiaxun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.18374)  

**Abstract**: This paper establishes the first explicit and closed-form upper bound on the Hausdorff distance between the truncated minimal robust positively invariant (mRPI) set and its infinite-horizon limit. While existing mRPI approximations guarantee asymptotic convergence through geometric or norm-based arguments, none provides a computable expression that quantifies the truncation error for a given horizon. We show that the error satisfies \( d_H(\mathcal{E}_N,\mathcal{E}_\infty) \le r_W\,\gamma^{N+1}/(1-\gamma), \) where $\gamma<1$ is the induced-norm contraction factor and $r_W$ depends only on the disturbance set. The bound is fully analytic, requires no iterative set computations, and directly characterizes the decay rate of the truncated Minkowski series. We further demonstrate that the choice of vector norm serves as a design parameter that accelerates convergence, enabling substantially tighter horizon selection for robust invariant-set computations and tube-based MPC. Numerical experiments validate the sharpness, scalability, and practical relevance of the proposed bound. 

**Abstract (ZH)**: 本文建立了截断的最小鲁棒正不变集（mRPI）与其无穷远限集之间的豪斯多夫距离的第一个显式和闭式上界。现有的mRPI近似通过几何或范数方法保证渐近收敛，但 none 提供一个可计算的表达式来量化给定时间范围内的截断误差。我们证明了误差满足 \( d_H(\mathcal{E}_N,\mathcal{E}_\infty) \le r_W\,\gamma^{N+1}/(1-\gamma) \)，其中 \(\gamma<1\) 是诱导范数收缩因子，且 \(r_W\) 仅依赖于扰动集。该界是完全解析的，无需迭代集计算，并直接表征截断闵可夫斯基级数的衰减率。此外，我们证明向量范数的选择作为设计参数能够加速收敛，从而使得鲁棒不变集合计算和基于管的模型预测控制中的时间范围选择更为紧凑。数值实验验证了所提界的确切性、可扩展性和实际相关性。 

---
# Enhancing UAV Search under Occlusion using Next Best View Planning 

**Title (ZH)**: 基于最佳视图规划的遮挡下无人机搜索增强 

**Authors**: Sigrid Helene Strand, Thomas Wiedemann, Bram Burczek, Dmitriy Shutin  

**Link**: [PDF](https://arxiv.org/pdf/2511.18353)  

**Abstract**: Search and rescue missions are often critical following sudden natural disasters or in high-risk environmental situations. The most challenging search and rescue missions involve difficult-to-access terrains, such as dense forests with high occlusion. Deploying unmanned aerial vehicles for exploration can significantly enhance search effectiveness, facilitate access to challenging environments, and reduce search time. However, in dense forests, the effectiveness of unmanned aerial vehicles depends on their ability to capture clear views of the ground, necessitating a robust search strategy to optimize camera positioning and perspective. This work presents an optimized planning strategy and an efficient algorithm for the next best view problem in occluded environments. Two novel optimization heuristics, a geometry heuristic, and a visibility heuristic, are proposed to enhance search performance by selecting optimal camera viewpoints. Comparative evaluations in both simulated and real-world settings reveal that the visibility heuristic achieves greater performance, identifying over 90% of hidden objects in simulated forests and offering 10% better detection rates than the geometry heuristic. Additionally, real-world experiments demonstrate that the visibility heuristic provides better coverage under the canopy, highlighting its potential for improving search and rescue missions in occluded environments. 

**Abstract (ZH)**: 在突然自然灾害或高风险环境情况下，搜索救援任务往往至关重要。最具有挑战性的搜索救援任务涉及难以到达的地形，如茂密的森林中有较高的遮挡。部署无人飞行器进行勘探可以显著增强搜索效果，便于进入挑战性环境，并减少搜索时间。然而，在茂密的森林中，无人飞行器的效果依赖于其捕捉清晰地面视图的能力，需要一种 robust 的搜索策略来优化相机的位置和视角。本文提出了一种优化的规划策略和一种高效的 occluded 环境下的下一个最佳视角问题算法。通过提出几何启发式和可见性启发式两种新型优化启发式方法，以选择最优的相机视角来增强搜索性能。在模拟和实际环境中的比较评估表明，可见性启发式方法取得了更好的性能，在模拟森林中发现超过 90% 的隐藏物体，并且检测率比几何启发式方法高出 10%。此外，实际实验表明，可见性启发式方法在树冠下的覆盖性能更好，展示了其在 occluded 环境中的搜索救援任务中的潜在改进效果。 

---
# Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video 

**Title (ZH)**: 从视频学习具有视觉可解释性的振子网络以用于软连续体机器人 

**Authors**: Henrik Krauss, Johann Licher, Naoya Takeishi, Annika Raatz, Takehisa Yairi  

**Link**: [PDF](https://arxiv.org/pdf/2511.18322)  

**Abstract**: Data-driven learning of soft continuum robot (SCR) dynamics from high-dimensional observations offers flexibility but often lacks physical interpretability, while model-based approaches require prior knowledge and can be computationally expensive. We bridge this gap by introducing (1) the Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds. (2) By coupling these attention maps to 2D oscillator networks, we enable direct on-image visualization of learned dynamics (masses, stiffness, and forces) without prior knowledge. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy: 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on the two-segment robot. The learned oscillator network autonomously discovers a chain structure of oscillators. Unlike standard methods, ABCD models enable smooth latent space extrapolation beyond training data. This fully data-driven approach yields compact, physically interpretable models suitable for control applications. 

**Abstract (ZH)**: 基于注意力广播解码器的软连续机器人动态数据驱动学习：结合振子网络实现直接图像可视化 

---
# MicCheck: Repurposing Off-the-Shelf Pin Microphones for Easy and Low-Cost Contact Sensing 

**Title (ZH)**: MicCheck: 转换即用型Pin微音器进行简易低成本接触感知 

**Authors**: Steven Oh, Tai Inui, Magdeline Kuan, Jia-Yeu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.18299)  

**Abstract**: Robotic manipulation tasks are contact-rich, yet most imitation learning (IL) approaches rely primarily on vision, which struggles to capture stiffness, roughness, slip, and other fine interaction cues. Tactile signals can address this gap, but existing sensors often require expensive, delicate, or integration-heavy hardware. In this work, we introduce MicCheck, a plug-and-play acoustic sensing approach that repurposes an off-the-shelf Bluetooth pin microphone as a low-cost contact sensor. The microphone clips into a 3D-printed gripper insert and streams audio via a standard USB receiver, requiring no custom electronics or drivers. Despite its simplicity, the microphone provides signals informative enough for both perception and control. In material classification, it achieves 92.9% accuracy on a 10-class benchmark across four interaction types (tap, knock, slow press, drag). For manipulation, integrating pin microphone into an IL pipeline with open source hardware improves the success rate on picking and pouring task from 0.40 to 0.80 and enables reliable execution of contact-rich skills such as unplugging and sound-based sorting. Compared with high-resolution tactile sensors, pin microphones trade spatial detail for cost and ease of integration, offering a practical pathway for deploying acoustic contact sensing in low-cost robot setups. 

**Abstract (ZH)**: 基于声学感知的MicCheck：一种低成本即插即用手动传感器 

---
# AIA-UltraNeRF:Acoustic-Impedance-Aware Neural Radiance Field with Hash Encodings for Robotic Ultrasound Reconstruction and Localization 

**Title (ZH)**: 基于声阻抗意识哈希编码的神经辐射场：用于机器人超声重建与定位的AIA-UltraNeRF 

**Authors**: Shuai Zhang, Jingsong Mu, Cancan Zhao, Leiqi Tian, Zhijun Xing, Bo Ouyang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.18293)  

**Abstract**: Neural radiance field (NeRF) is a promising approach for reconstruction and new view synthesis. However, previous NeRF-based reconstruction methods overlook the critical role of acoustic impedance in ultrasound imaging. Localization methods face challenges related to local minima due to the selection of initial poses. In this study, we design a robotic ultrasound system (RUSS) with an acoustic-impedance-aware ultrasound NeRF (AIA-UltraNeRF) to decouple the scanning and diagnostic processes. Specifically, AIA-UltraNeRF models a continuous function of hash-encoded spatial coordinates for the 3D ultrasound map, allowing for the storage of acoustic impedance without dense sampling. This approach accelerates both reconstruction and inference speeds. We then propose a dual-supervised network that leverages teacher and student models to hash-encode the rendered ultrasound images from the reconstructed map. AIA-UltraNeRF retrieves the most similar hash values without the need to render images again, providing an offline initial image position for localization. Moreover, we develop a RUSS with a spherical remote center of motion mechanism to hold the probe, implementing operator-independent scanning modes that separate image acquisition from diagnostic workflows. Experimental results on a phantom and human subjects demonstrate the effectiveness of acoustic impedance in implicitly characterizing the color of ultrasound images. AIAUltraNeRF achieves both reconstruction and localization with inference speeds that are 9.9 faster than those of vanilla NeRF. 

**Abstract (ZH)**: 基于声阻抗aware的神经辐射场在机器人超声系统中的应用：AIA-UltraNeRF 

---
# Skypilot: Fine-Tuning LLM with Physical Grounding for AAV Coverage Search 

**Title (ZH)**: Skypilot: 基于物理关联的大型语言模型微调以实现AAV覆盖搜索 

**Authors**: Zhongkai Chen, Yihao Sun, Chao Yan, Han Zhou, Xiaojia Xiang, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18270)  

**Abstract**: Autonomous aerial vehicles (AAVs) have played a pivotal role in coverage operations and search missions. Recent advances in large language models (LLMs) offer promising opportunities to augment AAV intelligence. These advances help address complex challenges like area coverage optimization, dynamic path planning, and adaptive decision-making. However, the absence of physical grounding in LLMs leads to hallucination and reproducibility problems in spatial reasoning and decision-making. To tackle these issues, we present Skypilot, an LLM-enhanced two-stage framework that grounds language models in physical reality by integrating monte carlo tree search (MCTS). In the first stage, we introduce a diversified action space that encompasses generate, regenerate, fine-tune, and evaluate operations, coupled with physics-informed reward functions to ensure trajectory feasibility. In the second stage, we fine-tune Qwen3-4B on 23,000 MCTS-generated samples, achieving substantial inference acceleration while maintaining solution quality. Extensive numerical simulations and real-world flight experiments validate the efficiency and superiority of our proposed approach. Detailed information and experimental results are accessible at this https URL. 

**Abstract (ZH)**: 自主飞行器（AAVs）在覆盖操作和搜索任务中起到了关键作用。近年来，大型语言模型（LLMs）的进步为增强AAVs的智能提供了 promising 的机会。这些进步有助于解决区域覆盖优化、动态路径规划和自适应决策等复杂挑战。然而，LLMs缺乏物理基线，导致在空间推理和决策中出现幻觉和可重复性问题。为了应对这些挑战，我们提出了一种名为Skypilot的LLMs增强两阶段框架，通过集成蒙特卡洛树搜索（MCTS）将语言模型与物理现实相连接。在第一阶段，我们引入了一个多元化的动作空间，包括生成、再生、微调和评估操作，并结合物理指导的奖励函数以确保轨迹的可行性。在第二阶段，我们在23,000个MCTS生成的样本上对Qwen3-4B进行了微调，实现了显著的推理加速，同时保持了解决方案的质量。广泛的数值仿真和实地飞行实验验证了我们提出方法的有效性和优越性。详细信息和实验结果可在以下链接获取：this https URL。 

---
# Dreaming Falcon: Physics-Informed Model-Based Reinforcement Learning for Quadcopters 

**Title (ZH)**: 梦鹰计划：基于物理信息的模型驱动强化学习在四轴飞行器上的应用 

**Authors**: Eashan Vytla, Bhavanishankar Kalavakolanu, Andrew Perrault, Matthew McCrink  

**Link**: [PDF](https://arxiv.org/pdf/2511.18243)  

**Abstract**: Current control algorithms for aerial robots struggle with robustness in dynamic environments and adverse conditions. Model-based reinforcement learning (RL) has shown strong potential in handling these challenges while remaining sample-efficient. Additionally, Dreamer has demonstrated that online model-based RL can be achieved using a recurrent world model trained on replay buffer data. However, applying Dreamer to aerial systems has been quite challenging due to its sample inefficiency and poor generalization of dynamics models. Our work explores a physics-informed approach to world model learning and improves policy performance. The world model treats the quadcopter as a free-body system and predicts the net forces and moments acting on it, which are then passed through a 6-DOF Runge-Kutta integrator (RK4) to predict future state rollouts. In this paper, we compare this physics-informed method to a standard RNN-based world model. Although both models perform well on the training data, we observed that they fail to generalize to new trajectories, leading to rapid divergence in state rollouts, preventing policy convergence. 

**Abstract (ZH)**: 基于物理约束的世界模型学习方法在旋翼无人机上的应用及性能改进 

---
# APULSE: A Scalable Hybrid Algorithm for the RCSPP on Large-Scale Dense Graphs 

**Title (ZH)**: APULSE：大规模密集图上RCSPP的可扩展混合算法 

**Authors**: Nuno Soares, António Grilo  

**Link**: [PDF](https://arxiv.org/pdf/2511.18236)  

**Abstract**: The resource-constrained shortest path problem (RCSPP) is a fundamental NP-hard optimization challenge with broad applications, from network routing to autonomous navigation. This problem involves finding a path that minimizes a primary cost subject to a budget on a secondary resource. While various RCSPP solvers exist, they often face critical scalability limitations when applied to the large, dense graphs characteristic of complex, real-world scenarios, making them impractical for time-critical planning. This challenge is particularly acute in domains like mission planning for unmanned ground vehicles (UGVs), which demand solutions on large-scale terrain graphs. This paper introduces APULSE, a hybrid label-setting algorithm designed to efficiently solve the RCSPP on such challenging graphs. APULSE integrates a best-first search guided by an A* heuristic with aggressive, Pulse-style pruning mechanisms and a time-bucketing strategy for effective state-space reduction. A computational study, using a large-scale UGV planning scenario, benchmarks APULSE against state-of-the-art algorithms. The results demonstrate that APULSE consistently finds near-optimal solutions while being orders of magnitude faster and more robust, particularly on large problem instances where competing methods fail. This superior scalability establishes APULSE as an effective solution for RCSPP in complex, large-scale environments, enabling capabilities such as interactive decision support and dynamic replanning. 

**Abstract (ZH)**: 受资源约束的最短路径问题（RCSPP）是一个基本的NP-hard优化挑战，具有广泛的应用，从网络路由到自主导航。本文介绍了一种名为APULSE的混合标签设定算法，旨在高效解决此类具有挑战性的大型密集图上的RCSPP问题。APULSE结合了由A*启发式引导的最佳首先搜索和Pulse风格的严厉剪枝机制以及时间分桶策略，以有效减少状态空间。通过一个大规模无人地面车辆（UGV）规划场景的计算研究，APULSE与最先进的算法进行了基准测试。结果表明，APULSE在找到近最优解的同时，比竞品快得多且更可靠，尤其是在大型问题实例上表现尤为突出。这种卓越的可扩展性使APULSE成为复杂大尺度环境中解决RCSPP的有效方案，从而支持交互式决策支持和动态重规划等功能。 

---
# AFT: Appearance-Based Feature Tracking for Markerless and Training-Free Shape Reconstruction of Soft Robots 

**Title (ZH)**: 基于外观的特征跟踪软机器人无标记且无需训练的形状重构 

**Authors**: Shangyuan Yuan, Preston Fairchild, Yu Mei, Xinyu Zhou, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.18215)  

**Abstract**: Accurate shape reconstruction is essential for precise control and reliable operation of soft robots. Compared to sensor-based approaches, vision-based methods offer advantages in cost, simplicity, and ease of deployment. However, existing vision-based methods often rely on complex camera setups, specific backgrounds, or large-scale training datasets, limiting their practicality in real-world scenarios. In this work, we propose a vision-based, markerless, and training-free framework for soft robot shape reconstruction that directly leverages the robot's natural surface appearance. These surface features act as implicit visual markers, enabling a hierarchical matching strategy that decouples local partition alignment from global kinematic optimization. Requiring only an initial 3D reconstruction and kinematic alignment, our method achieves real-time shape tracking across diverse environments while maintaining robustness to occlusions and variations in camera viewpoints. Experimental validation on a continuum soft robot demonstrates an average tip error of 2.6% during real-time operation, as well as stable performance in practical closed-loop control tasks. These results highlight the potential of the proposed approach for reliable, low-cost deployment in dynamic real-world settings. 

**Abstract (ZH)**: 基于视觉的无标记无需训练软机器人形状重建框架 

---
# SkillWrapper: Generative Predicate Invention for Skill Abstraction 

**Title (ZH)**: SkillWrapper: 生成谓词发明以实现技能抽象 

**Authors**: Ziyi Yang, Benned Hedegaard, Ahmed Jaafar, Yichen Wei, Skye Thompson, Shreyas S. Raman, Haotian Fu, Stefanie Tellex, George Konidaris, David Paulius, Naman Shah  

**Link**: [PDF](https://arxiv.org/pdf/2511.18203)  

**Abstract**: Generalizing from individual skill executions to solving long-horizon tasks remains a core challenge in building autonomous agents. A promising direction is learning high-level, symbolic abstractions of the low-level skills of the agents, enabling reasoning and planning independent of the low-level state space. Among possible high-level representations, object-centric skill abstraction with symbolic predicates has been proven to be efficient because of its compatibility with domain-independent planners. Recent advances in foundation models have made it possible to generate symbolic predicates that operate on raw sensory inputs, a process we call generative predicate invention, to facilitate downstream abstraction learning. However, it remains unclear which formal properties the learned representations must satisfy, and how they can be learned to guarantee these properties. In this paper, we address both questions by presenting a formal theory of generative predicate invention for skill abstraction, resulting in symbolic operators that can be used for provably sound and complete planning. Within this framework, we propose SkillWrapper, a method that leverages foundation models to actively collect robot data and learn human-interpretable, plannable representations of black-box skills, using only RGB image observations. Our extensive empirical evaluation in simulation and on real robots shows that SkillWrapper learns abstract representations that enable solving unseen, long-horizon tasks in the real world with black-box skills. 

**Abstract (ZH)**: 基于生成式谓词发明的技能抽象形式理论 

---
# Off-Road Navigation via Implicit Neural Representation of Terrain Traversability 

**Title (ZH)**: 离路导航：基于地形通过性的隐式神经表示 

**Authors**: Yixuan Jia, Qingyuan Li, Jonathan P. How  

**Link**: [PDF](https://arxiv.org/pdf/2511.18183)  

**Abstract**: Autonomous off-road navigation requires robots to estimate terrain traversability from onboard sensors and plan accordingly. Conventional approaches typically rely on sampling-based planners such as MPPI to generate short-term control actions that aim to minimize traversal time and risk measures derived from the traversability estimates. These planners can react quickly but optimize only over a short look-ahead window, limiting their ability to reason about the full path geometry, which is important for navigating in challenging off-road environments. Moreover, they lack the ability to adjust speed based on the terrain bumpiness, which is important for smooth navigation on challenging terrains. In this paper, we introduce TRAIL (Traversability with an Implicit Learned Representation), an off-road navigation framework that leverages an implicit neural representation to continuously parameterize terrain properties. This representation yields spatial gradients that enable integration with a novel gradient-based trajectory optimization method that adapts the path geometry and speed profile based on terrain traversability. 

**Abstract (ZH)**: 自主离路面导航要求机器人利用车载传感器估计地形越障能力并据此规划路径。传统方法通常依赖于基于采样的规划器如MPPI来生成旨在最小化越障时间和风险度量的短期控制动作。这些规划器能够快速响应，但仅优化短暂的视野窗口内的路径，限制了它们对完整路径几何形状进行推理的能力，这对于在具有挑战性的离路面环境中导航至关重要。此外，它们无法根据地形起伏调整速度，这在应对具有挑战性的地形时对于平滑导航是至关重要的。本文提出了一种名为TRAIL（地形越障的隐式学习表示）的离路面导航框架，该框架利用隐式神经表示连续参数化地形属性。这种表示生成的空间梯度使得能够与一种基于梯度的轨迹优化方法结合，该方法可根据地形越障能力适应路径几何形状和速度轮廓。 

---
# Time-aware Motion Planning in Dynamic Environments with Conformal Prediction 

**Title (ZH)**: 动态环境中基于齐性预测的时间感知运动规划 

**Authors**: Kaier Liang, Licheng Luo, Yixuan Wang, Mingyu Cai, Cristian Ioan Vasile  

**Link**: [PDF](https://arxiv.org/pdf/2511.18170)  

**Abstract**: Safe navigation in dynamic environments remains challenging due to uncertain obstacle behaviors and the lack of formal prediction guarantees. We propose two motion planning frameworks that leverage conformal prediction (CP): a global planner that integrates Safe Interval Path Planning (SIPP) for uncertainty-aware trajectory generation, and a local planner that performs online reactive planning. The global planner offers distribution-free safety guarantees for long-horizon navigation, while the local planner mitigates inaccuracies in obstacle trajectory predictions through adaptive CP, enabling robust and responsive motion in dynamic environments. To further enhance trajectory feasibility, we introduce an adaptive quantile mechanism in the CP-based uncertainty quantification. Instead of using a fixed confidence level, the quantile is automatically tuned to the optimal value that preserves trajectory feasibility, allowing the planner to adaptively tighten safety margins in regions with higher uncertainty. We validate the proposed framework through numerical experiments conducted in dynamic and cluttered environments. The project page is available at this https URL 

**Abstract (ZH)**: 在动态环境中的安全导航仍具有挑战性，因不确定性障碍行为和缺乏正式的预测保证。我们提出了两种利用符合性预测（CP）的运动规划框架：一个全局规划器，结合Safe Interval Path Planning (SIPP) 实现不确定性意识的轨迹生成；一个局部规划器，进行在线反应性规划。全局规划器为长期导航提供无分布的安全保证，而局部规划器通过自适应CP减轻障碍物轨迹预测的不准确性，从而在动态环境中实现稳健和响应迅速的运动。为进一步增强轨迹可行性，我们在基于CP的不确定性量化中引入了自适应分位数机制。分位数不是固定使用，而是自适应调整到保持轨迹可行性的最优值，使规划器能够适应性缩小安全性裕度，在高不确定性区域更加谨慎。我们通过在动态和拥挤环境中进行的数值实验验证了所提出的框架。项目页面参见此 [链接]。 

---
# A Coordinated Dual-Arm Framework for Delicate Snap-Fit Assemblies 

**Title (ZH)**: 协调双臂框架用于精细卡扣装配 

**Authors**: Shreyas Kumar, Barat S, Debojit Das, Yug Desai, Siddhi Jain, Rajesh Kumar, Harish J. Palanthandalam-Madapusi  

**Link**: [PDF](https://arxiv.org/pdf/2511.18153)  

**Abstract**: Delicate snap-fit assemblies, such as inserting a lens into an eye-wear frame or during electronics assembly, demand timely engagement detection and rapid force attenuation to prevent overshoot-induced component damage or assembly failure. We address these challenges with two key contributions. First, we introduce SnapNet, a lightweight neural network that detects snap-fit engagement from joint-velocity transients in real-time, showing that reliable detection can be achieved using proprioceptive signals without external sensors. Second, we present a dynamical-systems-based dual-arm coordination framework that integrates SnapNet driven detection with an event-triggered impedance modulation, enabling accurate alignment and compliant insertion during delicate snap-fit assemblies. Experiments across diverse geometries on a heterogeneous bimanual platform demonstrate high detection accuracy (over 96% recall) and up to a 30% reduction in peak impact forces compared to standard impedance control. 

**Abstract (ZH)**: 精确的卡扣组装，如将镜片插入眼镜框或电子组装过程中，需要实现及时的配合检测和快速力衰减，以防止因超调引起的组件损坏或组装失败。我们通过两项关键贡献应对这些挑战。首先，我们引入了SnapNet，这是一种轻量级神经网络，可在实时检测来自关节速度瞬变的卡扣配合情况，表明可以使用本体感受信号实现可靠的检测，无需外部传感器。其次，我们提出了基于动态系统的双臂协调框架，该框架将SnapNet驱动的检测与事件触发的阻抗调节相结合，实现了精密卡扣组装过程中的准确对齐和柔顺插入。在异构双臂平台上对多种几何形状进行的实验表明，该方法的检测准确率超过96%，且峰值冲击力比标准阻抗控制降低了30%。 

---
# Observer Actor: Active Vision Imitation Learning with Sparse View Gaussian Splatting 

**Title (ZH)**: 观察者行为者：基于稀疏视图高斯点云的主动视觉imitation学习 

**Authors**: Yilong Wang, Cheng Qian, Ruomeng Fan, Edward Johns  

**Link**: [PDF](https://arxiv.org/pdf/2511.18140)  

**Abstract**: We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at this https URL. 

**Abstract (ZH)**: 我们提出了一种新型的主动视觉imitation learning框架Observer Actor (ObAct)，其中观察者移动到有利于actor的最优视觉观察位置。我们研究了安装有腕部摄像头的双臂机器人系统上的ObAct。测试时，ObAct动态分配观察者和执行者角色：观察者手臂从三张图片构建3D高斯斑图表示（3DGS），虚拟探索以找到最优相机姿态，然后移动到该姿态；执行者手臂则使用观察者采集的观察执行策略。这种表达方式增强了策略观察中物体和夹持器的清晰度和可见性。因此，我们在与无遮挡训练分布更为接近的观察中训练了双足足迹策略，从而提高了策略的鲁棒性。我们使用两种现有的imitation learning方法——轨迹传输和行为克隆——研究了这种表达方式。实验表明，与静态摄像头设置相比，ObAct显著表现更优：在无遮挡情况下轨迹传输提高了145%，有遮挡情况下提高了233%；行为克隆分别提高了75%和143%。相关视频可在此网址获取。 

---
# EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation 

**Title (ZH)**: EchoVLA：协同声明记忆驱动的机器人视觉-语言-行动模型及其在移动操作中的应用 

**Authors**: Min Lin, Xiwen Liang, Bingqian Lin, Liu Jingzhi, Zijian Jiao, Kehan Li, Yuhan Ma, Yuecheng Liu, Shen Zhao, Yuzheng Zhuang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18112)  

**Abstract**: Recent progress in Vision-Language-Action (VLA) models has enabled embodied agents to interpret multimodal instructions and perform complex tasks. However, existing VLAs are mostly confined to short-horizon, table-top manipulation, lacking the memory and reasoning capability required for long-horizon mobile manipulation, where agents must coordinate navigation and manipulation under changing spatial contexts. In this work, we present EchoVLA, a memory-aware VLA model for long-horizon mobile manipulation. EchoVLA incorporates a synergistic declarative memory inspired by the human brain, consisting of a scene memory that maintains a collection of spatial-semantic maps and an episodic memory that stores task-level experiences with multimodal contextual features. During both training and inference, the two memories are individually stored, updated, and retrieved based on current observations, task history, and instructions, and their retrieved representations are fused via coarse- and fine-grained attention to guide mobile-arm diffusion policies. To support large-scale training and evaluation, we further introduce MoMani, an automated benchmark that generates expert-level long-horizon trajectories through multimodal large language model (MLLM)-guided planning and feedback-driven refinement, supplemented with real-robot demonstrations. Experiments in simulated and real-world settings show that EchoVLA improves long-horizon performance, reaching 0.52 SR on manipulation/navigation and 0.31 on mobile manipulation, exceeding $\pi_{0.5}$ by +0.08 and +0.11. 

**Abstract (ZH)**: Recent进展在视觉-语言-行动(VLA)模型方面使得具身代理能够解释多模态指令并执行复杂任务。然而，现有的VLAs主要局限于短时 horizon桌面操作，缺乏进行长时 horizon移动操作所需的记忆和推理能力，其中代理需要在不断变化的空间背景下协调导航和操作。在此工作中，我们提出了EchoVLA，一种面向长时 horizon移动操作的记忆感知VLA模型。EchoVLA结合了一种受人脑启发的协同声明性记忆，包括用于维持空间语义地图集合的场景记忆，以及用于存储具有多模态上下文特征的任务级经历的事件记忆。在训练和推理过程中，两个记忆分别存储、更新和基于当前观察、任务历史和指令检索，他们的检索表示通过粗粒度和细粒度注意机制融合以指导移动臂弥散策略。为了支持大规模训练与评估，我们进一步引入了MoMani，一种通过基于多模态大型语言模型(MLLM)规划与反馈驱动精炼生成专家级长时 horizon轨迹的自动基准，并辅以真实机器人演示。在模拟和真实环境中的实验表明，EchoVLA提高了长时 horizon表现，分别在操作/导航和移动操作中达到0.52 SR和0.31，超过$\pi_{0.5}$分别0.08和0.11。 

---
# A Unified Multi-Dynamics Framework for Perception-Oriented Modeling in Tendon-Driven Continuum Robots 

**Title (ZH)**: 基于肌腱驱动连续机器人感知导向建模的统一多动力学框架 

**Authors**: Ibrahim Alsarraj, Yuhao Wang, Abdalla Swikir, Cesare Stefanini, Dezhen Song, Zhanchi Wang, Ke Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18088)  

**Abstract**: Tendon-driven continuum robots offer intrinsically safe and contact-rich interactions owing to their kinematic redundancy and structural compliance. However, their perception often depends on external sensors, which increase hardware complexity and limit scalability. This work introduces a unified multi-dynamics modeling framework for tendon-driven continuum robotic systems, exemplified by a spiral-inspired robot named Spirob. The framework integrates motor electrical dynamics, motor-winch dynamics, and continuum robot dynamics into a coherent system model. Within this framework, motor signals such as current and angular displacement are modeled to expose the electromechanical signatures of external interactions, enabling perception grounded in intrinsic dynamics. The model captures and validates key physical behaviors of the real system, including actuation hysteresis and self-contact at motion limits. Building on this foundation, the framework is applied to environmental interaction: first for passive contact detection, verified experimentally against simulation data; then for active contact sensing, where control and perception strategies from simulation are successfully applied to the real robot; and finally for object size estimation, where a policy learned in simulation is directly deployed on hardware. The results demonstrate that the proposed framework provides a physically grounded way to interpret interaction signatures from intrinsic motor signals in tendon-driven continuum robots. 

**Abstract (ZH)**: tendon驱动连续体机器人由于其运动冗余度和结构柔顺性，能够提供固有的安全性和丰富的接触交互。然而，它们的感知往往依赖于外部传感器，这增加了硬件复杂性并限制了可扩展性。本文提出了一种统一的多动力学建模框架，用于腱驱动连续体机器人系统，以螺旋启发式机器人Spirob为例。该框架将电机电气动力学、电机滑轮动力学和连续体机器人动力学整合到一个一致的系统模型中。在该框架内，通过建模电机信号如电流和角位移来暴露外部交互的电磁特征，从而实现基于内在动力学的感知。该模型捕获并验证了实际系统的关键物理行为，包括执行器滞回和运动极限处的自接触。在此基础上，该框架被应用于环境交互：首先进行被动接触检测，通过实验数据验证；然后进行主动接触感知，其中从仿真中开发的控制和感知策略被成功应用于实际机器人；最后进行物体大小估计，其中在仿真中学习的策略直接部署在硬件上。结果表明，提出的框架为解释腱驱动连续体机器人内在电机信号的交互特征提供了一种物理基础的方法。 

---
# Anti-Jamming based on Null-Steering Antennas and Intelligent UAV Swarm Behavior 

**Title (ZH)**: 基于空洞定向天线和智能无人机 swarm 行为的抗干扰技术 

**Authors**: Miguel Lourenço, António Grilo  

**Link**: [PDF](https://arxiv.org/pdf/2511.18086)  

**Abstract**: Unmanned Aerial Vehicle (UAV) swarms represent a key advancement in autonomous systems, enabling coordinated missions through inter-UAV communication. However, their reliance on wireless links makes them vulnerable to jamming, which can disrupt coordination and mission success. This work investigates whether a UAV swarm can effectively overcome jamming while maintaining communication and mission efficiency.
To address this, a unified optimization framework combining Genetic Algorithms (GA), Supervised Learning (SL), and Reinforcement Learning (RL) is proposed. The mission model, structured into epochs and timeslots, allows dynamic path planning, antenna orientation, and swarm formation while progressively enforcing collision rules. Null-steering antennas enhance resilience by directing antenna nulls toward interference sources.
Results show that the GA achieved stable, collision-free trajectories but with high computational cost. SL models replicated GA-based configurations but struggled to generalize under dynamic or constrained settings. RL, trained via Proximal Policy Optimization (PPO), demonstrated adaptability and real-time decision-making with consistent communication and lower computational demand. Additionally, the Adaptive Movement Model generalized UAV motion to arbitrary directions through a rotation-based mechanism, validating the scalability of the proposed system.
Overall, UAV swarms equipped with null-steering antennas and guided by intelligent optimization algorithms effectively mitigate jamming while maintaining communication stability, formation cohesion, and collision safety. The proposed framework establishes a unified, flexible, and reproducible basis for future research on resilient swarm communication systems. 

**Abstract (ZH)**: 无人机蜂群在自主系统中的突破性进展：克服干扰保持通信与任务效率的统一优化框架 

---
# Continually Evolving Skill Knowledge in Vision Language Action Model 

**Title (ZH)**: 持续演化技能知识的视觉语言行动模型 

**Authors**: Yuxuan Wu, Guangming Wang, Zhiheng Yang, Maoqing Yao, Brian Sheil, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18085)  

**Abstract**: Developing general robot intelligence in open environments requires continual skill learning. Recent Vision-Language-Action (VLA) models leverage massive pretraining data to support diverse manipulation tasks, but they still depend heavily on task-specific fine-tuning, revealing a lack of continual learning capability. Existing continual learning methods are also resource-intensive to scale to VLA models. We propose Stellar VLA, a knowledge-driven continual learning framework with two variants: T-Stellar, modeling task-centric knowledge space, and TS-Stellar, capturing hierarchical task-skill structure. Stellar VLA enables self-supervised knowledge evolution through joint learning of task latent representation and the knowledge space, reducing annotation needs. Knowledge-guided expert routing provide task specialization without extra network parameters, lowering training this http URL on the LIBERO benchmark and real-world tasks show over 50 percentage average improvement in final success rates relative to baselines. TS-Stellar further excels in complex action inference, and in-depth analyses verify effective knowledge retention and discovery. Our code will be released soon. 

**Abstract (ZH)**: 在开放环境下开发通用机器人智能需要持续技能学习。基于Vision-Language-Action (VLA)的模型利用大量预训练数据支持多种操作任务，但仍高度依赖于特定任务的微调，显示出缺乏持续学习能力。现有持续学习方法在扩展到VLA模型时也消耗大量资源。我们提出了Stellar VLA，这是一种知识驱动的持续学习框架，包含两种变体：T-Stellar，建模任务中心的知识空间，和TS-Stellar，捕获层级的任务技能结构。Stellar VLA 通过联合学习任务潜在表示和知识空间实现自我监督的知识进化，减少标注需求。知识引导的专家路由提供任务专业化而不增加网络参数，降低训练成本。在LIBERO基准和真实世界任务上，与基线相比，最终成功率平均提高超过50个百分点。TS-Stellar 在复杂动作推理方面表现尤为出色，深入分析验证了有效知识保留和发现。我们的代码即将发布。 

---
# Unobservable Subspace Evolution and Alignment for Consistent Visual-Inertial Navigation 

**Title (ZH)**: 不可观测子空间演化与对齐以实现一致的视觉-惯性导航 

**Authors**: Chungeng Tian, Fenghua He, Ning Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17992)  

**Abstract**: The inconsistency issue in the Visual-Inertial Navigation System (VINS) is a long-standing and fundamental challenge. While existing studies primarily attribute the inconsistency to observability mismatch, these analyses are often based on simplified theoretical formulations that consider only prediction and SLAM correction. Such formulations fail to cover the non-standard estimation steps, such as MSCKF correction and delayed initialization, which are critical for practical VINS estimators. Furthermore, the lack of a comprehensive understanding of how inconsistency dynamically emerges across estimation steps has hindered the development of precise and efficient solutions. As a result, current approaches often face a trade-off between estimator accuracy, consistency, and implementation complexity. To address these limitations, this paper proposes a novel analysis framework termed Unobservable Subspace Evolution (USE), which systematically characterizes how the unobservable subspace evolves throughout the entire estimation pipeline by explicitly tracking changes in its evaluation points. This perspective sheds new light on how individual estimation steps contribute to inconsistency. Our analysis reveals that observability misalignment induced by certain steps is the antecedent of observability mismatch. Guided by this insight, we propose a simple yet effective solution paradigm, Unobservable Subspace Alignment (USA), which eliminates inconsistency by selectively intervening only in those estimation steps that induce misalignment. We design two USA methods: transformation-based and re-evaluation-based, both offering accurate and computationally lightweight solutions. Extensive simulations and real-world experiments validate the effectiveness of the proposed methods. 

**Abstract (ZH)**: 视觉-惯性导航系统（VINS）中的不可观测性问题是一个长期存在的根本挑战。现有的研究主要将不可观测性归因于可观性匹配不良，但这些分析往往基于简化理论模型，仅考虑预测和SLAM校正步骤。这些模型未能涵盖如MSCKF校正和延迟初始化等非标准估计步骤，这些步骤对于实际的VINS估计器至关重要。此外，缺乏对不可观测性如何在整个估计过程中动态出现的全面理解，阻碍了精确高效解决方案的发展。因此，当前的方法往往在估计器精度、一致性和实现复杂性之间寻求权衡。为解决这些限制，本文提出了一种新颖的分析框架——不可观测子空间演化（USE），该框架系统地表征了在整个整个估计管道中不可观测子空间如何演变，并明确跟踪其评估点的变化情况。从这个角度看，我们揭示了个体估计步骤如何贡献于不可观测性。我们的分析表明，由某些步骤引起的可观性对齐不良是可观性匹配不良的先兆。根据这一见解，我们提出了一种简单而有效的解决方案范式——不可观测子空间对齐（USA），该方法通过在仅在那些引起对齐不良的估计步骤中选择性地介入来消除不可观测性。我们设计了两种USA方法：基于变换的方法和基于重新评估的方法，两者都提供了准确且计算高效的解决方案。广泛的仿真实验和实际实验验证了所提出方法的有效性。 

---
# RoboArmGS: High-Quality Robotic Arm Splatting via Bézier Curve Refinement 

**Title (ZH)**: RoboArmGS: 高质量的Bezier曲线精化机器人手臂绘制 

**Authors**: Hao Wang, Xiaobao Wei, Ying Li, Qingpo Wuwu, Dongli Wu, Jiajun Cao, Ming Lu, Wenzhao Zheng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17961)  

**Abstract**: Building high-quality digital assets of robotic arms is crucial yet challenging for the Real2Sim2Real pipeline. Current approaches naively bind static 3D Gaussians according to URDF links, forcing them to follow an URDF-rigged motion passively. However, real-world arm motion is noisy, and the idealized URDF-rigged motion cannot accurately model it, leading to severe rendering artifacts in 3D Gaussians. To address these challenges, we propose RoboArmGS, a novel hybrid representation that refines the URDF-rigged motion with learnable Bézier curves, enabling more accurate real-world motion modeling. To be more specific, we present a learnable Bézier Curve motion refiner that corrects per-joint residuals to address mismatches between real-world motion and URDF-rigged motion. RoboArmGS enables the learning of more accurate real-world motion while achieving a coherent binding of 3D Gaussians across arm parts. To support future research, we contribute a carefully collected dataset named RoboArm4D, which comprises several widely used robotic arms for evaluating the quality of building high-quality digital assets. We evaluate our approach on RoboArm4D, and RoboArmGS achieves state-of-the-art performance in real-world motion modeling and rendering quality. The code and dataset will be released. 

**Abstract (ZH)**: 构建高质量的机器人手臂数字资产对于Real2Sim2Real管道至关重要且具有挑战性。当前的方法根据URDF链接生硬地绑定静态3D高斯分布，使其被动地跟随URDF定义的运动。然而，真实世界的手臂运动是嘈杂的，理想化的URDF定义的运动无法准确建模，导致3D高斯分布中出现严重渲染伪影。为解决这些挑战，我们提出了RoboArmGS，这是一种新型混合表示方法，通过可学习的Bézier曲线细化URDF定义的运动，从而更准确地建模现实世界运动。具体而言，我们提出了一种可学习的Bézier曲线运动细化器，用于修正关节残差以解决现实世界运动与URDF定义运动之间的不匹配。RoboArmGS使得能够学习更准确的现实世界运动，并实现手臂各部分3D高斯分布的协调绑定。为了支持未来的研究，我们贡献了一个精心收集的数据集RoboArm4D，该数据集包含多个广泛使用的机器人手臂，用于评估构建高质量数字资产的质量。我们在RoboArm4D上评估了我们的方法，并发现RoboArmGS在现实世界运动建模和渲染质量方面达到了最先进的性能。我们将发布代码和数据集。 

---
# Switch-JustDance: Benchmarking Whole Body Motion Tracking Policies Using a Commercial Console Game 

**Title (ZH)**: Switch-JustDance：基于商业console游戏的整体身体动作追踪策略评估 

**Authors**: Jeonghwan Kim, Wontaek Kim, Yidan Lu, Jin Cheng, Fatemeh Zargarbashi, Zicheng Zeng, Zekun Qi, Zhiyang Dou, Nitish Sontakke, Donghoon Baek, Sehoon Ha, Tianyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.17925)  

**Abstract**: Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations. 

**Abstract (ZH)**: Recent Advances in Whole-Body Robot Control Have Enabled Humanoid and Legged Robots to Perform Increasingly Agile and Coordinated Motions. However, Standardized Benchmarks for Evaluating These Capabilities in Real-World Settings and in Direct Comparison to Humans Remain Scarce. We Present Switch-JustDance, a Low-Cost and Reproducible Benchmarking Pipeline That Leverages Motion-Sensing Console Games, Just Dance on the Nintendo Switch, to Evaluate Robot Whole-Body Control. 

---
# L1 Sample Flow for Efficient Visuomotor Learning 

**Title (ZH)**: L1 样本流用于高效视听运动学习 

**Authors**: Weixi Song, Zhetao Chen, Tao Xu, Xianchao Zeng, Xinyu Zhou, Lixin Yang, Donglin Wang, Cewu Lu, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.17898)  

**Abstract**: Denoising-based models, such as diffusion and flow matching, have been a critical component of robotic manipulation for their strong distribution-fitting and scaling capacity. Concurrently, several works have demonstrated that simple learning objectives, such as L1 regression, can achieve performance comparable to denoising-based methods on certain tasks, while offering faster convergence and inference. In this paper, we focus on how to combine the advantages of these two paradigms: retaining the ability of denoising models to capture multi-modal distributions and avoid mode collapse while achieving the efficiency of the L1 regression objective. To achieve this vision, we reformulate the original v-prediction flow matching and transform it into sample-prediction with the L1 training objective. We empirically show that the multi-modality can be expressed via a single ODE step. Thus, we propose \textbf{L1 Flow}, a two-step sampling schedule that generates a suboptimal action sequence via a single integration step and then reconstructs the precise action sequence through a single prediction. The proposed method largely retains the advantages of flow matching while reducing the iterative neural function evaluations to merely two and mitigating the potential performance degradation associated with direct sample regression. We evaluate our method with varying baselines and benchmarks, including 8 tasks in MimicGen, 5 tasks in RoboMimic \& PushT Bench, and one task in the real-world scenario. The results show the advantages of the proposed method with regard to training efficiency, inference speed, and overall performance. \href{this https URL}{Project Website.} 

**Abstract (ZH)**: 基于去噪的模型，如扩散和流匹配模型，因其强大的分布拟合能力和扩展能力而在机器人操纵中起到了关键作用。同时，有多项研究表明，简单的学习目标，如L1回归，可以在某些任务上达到与去噪基于方法相当的性能，同时提供更快的收敛速度和推理速度。在本文中，我们关注如何结合这两种范式的优点：保持去噪模型捕捉多模态分布和避免模式崩塌的能力，同时实现L1回归目标的有效性。为了实现这一愿景，我们将原始的v-预测流匹配重新 formulized，并转换为使用L1训练目标的样本预测。我们通过实验表明，多模态可以通过单一的微分方程步骤来表达。因此，我们提出了一种名为L1 Flow的两步采样计划，通过单一积分步骤生成次优动作序列，然后通过单一预测重构精确的动作序列。所提出的方法在保留流匹配优点的同时，将迭代神经函数评估减少到仅两次，并减轻了直接样本回归可能引起的性能下降。我们使用不同的基线和基准对方法进行了评估，包括MimicGen中的8个任务、RoboMimic & PushT Bench中的5个任务以及一个真实场景中的任务。结果显示，所提出的方法在训练效率、推理速度和总体性能方面具有优势。项目网站：<this https URL>。 

---
# MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots 

**Title (ZH)**: MobileVLA-R1：增强视觉-语言-行动能力的移动机器人算法 

**Authors**: Ting Huang, Dongjian Li, Rui Yang, Zeyu Zhang, Zida Yang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17889)  

**Abstract**: Grounding natural-language instructions into continuous control for quadruped robots remains a fundamental challenge in vision language action. Existing methods struggle to bridge high-level semantic reasoning and low-level actuation, leading to unstable grounding and weak generalization in the real world. To address these issues, we present MobileVLA-R1, a unified vision-language-action framework that enables explicit reasoning and continuous control for quadruped robots. We construct MobileVLA-CoT, a large-scale dataset of multi-granularity chain-of-thought (CoT) for embodied trajectories, providing structured reasoning supervision for alignment. Built upon this foundation, we introduce a two-stage training paradigm that combines supervised CoT alignment with GRPO reinforcement learning to enhance reasoning consistency, control stability, and long-horizon execution. Extensive evaluations on VLN and VLA tasks demonstrate superior performance over strong baselines, with approximately a 5% improvement. Real-world deployment on a quadruped robot validates robust performance in complex environments. Code: this https URL. Website: this https URL. 

**Abstract (ZH)**: 将自然语言指令 grounding 至四足机器人的连续控制在视觉语言行动领域仍然是一个基本挑战。现有方法难以桥接高层语义推理和低层执行，导致实际世界中的不稳定 grounding 和弱泛化能力。为了解决这些问题，我们提出了 MobileVLA-R1，一个统一的视觉语言行动框架，能够为四足机器人提供明确的推理和连续控制。我们构建了 MobileVLA-CoT，一个大规模的多粒度思维链（CoT）数据集，为实体轨迹提供结构化的推理监督。在此基础上，我们引入了一种两阶段训练范式，结合监督 CoT 对齐和 GRPO 强化学习，增强推理一致性、控制稳定性和长时执行能力。在 VLN 和 VLA 任务上的广泛评估证明了相对于强基线的优越性能，大约提高了 5%。在四足机器人上的实际部署验证了其在复杂环境中的鲁棒性能。代码：this https URL。网站：this https URL。 

---
# SM$^2$ITH: Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control 

**Title (ZH)**: SM$^2$ITH: 安全移动操作与基于任务分级双层模型预测控制的交互式人类预测 

**Authors**: Francesco D'Orazio, Sepehr Samavi, Xintong Du, Siqi Zhou, Giuseppe Oriolo, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2511.17798)  

**Abstract**: Mobile manipulators are designed to perform complex sequences of navigation and manipulation tasks in human-centered environments. While recent optimization-based methods such as Hierarchical Task Model Predictive Control (HTMPC) enable efficient multitask execution with strict task priorities, they have so far been applied mainly to static or structured scenarios. Extending these approaches to dynamic human-centered environments requires predictive models that capture how humans react to the actions of the robot. This work introduces Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control (SM$^2$ITH), a unified framework that combines HTMPC with interactive human motion prediction through bilevel optimization that jointly accounts for robot and human dynamics. The framework is validated on two different mobile manipulators, the Stretch 3 and the Ridgeback-UR10, across three experimental settings: (i) delivery tasks with different navigation and manipulation priorities, (ii) sequential pick-and-place tasks with different human motion prediction models, and (iii) interactions involving adversarial human behavior. Our results highlight how interactive prediction enables safe and efficient coordination, outperforming baselines that rely on weighted objectives or open-loop human models. 

**Abstract (ZH)**: 基于交互式人类预测的安全移动操作与任务层次 bilevel 模型预测控制（SM$^2$ITH） 

---
# SAFE-SMART: Safety Analysis and Formal Evaluation using STL Metrics for Autonomous RoboTs 

**Title (ZH)**: SAFE-SMART：基于STL度量的安全分析与形式化评估自主机器人 

**Authors**: Kristy Sakano, Jianyu An, Dinesh Manocha, Huan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17781)  

**Abstract**: We present a novel, regulator-driven approach for post hoc safety evaluation of learning-based, black-box autonomous mobile robots, ensuring ongoing compliance with evolving, human-defined safety rules. In our iterative workflow, human safety requirements are translated by regulators into Signal Temporal Logic (STL) specifications. Rollout traces from the black-box model are externally verified for compliance, yielding quantitative safety metrics, Total Robustness Value (TRV) and Largest Robustness Value (LRV), which measure average and worst-case specification adherence. These metrics inform targeted retraining and iterative improvement by model designers. We apply our method across two different applications: a virtual driving scenario and an autonomous mobile robot navigating a complex environment, and observe statistically significant improvements across both scenarios. In the virtual driving scenario, we see a 177% increase in traces adhering to the simulation speed limit, a 1138% increase in traces minimizing off-road driving, and a 16% increase in traces successfully reaching the goal within the time limit. In the autonomous navigation scenario, there is a 300% increase in traces avoiding sharp turns, a 200% increase in traces reaching the goal within the time limit, and a 49% increase in traces minimizing time spent near obstacles. Finally, we validate our approach on a TurtleBot3 robot in the real world, and demonstrate improved obstacle navigation with safety buffers. 

**Abstract (ZH)**: 一种调节器驱动的后验安全评估方法：基于学习的黑盒自主移动机器人的持续合规性验证 

---
# See, Plan, Cut: MPC-Based Autonomous Volumetric Robotic Laser Surgery with OCT Guidance 

**Title (ZH)**: 看、计划、切割：基于MPC的自主体轮廓激光手术系统及OCT引导 

**Authors**: Ravi Prakash, Vincent Y. Wang, Arpit Mishra, Devi Yuliarti, Pei Zhong, Ryan P. McNabb, Patrick J. Codd, Leila J. Bridgeman  

**Link**: [PDF](https://arxiv.org/pdf/2511.17777)  

**Abstract**: Robotic laser systems offer the potential for sub-millimeter, non-contact, high-precision tissue resection, yet existing platforms lack volumetric planning and intraoperative feedback. We present RATS (Robot-Assisted Tissue Surgery), an intelligent opto-mechanical, optical coherence tomography (OCT)-guided robotic platform designed for autonomous volumetric soft tissue resection in surgical applications. RATS integrates macro-scale RGB-D imaging, micro-scale OCT, and a fiber-coupled surgical laser, calibrated through a novel multistage alignment pipeline that achieves OCT-to-laser calibration accuracy of 0.161+-0.031mm on tissue phantoms and ex vivo porcine tissue. A super-Gaussian laser-tissue interaction (LTI) model characterizes ablation crater morphology with an average RMSE of 0.231+-0.121mm, outperforming Gaussian baselines. A sampling-based model predictive control (MPC) framework operates directly on OCT voxel data to generate constraint-aware resection trajectories with closed-loop feedback, achieving 0.842mm RMSE and improving intersection-over-union agreement by 64.8% compared to feedforward execution. With OCT, RATS detects subsurface structures and modifies the planner's objective to preserve them, demonstrating clinical feasibility. 

**Abstract (ZH)**: 基于机器人的激光系统有望实现亚毫米级、无接触、高精度的组织切除，但现有平台缺乏体积规划和术中反馈。我们提出了RATS（机器人辅助组织手术）智能光学-机械平台，该平台结合光学相干断层成像（OCT）指导，设计用于手术应用中自主体积软组织切除。RATS集成了宏观RGB-D成像、微观OCT和光纤耦合手术激光，并通过一种新颖的多阶段对准管道校准，实现了在组织模型和离体猪组织上的OCT到激光校准精度为0.161±0.031mm。超高斯激光-组织相互作用（LTI）模型以平均RMSE为0.231±0.121mm描述消融坑形态，优于高斯基线。基于采样的模型预测控制（MPC）框架直接作用于OCT体素数据，生成约束感知的切除轨迹，并提供闭环反馈，实现了0.842mm RMSE，并将交并比一致性提高了64.8%。通过OCT，RATS检测到亚表面结构并修改规划目标以保留它们，展示了临床可行性。 

---
# Learning Diffusion Policies for Robotic Manipulation of Timber Joinery under Fabrication Uncertainty 

**Title (ZH)**: 学习在制造不确定性条件下进行木材接合操作的扩散策略 

**Authors**: Salma Mozaffari, Daniel Ruan, William van den Bogert, Nima Fazeli, Sigrid Adriaenssens, Arash Adel  

**Link**: [PDF](https://arxiv.org/pdf/2511.17774)  

**Abstract**: Construction uncertainties such as fabrication inaccuracies and material imperfections pose a significant challenge to contact-rich robotic manipulation by hindering precise and robust assembly. In this paper, we explore the performance and robustness of diffusion policy learning as a promising solution for contact-sensitive robotic assembly at construction scale, using timber mortise and tenon joints as a case study. A two-phase study is conducted: first, to evaluate policy performance and applicability; second, to assess robustness in handling fabrication uncertainties simulated as randomized perturbations to the mortise position. The best-performing policy achieved a total average success rate of 75% with perturbations up to 10 mm, including 100% success in unperturbed cases. The results demonstrate the potential of sensory-motor diffusion policies to generalize to a wide range of complex, contact-rich assembly tasks across construction and manufacturing, advancing robotic construction under uncertainty and contributing to safer, more efficient building practices. 

**Abstract (ZH)**: 构造不确定性如制造不准确和材料缺陷给接触密集型机器人操作带来了显著挑战，这妨碍了精确和鲁棒的装配。本文探讨了作为接触敏感装配问题潜在解决方案的扩散策略学习的性能和鲁棒性，以木材榫卯连接为例进行了研究。这项研究分为两个阶段：首先评估策略性能和适用性；其次评估在模拟榫位置随机扰动下处理制造不确定性时的鲁棒性。表现最佳的策略在扰动不超过10毫米的情况下实现了75%的总平均成功率，包括未扰动情况下的100%成功。研究结果展示了感觉运动扩散策略在不同构造和制造业复杂、接触密集型装配任务中的泛化潜力，推动了在不确定性下的机器人建造，并促进了更安全、更高效的建筑实践。 

---
# LEARN: Learning End-to-End Aerial Resource-Constrained Multi-Robot Navigation 

**Title (ZH)**: LEARN: 学习端到端受限资源多机器人航迹规划 

**Authors**: Darren Chiu, Zhehui Huang, Ruohai Ge, Gaurav S. Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2511.17765)  

**Abstract**: Nano-UAV teams offer great agility yet face severe navigation challenges due to constrained onboard sensing, communication, and computation. Existing approaches rely on high-resolution vision or compute-intensive planners, rendering them infeasible for these platforms. We introduce LEARN, a lightweight, two-stage safety-guided reinforcement learning (RL) framework for multi-UAV navigation in cluttered spaces. Our system combines low-resolution Time-of-Flight (ToF) sensors and a simple motion planner with a compact, attention-based RL policy. In simulation, LEARN outperforms two state-of-the-art planners by $10\%$ while using substantially fewer resources. We demonstrate LEARN's viability on six Crazyflie quadrotors, achieving fully onboard flight in diverse indoor and outdoor environments at speeds up to $2.0 m/s$ and traversing $0.2 m$ gaps. 

**Abstract (ZH)**: nano-UAV队列提供了巨大的机动性，但由于机载传感、通信和计算能力受限，面临严峻的导航挑战。现有方法依赖高分辨率视觉或计算密集型规划器，使得这些平台无法采用。我们提出LEARN，一种轻量级的两阶段安全引导强化学习(RL)框架，用于拥挤空间中的多UAV导航。该系统结合了低分辨率飞行时间（ToF）传感器和简单的运动规划器，以及一种紧凑的注意力基于的RL策略。在仿真中，LEARN在资源使用显著减少的情况下，性能比两个最先进的规划器高出10%。我们在六架Crazyflie四旋翼飞行器上展示了LEARN的可行性，在多种室内外环境中实现了最高2.0 m/s的速度飞行和跨越0.2 m的缝隙，全部在机载平台上完成。 

---
# Vision-Guided Optic Flow Navigation for Small Lunar Missions 

**Title (ZH)**: 视觉引导视网膜流动导航小型月球任务 

**Authors**: Sean Cowan, Pietro Fanti, Leon B. S. Williams, Chit Hong Yam, Kaneyasu Asakuma, Yuichiro Nada, Dario Izzo  

**Link**: [PDF](https://arxiv.org/pdf/2511.17720)  

**Abstract**: Private lunar missions are faced with the challenge of robust autonomous navigation while operating under stringent constraints on mass, power, and computational resources. This work proposes a motion-field inversion framework that uses optical flow and rangefinder-based depth estimation as a lightweight CPU-based solution for egomotion estimation during lunar descent. We extend classical optical flow formulations by integrating them with depth modeling strategies tailored to the geometry for lunar/planetary approach, descent, and landing, specifically, planar and spherical terrain approximations parameterized by a laser rangefinder. Motion field inversion is performed through a least-squares framework, using sparse optical flow features extracted via the pyramidal Lucas-Kanade algorithm. We verify our approach using synthetically generated lunar images over the challenging terrain of the lunar south pole, using CPU budgets compatible with small lunar landers. The results demonstrate accurate velocity estimation from approach to landing, with sub-10% error for complex terrain and on the order of 1% for more typical terrain, as well as performances suitable for real-time applications. This framework shows promise for enabling robust, lightweight on-board navigation for small lunar missions. 

**Abstract (ZH)**: 私有月球任务在严格的质量、功耗和计算资源约束下面临着 robust 自主导航的挑战。本文提出了一种基于光学流和激光测距仪深度估计的运动场反演框架，作为轻量级 CPU 基础解决方案，用于月球下降过程中自运动估计。我们通过将经典光学流公式与适应月球/星球进场、下降和着陆几何的深度建模策略相结合，特别是基于激光测距仪的平面和球面地形近似，扩展了经典的光学流公式。通过最小二乘框架进行运动场反演，使用通过 pyramidal Lucas-Kanade 算法提取的稀疏光学流特征。在月球南极复杂地形上使用与小型月球着陆器兼容的 CPU 预算验证了我们的方法，结果表明从进场到着陆的准确速度估计，复杂地形下的误差小于 10%，典型地形下的误差为 1%，并且适用于实时应用。该框架展示了为小型月球任务实现 robust 和轻量级的机载导航的潜力。 

---
# Robot joint characterisation and control using a magneto-optical rotary encoder 

**Title (ZH)**: 使用磁光旋转编码器的机器人关节特性和控制研究 

**Authors**: Yunlong Guo, John Canning, Zenon Chaczko, Gang-Ding Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.17608)  

**Abstract**: A robust and compact magneto-optical rotary encoder for the characterisation of robotic rotary joints is demonstrated. The system employs magnetic field-induced optical attenuation in a double-pass configuration using rotating nonuniform magnets around an optical circulator operating in reflection. The encoder tracks continuous 360° rotation with rotation sweep rates from {\nu} = 135 °/s to {\nu} = 370 °/s, and an angular resolution of {\Delta}{\theta} = 0.3°. This offers a low-cost and reliable alternative to conventional robot rotation encoders while maintaining competitive performance. 

**Abstract (ZH)**: 一种用于机器人旋转关节表征的稳健紧凑型磁光旋转编码器的示例性展示 

---
# Translating Cultural Choreography from Humanoid Forms to Robotic Arm 

**Title (ZH)**: 将文化舞动从类人形转化为机器人臂łowongan 

**Authors**: Chelsea-Xi Chen, Zhe Zhang, Aven-Le Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.17603)  

**Abstract**: Robotic arm choreography often reproduces trajectories while missing cultural semantics. This study examines whether symbolic posture transfer with joint space compatible notation can preserve semantic fidelity on a six-degree-of-freedom arm and remain portable across morphologies. We implement ROPERA, a three-stage pipeline for encoding culturally codified postures, composing symbolic sequences, and decoding to servo commands. A scene from Kunqu opera, \textit{The Peony Pavilion}, serves as the material for evaluation. The procedure includes corpus-based posture selection, symbolic scoring, direct joint angle execution, and a visual layer with light painting and costume-informed colors. Results indicate reproducible execution with intended timing and cultural legibility reported by experts and audiences. The study points to non-anthropocentric cultural preservation and portable authoring workflows. Future work will design dance-informed transition profiles, extend the notation to locomotion with haptic, musical, and spatial cues, and test portability across platforms. 

**Abstract (ZH)**: 机器人手臂编排往往复制轨迹而忽略文化语义。本研究探讨是否可以通过关节空间兼容的符号姿势转移方法，在六自由度手臂上保持语义准确性，并实现跨不同形态的移植。我们实现了一种三阶段流水线：ROPERA，用于编码文化编码的姿势、编排符号序列以及解码为伺服命令。昆曲场景《牡丹亭》用作评估材料。该过程包括基于语料库的姿势选择、符号评分、直接关节角度执行以及带有光绘和服饰启发色彩的视觉层。结果表明，执行具有预期的时间性和专家及观众的文化可读性。该研究指出了非人本主义的文化保存和可移植的作者工作流程。未来工作将设计舞蹈启发的过渡曲线、将符号扩展到包含触觉、音乐和空间线索的运动，并测试跨平台的可移植性。 

---
# Implicit Neural Field-Based Process Planning for Multi-Axis Manufacturing: Direct Control over Collision Avoidance and Toolpath Geometry 

**Title (ZH)**: 基于隐式神经场的多轴制造工艺规划：直接控制碰撞避免和刀轨几何 

**Authors**: Neelotpal Dutta, Tianyu Zhang, Tao Liu, Yongxue Chen, Charlie C.L. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17578)  

**Abstract**: Existing curved-layer-based process planning methods for multi-axis manufacturing address collisions only indirectly and generate toolpaths in a post-processing step, leaving toolpath geometry uncontrolled during optimization. We present an implicit neural field-based framework for multi-axis process planning that overcomes these limitations by embedding both layer generation and toolpath design within a single differentiable pipeline. Using sinusoidally activated neural networks to represent layers and toolpaths as implicit fields, our method enables direct evaluation of field values and derivatives at any spatial point, thereby allowing explicit collision avoidance and joint optimization of manufacturing layers and toolpaths. We further investigate how network hyperparameters and objective definitions influence singularity behavior and topology transitions, offering built-in mechanisms for regularization and stability control. The proposed approach is demonstrated on examples in both additive and subtractive manufacturing, validating its generality and effectiveness. 

**Abstract (ZH)**: 基于隐神经场的多轴制造工艺规划方法 

---
# AUTOSAR AP and ROS 2 Collaboration Framework 

**Title (ZH)**: AUTOSAR AP与ROS 2协作框架 

**Authors**: Ryudai Iwakami, Bo Peng, Hiroyuki Hanyu, Tasuku Ishigooka, Takuya Azumi  

**Link**: [PDF](https://arxiv.org/pdf/2511.17540)  

**Abstract**: The field of autonomous vehicle research is advancing rapidly, necessitating platforms that meet real-time performance, safety, and security requirements for practical deployment. AUTOSAR Adaptive Platform (AUTOSAR AP) is widely adopted in development to meet these criteria; however, licensing constraints and tool implementation challenges limit its use in research. Conversely, Robot Operating System 2 (ROS 2) is predominantly used in research within the autonomous driving domain, leading to a disparity between research and development platforms that hinders swift commercialization. This paper proposes a collaboration framework that enables AUTOSAR AP and ROS 2 to communicate with each other using a Data Distribution Service for Real-Time Systems (DDS). In contrast, AUTOSAR AP uses Scalable service-Oriented Middleware over IP (SOME/IP) for communication. The proposed framework bridges these protocol differences, ensuring seamless interaction between the two platforms. We validate the functionality and performance of our bridge converter through empirical analysis, demonstrating its efficiency in conversion time and ease of integration with ROS 2 tools. Furthermore, the availability of the proposed collaboration framework is improved by automatically generating a configuration file for the proposed bridge converter. 

**Abstract (ZH)**: 自主驾驶车辆领域正在迅速发展，需要能够满足实时性能、安全性和安全性要求的平台以便实际部署。AUTOSAR自适应平台（AUTOSAR AP）在开发中广泛采用以满足这些标准；然而，许可限制和工具实现挑战限制了其在研究中的应用。相比之下，机器人操作系统2（ROS 2）在自主驾驶领域主要被用于研究，导致研究平台与开发平台之间存在分歧，阻碍了快速商业化。本文提出了一种协作框架，该框架允许AUTOSAR AP和ROS 2通过实时系统数据分布服务（DDS）进行通信。相反，AUTOSAR AP使用基于IP的可扩展服务导向中间件（SOME/IP）进行通信。该提出的框架消除了这些协议差异，确保了两个平台之间的无缝交互。我们通过实证分析验证了桥接转换器的功能和性能，展示了其在转换时间和与ROS 2工具集成的效率。此外，通过自动生成桥接转换器的配置文件，提高了该协作框架的可用性。 

---
# Leveraging LLMs for reward function design in reinforcement learning control tasks 

**Title (ZH)**: 利用大规模语言模型设计强化学习控制任务中的奖励函数 

**Authors**: Franklin Cardenoso, Wouter Caarls  

**Link**: [PDF](https://arxiv.org/pdf/2511.19355)  

**Abstract**: The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability. 

**Abstract (ZH)**: 基于大语言模型的奖励函数优化框架LEARN-Opt：自动评估与选择 

---
# Three-Dimensional Anatomical Data Generation Based on Artificial Neural Networks 

**Title (ZH)**: 基于人工神经网络的三维解剖数据生成 

**Authors**: Ann-Sophia Müller, Moonkwang Jeong, Meng Zhang, Jiyuan Tian, Arkadiusz Miernik, Stefanie Speidel, Tian Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19198)  

**Abstract**: Surgical planning and training based on machine learning requires a large amount of 3D anatomical models reconstructed from medical imaging, which is currently one of the major bottlenecks. Obtaining these data from real patients and during surgery is very demanding, if even possible, due to legal, ethical, and technical challenges. It is especially difficult for soft tissue organs with poor imaging contrast, such as the prostate. To overcome these challenges, we present a novel workflow for automated 3D anatomical data generation using data obtained from physical organ models. We additionally use a 3D Generative Adversarial Network (GAN) to obtain a manifold of 3D models useful for other downstream machine learning tasks that rely on 3D data. We demonstrate our workflow using an artificial prostate model made of biomimetic hydrogels with imaging contrast in multiple zones. This is used to physically simulate endoscopic surgery. For evaluation and 3D data generation, we place it into a customized ultrasound scanner that records the prostate before and after the procedure. A neural network is trained to segment the recorded ultrasound images, which outperforms conventional, non-learning-based computer vision techniques in terms of intersection over union (IoU). Based on the segmentations, a 3D mesh model is reconstructed, and performance feedback is provided. 

**Abstract (ZH)**: 基于机器学习的手术规划与训练需要大量从医学影像重建的3D解剖模型，这是目前的主要瓶颈之一。由于法律、伦理和技术挑战，从真实患者和手术中获得这些数据非常困难，甚至不可能。对于反差差的软组织器官（如前列腺）更是如此。为克服这些挑战，我们提出了一种新的自动化3D解剖数据生成工作流，使用物理器官模型获得的数据。此外，我们使用3D生成对抗网络（GAN）获取适用于其他依赖3D数据的下游机器学习任务的数据流形。我们使用生物仿生水凝胶制成的人工前列腺模型，该模型具有多个区域的影像对比度，并用于物理模拟内镜手术。在评估和3D数据生成过程中，我们将模型置于定制超声扫描仪中，记录手术前后的前列腺影像。训练神经网络进行分割，其在交并比（IoU）方面优于传统的非学习计算机视觉技术。基于分割结果，重建3D网格模型，并提供性能反馈。 

---
# First-order Sobolev Reinforcement Learning 

**Title (ZH)**: 一阶Sobolev强化学习 

**Authors**: Fabian Schramm, Nicolas Perrin-Gilbert, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2511.19165)  

**Abstract**: We propose a refinement of temporal-difference learning that enforces first-order Bellman consistency: the learned value function is trained to match not only the Bellman targets in value but also their derivatives with respect to states and actions. By differentiating the Bellman backup through differentiable dynamics, we obtain analytically consistent gradient targets. Incorporating these into the critic objective using a Sobolev-type loss encourages the critic to align with both the value and local geometry of the target function. This first-order TD matching principle can be seamlessly integrated into existing algorithms, such as Q-learning or actor-critic methods (e.g., DDPG, SAC), potentially leading to faster critic convergence and more stable policy gradients without altering their overall structure. 

**Abstract (ZH)**: 我们提出了一种强化时差学习的方法，使其满足一阶贝尔曼一致性：学习的价值函数不仅需要匹配值的贝尔曼目标，还需匹配其对状态和动作的导数。通过可微动力学对贝尔曼备份进行微分，我们获得了分析上一致的梯度目标。将这些目标结合到评论者目标中使用Sobolev型损失，促使评论者与目标函数的值和局部几何相匹配。这一一阶TD匹配原则可以无缝集成到现有的算法（如Q学习或演员-评论者方法，例如DDPG、SAC）中，可能加速评论者的收敛并提供更稳定的压力梯度，而不改变其整体结构。 

---
# AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention 

**Title (ZH)**: AVA-VLA：通过主动视觉注意力提高视觉-语言-行动模型性能 

**Authors**: Lei Xiao, Jifeng Li, Juntao Gao, Feiyang Ye, Yan Jin, Jingjing Qian, Jing Zhang, Yong Wu, Xiaoyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18960)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability. 

**Abstract (ZH)**: 基于视觉-语言-动作的POMDP视角框架：AVA-VLA 

---
# GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction 

**Title (ZH)**: GContextFormer：一种考虑全局上下文的混合多头注意力方法及其标度加性聚合在多模态轨迹预测中的应用 

**Authors**: Yuzhi Chen, Yuanchang Xie, Lei Zhao, Pan Liu, Yajie Zou, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18874)  

**Abstract**: Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: this https URL. 

**Abstract (ZH)**: 基于全局上下文的变换器架构GContextFormer实现意图对齐的多模态轨迹预测 

---
# Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents 

**Title (ZH)**: 超越描述：细粒度动作的认知基准测试对具身智能体的应用 

**Authors**: Dayong Liu, Chao Xu, Weihong Chen, Suyu Zhang, Juncheng Wang, Jiankang Deng, Baigui Sun, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18685)  

**Abstract**: Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）作为在复杂物理环境中运行的实体代理的决策引擎显示出 promising 的结果。然而，现有的基准测试往往侧重于高层规划或空间推理，而实体物理交互所需的细粒度操作智能则未被充分探索。为解决这一问题，我们引入了CFG-Bench，一个旨在系统性评估这一关键能力的新基准测试。CFG-Bench 包含 1,368 个精挑细选的视频，配对有 19,562 个三模态问答对，针对四种认知能力进行目标定向：1）物理交互，2）时间因果关系，3）意图理解，4）评估判断。这些维度共同提供了一个系统框架，用于评估模型将视觉观察转化为可执行知识的能力，超越了仅仅表面的识别。通过对 CFG-Bench 的全面评估表明，领先的 MLLMs 在生成关于物理交互的详细指令方面挣扎，并在关于意图和评估的高级推理方面表现出巨大的局限性。此外，我们在数据上的监督微调（SFT）表明，教会 MLLMs 表述细粒度操作可以显著提高其在现有实体基准测试中的表现。我们的分析指出了这些局限性，并为开发更强大和更贴地气的实体代理提供了见解。 

---
# Connectivity-Preserving Multi-Agent Area Coverage via Optimal-Transport-Based Density-Driven Optimal Control (D2OC) 

**Title (ZH)**: 基于最优运输的密度驱动最优控制（D2OC）的连接保持多agent区域覆盖 

**Authors**: Kooktae Lee, Ethan Brook  

**Link**: [PDF](https://arxiv.org/pdf/2511.18579)  

**Abstract**: Multi-agent systems play a central role in area coverage tasks across search-and-rescue, environmental monitoring, and precision agriculture. Achieving non-uniform coverage, where spatial priorities vary across the domain, requires coordinating agents while respecting dynamic and communication constraints. Density-driven approaches can distribute agents according to a prescribed reference density, but existing methods do not ensure connectivity. This limitation often leads to communication loss, reduced coordination, and degraded coverage performance.
This letter introduces a connectivity-preserving extension of the Density-Driven Optimal Control (D2OC) framework. The coverage objective, defined using the Wasserstein distance between the agent distribution and the reference density, admits a convex quadratic program formulation. Communication constraints are incorporated through a smooth connectivity penalty, which maintains strict convexity, supports distributed implementation, and preserves inter-agent communication without imposing rigid formations.
Simulation studies show that the proposed method consistently maintains connectivity, improves convergence speed, and enhances non-uniform coverage quality compared with density-driven schemes that do not incorporate explicit connectivity considerations. 

**Abstract (ZH)**: 多智能体系统在搜救、环境监测和精准农业领域的区域覆盖任务中发挥着核心作用。实现非均匀覆盖，其中空间优先级在区域内变化，需要在尊重动态和通信约束的情况下协调智能体。基于密度的方法可以根据预设的参考密度分布智能体，但现有方法不保证连通性。这一限制通常会导致通信损失、协调减少和覆盖性能下降。

本文提出了一种保持连通性的Density-Driven Optimal Control (D2OC)框架的扩展。覆盖目标由智能体分布与参考密度之间的Wasserstein距离定义，可以表示为凸二次规划问题。通过一个光滑的连通性惩罚项将通信约束纳入其中，该惩罚项保持严格的凸性，支持分布式实现，并保持智能体间的通信而不强制形成固定的队形。

模拟研究显示，所提出的方法能够一致地保持连通性、提高收敛速度并增强非均匀覆盖质量，相较于未考虑显式连通性约束的基于密度的方法。 

---
# PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation 

**Title (ZH)**: PhysGS: 基于贝叶斯推断的高斯点云物理性质估计 

**Authors**: Samarth Chopra, Jing Liang, Gershom Seneviratne, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2511.18570)  

**Abstract**: Understanding physical properties such as friction, stiffness, hardness, and material composition is essential for enabling robots to interact safely and effectively with their surroundings. However, existing 3D reconstruction methods focus on geometry and appearance and cannot infer these underlying physical properties. We present PhysGS, a Bayesian-inferred extension of 3D Gaussian Splatting that estimates dense, per-point physical properties from visual cues and vision--language priors. We formulate property estimation as Bayesian inference over Gaussian splats, where material and property beliefs are iteratively refined as new observations arrive. PhysGS also models aleatoric and epistemic uncertainties, enabling uncertainty-aware object and scene interpretation. Across object-scale (ABO-500), indoor, and outdoor real-world datasets, PhysGS improves accuracy of the mass estimation by up to 22.8%, reduces Shore hardness error by up to 61.2%, and lowers kinetic friction error by up to 18.1% compared to deterministic baselines. Our results demonstrate that PhysGS unifies 3D reconstruction, uncertainty modeling, and physical reasoning in a single, spatially continuous framework for dense physical property estimation. Additional results are available at this https URL. 

**Abstract (ZH)**: 理解摩擦、刚度、硬度和材料组成等物理属性对于使机器人能够安全有效地与周围环境互动至关重要。然而，现有三维重建方法主要关注几何和外观，无法推断这些底层物理属性。我们提出了PhysGS，一个基于贝叶斯推断的三维高斯点扩展方法，能够从视觉线索和视觉-语言先验中估计密集的、每点的物理属性。我们将属性估计公式化为高斯点的贝叶斯推断过程，随着新观测数据的 arriving，物质和属性信念将逐步优化。PhysGS 还模型化了抽样不确定性和知识不确定性，使对象和场景解释具有不确定性意识。在对象尺度（ABO-500）、室内和室外真实世界数据集上，PhysGS 的质量估计准确度提高了最多 22.8%，邵氏硬度误差降低了最多 61.2%，动摩擦系数误差降低了最多 18.1%，优于确定性基线。我们的结果表明，PhysGS 将三维重建、不确定性建模和物理推理统一在一个连续的空间框架中，用于密集物理属性估计。更多结果请参阅此链接。 

---
# Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems 

**Title (ZH)**: 范畴对称深度学习：范畴对称神经网络与普遍逼近定理 

**Authors**: Yoshihiro Maruyama  

**Link**: [PDF](https://arxiv.org/pdf/2511.18417)  

**Abstract**: We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures, formulating linear and nonlinear layers in the categorical setup. We prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries. 

**Abstract (ZH)**: 我们开发了一种范畴齐变神经网络（CENNs）的理论，它统一了群/群丛齐变网络、偏序集/格齐变网络、图神经网络和层神经网络。齐变性被形式化为拓扑范畴中带有Radon测度的自然性，在范畴框架下定义了线性和非线性层。我们证明了在一般情况下齐变通用近似定理：有限深度的CENNs类在连续齐变变换的空间中稠密。我们为群/群丛、偏序集/格、图和细胞层神经网络实例化了该框架，系统地推导了它们的通用近似定理。因此，范畴齐变深度学习使得我们能够将齐变深度学习的视野扩展到超越群作用的范围，不仅包括几何对称性，还包括上下文和组合对称性。 

---
# scipy.spatial.transform: Differentiable Framework-Agnostic 3D Transformations in Python 

**Title (ZH)**: scipy.spatial.transform: 不依赖于框架的可微分3D变换在Python中 

**Authors**: Martin Schuck, Alexander von Rohr, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2511.18157)  

**Abstract**: Three-dimensional rigid-body transforms, i.e. rotations and translations, are central to modern differentiable machine learning pipelines in robotics, vision, and simulation. However, numerically robust and mathematically correct implementations, particularly on SO(3), are error-prone due to issues such as axis conventions, normalizations, composition consistency and subtle errors that only appear in edge cases. SciPy's this http URL module is a rigorously tested Python implementation. However, it historically only supported NumPy, limiting adoption in GPU-accelerated and autodiff-based workflows. We present a complete overhaul of SciPy's this http URL functionality that makes it compatible with any array library implementing the Python array API, including JAX, PyTorch, and CuPy. The revised implementation preserves the established SciPy interface while enabling GPU/TPU execution, JIT compilation, vectorized batching, and differentiation via native autodiff of the chosen backend. We demonstrate how this foundation supports differentiable scientific computing through two case studies: (i) scalability of 3D transforms and rotations and (ii) a JAX drone simulation that leverages SciPy's Rotation for accurate integration of rotational dynamics. Our contributions have been merged into SciPy main and will ship in the next release, providing a framework-agnostic, production-grade basis for 3D spatial math in differentiable systems and ML. 

**Abstract (ZH)**: 三维刚体变换，即旋转和平移，是现代机器人学、视觉和模拟中可微机器学习管道的核心。然而，特别是在SO(3)上，由于轴约定、规范化、组合一致性以及仅在边界情况下出现的细微错误，其数值稳健且数学正确的实现容易出错。SciPy的this http URL模块是一个严格测试的Python实现。然而，它历史上仅支持NumPy，限制了其在GPU加速和自动求导工作流中的采用。我们提供了一个全新的SciPy this http URL功能，使其与任何实现Python数组API的数组库兼容，包括JAX、PyTorch和CuPy。修订后的实现保留了Scipy现有的接口，同时允许GPU/TPU执行、JIT编译、向量化批量处理以及通过所选后端的本征自动求导进行求导。我们通过两个案例研究展示了这一基础如何支持差异化的科学计算：（i）3D变换和旋转的 scalability 以及（ii）一个利用SciPy旋转进行旋转动力学准确集成的JAX无人机模拟。我们的贡献已被合并到SciPy主分支，并将在下一个版本中发布，为差异化的系统和机器学习中的3D空间数学提供了一个框架无关的、企业级的基础。 

---
# ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models 

**Title (ZH)**: ActDistill: 一般动作引导的自我衍生distillation方法以提升高效视觉-语言-动作模型 

**Authors**: Wencheng Ye, Tianshi Wang, Lei Zhu, Fengling Li, Guoli Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18082)  

**Abstract**: Recent Vision-Language-Action (VLA) models have shown impressive flexibility and generalization, yet their deployment in robotic manipulation remains limited by heavy computational overhead and inference latency. In this work, we present ActDistill, a general action-guided self-derived distillation framework that transfers the action prediction capability of any existing VLA model to a lightweight counterpart. Unlike previous efficiency strategies that primarily emphasize vision-language correlations, ActDistill leverages action priors to guide knowledge transfer and model compression, achieving action-oriented efficiency for VLA models. Specifically, we employ a well-trained VLA model as the teacher and introduce a graph-structured encapsulation strategy to explicitly model the hierarchical evolution of action prediction. The student model, derived from the graph-encapsulated teacher, is further equipped with a dynamic router that adaptively selects computation paths based on action prediction demands, guided by hierarchical graph-informed supervision to ensure smooth and efficient evolution. During inference, graph-related auxiliary components are removed, allowing the student to execute only dynamically routed layers and predict high-precision actions with minimal computation and latency. Experiments on embodied benchmarks demonstrate that ActDistill achieves comparable or superior performance to full-scale VLA models while reducing computation by over 50% with up to 1.67 times speedup, thereby establishing a general paradigm toward efficient embodied intelligence. 

**Abstract (ZH)**: 基于动作引导的自衍生精简框架ActDistill：面向视觉-语言-动作模型的高效化探索 

---
# CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for Multimodal Scene Representation 

**Title (ZH)**: CUS-GS: 一种紧凑的统一结构高斯斑点表示框架用于多模态场景表示 

**Authors**: Yuhang Ming, Chenxin Fang, Xingyuan Yu, Fan Zhang, Weichen Dai, Wanzeng Kong, Guofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17904)  

**Abstract**: Recent advances in Gaussian Splatting based 3D scene representation have shown two major trends: semantics-oriented approaches that focus on high-level understanding but lack explicit 3D geometry modeling, and structure-oriented approaches that capture spatial structures yet provide limited semantic abstraction. To bridge this gap, we present CUS-GS, a compact unified structured Gaussian Splatting representation, which connects multimodal semantic features with structured 3D geometry. Specifically, we design a voxelized anchor structure that constructs a spatial scaffold, while extracting multimodal semantic features from a set of foundation models (e.g., CLIP, DINOv2, SEEM). Moreover, we introduce a multimodal latent feature allocation mechanism to unify appearance, geometry, and semantics across heterogeneous feature spaces, ensuring a consistent representation across multiple foundation models. Finally, we propose a feature-aware significance evaluation strategy to dynamically guide anchor growing and pruning, effectively removing redundant or invalid anchors while maintaining semantic integrity. Extensive experiments show that CUS-GS achieves competitive performance compared to state-of-the-art methods using as few as 6M parameters - an order of magnitude smaller than the closest rival at 35M - highlighting the excellent trade off between performance and model efficiency of the proposed framework. 

**Abstract (ZH)**: 基于Gaussian Splatting的三维场景表示recent进展表明了两大趋势：面向语义的方法注重高层次理解但缺乏显式的三维几何建模，面向结构的方法捕捉空间结构但提供有限的语义抽象。为弥合这一差距，我们提出了一种紧凑的统一结构化Gaussian Splatting表示CUS-GS，将多模态语义特征与结构化的3D几何联系起来。具体而言，我们设计了一种体素化锚结构来构建空间骨架，同时从一组基础模型（如CLIP、DINOv2、SEEM）中提取多模态语义特征。此外，我们引入了一种多模态潜在特征分配机制，以统一不同特征空间中的外观、几何和语义特征，确保在多个基础模型中保持一致的表示。最后，我们提出了一种基于特征的显著性评估策略，以动态指导锚点的生长和修剪，有效地移除冗余或无效的锚点，同时保持语义完整性。广泛实验表明，CUS-GS使用600万参数即可达到与最先进的方法相当的性能——相比最接近的竞争者3500万参数，性能和模型效率达到了理想的权衡。 

---
# ArticFlow: Generative Simulation of Articulated Mechanisms 

**Title (ZH)**: articFlow：刚体机制生成模拟 

**Authors**: Jiong Lin, Jinchen Ruan, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2511.17883)  

**Abstract**: Recent advances in generative models have produced strong results for static 3D shapes, whereas articulated 3D generation remains challenging due to action-dependent deformations and limited datasets. We introduce ArticFlow, a two-stage flow matching framework that learns a controllable velocity field from noise to target point sets under explicit action control. ArticFlow couples (i) a latent flow that transports noise to a shape-prior code and (ii) a point flow that transports points conditioned on the action and the shape prior, enabling a single model to represent diverse articulated categories and generalize across actions. On MuJoCo Menagerie, ArticFlow functions both as a generative model and as a neural simulator: it predicts action-conditioned kinematics from a compact prior and synthesizes novel morphologies via latent interpolation. Compared with object-specific simulators and an action-conditioned variant of static point-cloud generators, ArticFlow achieves higher kinematic accuracy and better shape quality. Results show that action-conditioned flow matching is a practical route to controllable and high-quality articulated mechanism generation. 

**Abstract (ZH)**: Recent advances in生成模型在静态3D形状上取得了显著成果，但由于动作相关的变形和数据集有限， articulated 3D生成仍具有挑战性。我们介绍了一种两阶段流动匹配框架ArticFlow，该框架在明确动作控制下从噪声学习可控的速度场以目标点集为目标配准。ArticFlow结合了(i)一个潜藏的流，将噪声输运到形状先验编码，以及(ii)一个点流，基于动作和形状先验条件运输点，从而使单一模型能够表示多种articulated类别并在动作之间进行泛化。在MuJoCo Menagerie中，ArticFlow既作为一个生成模型也作为一个神经模拟器：它从紧凑的先验中预测条件动作的运动学并借助潜藏插值合成新的形态。与针对特定对象的模拟器和条件动作的静态点云生成变体相比，ArticFlow在运动学准确性和形状质量方面表现更佳。结果表明，条件动作的流动匹配是可控和高质量articulated机制生成的实用途径。 

---
# QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents 

**Title (ZH)**: QuickLAP: 快速语言与动作偏好学习方法在自动驾驶代理中的应用 

**Authors**: Jordan Abi Nader, David Lee, Nathaniel Dennler, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17855)  

**Abstract**: Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL. 

**Abstract (ZH)**: 机器人必须从人们的行动和言辞中学习，但单靠其中任何一种方式往往是不完整的：物理纠正虽然具体但意图不明确，而语言则能表达高层次的目标但缺乏物理基础。我们 introduce QuickLAP: 快速语言-行动偏好学习，这是一种贝叶斯框架，能够融合物理反馈和语言反馈以实现实时奖励函数的推断。我们的关键见解是将语言视为用户潜在偏好的概率性观测，这有助于明确哪些奖励特征重要以及如何解释物理纠正。QuickLAP 使用大型语言模型（LLMs）从自由格式的陈述中提取奖励特征注意力掩码和偏好偏移，并将这些内容与物理反馈结合到一个封闭形式的更新规则中。这使得快速、实时且稳健的奖励学习成为可能，并能够处理含糊不清的反馈。在半自主驾驶模拟器中，QuickLAP 在奖励学习误差上比仅基于物理反馈和启发式多模态基准降低了超过 70%。一项由 15 名参与者参与的用户研究进一步验证了我们的方法：参与者发现 QuickLAP 更易于理解和更具协作性，并更偏好其学习到的行为。代码可在以下链接获取。 

---
# QAL: A Loss for Recall Precision Balance in 3D Reconstruction 

**Title (ZH)**: QAL：三维重建中召回精度平衡的损失函数 

**Authors**: Pranay Meshram, Yash Turkar, Kartikeya Singh, Praveen Raj Masilamani, Charuvahan Adhivarahan, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17824)  

**Abstract**: Volumetric learning underpins many 3D vision tasks such as completion, reconstruction, and mesh generation, yet training objectives still rely on Chamfer Distance (CD) or Earth Mover's Distance (EMD), which fail to balance recall and precision. We propose Quality-Aware Loss (QAL), a drop-in replacement for CD/EMD that combines a coverage-weighted nearest-neighbor term with an uncovered-ground-truth attraction term, explicitly decoupling recall and precision into tunable components.
Across diverse pipelines, QAL achieves consistent coverage gains, improving by an average of +4.3 pts over CD and +2.8 pts over the best alternatives. Though modest in percentage, these improvements reliably recover thin structures and under-represented regions that CD/EMD overlook. Extensive ablations confirm stable performance across hyperparameters and across output resolutions, while full retraining on PCN and ShapeNet demonstrates generalization across datasets and backbones. Moreover, QAL-trained completions yield higher grasp scores under GraspNet evaluation, showing that improved coverage translates directly into more reliable robotic manipulation.
QAL thus offers a principled, interpretable, and practical objective for robust 3D vision and safety-critical robotics pipelines 

**Abstract (ZH)**: Volumetric Learning underpins Many 3D Vision Tasks Such as Completion, Reconstruction, and Mesh Generation: Quality-Aware Loss (QAL) Explicitly Decouples Recall and Precision for Improved Coverage 

---
# Target-Bench: Can World Models Achieve Mapless Path Planning with Semantic Targets? 

**Title (ZH)**: Target-Bench: 语义目标下世界模型能否实现无地图路径规划？ 

**Authors**: Dingrui Wang, Hongyuan Ye, Zhihao Liang, Zhexiao Sun, Zhaowei Lu, Yuchen Zhang, Yuyu Zhao, Yuan Gao, Marvin Seegert, Finn Schäfer, Haotong Qin, Wei Li, Luigi Palmieri, Felix Jahncke, Mattia Piccinini, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2511.17792)  

**Abstract**: While recent world models generate highly realistic videos, their ability to perform robot path planning remains unclear and unquantified. We introduce Target-Bench, the first benchmark specifically designed to evaluate world models on mapless path planning toward semantic targets in real-world environments. Target-Bench provides 450 robot-collected video sequences spanning 45 semantic categories with SLAM-based ground truth trajectories. Our evaluation pipeline recovers camera motion from generated videos and measures planning performance using five complementary metrics that quantify target-reaching capability, trajectory accuracy, and directional consistency. We evaluate state-of-the-art models including Sora 2, Veo 3.1, and the Wan series. The best off-the-shelf model (Wan2.2-Flash) achieves only 0.299 overall score, revealing significant limitations in current world models for robotic planning tasks. We show that fine-tuning an open-source 5B-parameter model on only 325 scenarios from our dataset achieves 0.345 overall score -- an improvement of more than 400% over its base version (0.066) and 15% higher than the best off-the-shelf model. We will open-source the code and dataset. 

**Abstract (ZH)**: Target-Bench: 一种专门用于评估世界模型在真实环境无地图路径规划中朝向语义目标进行规划能力的基准 

---
# Multi-Agent Coordination in Autonomous Vehicle Routing: A Simulation-Based Study of Communication, Memory, and Routing Loops 

**Title (ZH)**: 自主车辆路线规划中的多agent协调：基于通信、记忆和路由循环的仿真研究 

**Authors**: KM Khalid Saifullah, Daniel Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.17656)  

**Abstract**: Multi-agent coordination is critical for next-generation autonomous vehicle (AV) systems, yet naive implementations of communication-based rerouting can lead to catastrophic performance degradation. This study investigates a fundamental problem in decentralized multi-agent navigation: routing loops, where vehicles without persistent obstacle memory become trapped in cycles of inefficient path recalculation. Through systematic simulation experiments involving 72 unique configurations across varying vehicle densities (15, 35, 55 vehicles) and obstacle frequencies (6, 20 obstacles), we demonstrate that memory-less reactive rerouting increases average travel time by up to 682% compared to baseline conditions. To address this, we introduce Object Memory Management (OMM), a lightweight mechanism enabling agents to retain and share knowledge of previously encountered obstacles. OMM operates by maintaining a distributed blacklist of blocked nodes, which each agent consults during Dijkstra-based path recalculation, effectively preventing redundant routing attempts. Our results show that OMM-enabled coordination reduces average travel time by 75.7% and wait time by 88% compared to memory-less systems, while requiring only 1.67 route recalculations per vehicle versus 9.83 in memory-less scenarios. This work provides empirical evidence that persistent, shared memory is not merely beneficial but essential for robust multi-agent coordination in dynamic environments. The findings have implications beyond autonomous vehicles, informing the design of decentralized systems in robotics, network routing, and distributed AI. We provide a comprehensive experimental analysis, including detailed scenario breakdowns, scalability assessments, and visual documentation of the routing loop phenomenon, demonstrating OMM's critical role in preventing detrimental feedback cycles in cooperative multi-agent systems. 

**Abstract (ZH)**: 基于对象记忆管理的多Agent协调研究：从通信重路由到去耦循环 

---
# SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios 

**Title (ZH)**: SWITCH：长时身临其境场景中实体界面建模与处理的基准研究 

**Authors**: Jieru Lin, Zhiwei Yu, Börje F. Karlsson  

**Link**: [PDF](https://arxiv.org/pdf/2511.17649)  

**Abstract**: Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities:task-aware VQA, semantic UI grounding, action generation, state-transition prediction, and result verification, under egocentric RGB video input and device diversity. Across 351 tasks spanning 98 real devices and appliances, commercial and open LMMMs exhibit inconsistent performance even on single-step interactions, often over-relying on textual cues and under-using visual or video evidence (and high aggregate scores can mask such failures). SWITCH provides data, code, and held-out splits to enable reproducible evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of training datasets. Benchmark resources are available at: this https URL. 

**Abstract (ZH)**: 自主智能不仅需要感知和推理，而且关键地需要与现有世界及其基础设施进行有效的交互。日常生活环境充满了实体控制接口（TCIs），例如开关、家电面板和嵌入式GUI，这些接口要求常识和物理推理，同时也需要时间和空间中的因果预测和结果验证（例如，延迟加热、远程灯光）。此外，这些领域的失败可能具有潜在的安全影响，但当前的评估基准很少测试地面真实情况、部分可观测性（视频）或情境下的事后验证。我们引入了SWITCH（语义世界接口任务集，用于控制和处理），这是一种通过迭代发布创建的基于身体的、任务驱动的基准，旨在探究这些差距。其首次迭代SWITCH-Basic评估了五种互补的能力：任务感知型VQA、语义UI接地、行动生成、状态转换预测和结果验证，在第一人称RGB视频输入和设备多样性条件下进行。在涵盖98种真实设备和家电的351个任务中，即使是单一步骤交互，商业和开源LMMMs的表现也不一致，经常过度依赖文本提示，而未充分利用视觉或视频证据（高整体分数可能会掩盖这些失败）。SWITCH提供了数据、代码和保留集以实现可重复评估并促进社区贡献，以支持更具挑战性的未来基准迭代和训练数据集的创建。基准资源可在以下链接获取：[this https URL]。 

---
# Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments 

**Title (ZH)**: Rad-GS: 雷达-视觉集成的室外环境三维高斯点云SLAM 

**Authors**: Renxiang Xiao, Wei Liu, Yuanfan Zhang, Yushuai Chen, Jinming Chen, Zilu Wang, Liang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16091)  

**Abstract**: We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction. 

**Abstract (ZH)**: Rad-GS：一种适用于千米级户外环境的4D雷达-摄像机SLAM系统 

---
