# Bayesian Optimization applied for accelerated Virtual Validation of the Autonomous Driving Function 

**Title (ZH)**: 应用于自主驾驶功能加速虚拟验证的贝叶斯优化方法 

**Authors**: Satyesh Shanker Awasthi, Mohammed Irshadh Ismaaeel Sathyamangalam Imran, Stefano Arrigoni, Francesco Braghin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22769)  

**Abstract**: Rigorous Verification and Validation (V&V) of Autonomous Driving Functions (ADFs) is paramount for ensuring the safety and public acceptance of Autonomous Vehicles (AVs). Current validation relies heavily on simulation to achieve sufficient test coverage within the Operational Design Domain (ODD) of a vehicle, but exhaustively exploring the vast parameter space of possible scenarios is computationally expensive and time-consuming. This work introduces a framework based on Bayesian Optimization (BO) to accelerate the discovery of critical scenarios. We demonstrate the effectiveness of the framework on an Model Predictive Controller (MPC)-based motion planner, showing that it identifies hazardous situations, such as off-road events, using orders of magnitude fewer simulations than brute-force Design of Experiments (DoE) methods. Furthermore, this study investigates the scalability of the framework in higher-dimensional parameter spaces and its ability to identify multiple, distinct critical regions within the ODD of the motion planner used as the case study . 

**Abstract (ZH)**: 严格验证与验证（V&V）对于确保自动驾驶功能（ADFs）的安全性和公众接受度至关重要。当前的验证主要依赖于模拟来实现对车辆操作设计域（ODD）的充分测试覆盖，但全面探索可能场景的庞大参数空间在计算上非常昂贵且耗时。本研究提出了一种基于贝叶斯优化（BO）的框架，以加速关键场景的发现。我们通过在基于模型预测控制（MPC）的运动规划器上验证该框架的有效性，结果显示它使用比暴力设计实验（DoE）方法小数量级的模拟就能识别出危险情况，如脱道路事件。此外，本研究还探讨了该框架在高维参数空间中的可扩展性及其在所研究的运动规划器ODD中识别多个独立关键区域的能力。 

---
# UniLegs: Universal Multi-Legged Robot Control through Morphology-Agnostic Policy Distillation 

**Title (ZH)**: UniLegs: 基于形态无关策略蒸馏的通用多足机器人控制 

**Authors**: Weijie Xi, Zhanxiang Cao, Chenlin Ming, Jianying Zheng, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22653)  

**Abstract**: Developing controllers that generalize across diverse robot morphologies remains a significant challenge in legged locomotion. Traditional approaches either create specialized controllers for each morphology or compromise performance for generality. This paper introduces a two-stage teacher-student framework that bridges this gap through policy distillation. First, we train specialized teacher policies optimized for individual morphologies, capturing the unique optimal control strategies for each robot design. Then, we distill this specialized expertise into a single Transformer-based student policy capable of controlling robots with varying leg configurations. Our experiments across five distinct legged morphologies demonstrate that our approach preserves morphology-specific optimal behaviors, with the Transformer architecture achieving 94.47\% of teacher performance on training morphologies and 72.64\% on unseen robot designs. Comparative analysis reveals that Transformer-based architectures consistently outperform MLP baselines by leveraging attention mechanisms to effectively model joint relationships across different kinematic structures. We validate our approach through successful deployment on a physical quadruped robot, demonstrating the practical viability of our morphology-agnostic control framework. This work presents a scalable solution for developing universal legged robot controllers that maintain near-optimal performance while generalizing across diverse morphologies. 

**Abstract (ZH)**: 跨多样机器人形态实现通用控制器的设计仍然是 legged 机器人类足运动控制中的一个重大挑战。传统的做法要么为每种形态设计专门的控制器，要么牺牲通用性以提升性能。本文提出了一种两阶段教师-学生框架，通过策略蒸馏弥合这一差距。首先，我们训练针对 individual 形态优化的专门教师策略，捕捉每种机器人设计的独特最优控制策略。然后，将这种专门化的专业知识蒸馏到一个能够控制具有不同腿部配置的机器人的 Transformer 基础学生策略中。我们在五种不同的腿足形态的实验中展示了该方法保留形态特有的最优行为，Transformer 架构在训练形态上的性能达到教师的 94.47%，在未见过的机器人设计上达到 72.64%。比较分析显示，基于 Transformer 的架构通过利用注意力机制有效建模不同运动学结构之间的关节关系，连续优于 MLP 基线。我们通过成功部署在物理四足机器人上验证了该方法，证明了我们的形态无关控制框架的实用性。本文展示了开发能够保持接近最优性能并在多样化形态中泛化的通用腿足机器人控制器的可扩展解决方案。 

---
# Explainable Deep Anomaly Detection with Sequential Hypothesis Testing for Robotic Sewer Inspection 

**Title (ZH)**: 基于序列假设检验的可解释深度异常检测在机器人 sewer 检查中的应用 

**Authors**: Alex George, Will Shepherd, Simon Tait, Lyudmila Mihaylova, Sean R. Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2507.22546)  

**Abstract**: Sewer pipe faults, such as leaks and blockages, can lead to severe consequences including groundwater contamination, property damage, and service disruption. Traditional inspection methods rely heavily on the manual review of CCTV footage collected by mobile robots, which is inefficient and susceptible to human error. To automate this process, we propose a novel system incorporating explainable deep learning anomaly detection combined with sequential probability ratio testing (SPRT). The anomaly detector processes single image frames, providing interpretable spatial localisation of anomalies, whilst the SPRT introduces temporal evidence aggregation, enhancing robustness against noise over sequences of image frames. Experimental results demonstrate improved anomaly detection performance, highlighting the benefits of the combined spatiotemporal analysis system for reliable and robust sewer inspection. 

**Abstract (ZH)**: Sewer 管道故障（如泄漏和堵塞）可能导致地下水污染、财产损失和服务中断。传统检查方法主要依赖移动机器人收集的CCTV footage的手动审查，效率低且易出错。为自动化这一过程，我们提出了一种结合可解释的深度学习异常检测和序列概率比率测试（SPRT）的新型系统。异常检测器处理单张图像帧，提供可解释的空间定位，而SPRT引入了时间证据聚合，增强了对图像帧序列中噪声的鲁棒性。实验结果表明，这种结合时空分析系统的异常检测性能改进，突显了其在可靠和 robust 管道检查中的优势。 

---
# A Two-Stage Lightweight Framework for Efficient Land-Air Bimodal Robot Autonomous Navigation 

**Title (ZH)**: 一种两阶段轻量级框架，实现高效地面-空中双模式机器人自主导航 

**Authors**: Yongjie Li, Zhou Liu, Wenshuai Yu, Zhangji Lu, Chenyang Wang, Fei Yu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22473)  

**Abstract**: Land-air bimodal robots (LABR) are gaining attention for autonomous navigation, combining high mobility from aerial vehicles with long endurance from ground vehicles. However, existing LABR navigation methods are limited by suboptimal trajectories from mapping-based approaches and the excessive computational demands of learning-based methods. To address this, we propose a two-stage lightweight framework that integrates global key points prediction with local trajectory refinement to generate efficient and reachable trajectories. In the first stage, the Global Key points Prediction Network (GKPN) was used to generate a hybrid land-air keypoint path. The GKPN includes a Sobel Perception Network (SPN) for improved obstacle detection and a Lightweight Attention Planning Network (LAPN) to improves predictive ability by capturing contextual information. In the second stage, the global path is segmented based on predicted key points and refined using a mapping-based planner to create smooth, collision-free trajectories. Experiments conducted on our LABR platform show that our framework reduces network parameters by 14\% and energy consumption during land-air transitions by 35\% compared to existing approaches. The framework achieves real-time navigation without GPU acceleration and enables zero-shot transfer from simulation to reality during 

**Abstract (ZH)**: 陆空两用机器人（LABR）的双阶段轻量级导航框架：结合全局关键点预测与局部路径细化生成高效可达轨迹 

---
# Operationalization of Scenario-Based Safety Assessment of Automated Driving Systems 

**Title (ZH)**: 基于场景的安全评估自动化驾驶系统操作化 

**Authors**: Olaf Op den Camp, Erwin de Gelder  

**Link**: [PDF](https://arxiv.org/pdf/2507.22433)  

**Abstract**: Before introducing an Automated Driving System (ADS) on the road at scale, the manufacturer must conduct some sort of safety assurance. To structure and harmonize the safety assurance process, the UNECE WP.29 Working Party on Automated/Autonomous and Connected Vehicles (GRVA) is developing the New Assessment/Test Method (NATM) that indicates what steps need to be taken for safety assessment of an ADS. In this paper, we will show how to practically conduct safety assessment making use of a scenario database, and what additional steps must be taken to fully operationalize the NATM. In addition, we will elaborate on how the use of scenario databases fits with methods developed in the Horizon Europe projects that focus on safety assessment following the NATM approach. 

**Abstract (ZH)**: 在大规模部署自动驾驶系统之前，制造商必须开展某种形式的安全保证工作。为了结构化和协调安全保证过程，联合国经济及社会理事会 WP.29 自动/自主及连接车辆工作组（GRVA）正在开发新的评估/测试方法（NATM），以指示对自动驾驶系统进行安全评估所必需的步骤。在本文中，我们将展示如何利用场景数据库进行实际的安全评估，并说明为了完全实现NATM还需要采取哪些额外步骤。此外，我们将详细阐述场景数据库的使用如何与 Horizon Europe 项目中开发的安全评估方法（遵循NATM方法）相契合。 

---
# Comparing Normalizing Flows with Kernel Density Estimation in Estimating Risk of Automated Driving Systems 

**Title (ZH)**: 比较归一化流与核密度估计在评估自动驾驶系统风险中的表现 

**Authors**: Erwin de Gelder, Maren Buermann, Olaf Op den Camp  

**Link**: [PDF](https://arxiv.org/pdf/2507.22429)  

**Abstract**: The development of safety validation methods is essential for the safe deployment and operation of Automated Driving Systems (ADSs). One of the goals of safety validation is to prospectively evaluate the risk of an ADS dealing with real-world traffic. Scenario-based assessment is a widely-used approach, where test cases are derived from real-world driving data. To allow for a quantitative analysis of the system performance, the exposure of the scenarios must be accurately estimated. The exposure of scenarios at parameter level is expressed using a Probability Density Function (PDF). However, assumptions about the PDF, such as parameter independence, can introduce errors, while avoiding assumptions often leads to oversimplified models with limited parameters to mitigate the curse of dimensionality.
This paper considers the use of Normalizing Flows (NF) for estimating the PDF of the parameters. NF are a class of generative models that transform a simple base distribution into a complex one using a sequence of invertible and differentiable mappings, enabling flexible, high-dimensional density estimation without restrictive assumptions on the PDF's shape. We demonstrate the effectiveness of NF in quantifying risk and risk uncertainty of an ADS, comparing its performance with Kernel Density Estimation (KDE), a traditional method for non-parametric PDF estimation. While NF require more computational resources compared to KDE, NF is less sensitive to the curse of dimensionality. As a result, NF can improve risk uncertainty estimation, offering a more precise assessment of an ADS's safety.
This work illustrates the potential of NF in scenario-based safety. Future work involves experimenting more with using NF for scenario generation and optimizing the NF architecture, transformation types, and training hyperparameters to further enhance their applicability. 

**Abstract (ZH)**: 基于Normalizing Flows的安全评估方法在自动驾驶系统中的应用研究 

---
# Safety Evaluation of Motion Plans Using Trajectory Predictors as Forward Reachable Set Estimators 

**Title (ZH)**: 使用轨迹预测器作为前方可达集估计器的运动计划安全性评估 

**Authors**: Kaustav Chakraborty, Zeyuan Feng, Sushant Veer, Apoorva Sharma, Wenhao Ding, Sever Topan, Boris Ivanovic, Marco Pavone, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.22389)  

**Abstract**: The advent of end-to-end autonomy stacks - often lacking interpretable intermediate modules - has placed an increased burden on ensuring that the final output, i.e., the motion plan, is safe in order to validate the safety of the entire stack. This requires a safety monitor that is both complete (able to detect all unsafe plans) and sound (does not flag safe plans). In this work, we propose a principled safety monitor that leverages modern multi-modal trajectory predictors to approximate forward reachable sets (FRS) of surrounding agents. By formulating a convex program, we efficiently extract these data-driven FRSs directly from the predicted state distributions, conditioned on scene context such as lane topology and agent history. To ensure completeness, we leverage conformal prediction to calibrate the FRS and guarantee coverage of ground-truth trajectories with high probability. To preserve soundness in out-of-distribution (OOD) scenarios or under predictor failure, we introduce a Bayesian filter that dynamically adjusts the FRS conservativeness based on the predictor's observed performance. We then assess the safety of the ego vehicle's motion plan by checking for intersections with these calibrated FRSs, ensuring the plan remains collision-free under plausible future behaviors of others. Extensive experiments on the nuScenes dataset show our approach significantly improves soundness while maintaining completeness, offering a practical and reliable safety monitor for learned autonomy stacks. 

**Abstract (ZH)**: 端到端自主系统的发展 - 通常缺乏可解释的中间模块 - 已经增加了确保最终输出（即运动计划）的安全性以验证整个系统的安全性的负担。这需要一个既完备（能够检测所有不安全的计划）又sound（不标记安全的计划）的安全监控器。在本文中，我们提出了一种基于原理的安全监控器，利用现代多模态轨迹预测器来近似邻近代理的前方可达集（FRS）。通过形式化一个凸规划，我们可以直接从预测的状态分布中，根据场景上下文（如车道拓扑和代理历史）提取这些数据驱动的FRS。为了保证完备性，我们利用兼容预测来校准FRS，并以高概率保证真实轨迹的覆盖。为了在分布外（OOD）场景或预测器失效情况下保持soundness，我们引入了一个贝叶斯滤波器，根据预测器的观测性能动态调整FRS的保守性。然后，我们通过检查运动计划与这些校准后的FRS的交集来评估自主车辆的运动计划的安全性，从而确保在其他主体可能的未来行为下，计划保持无碰撞。在nuScenes数据集上的广泛实验表明，我们的方法在保持完备性的基础上显著提高了soundness，提供了一个实用且可靠的学习自主系统的安全监控器。 

---
# Improving Generalization Ability of Robotic Imitation Learning by Resolving Causal Confusion in Observations 

**Title (ZH)**: 通过解决观测中的因果混淆提高机器人 imitation 学习的泛化能力 

**Authors**: Yifei Chen, Yuzhe Zhang, Giovanni D'urso, Nicholas Lawrance, Brendan Tidd  

**Link**: [PDF](https://arxiv.org/pdf/2507.22380)  

**Abstract**: Recent developments in imitation learning have considerably advanced robotic manipulation. However, current techniques in imitation learning can suffer from poor generalization, limiting performance even under relatively minor domain shifts. In this work, we aim to enhance the generalization capabilities of complex imitation learning algorithms to handle unpredictable changes from the training environments to deployment environments. To avoid confusion caused by observations that are not relevant to the target task, we propose to explicitly learn the causal relationship between observation components and expert actions, employing a framework similar to [6], where a causal structural function is learned by intervention on the imitation learning policy. Disentangling the feature representation from image input as in [6] is hard to satisfy in complex imitation learning process in robotic manipulation, we theoretically clarify that this requirement is not necessary in causal relationship learning. Therefore, we propose a simple causal structure learning framework that can be easily embedded in recent imitation learning architectures, such as the Action Chunking Transformer [31]. We demonstrate our approach using a simulation of the ALOHA [31] bimanual robot arms in Mujoco, and show that the method can considerably mitigate the generalization problem of existing complex imitation learning algorithms. 

**Abstract (ZH)**: 近期模仿学习的发展显著推进了机器人操作技术。然而，当前模仿学习技术在泛化能力方面存在局限，即使在相对较轻微的领域变化下也不例外。在此项工作中，我们旨在增强复杂模仿学习算法的泛化能力，以应对从训练环境到部署环境中的不可预测变化。为了避免无关观测信息对目标任务造成的混淆，我们提出明确学习观测组件与专家动作之间的因果关系，采用类似于[6]的框架，在模仿学习策略上进行干预以学习因果结构函数。尽管在复杂机器人操作的模仿学习过程中解缠特征表示与图像输入的要求很难实现，但理论上我们澄清了这一要求在因果关系学习中并非必要。因此，我们提出了一种简单的因果结构学习框架，该框架可以方便地嵌入到最近的模仿学习架构中，如Action Chunking Transformer [31]。我们使用Mujoco中的ALOHA [31] 双臂机器人模拟来展示该方法，并证明该方法可以在显著缓解现有复杂模仿学习算法的泛化问题方面发挥重要作用。 

---
# In-Situ Soil-Property Estimation and Bayesian Mapping with a Simulated Compact Track Loader 

**Title (ZH)**: 基于模拟紧凑型轨道装载机的现场土壤性质估计与贝叶斯制图 

**Authors**: W. Jacob Wagner, Ahmet Soylemezoglu, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2507.22356)  

**Abstract**: Existing earthmoving autonomy is largely confined to highly controlled and well-characterized environments due to the complexity of vehicle-terrain interaction dynamics and the partial observability of the terrain resulting from unknown and spatially varying soil conditions. In this chapter, a a soil-property mapping system is proposed to extend the environmental state, in order to overcome these restrictions and facilitate development of more robust autonomous earthmoving. A GPU accelerated elevation mapping system is extended to incorporate a blind mapping component which traces the movement of the blade through the terrain to displace and erode intersected soil, enabling separately tracking undisturbed and disturbed soil. Each interaction is approximated as a flat blade moving through a locally homogeneous soil, enabling modeling of cutting forces using the fundamental equation of earthmoving (FEE). Building upon our prior work on in situ soil-property estimation, a method is devised to extract approximate geometric parameters of the model given the uneven terrain, and an improved physics infused neural network (PINN) model is developed to predict soil properties and uncertainties of these estimates. A simulation of a compact track loader (CTL) with a blade attachment is used to collect data to train the PINN model. Post-training, the model is leveraged online by the mapping system to track soil property estimates spatially as separate layers in the map, with updates being performed in a Bayesian manner. Initial experiments show that the system accurately highlights regions requiring higher relative interaction forces, indicating the promise of this approach in enabling soil-aware planning for autonomous terrain shaping. 

**Abstract (ZH)**: 基于土壤特性映射的自主土方作业扩展环境状态研究 

---
# FLORES: A Reconfigured Wheel-Legged Robot for Enhanced Steering and Adaptability 

**Title (ZH)**: FLORES：一种重构的轮腿机器人，以增强转向能力和适应性 

**Authors**: Zhicheng Song, Jinglan Xu, Chunxin Zheng, Yulin Li, Zhihai Bi, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.22345)  

**Abstract**: Wheel-legged robots integrate the agility of legs for navigating rough terrains while harnessing the efficiency of wheels for smooth surfaces. However, most existing designs do not fully capitalize on the benefits of both legged and wheeled structures, which limits overall system flexibility and efficiency. We present FLORES (reconfigured wheel-legged robot for enhanced steering and adaptability), a novel wheel-legged robot design featuring a distinctive front-leg configuration that sets it beyond standard design approaches. Specifically, FLORES replaces the conventional hip-roll degree of freedom (DoF) of the front leg with hip-yaw DoFs, and this allows for efficient movement on flat surfaces while ensuring adaptability when navigating complex terrains. This innovative design facilitates seamless transitions between different locomotion modes (i.e., legged locomotion and wheeled locomotion) and optimizes the performance across varied environments. To fully exploit FLORES's mechanical capabilities, we develop a tailored reinforcement learning (RL) controller that adapts the Hybrid Internal Model (HIM) with a customized reward structure optimized for our unique mechanical configuration. This framework enables the generation of adaptive, multi-modal locomotion strategies that facilitate smooth transitions between wheeled and legged movements. Furthermore, our distinctive joint design enables the robot to exhibit novel and highly efficient locomotion gaits that capitalize on the synergistic advantages of both locomotion modes. Through comprehensive experiments, we demonstrate FLORES's enhanced steering capabilities, improved navigation efficiency, and versatile locomotion across various terrains. The open-source project can be found at this https URL. 

**Abstract (ZH)**: 基于改进转向与适应性的可重构轮腿机器人FLORES 

---
# Deployment of Objects with a Soft Everting Robot 

**Title (ZH)**: 软变形机器人中对象的部署 

**Authors**: Ethan DeVries, Jack Ferlazzo, Mustafa Ugur, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2507.22188)  

**Abstract**: Soft everting robots present significant advantages over traditional rigid robots, including enhanced dexterity, improved environmental interaction, and safe navigation in unpredictable environments. While soft everting robots have been widely demonstrated for exploration type tasks, their potential to move and deploy payloads in such tasks has been less investigated, with previous work focusing on sensors and tools for the robot. Leveraging the navigation capabilities, and deployed body, of the soft everting robot to deliver payloads in hazardous areas, e.g. carrying a water bottle to a person stuck under debris, would represent a significant capability in many applications. In this work, we present an analysis of how soft everting robots can be used to deploy larger, heavier payloads through the inside of the robot. We analyze both what objects can be deployed and what terrain features they can be carried through. Building on existing models, we present methods to quantify the effects of payloads on robot growth and self-support, and develop a model to predict payload slip. We then experimentally quantify payload transport using soft everting robot with a variety of payload shapes, sizes, and weights and though a series of tasks: steering, vertical transport, movement through holes, and movement across gaps. Overall, the results show that we can transport payloads in a variety of shapes and up to 1.5kg in weight and that we can move through circular apertures with as little as 0.01cm clearance around payloads, carry out discrete turns up to 135 degrees, and move across unsupported gaps of 1.15m in length. 

**Abstract (ZH)**: 软胀出机器人在部署更大、更重载荷方面的应用分析 

---
# Viser: Imperative, Web-based 3D Visualization in Python 

**Title (ZH)**: Viser: 基于Web的Python imperative 3D可视化 

**Authors**: Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22885)  

**Abstract**: We present Viser, a 3D visualization library for computer vision and robotics. Viser aims to bring easy and extensible 3D visualization to Python: we provide a comprehensive set of 3D scene and 2D GUI primitives, which can be used independently with minimal setup or composed to build specialized interfaces. This technical report describes Viser's features, interface, and implementation. Key design choices include an imperative-style API and a web-based viewer, which improve compatibility with modern programming patterns and workflows. 

**Abstract (ZH)**: Viser：一种面向计算机视觉和机器人领域的3D可视化库 

---
# Recognizing Actions from Robotic View for Natural Human-Robot Interaction 

**Title (ZH)**: 从机器人视角识别动作以实现自然人机交互 

**Authors**: Ziyi Wang, Peiming Li, Hong Liu, Zhichao Deng, Can Wang, Jun Liu, Junsong Yuan, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22522)  

**Abstract**: Natural Human-Robot Interaction (N-HRI) requires robots to recognize human actions at varying distances and states, regardless of whether the robot itself is in motion or stationary. This setup is more flexible and practical than conventional human action recognition tasks. However, existing benchmarks designed for traditional action recognition fail to address the unique complexities in N-HRI due to limited data, modalities, task categories, and diversity of subjects and environments. To address these challenges, we introduce ACTIVE (Action from Robotic View), a large-scale dataset tailored specifically for perception-centric robotic views prevalent in mobile service robots. ACTIVE comprises 30 composite action categories, 80 participants, and 46,868 annotated video instances, covering both RGB and point cloud modalities. Participants performed various human actions in diverse environments at distances ranging from 3m to 50m, while the camera platform was also mobile, simulating real-world scenarios of robot perception with varying camera heights due to uneven ground. This comprehensive and challenging benchmark aims to advance action and attribute recognition research in N-HRI. Furthermore, we propose ACTIVE-PC, a method that accurately perceives human actions at long distances using Multilevel Neighborhood Sampling, Layered Recognizers, Elastic Ellipse Query, and precise decoupling of kinematic interference from human actions. Experimental results demonstrate the effectiveness of ACTIVE-PC. Our code is available at: this https URL. 

**Abstract (ZH)**: 自然人类-机器人交互（N-HRI）要求机器人能够在不同距离和状态下识别人类动作，无论机器人本身是移动还是静止。这种设置比传统的动作识别任务更具灵活性和实用性。然而，现有的用于传统动作识别的基准数据集由于数据量、模态、任务类别以及参与者的多样性和环境的多样性有限，未能解决N-HRI中特有的复杂性。为应对这些挑战，我们介绍了ACTIVE（基于机器人视角的动作识别）数据集，该数据集专门针对移动服务机器人中常见的感知中心机器人视角。ACTIVE包含30个复合动作类别、80名参与者和46,868个标注视频实例，涵盖RGB和点云模态。参与者在多种环境条件下，在3米至50米的距离范围内执行各种人类动作，同时摄像平台也具有移动性，模拟不同地面高度导致的机器人感知真实性场景。这一全面且具有挑战性的基准旨在促进N-HRI中的动作和属性识别研究。此外，我们提出了ACTIVE-PC方法，该方法使用多级邻域采样、分层识别器、弹性椭圆查询和精确的动力学干扰解耦，准确识别远距离的人类动作。实验证明了ACTIVE-PC的有效性。我们的代码可在以下链接获取：this https URL。 

---
# Multi-Agent Path Finding Among Dynamic Uncontrollable Agents with Statistical Safety Guarantees 

**Title (ZH)**: 动态不可控代理中的统计安全性保证多代理路径寻找 

**Authors**: Kegan J. Strawn, Thomy Phan, Eric Wang, Nora Ayanian, Sven Koenig, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2507.22282)  

**Abstract**: Existing multi-agent path finding (MAPF) solvers do not account for uncertain behavior of uncontrollable agents. We present a novel variant of Enhanced Conflict-Based Search (ECBS), for both one-shot and lifelong MAPF in dynamic environments with uncontrollable agents. Our method consists of (1) training a learned predictor for the movement of uncontrollable agents, (2) quantifying the prediction error using conformal prediction (CP), a tool for statistical uncertainty quantification, and (3) integrating these uncertainty intervals into our modified ECBS solver. Our method can account for uncertain agent behavior, comes with statistical guarantees on collision-free paths for one-shot missions, and scales to lifelong missions with a receding horizon sequence of one-shot instances. We run our algorithm, CP-Solver, across warehouse and game maps, with competitive throughput and reduced collisions. 

**Abstract (ZH)**: 不确定行为的不可控代理下的多代理路径寻找：Enhanced Conflict-Based Search 的新型变体及其在动态环境中的应用 

---
# Modified Smith predictor for unstable linear systems 

**Title (ZH)**: 不稳定线性系统的修改Smith预估器 

**Authors**: Anton Pyrkin, Konstantin Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22243)  

**Abstract**: The paper presents a new control algorithm for unstable linear systems with input delay. In comparison with known analogues, the control law has been designed, which is a modification of the Smith predictor, and is the simplest one to implement without requiring complex integration methods. At the same time, the problem of stabilization of a closed system is effectively solved, ensuring the boundedness of all state variables and the exponential stability of the equilibrium point. 

**Abstract (ZH)**: 一种用于具有输入延迟的不稳定线性系统的控制算法及其实现 

---
# Toward Trusted Onboard AI: Advancing Small Satellite Operations using Reinforcement Learning 

**Title (ZH)**: 面向可信机载AI：增强学习推动小型卫星运行技术发展 

**Authors**: Cannon Whitney, Joseph Melville  

**Link**: [PDF](https://arxiv.org/pdf/2507.22198)  

**Abstract**: A RL (Reinforcement Learning) algorithm was developed for command automation onboard a 3U CubeSat. This effort focused on the implementation of macro control action RL, a technique in which an onboard agent is provided with compiled information based on live telemetry as its observation. The agent uses this information to produce high-level actions, such as adjusting attitude to solar pointing, which are then translated into control algorithms and executed through lower-level instructions. Once trust in the onboard agent is established, real-time environmental information can be leveraged for faster response times and reduced reliance on ground control. The approach not only focuses on developing an RL algorithm for a specific satellite but also sets a precedent for integrating trusted AI into onboard systems. This research builds on previous work in three areas: (1) RL algorithms for issuing high-level commands that are translated into low-level executable instructions; (2) the deployment of AI inference models interfaced with live operational systems, particularly onboard spacecraft; and (3) strategies for building trust in AI systems, especially for remote and autonomous applications. Existing RL research for satellite control is largely limited to simulation-based experiments; in this work, these techniques are tailored by constructing a digital twin of a specific spacecraft and training the RL agent to issue macro actions in this simulated environment. The policy of the trained agent is copied to an isolated environment, where it is fed compiled information about the satellite to make inference predictions, thereby demonstrating the RL algorithm's validity on orbit without granting it command authority. This process enables safe comparison of the algorithm's predictions against actual satellite behavior and ensures operation within expected parameters. 

**Abstract (ZH)**: 一种强化学习算法被开发用于3U立方星上的命令自动化宏控制动作强化学习技术的研究：建立可信赖的人工智能在星上系统中的先例 

---
# Temporally Consistent Unsupervised Segmentation for Mobile Robot Perception 

**Title (ZH)**: 移动机器人感知中的时间一致无监督分割 

**Authors**: Christian Ellis, Maggie Wigness, Craig Lennon, Lance Fiondella  

**Link**: [PDF](https://arxiv.org/pdf/2507.22194)  

**Abstract**: Rapid progress in terrain-aware autonomous ground navigation has been driven by advances in supervised semantic segmentation. However, these methods rely on costly data collection and labor-intensive ground truth labeling to train deep models. Furthermore, autonomous systems are increasingly deployed in unrehearsed, unstructured environments where no labeled data exists and semantic categories may be ambiguous or domain-specific. Recent zero-shot approaches to unsupervised segmentation have shown promise in such settings but typically operate on individual frames, lacking temporal consistency-a critical property for robust perception in unstructured environments. To address this gap we introduce Frontier-Seg, a method for temporally consistent unsupervised segmentation of terrain from mobile robot video streams. Frontier-Seg clusters superpixel-level features extracted from foundation model backbones-specifically DINOv2-and enforces temporal consistency across frames to identify persistent terrain boundaries or frontiers without human supervision. We evaluate Frontier-Seg on a diverse set of benchmark datasets-including RUGD and RELLIS-3D-demonstrating its ability to perform unsupervised segmentation across unstructured off-road environments. 

**Abstract (ZH)**: 基于地形感知的自主地面导航快速进展得益于监督语义分割技术的进步。然而，这些方法依赖于昂贵的数据收集和劳动密集型的地面真实标签来进行模型训练。此外，自主系统 increasingly 部署于未预演的、结构化程度低的环境中，这些环境中不存在标记数据，且语义类别可能模糊或具有领域特定性。最近的零样本无监督分割方法在这些环境中展现了前景，但这些方法通常在单帧上运行，缺乏时间连贯性—在结构化环境中进行鲁棒感知的一个关键属性。为了填补这个缺口，我们引入了 Frontier-Seg 方法，这是一种基于移动机器人视频流的地形无监督分割方法，能够保持时间连贯性。Frontier-Seg 对基础模型主干提取的超像素级特征进行聚类，并在帧之间施加时间连贯性约束，以在无需人工监督的情况下识别持久的地形边界或前沿。我们在包括 RUGD 和 RELLIS-3D 在内的多样性基准数据集上评估了 Frontier-Seg，展示了其在非结构化离路环境中的无监督分割能力。 

---
# Emergent interactions lead to collective frustration in robotic matter 

**Title (ZH)**: Emergent interactions导致机器人物质中的集体挫败感 

**Authors**: Onurcan Bektas, Adolfo Alsina, Steffen Rulands  

**Link**: [PDF](https://arxiv.org/pdf/2507.22148)  

**Abstract**: Current artificial intelligence systems show near-human-level capabilities when deployed in isolation. Systems of a few collaborating intelligent agents are being engineered to perform tasks collectively. This raises the question of whether robotic matter, where many learning and intelligent agents interact, shows emergence of collective behaviour. And if so, which kind of phenomena would such systems exhibit? Here, we study a paradigmatic model for robotic matter: a stochastic many-particle system in which each particle is endowed with a deep neural network that predicts its transitions based on the particles' environments. For a one-dimensional model, we show that robotic matter exhibits complex emergent phenomena, including transitions between long-lived learning regimes, the emergence of particle species, and frustration. We also find a density-dependent phase transition with signatures of criticality. Using active matter theory, we show that this phase transition is a consequence of self-organisation mediated by emergent inter-particle interactions. Our simple model captures key features of more complex forms of robotic systems. 

**Abstract (ZH)**: 当前孤立部署的人工智能系统展示了接近人类的水平能力。几个协作智能代理的系统正在被工程化以共同执行任务。这引发了关于在众多学习和智能代理相互作用的机器人物质中是否会出现集体行为的问题。如果出现，这样的系统将表现出哪些现象？在这里，我们研究了一个典型的机器人物质模型：一个随机的多粒子系统，其中每个粒子都配备了基于自身环境预测状态转换的深度神经网络。对于一维模型，我们显示机器人物质表现出复杂的涌现现象，包括长期学习模式之间的转换、粒子物种的涌现以及挫败感。我们还发现密度依赖的相转换，并且带有临界性的特征。利用活性物质理论，我们证明这一相转换是通过涌现的粒子间相互作用介导的自我组织的后果。我们的简单模型捕捉到了更复杂形式的机器人系统的关键特征。 

---
