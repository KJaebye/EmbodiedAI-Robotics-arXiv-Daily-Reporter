# EmoACT: a Framework to Embed Emotions into Artificial Agents Based on Affect Control Theory 

**Title (ZH)**: EmoACT：基于情感控制理论的情感嵌入人工代理框架 

**Authors**: Francesca Corrao, Alice Nardelli, Jennifer Renoux, Carmine Tommaso Recchiuto  

**Link**: [PDF](https://arxiv.org/pdf/2504.12125)  

**Abstract**: As robots and artificial agents become increasingly integrated into daily life, enhancing their ability to interact with humans is essential. Emotions, which play a crucial role in human interactions, can improve the naturalness and transparency of human-robot interactions (HRI) when embodied in artificial agents. This study aims to employ Affect Control Theory (ACT), a psychological model of emotions deeply rooted in interaction, for the generation of synthetic emotions. A platform-agnostic framework inspired by ACT was developed and implemented in a humanoid robot to assess its impact on human perception. Results show that the frequency of emotional displays impacts how users perceive the robot. Moreover, appropriate emotional expressions seem to enhance the robot's perceived emotional and cognitive agency. The findings suggest that ACT can be successfully employed to embed synthetic emotions into robots, resulting in effective human-robot interactions, where the robot is perceived more as a social agent than merely a machine. 

**Abstract (ZH)**: 随着机器人和人工代理越来越多地融入日常生活，提高它们与人类互动的能力变得越来越重要。将情绪体现在人工代理中可以增强人类与机器人互动（HRI）的自然性和透明度。本研究旨在利用情绪控制理论（ACT），一种深深植根于互动的心理学情绪模型，生成合成情绪。受ACT启发的一个平台无关的框架已在人形机器人中开发和实现，以评估其对人类感知的影响。研究结果表明，情绪展示的频率会影响用户对机器人的感知。此外，适当的的情绪表达似乎增强了机器人在感知中的情感和认知代理能力。研究发现表明，ACT可以成功应用于将合成情绪嵌入机器人中，从而在人机互动中实现有效的交互，使机器人被更多地视为社会代理而非 merely 机器。 

---
# GripMap: An Efficient, Spatially Resolved Constraint Framework for Offline and Online Trajectory Planning in Autonomous Racing 

**Title (ZH)**: GripMap：一种高效的空间解析约束框架，用于自主赛车的离线和在线轨迹规划 

**Authors**: Frederik Werner, Ann-Kathrin Schwehn, Markus Lienkamp, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.12115)  

**Abstract**: Conventional trajectory planning approaches for autonomous vehicles often assume a fixed vehicle model that remains constant regardless of the vehicle's location. This overlooks the critical fact that the tires and the surface are the two force-transmitting partners in vehicle dynamics; while the tires stay with the vehicle, surface conditions vary with location. Recognizing these challenges, this paper presents a novel framework for spatially resolving dynamic constraints in both offline and online planning algorithms applied to autonomous racing. We introduce the GripMap concept, which provides a spatial resolution of vehicle dynamic constraints in the Frenet frame, allowing adaptation to locally varying grip conditions. This enables compensation for location-specific effects, more efficient vehicle behavior, and increased safety, unattainable with spatially invariant vehicle models. The focus is on low storage demand and quick access through perfect hashing. This framework proved advantageous in real-world applications in the presented form. Experiments inspired by autonomous racing demonstrate its effectiveness. In future work, this framework can serve as a foundational layer for developing future interpretable learning algorithms that adjust to varying grip conditions in real-time. 

**Abstract (ZH)**: 基于空间分辨率的动力学约束框架在自主竞速中的应用 

---
# An Extended Generalized Prandtl-Ishlinskii Hysteresis Model for I2RIS Robot 

**Title (ZH)**: 扩展的广义普朗特-伊谢林斯基滞回模型用于I2RIS机器人 

**Authors**: Yiyao Yue, Mojtaba Esfandiari, Pengyuan Du, Peter Gehlbach, Makoto Jinno, Adnan Munawar, Peter Kazanzides, Iulian Iordachita  

**Link**: [PDF](https://arxiv.org/pdf/2504.12114)  

**Abstract**: Retinal surgery requires extreme precision due to constrained anatomical spaces in the human retina. To assist surgeons achieve this level of accuracy, the Improved Integrated Robotic Intraocular Snake (I2RIS) with dexterous capability has been developed. However, such flexible tendon-driven robots often suffer from hysteresis problems, which significantly challenges precise control and positioning. In particular, we observed multi-stage hysteresis phenomena in the small-scale I2RIS. In this paper, we propose an Extended Generalized Prandtl-Ishlinskii (EGPI) model to increase the fitting accuracy of the hysteresis. The model incorporates a novel switching mechanism that enables it to describe multi-stage hysteresis in the regions of monotonic input. Experimental validation on I2RIS data demonstrate that the EGPI model outperforms the conventional Generalized Prandtl-Ishlinskii (GPI) model in terms of RMSE, NRMSE, and MAE across multiple motor input directions. The EGPI model in our study highlights the potential in modeling multi-stage hysteresis in minimally invasive flexible robots. 

**Abstract (ZH)**: 微创柔性手术机器人中多阶段滞回现象的扩展广义普朗特-伊尔欣斯基模型研究 

---
# Self-Supervised Traversability Learning with Online Prototype Adaptation for Off-Road Autonomous Driving 

**Title (ZH)**: 基于在线原型适应的自监督通过性学习在非道路自主驾驶中的应用 

**Authors**: Yafeng Bu, Zhenping Sun, Xiaohui Li, Jun Zeng, Xin Zhang, Hui Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12109)  

**Abstract**: Achieving reliable and safe autonomous driving in off-road environments requires accurate and efficient terrain traversability analysis. However, this task faces several challenges, including the scarcity of large-scale datasets tailored for off-road scenarios, the high cost and potential errors of manual annotation, the stringent real-time requirements of motion planning, and the limited computational power of onboard units. To address these challenges, this paper proposes a novel traversability learning method that leverages self-supervised learning, eliminating the need for manual annotation. For the first time, a Birds-Eye View (BEV) representation is used as input, reducing computational burden and improving adaptability to downstream motion planning. During vehicle operation, the proposed method conducts online analysis of traversed regions and dynamically updates prototypes to adaptively assess the traversability of the current environment, effectively handling dynamic scene changes. We evaluate our approach against state-of-the-art benchmarks on both public datasets and our own dataset, covering diverse seasons and geographical locations. Experimental results demonstrate that our method significantly outperforms recent approaches. Additionally, real-world vehicle experiments show that our method operates at 10 Hz, meeting real-time requirements, while a 5.5 km autonomous driving experiment further validates the generated traversability cost maps compatibility with downstream motion planning. 

**Abstract (ZH)**: 在离线环境中实现可靠和安全的自主驾驶需要准确且高效地进行地形穿越性分析。然而，这一任务面临着多个挑战，包括专门针对离线场景的大规模数据集稀缺、手动标注的高成本和潜在错误、运动规划的严格实时要求以及车载单元的计算能力有限。为应对这些挑战，本文提出了一种新颖的自监督学习穿越性学习方法，无需手动标注。首次使用空中鸟瞰视图（BEV）表示法作为输入，减少了计算负担并增强了对下游运动规划的适应性。在车辆运行过程中，所提出的方法进行在线分析已穿越区域并动态更新原型，以适应性评估当前环境的穿越性，有效应对动态场景变化。我们在公共数据集和我们自己的数据集上（涵盖不同季节和地理区域）对我们的方法与最新基准进行了评估。实验结果表明，我们的方法在性能上显著优于最近的方法。此外，实际车辆实验表明，我们的方法以10 Hz 的速度运行，满足了实时要求，而长达5.5 km 的自主驾驶试验进一步验证了生成的穿越性成本地图与下游运动规划的一致性。 

---
# A Graph-Based Reinforcement Learning Approach with Frontier Potential Based Reward for Safe Cluttered Environment Exploration 

**Title (ZH)**: 基于图形的强化学习方法：结合前端潜力奖励的安全拥挤环境探索 

**Authors**: Gabriele Calzolari, Vidya Sumathy, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.11907)  

**Abstract**: Autonomous exploration of cluttered environments requires efficient exploration strategies that guarantee safety against potential collisions with unknown random obstacles. This paper presents a novel approach combining a graph neural network-based exploration greedy policy with a safety shield to ensure safe navigation goal selection. The network is trained using reinforcement learning and the proximal policy optimization algorithm to maximize exploration efficiency while reducing the safety shield interventions. However, if the policy selects an infeasible action, the safety shield intervenes to choose the best feasible alternative, ensuring system consistency. Moreover, this paper proposes a reward function that includes a potential field based on the agent's proximity to unexplored regions and the expected information gain from reaching them. Overall, the approach investigated in this paper merges the benefits of the adaptability of reinforcement learning-driven exploration policies and the guarantee ensured by explicit safety mechanisms. Extensive evaluations in simulated environments demonstrate that the approach enables efficient and safe exploration in cluttered environments. 

**Abstract (ZH)**: 基于图神经网络的高效安全探索策略在拥挤环境中的自主探索 

---
# Causality-enhanced Decision-Making for Autonomous Mobile Robots in Dynamic Environments 

**Title (ZH)**: 因果增强的自主移动机器人在动态环境中的决策-making 

**Authors**: Luca Castri, Gloria Beraldo, Nicola Bellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11901)  

**Abstract**: The growing integration of robots in shared environments -- such as warehouses, shopping centres, and hospitals -- demands a deep understanding of the underlying dynamics and human behaviours, including how, when, and where individuals engage in various activities and interactions. This knowledge goes beyond simple correlation studies and requires a more comprehensive causal analysis. By leveraging causal inference to model cause-and-effect relationships, we can better anticipate critical environmental factors and enable autonomous robots to plan and execute tasks more effectively. To this end, we propose a novel causality-based decision-making framework that reasons over a learned causal model to predict battery usage and human obstructions, understanding how these factors could influence robot task execution. Such reasoning framework assists the robot in deciding when and how to complete a given task. To achieve this, we developed also PeopleFlow, a new Gazebo-based simulator designed to model context-sensitive human-robot spatial interactions in shared workspaces. PeopleFlow features realistic human and robot trajectories influenced by contextual factors such as time, environment layout, and robot state, and can simulate a large number of agents. While the simulator is general-purpose, in this paper we focus on a warehouse-like environment as a case study, where we conduct an extensive evaluation benchmarking our causal approach against a non-causal baseline. Our findings demonstrate the efficacy of the proposed solutions, highlighting how causal reasoning enables autonomous robots to operate more efficiently and safely in dynamic environments shared with humans. 

**Abstract (ZH)**: 机器人在共享环境中的日益融合——如仓库、购物中心和医院——要求深刻理解其背后的动力学和人类行为，包括个体在不同时间和地点进行的各种活动和互动方式。这种知识超越了简单的相关性研究，需要更全面的因果分析。通过利用因果推断来建模因果关系，我们可以更好地预见关键环境因素，并使自主机器人更有效地规划和执行任务。为此，我们提出了一种基于因果关系的决策框架，该框架通过推理学习到的因果模型来预测电池使用和人类障碍，理解这些因素如何影响机器人任务执行。这种推理框架帮助机器人决定何时以及如何完成给定任务。为了实现这一点，我们还开发了PeopleFlow，这是一种基于Gazebo的新模拟器，用于模拟共享工作空间中上下文敏感的人机空间交互。PeopleFlow 能够模拟受时间、环境布局和机器人状态等因素影响的现实的人类和机器人轨迹，并可以模拟大量智能体。尽管模拟器具有通用性，但在本文中，我们以仓库环境作为案例研究，开展了广泛的评估，将我们提出的因果方法与非因果基线进行基准测试。我们的研究结果表明，所提出解决方案的有效性，强调了因果推理如何使自主机器人在与人类共存的动态环境中更高效、更安全地运行。 

---
# Real-Time Shape Estimation of Tensegrity Structures Using Strut Inclination Angles 

**Title (ZH)**: 基于杆件倾斜角的 tensegrity 结构实时形状估计方法 

**Authors**: Tufail Ahmad Bhat, Yuhei Yoshimitsu, Kazuki Wada, Shuhei Ikemoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11868)  

**Abstract**: Tensegrity structures are becoming widely used in robotics, such as continuously bending soft manipulators and mobile robots to explore unknown and uneven environments dynamically. Estimating their shape, which is the foundation of their state, is essential for establishing control. However, on-board sensor-based shape estimation remains difficult despite its importance, because tensegrity structures lack well-defined joints, which makes it challenging to use conventional angle sensors such as potentiometers or encoders for shape estimation. To our knowledge, no existing work has successfully achieved shape estimation using only onboard sensors such as Inertial Measurement Units (IMUs). This study addresses this issue by proposing a novel approach that uses energy minimization to estimate the shape. We validated our method through experiments on a simple Class 1 tensegrity structure, and the results show that the proposed algorithm can estimate the real-time shape of the structure using onboard sensors, even in the presence of external disturbances. 

**Abstract (ZH)**: tensegrity结构在机器人领域中的应用日益广泛，如连续弯曲的柔软 manipulator 和移动机器人用于动态探索未知和不平坦的环境。估计其形状是确定其状态的基础，并且对于建立控制至关重要。然而，基于机载传感器的形状估计仍然具有挑战性，尽管它非常重要，因为 tensegrity 结构缺乏明确的关节，使得使用传统的角度传感器（如电位计或编码器）进行形状估计变得困难。据我们所知，现有工作中尚无使用惯性测量单元（IMUs）等机载传感器成功实现形状估计的实例。本研究通过提出一种基于能量最小化的新型方法来解决这一问题。我们通过对一个简单的 Class 1 tensegrity 结构的实验验证了该方法，并结果表明，所提出的方法可以在存在外部干扰的情况下，使用机载传感器实时估计结构的形状。 

---
# Towards Forceful Robotic Foundation Models: a Literature Survey 

**Title (ZH)**: 面向力ful机器人基础模型：文献综述 

**Authors**: William Xie, Nikolaus Correll  

**Link**: [PDF](https://arxiv.org/pdf/2504.11827)  

**Abstract**: This article reviews contemporary methods for integrating force, including both proprioception and tactile sensing, in robot manipulation policy learning. We conduct a comparative analysis on various approaches for sensing force, data collection, behavior cloning, tactile representation learning, and low-level robot control. From our analysis, we articulate when and why forces are needed, and highlight opportunities to improve learning of contact-rich, generalist robot policies on the path toward highly capable touch-based robot foundation models. We generally find that while there are few tasks such as pouring, peg-in-hole insertion, and handling delicate objects, the performance of imitation learning models is not at a level of dynamics where force truly matters. Also, force and touch are abstract quantities that can be inferred through a wide range of modalities and are often measured and controlled implicitly. We hope that juxtaposing the different approaches currently in use will help the reader to gain a systemic understanding and help inspire the next generation of robot foundation models. 

**Abstract (ZH)**: 本文回顾了结合本体感受和触觉感知的当代机器人 manipulaton 力控制方法。我们对各种力感知方法、数据收集、行为克隆、触觉表示学习以及低级机器人控制的策略进行了比较分析。通过分析，我们阐述了何时以及为何需要力，强调了在通向基于触觉的高能力机器人基础模型过程中改进接触丰富的一般机器人策略的机会。我们发现，虽然有些任务如倒液体、孔内插入钉子和处理精细物体需要力，但模仿学习模型的性能还未达到足够细致的动力学特性，使得力的实际作用不显著。此外，力和触觉是可以通过多种模态推断出的抽象量，通常会隐式地进行测量和控制。我们希望通过对比当前使用的不同方法，帮助读者获得系统的理解，并启发下一代机器人基础模型的研发。 

---
# Multi-goal Rapidly Exploring Random Tree with Safety and Dynamic Constraints for UAV Cooperative Path Planning 

**Title (ZH)**: 具有安全和动态约束的多目标快速探索随机树无人机协同路径规划 

**Authors**: Thu Hang Khuat, Duy-Nam Bui, Hoa TT. Nguyen, Mien L. Trinh, Minh T. Nguyen, Manh Duong Phung  

**Link**: [PDF](https://arxiv.org/pdf/2504.11823)  

**Abstract**: Cooperative path planning is gaining its importance due to the increasing demand on using multiple unmanned aerial vehicles (UAVs) for complex missions. This work addresses the problem by introducing a new algorithm named MultiRRT that extends the rapidly exploring random tree (RRT) to generate paths for a group of UAVs to reach multiple goal locations at the same time. We first derive the dynamics constraint of the UAV and include it in the problem formulation. MultiRRT is then developed, taking into account the cooperative requirements and safe constraints during its path-searching process. The algorithm features two new mechanisms, node reduction and Bezier interpolation, to ensure the feasibility and optimality of the paths generated. Importantly, the interpolated paths are proven to meet the safety and dynamics constraints imposed by obstacles and the UAVs. A number of simulations, comparisons, and experiments have been conducted to evaluate the performance of the proposed approach. The results show that MultiRRT can generate collision-free paths for multiple UAVs to reach their goals with better scores in path length and smoothness metrics than state-of-the-art RRT variants including Theta-RRT, FN-RRT, RRT*, and RRT*-Smart. The generated paths are also tested in practical flights with real UAVs to evaluate their validity for cooperative tasks. The source code of the algorithm is available at this https URL 

**Abstract (ZH)**: 基于多无人飞行器协同的路径规划 Método de planificación de trayectorias cooperativa para un grupo de vehículos aéreos no tripulados mediante un nuevo algoritmo MultiRRT 

---
# Steerable rolling of a 1-DoF robot using an internal pendulum 

**Title (ZH)**: 使用内部摆锤实现单自由度机器人可引导的滚动 

**Authors**: Christopher Y. Xu, Jack Yan, Kathleen Lum, Justin K. Yim  

**Link**: [PDF](https://arxiv.org/pdf/2504.11748)  

**Abstract**: We present ROCK (Rolling One-motor Controlled rocK), a 1 degree-of-freedom robot consisting of a round shell and an internal pendulum. An uneven shell surface enables steering by using only the movement of the pendulum, allowing for mechanically simple designs that may be feasible to scale to large quantities or small sizes. We train a control policy using reinforcement learning in simulation and deploy it onto the robot to complete a rectangular trajectory. 

**Abstract (ZH)**: ROCK (Rolling One-motor Controlled rocK): 一个由圆形外壳和内部摆动机构组成的单自由度机器人 

---
# Inversion of biological strategies in engineering technology: in case underwater soft robot 

**Title (ZH)**: 生物策略在工程技术中的逆向应用：以水下软体机器人为例 

**Authors**: Siqing Chen, He Xua, Xueyu Zhang, Zhen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.11722)  

**Abstract**: This paper proposes a biomimetic design framework based on biological strategy inversion, aiming to systematically map solutions evolved in nature to the engineering field. By constructing a "Function-Behavior-Feature-Environment" (F-B-Cs in E) knowledge model, combined with natural language processing (NLP) and multi-criteria decision-making methods, it achieves efficient conversion from biological strategies to engineering solutions. Using underwater soft robot design as a case study, the effectiveness of the framework in optimizing drive mechanisms, power distribution, and motion pattern design is verified. This research provides scalable methodological support for interdisciplinary biomimetic innovation. 

**Abstract (ZH)**: 基于生物策略反转的生物仿生设计框架：从自然演化方案到工程解决方案的系统映射及其在水下软体机器人设计中的应用验证 

---
# Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports 

**Title (ZH)**: 安全与自主：以人为本的安全过滤器及其在智能辅助赛车运动中的应用 

**Authors**: Donggeon David Oh, Justin Lidard, Haimin Hu, Himani Sinhmar, Elle Lazarski, Deepak Gopinath, Emily S. Sumner, Jonathan A. DeCastro, Guy Rosman, Naomi Ehrich Leonard, Jaime Fernández Fisac  

**Link**: [PDF](https://arxiv.org/pdf/2504.11717)  

**Abstract**: We propose a human-centered safety filter (HCSF) for shared autonomy that significantly enhances system safety without compromising human agency. Our HCSF is built on a neural safety value function, which we first learn scalably through black-box interactions and then use at deployment to enforce a novel quality control barrier function (Q-CBF) safety constraint. Since this Q-CBF safety filter does not require any knowledge of the system dynamics for both synthesis and runtime safety monitoring and intervention, our method applies readily to complex, black-box shared autonomy systems. Notably, our HCSF's CBF-based interventions modify the human's actions minimally and smoothly, avoiding the abrupt, last-moment corrections delivered by many conventional safety filters. We validate our approach in a comprehensive in-person user study using Assetto Corsa-a high-fidelity car racing simulator with black-box dynamics-to assess robustness in "driving on the edge" scenarios. We compare both trajectory data and drivers' perceptions of our HCSF assistance against unassisted driving and a conventional safety filter. Experimental results show that 1) compared to having no assistance, our HCSF improves both safety and user satisfaction without compromising human agency or comfort, and 2) relative to a conventional safety filter, our proposed HCSF boosts human agency, comfort, and satisfaction while maintaining robustness. 

**Abstract (ZH)**: 基于人类中心的安全过滤器：提升共享自主性系统的安全性和用户体验 

---
# An Online Adaptation Method for Robust Depth Estimation and Visual Odometry in the Open World 

**Title (ZH)**: 开放世界中的稳健深度估计和视觉里程杆在线适应方法 

**Authors**: Xingwu Ji, Haochen Niu, Dexin Duan, Rendong Ying, Fei Wen, Peilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11698)  

**Abstract**: Recently, learning-based robotic navigation systems have gained extensive research attention and made significant progress. However, the diversity of open-world scenarios poses a major challenge for the generalization of such systems to practical scenarios. Specifically, learned systems for scene measurement and state estimation tend to degrade when the application scenarios deviate from the training data, resulting to unreliable depth and pose estimation. Toward addressing this problem, this work aims to develop a visual odometry system that can fast adapt to diverse novel environments in an online manner. To this end, we construct a self-supervised online adaptation framework for monocular visual odometry aided by an online-updated depth estimation module. Firstly, we design a monocular depth estimation network with lightweight refiner modules, which enables efficient online adaptation. Then, we construct an objective for self-supervised learning of the depth estimation module based on the output of the visual odometry system and the contextual semantic information of the scene. Specifically, a sparse depth densification module and a dynamic consistency enhancement module are proposed to leverage camera poses and contextual semantics to generate pseudo-depths and valid masks for the online adaptation. Finally, we demonstrate the robustness and generalization capability of the proposed method in comparison with state-of-the-art learning-based approaches on urban, in-house datasets and a robot platform. Code is publicly available at: this https URL. 

**Abstract (ZH)**: 近期，基于学习的机器人导航系统获得了广泛的研究关注并取得了显著进展。然而，开放世界场景的多样性对这类系统在实际场景中的泛化能力构成了重大挑战。具体而言，用于场景测量和状态估计的学到的系统在应用场景偏离训练数据时往往会退化，导致不稳定的深度和姿态估计。为解决这一问题，本工作旨在开发一种能够在线快速适应多样化新型环境的视觉里程计系统。为此，我们构建了一个由在线更新的深度估计模块辅助的半监督在线适应框架。首先，我们设计了一种配有轻量级细化模块的单目深度估计网络，以实现高效的在线适应。然后，我们基于视觉里程计系统输出和场景的上下文语义信息构建了一个半监督学习目标。具体来说，我们提出了稀疏深度稠密化模块和动态一致性增强模块，利用相机姿态和上下文语义生成伪深度和有效的掩码以实现在线适应。最后，我们通过在城市、室内数据集和机器人平台上与现有最先进的学习方法进行比较，展示了所提出方法的鲁棒性和泛化能力。相关代码已公开于此：this https URL。 

---
# DM-OSVP++: One-Shot View Planning Using 3D Diffusion Models for Active RGB-Based Object Reconstruction 

**Title (ZH)**: DM-OSVP++: 基于3D扩散模型的一次成像视图规划方法用于基于RGB的对象重建 

**Authors**: Sicong Pan, Liren Jin, Xuying Huang, Cyrill Stachniss, Marija Popović, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.11674)  

**Abstract**: Active object reconstruction is crucial for many robotic applications. A key aspect in these scenarios is generating object-specific view configurations to obtain informative measurements for reconstruction. One-shot view planning enables efficient data collection by predicting all views at once, eliminating the need for time-consuming online replanning. Our primary insight is to leverage the generative power of 3D diffusion models as valuable prior information. By conditioning on initial multi-view images, we exploit the priors from the 3D diffusion model to generate an approximate object model, serving as the foundation for our view planning. Our novel approach integrates the geometric and textural distributions of the object model into the view planning process, generating views that focus on the complex parts of the object to be reconstructed. We validate the proposed active object reconstruction system through both simulation and real-world experiments, demonstrating the effectiveness of using 3D diffusion priors for one-shot view planning. 

**Abstract (ZH)**: 基于3D扩散模型的主动物体重建中的单次视图规划关键技术 

---
# Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial Simultaneous Localization and Mapping 

**Title (ZH)**: 多普勒-SLAM: 多普勒辅助雷达-惯性与LiDAR-惯性同时定位与建图 

**Authors**: Dong Wang, Hannes Haag, Daniel Casado Herraez, Stefan May, Cyrill Stachniss, Andreas Nuechter  

**Link**: [PDF](https://arxiv.org/pdf/2504.11634)  

**Abstract**: Simultaneous localization and mapping (SLAM) is a critical capability for autonomous systems. Traditional SLAM approaches, which often rely on visual or LiDAR sensors, face significant challenges in adverse conditions such as low light or featureless environments. To overcome these limitations, we propose a novel Doppler-aided radar-inertial and LiDAR-inertial SLAM framework that leverages the complementary strengths of 4D radar, FMCW LiDAR, and inertial measurement units. Our system integrates Doppler velocity measurements and spatial data into a tightly-coupled front-end and graph optimization back-end to provide enhanced ego velocity estimation, accurate odometry, and robust mapping. We also introduce a Doppler-based scan-matching technique to improve front-end odometry in dynamic environments. In addition, our framework incorporates an innovative online extrinsic calibration mechanism, utilizing Doppler velocity and loop closure to dynamically maintain sensor alignment. Extensive evaluations on both public and proprietary datasets show that our system significantly outperforms state-of-the-art radar-SLAM and LiDAR-SLAM frameworks in terms of accuracy and robustness. To encourage further research, the code of our Doppler-SLAM and our dataset are available at: this https URL. 

**Abstract (ZH)**: 基于多传感器融合的 Doppler 辅助雷达-惯导和 LiDAR-惯导 SLAM 框架 

---
# RESPLE: Recursive Spline Estimation for LiDAR-Based Odometry 

**Title (ZH)**: 基于LiDAR的递归 spline 估计里程计(RESPLE) 

**Authors**: Ziyu Cao, William Talbot, Kailai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11580)  

**Abstract**: We present a novel recursive Bayesian estimation framework for continuous-time six-DoF dynamic motion estimation using B-splines. The state vector consists of a recurrent set of position control points and orientation control point increments, enabling a straightforward modification of the iterated extended Kalman filter without involving the error-state formulation. The resulting recursive spline estimator (RESPLE) provides a versatile, pragmatic and lightweight solution for motion estimation and is further exploited for direct LiDAR-based odometry, supporting integration of one or multiple LiDARs and an IMU. We conduct extensive real-world benchmarking based on public datasets and own experiments, covering aerial, wheeled, legged, and wearable platforms operating in indoor, urban, wild environments with diverse LiDARs. RESPLE-based solutions achieve superior estimation accuracy and robustness over corresponding state-of-the-art systems, while attaining real-time performance. Notably, our LiDAR-only variant outperforms existing LiDAR-inertial systems in scenarios without significant LiDAR degeneracy, and showing further improvements when additional LiDAR and inertial sensors are incorporated for more challenging conditions. We release the source code and own experimental datasets at this https URL . 

**Abstract (ZH)**: 一种基于B样条的新型递归贝叶斯连续时间六自由度动态运动估计框架 

---
# Probabilistic Task Parameterization of Tool-Tissue Interaction via Sparse Landmarks Tracking in Robotic Surgery 

**Title (ZH)**: 基于稀疏地标跟踪的机器人手术中工具-组织交互的概率任务参数化 

**Authors**: Yiting Wang, Yunxin Fan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11495)  

**Abstract**: Accurate modeling of tool-tissue interactions in robotic surgery requires precise tracking of deformable tissues and integration of surgical domain knowledge. Traditional methods rely on labor-intensive annotations or rigid assumptions, limiting flexibility. We propose a framework combining sparse keypoint tracking and probabilistic modeling that propagates expert-annotated landmarks across endoscopic frames, even with large tissue deformations. Clustered tissue keypoints enable dynamic local transformation construction via PCA, and tool poses, tracked similarly, are expressed relative to these frames. Embedding these into a Task-Parameterized Gaussian Mixture Model (TP-GMM) integrates data-driven observations with labeled clinical expertise, effectively predicting relative tool-tissue poses and enhancing visual understanding of robotic surgical motions directly from video data. 

**Abstract (ZH)**: 精确 modeling of 工具-组织交互在机器人手术中的建模需要精确跟踪可变形组织并整合手术领域知识。传统方法依赖于劳动密集型注释或刚性假设，限制了灵活性。我们提出了一种结合稀疏关键点跟踪和概率建模的框架，该框架能够在大量组织变形的情况下，将专家标注的关键点传播到内镜帧中。聚类组织关键点通过PCA_enable动态局部变换的构造，并以类似方式跟踪的工具姿态相对于这些框架进行表达。将这些内容嵌入到任务参数化高斯混合模型（TP-GMM）中，将数据驱动的观察与标记的临床专业知识整合起来，有效地预测相对工具-组织姿态并直接从视频数据中增强对手术运动的视觉理解。 

---
# Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning 

**Title (ZH)**: 通过多模态示范学习实现人类与机器人动作对齐 

**Authors**: Azizul Zahid, Jie Fan, Farong Wang, Ashton Dy, Sai Swaminathan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11493)  

**Abstract**: Understanding action correspondence between humans and robots is essential for evaluating alignment in decision-making, particularly in human-robot collaboration and imitation learning within unstructured environments. We propose a multimodal demonstration learning framework that explicitly models human demonstrations from RGB video with robot demonstrations in voxelized RGB-D space. Focusing on the "pick and place" task from the RH20T dataset, we utilize data from 5 users across 10 diverse scenes. Our approach combines ResNet-based visual encoding for human intention modeling and a Perceiver Transformer for voxel-based robot action prediction. After 2000 training epochs, the human model reaches 71.67% accuracy, and the robot model achieves 71.8% accuracy, demonstrating the framework's potential for aligning complex, multimodal human and robot behaviors in manipulation tasks. 

**Abstract (ZH)**: 理解人类与机器人之间的动作对应对于评估决策一致性至关重要，特别是在未结构化环境中的人机协作和模仿学习中。我们提出了一种多模态示范学习框架，该框架明确地从RGB视频中建模人类示范并在体素化RGB-D空间中建模机器人示范。我们利用RH20T数据集中5名用户在10个不同场景下的数据，重点关注“ pick and place ”任务。我们的方法结合了基于ResNet的视觉编码进行人类意图建模以及基于体素的机器人动作预测的Perceiver Transformer。经过2000个训练周期后，人类模型的准确率为71.67%，机器人模型的准确率为71.8%，表明该框架在操作任务中实现复杂多模态人类与机器人行为对齐的潜力。 

---
# Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions 

**Title (ZH)**: 空中安全：无人机反制方法综述、基准测试及未来方向 

**Authors**: Yifei Dong, Fengyi Wu, Sanjian Zhang, Guangyu Chen, Yuzhi Hu, Masumi Yano, Jingdong Sun, Siyu Huang, Feng Liu, Qi Dai, Zhi-Qi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11967)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs. 

**Abstract (ZH)**: 无人机(UAVs)在基础设施检测、监控及相关任务中不可或缺，但也带来了关键的安全挑战。本文综述了反无人机领域，聚焦于分类、检测和追踪三大核心目标，详细介绍了诸如基于扩散的数据合成、多模态融合、视觉-语言建模、半监督学习和强化学习等新兴方法。我们系统评估了单模态和多传感器管道（涵盖RGB、红外、音频、雷达和RF）中的最新解决方案，并讨论了大规模及对抗导向的基准测试。我们的分析揭示了实时性能、隐蔽检测和群嘲场景中的持续差距，强调了需要开发稳健、适应性强的反无人机系统。通过突出开放的研究方向，我们旨在促进创新并指导在广泛使用无人机的时代开发下一代防御策略。 

---
# Exploring Video-Based Driver Activity Recognition under Noisy Labels 

**Title (ZH)**: 基于嘈杂标签的视频驱动活动识别探究 

**Authors**: Linjuan Fan, Di Wen, Kunyu Peng, Kailun Yang, Jiaming Zhang, Ruiping Liu, Yufan Chen, Junwei Zheng, Jiamin Wu, Xudong Han, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11966)  

**Abstract**: As an open research topic in the field of deep learning, learning with noisy labels has attracted much attention and grown rapidly over the past ten years. Learning with label noise is crucial for driver distraction behavior recognition, as real-world video data often contains mislabeled samples, impacting model reliability and performance. However, label noise learning is barely explored in the driver activity recognition field. In this paper, we propose the first label noise learning approach for the driver activity recognition task. Based on the cluster assumption, we initially enable the model to learn clustering-friendly low-dimensional representations from given videos and assign the resultant embeddings into clusters. We subsequently perform co-refinement within each cluster to smooth the classifier outputs. Furthermore, we propose a flexible sample selection strategy that combines two selection criteria without relying on any hyperparameters to filter clean samples from the training dataset. We also incorporate a self-adaptive parameter into the sample selection process to enforce balancing across classes. A comprehensive variety of experiments on the public Drive&Act dataset for all granularity levels demonstrates the superior performance of our method in comparison with other label-denoising methods derived from the image classification field. The source code is available at this https URL. 

**Abstract (ZH)**: 深度学习领域中带有噪声标签的学习：一种驾驶活动识别中的噪声标签学习方法 

---
# GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision 

**Title (ZH)**: GrabS: 生成式具身代理用于无场景监督的3D物体分割 

**Authors**: Zihui Zhang, Yafei Yang, Hongtao Wen, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11754)  

**Abstract**: We study the hard problem of 3D object segmentation in complex point clouds without requiring human labels of 3D scenes for supervision. By relying on the similarity of pretrained 2D features or external signals such as motion to group 3D points as objects, existing unsupervised methods are usually limited to identifying simple objects like cars or their segmented objects are often inferior due to the lack of objectness in pretrained features. In this paper, we propose a new two-stage pipeline called GrabS. The core concept of our method is to learn generative and discriminative object-centric priors as a foundation from object datasets in the first stage, and then design an embodied agent to learn to discover multiple objects by querying against the pretrained generative priors in the second stage. We extensively evaluate our method on two real-world datasets and a newly created synthetic dataset, demonstrating remarkable segmentation performance, clearly surpassing all existing unsupervised methods. 

**Abstract (ZH)**: 我们研究在无需3D场景人工标签监督的情况下复杂点云中的3D物体分割难题。通过依赖预训练的2D特征相似性或外部信号（如运动）来分组3D点为物体，现有的无监督方法通常仅限于识别简单的物体，如汽车，或者由于预训练特征缺乏物体性，分割出的物体质量往往较差。本文提出了一种名为GrabS的新两阶段管道。我们方法的核心概念是在第一阶段从物体数据集中学习生成性和判别性物体中心先验，然后在第二阶段设计一个具身代理，通过查询预训练的生成性先验来学习发现多个物体。我们详细评估了该方法在两个真实世界数据集和一个新创建的合成数据集上的性能，展示了显著的分割性能，明显优于所有现有的无监督方法。 

---
# Linearity, Time Invariance, and Passivity of a Novice Person in Human Teleoperation 

**Title (ZH)**: 新手在人体遥控操作中的线性、时不变性和有界输入有界输出性 

**Authors**: David Black, Septimiu Salcudean  

**Link**: [PDF](https://arxiv.org/pdf/2504.11653)  

**Abstract**: Low-cost teleguidance of medical procedures is becoming essential to provide healthcare to remote and underserved communities. Human teleoperation is a promising new method for guiding a novice person with relatively high precision and efficiency through a mixed reality (MR) interface. Prior work has shown that the novice, or "follower", can reliably track the MR input with performance not unlike a telerobotic system. As a consequence, it is of interest to understand and control the follower's dynamics to optimize the system performance and permit stable and transparent bilateral teleoperation. To this end, linearity, time-invariance, inter-axis coupling, and passivity are important in teleoperation and controller design. This paper therefore explores these effects with regard to the follower person in human teleoperation. It is demonstrated through modeling and experiments that the follower can indeed be treated as approximately linear and time invariant, with little coupling and a large excess of passivity at practical frequencies. Furthermore, a stochastic model of the follower dynamics is derived. These results will permit controller design and analysis to improve the performance of human teleoperation. 

**Abstract (ZH)**: 低成本远程指导在医疗程序中的应用对于提供偏远和未充分服务社区的医疗服务变得至关重要。人类远程操作是一种有前景的新方法，通过混合现实接口，能够相对高精度和高效地引导新手人员。先前的研究表明，新手，即“追随者”，能够可靠地跟踪混合现实输入，其性能类似于遥控机器人系统。因此，了解和控制“追随者”的动力学对于优化系统性能、实现稳定透明的双边远程操作是重要的。本论文因此探讨了这些效应对人类远程操作中“追随者”人员的影响。通过建模和实验表明，“追随者”确实可以近似为线性且时间不变，具有少量耦合和在实用频率下存在大量过剩的耗散性。此外，还推导出“追随者”动力学的随机模型。这些结果将有助于控制器的设计与分析，以提高人类远程操作的性能。 

---
# LANGTRAJ: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation 

**Title (ZH)**: LANGTRAJ：基于语言条件的轨迹模拟扩散模型及数据集 

**Authors**: Wei-Jer Chang, Wei Zhan, Masayoshi Tomizuka, Manmohan Chandraker, Francesco Pittaluga  

**Link**: [PDF](https://arxiv.org/pdf/2504.11521)  

**Abstract**: Evaluating autonomous vehicles with controllability enables scalable testing in counterfactual or structured settings, enhancing both efficiency and safety. We introduce LangTraj, a language-conditioned scene-diffusion model that simulates the joint behavior of all agents in traffic scenarios. By conditioning on natural language inputs, LangTraj provides flexible and intuitive control over interactive behaviors, generating nuanced and realistic scenarios. Unlike prior approaches that depend on domain-specific guidance functions, LangTraj incorporates language conditioning during training, facilitating more intuitive traffic simulation control. We propose a novel closed-loop training strategy for diffusion models, explicitly tailored to enhance stability and realism during closed-loop simulation. To support language-conditioned simulation, we develop Inter-Drive, a large-scale dataset with diverse and interactive labels for training language-conditioned diffusion models. Our dataset is built upon a scalable pipeline for annotating agent-agent interactions and single-agent behaviors, ensuring rich and varied supervision. Validated on the Waymo Motion Dataset, LangTraj demonstrates strong performance in realism, language controllability, and language-conditioned safety-critical simulation, establishing a new paradigm for flexible and scalable autonomous vehicle testing. 

**Abstract (ZH)**: 基于可控性的自主车辆评估方法：在假设或结构化场景中实现可扩展测试，提高效率和安全性。我们引入了LangTraj，一种基于语言的场景扩散模型，用于模拟交通场景中所有代理的联合行为。通过基于自然语言输入，LangTraj提供了灵活且直观的交互行为控制，生成细腻且逼真的场景。与依赖特定领域指导函数的前ethod不同，LangTraj在训练过程中引入了语言条件，使交通模拟控制更加直观。我们提出了一种新的闭环训练策略，专门用于增强闭环仿真过程中的稳定性和真实性。为了支持基于语言的仿真，我们开发了Inter-Drive，一个包含多种互动标签的大型数据集，用于训练基于语言的扩散模型。该数据集基于可扩展的注释管道，确保丰富的监督信息。在Waymo Motion数据集上的验证表明，LangTraj在真实性、语言可控性和语言条件下的关键安全仿真方面表现出强劲性能，建立了灵活和可扩展的自主车辆测试的新范式。 

---
# Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models 

**Title (ZH)**: 利用机载部署的大语言模型增强自动驾驶系统 

**Authors**: Nicolas Baumann, Cheng Hu, Paviththiren Sivasothilingam, Haotong Qin, Lei Xie, Michele Magno, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2504.11514)  

**Abstract**: Neural Networks (NNs) trained through supervised learning struggle with managing edge-case scenarios common in real-world driving due to the intractability of exhaustive datasets covering all edge-cases, making knowledge-driven approaches, akin to how humans intuitively detect unexpected driving behavior, a suitable complement to data-driven methods. This work proposes a hybrid architecture combining low-level Model Predictive Controller (MPC) with locally deployed Large Language Models (LLMs) to enhance decision-making and Human Machine Interaction (HMI). The DecisionxLLM module evaluates robotic state information against natural language instructions to ensure adherence to desired driving behavior. The MPCxLLM module then adjusts MPC parameters based on LLM-generated insights, achieving control adaptability while preserving the safety and constraint guarantees of traditional MPC systems. Further, to enable efficient on-board deployment and to eliminate dependency on cloud connectivity, we shift processing to the on-board computing platform: We propose an approach that exploits Retrieval Augmented Generation (RAG), Low Rank Adaptation (LoRA) fine-tuning, and quantization. Experimental results demonstrate that these enhancements yield significant improvements in reasoning accuracy by up to 10.45%, control adaptability by as much as 52.2%, and up to 10.5x increase in computational efficiency (tokens/s), validating the proposed framework's practicality for real-time deployment even on down-scaled robotic platforms. This work bridges high-level decision-making with low-level control adaptability, offering a synergistic framework for knowledge-driven and adaptive Autonomous Driving Systems (ADS). 

**Abstract (ZH)**: 通过监督学习训练的神经网络（NNs）在处理真实驾驶场景中的边缘案例时存在困难，因为难以获得涵盖所有边缘案例的详尽数据集，因此知识驱动的方法，类似于人类如何直观检测意外驾驶行为，是数据驱动方法的合适补充。本文提出了一种结合低层级模型预测控制器（MPC）和局部部署的大语言模型（LLMs）的混合架构，以增强决策能力和人机交互（HMI）。DecisionxLLM模块通过评估机器人状态信息与自然语言指令的一致性来确保驾驶行为符合预期。MPCxLLM模块根据LLM生成的见解调整MPC参数，实现控制适应性的同时保持传统MPC系统的安全性和约束保障。此外，为了实现高效的车载部署并消除对云连接的依赖，我们将处理转移到车载计算平台：我们提出了一种利用检索增强生成（RAG）、低秩适应（LoRA）微调和量化的方法。实验结果表明，这些增强措施在推理准确性上提高了10.45%，控制适应性提高了52.2%，计算效率提高了10.5倍（每秒token数），验证了所提框架在即使在缩小规模的机器人平台上也能实现实时部署的实用性。本文将高层次决策与低层次控制适应性相结合，提供了一种知识驱动和自适应自动驾驶系统（ADS）的协同框架。 

---
# Cross-cultural Deployment of Autonomous Vehicles Using Data-light Inverse Reinforcement Learning 

**Title (ZH)**: 基于数据轻量级逆强化学习的跨文化自动驾驶车辆部署 

**Authors**: Hongliang Lu, Shuqi Shen, Junjie Yang, Chao Lu, Xinhu Zheng, Hai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11506)  

**Abstract**: More than the adherence to specific traffic regulations, driving culture touches upon a more implicit part - an informal, conventional, collective behavioral pattern followed by drivers - that varies across countries, regions, and even cities. Such cultural divergence has become one of the biggest challenges in deploying autonomous vehicles (AVs) across diverse regions today. The current emergence of data-driven methods has shown a potential solution to enable culture-compatible driving through learning from data, but what if some underdeveloped regions cannot provide sufficient local data to inform driving culture? This issue is particularly significant for a broader global AV market. Here, we propose a cross-cultural deployment scheme for AVs, called data-light inverse reinforcement learning, designed to re-calibrate culture-specific AVs and assimilate them into other cultures. First, we report the divergence in driving cultures through a comprehensive comparative analysis of naturalistic driving datasets on highways from three countries: Germany, China, and the USA. Then, we demonstrate the effectiveness of our scheme by testing the expeditious cross-cultural deployment across these three countries, with cumulative testing mileage of over 56084 km. The performance is particularly advantageous when cross-cultural deployment is carried out without affluent local data. Results show that we can reduce the dependence on local data by a margin of 98.67% at best. This study is expected to bring a broader, fairer AV global market, particularly in those regions that lack enough local data to develop culture-compatible AVs. 

**Abstract (ZH)**: 跨文化自主驾驶车辆部署方案：基于数据稀疏逆强化学习的方法 

---
# snnTrans-DHZ: A Lightweight Spiking Neural Network Architecture for Underwater Image Dehazing 

**Title (ZH)**: snnTrans-DHZ：一种用于水下图像去雾的轻量级脉冲神经网络架构 

**Authors**: Vidya Sudevan, Fakhreddine Zayer, Rizwana Kausar, Sajid Javed, Hamad Karki, Giulia De Masi, Jorge Dias  

**Link**: [PDF](https://arxiv.org/pdf/2504.11482)  

**Abstract**: Underwater image dehazing is critical for vision-based marine operations because light scattering and absorption can severely reduce visibility. This paper introduces snnTrans-DHZ, a lightweight Spiking Neural Network (SNN) specifically designed for underwater dehazing. By leveraging the temporal dynamics of SNNs, snnTrans-DHZ efficiently processes time-dependent raw image sequences while maintaining low power consumption. Static underwater images are first converted into time-dependent sequences by repeatedly inputting the same image over user-defined timesteps. These RGB sequences are then transformed into LAB color space representations and processed concurrently. The architecture features three key modules: (i) a K estimator that extracts features from multiple color space representations; (ii) a Background Light Estimator that jointly infers the background light component from the RGB-LAB images; and (iii) a soft image reconstruction module that produces haze-free, visibility-enhanced outputs. The snnTrans-DHZ model is directly trained using a surrogate gradient-based backpropagation through time (BPTT) strategy alongside a novel combined loss function. Evaluated on the UIEB benchmark, snnTrans-DHZ achieves a PSNR of 21.68 dB and an SSIM of 0.8795, and on the EUVP dataset, it yields a PSNR of 23.46 dB and an SSIM of 0.8439. With only 0.5670 million network parameters, and requiring just 7.42 GSOPs and 0.0151 J of energy, the algorithm significantly outperforms existing state-of-the-art methods in terms of efficiency. These features make snnTrans-DHZ highly suitable for deployment in underwater robotics, marine exploration, and environmental monitoring. 

**Abstract (ZH)**: 基于时空动态的轻量级脉冲神经网络 underwater image dehazing for vision-based marine operations: snnTrans-DHZ 

---
