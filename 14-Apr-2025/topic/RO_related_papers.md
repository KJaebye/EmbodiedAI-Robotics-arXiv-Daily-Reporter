# BiFlex: A Passive Bimodal Stiffness Flexible Wrist for Manipulation in Unstructured Environments 

**Title (ZH)**: BiFlex: 一种用于结构化环境中操作的被动双模态刚度可调手腕 

**Authors**: Gu-Cheol Jeong, Stefano Dalla Gasperina, Ashish D. Deshpande, Lillian Chin, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2504.08706)  

**Abstract**: Robotic manipulation in unstructured, humancentric environments poses a dual challenge: achieving the precision need for delicate free-space operation while ensuring safety during unexpected contact events. Traditional wrists struggle to balance these demands, often relying on complex control schemes or complicated mechanical designs to mitigate potential damage from force overload. In response, we present BiFlex, a flexible robotic wrist that uses a soft buckling honeycomb structure to provides a natural bimodal stiffness response. The higher stiffness mode enables precise household object manipulation, while the lower stiffness mode provides the compliance needed to adapt to external forces. We design BiFlex to maintain a fingertip deflection of less than 1 cm while supporting loads up to 500g and create a BiFlex wrist for many grippers, including Panda, Robotiq, and BaRiFlex. We validate BiFlex under several real-world experimental evaluations, including surface wiping, precise pick-and-place, and grasping under environmental constraints. We demonstrate that BiFlex simplifies control while maintaining precise object manipulation and enhanced safety in real-world applications. 

**Abstract (ZH)**: 非结构化、以人为中心环境中的人形机器人Manipulation面临的是一种双重挑战：需要实现对精细自由空间操作的精度要求，同时在意外接触事件中确保安全性。传统的手腕结构难以平衡这些需求，往往依赖于复杂的控制方案或复杂的机械设计来减轻负载过大的潜在损伤。为应对这一挑战，我们提出了BiFlex柔性机器人手腕，该手腕采用软折纸蜂窝结构，提供自然的双模刚度响应。更高的刚度模式能够实现对家用物体的精确操作，而较低的刚度模式则提供了适应外部力所需的柔顺性。我们设计BiFlex以在支持高达500g的负载时保持指尖形变小于1cm，并为多种握持器，包括Panda、Robotiq和BaRiFlex，设计了BiFlex手腕。我们通过多种实际实验验证了BiFlex，包括表面擦拭、精确拾放和环境约束下的抓取。我们展示了BiFlex在实际应用中简化控制的同时，保持了精确的物体操作和增强的安全性。 

---
# Performance Evaluation of Trajectory Tracking Controllers for a Quadruped Robot Leg 

**Title (ZH)**: 四足机器人腿部轨迹跟踪控制器性能评估 

**Authors**: Hossein Shojaei, Hamid Rahmanei, Seyed Hossein Sadati  

**Link**: [PDF](https://arxiv.org/pdf/2504.08698)  

**Abstract**: The complexities in the dynamic model of the legged robots make it necessary to utilize model-free controllers in the task of trajectory tracking. In This paper, an adaptive transpose Jacobian approach is proposed to deal with the dynamic model complexity, which utilizes an adaptive PI-algorithm to adjust the control gains. The performance of the proposed control algorithm is compared with the conventional transpose Jacobian and sliding mode control algorithms and evaluated by the root mean square of the errors and control input energy criteria. In order to appraise the effectiveness of the proposed control system, simulations are carried out in MATLAB/Simulink software for a quadruped robot leg for semi-elliptical path tracking. The obtained results show that the proposed adaptive transpose Jacobian reduces the overshoot and root mean square of the errors and at the same time, decreases the control input energy. Moreover, transpose Jacobin and adaptive transpose Jacobian are more robust to changes in initial conditions compared to the conventional sliding mode control. Furthermore, sliding mode control performs well up to 20% uncertainties in the parameters due to its model-based nature, whereas the transpose Jacobin and the proposed adaptive transpose Jacobian algorithms show promising results even in higher mass uncertainties. 

**Abstract (ZH)**: 具有自适应转置雅可比的轨迹跟踪控制算法研究 

---
# Pobogot -- An Open-Hardware Open-Source Low Cost Robot for Swarm Robotics 

**Title (ZH)**: Pobogot -- 一种用于群机器人lopen硬件开源低成本机器人 

**Authors**: Alessia Loi, Loona Macabre, Jérémy Fersula, Keivan Amini, Leo Cazenille, Fabien Caura, Alexandre Guerre, Stéphane Gourichon, Olivier Dauchot, Nicolas Bredeche  

**Link**: [PDF](https://arxiv.org/pdf/2504.08686)  

**Abstract**: This paper describes the Pogobot, an open-source and open-hardware platform specifically designed for research involving swarm robotics. Pogobot features vibration-based locomotion, infrared communication, and an array of sensors in a cost-effective package (approx. 250~euros/unit). The platform's modular design, comprehensive API, and extensible architecture facilitate the implementation of swarm intelligence algorithms and distributed online reinforcement learning algorithms. Pogobots offer an accessible alternative to existing platforms while providing advanced capabilities including directional communication between units. More than 200 Pogobots are already being used on a daily basis at Sorbonne Université and PSL to study self-organizing systems, programmable active matter, discrete reaction-diffusion-advection systems as well as models of social learning and evolution. 

**Abstract (ZH)**: 本文描述了Pogobot，一个专为 swarm robotics 研究设计的开源且开放硬件平台。Pogobot 具有基于振动的移动方式、红外通信以及成本效益高的传感器阵列（每单位约250~欧元）。该平台的模块化设计、全面的应用程序接口（API）和可扩展架构便于实现 swarm 智能算法和分布式在线强化学习算法。Pogobots 提供了一种现成平台的可访问替代方案，同时具备包括单元间方向性通信在内的高级功能。目前，已有超过200个Pogobots在索邦大学和巴黎文理研究大学每天用于研究自组织系统、可编程活性物质、离散反应-扩散-传输系统以及社会学习和进化的模型。 

---
# Safe Flow Matching: Robot Motion Planning with Control Barrier Functions 

**Title (ZH)**: 安全流匹配：基于控制屏障函数的机器人运动规划 

**Authors**: Xiaobing Dai, Dian Yu, Shanshan Zhang, Zewen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08661)  

**Abstract**: Recent advances in generative modeling have led to promising results in robot motion planning, particularly through diffusion and flow-based models that capture complex, multimodal trajectory distributions. However, these methods are typically trained offline and remain limited when faced with unseen environments or dynamic constraints, often lacking explicit mechanisms to ensure safety during deployment. In this work, we propose, Safe Flow Matching (SafeFM), a motion planning approach for trajectory generation that integrates flow matching with safety guarantees. By incorporating the proposed flow matching barrier functions, SafeFM ensures that generated trajectories remain within safe regions throughout the planning horizon, even in the presence of previously unseen obstacles or state-action constraints. Unlike diffusion-based approaches, our method allows for direct, efficient sampling of constraint-satisfying trajectories, making it well-suited for real-time motion planning. We evaluate SafeFM on a diverse set of tasks, including planar robot navigation and 7-DoF manipulation, demonstrating superior safety, generalization, and planning performance compared to state-of-the-art generative planners. Comprehensive resources are available on the project website: this https URL 

**Abstract (ZH)**: Recent advances in generative modeling have led to promising results in robot motion planning, particularly through diffusion and flow-based models that capture complex, multimodal trajectory distributions.然而，这些方法通常在离线训练，并且在面对未见过的环境或动态约束时受到限制，往往缺乏明确的机制来确保部署过程中的安全性。在这种情况下，我们提出了一种名为Safe Flow Matching (SafeFM)的运动规划方法，该方法将流匹配与安全性保证结合，通过引入所提出的流匹配障碍函数，SafeFM 确保生成的轨迹在整个计划时段内保持在安全区域内，即使存在未见过的障碍物或状态-动作约束。与基于扩散的方法不同，我们的方法允许直接、高效地抽样满足约束的轨迹，使其非常适合实时运动规划。我们通过一系列任务，包括平面机器人导航和7自由度操作，评估了SafeFM，结果显示其在安全性、泛化能力和规划性能方面均优于最先进的生成式规划器。更多资源可在项目网站获取：this https URL。 

---
# Enabling Safety for Aerial Robots: Planning and Control Architectures 

**Title (ZH)**: 确保空中机器人安全：规划与控制架构 

**Authors**: Kaleb Ben Naveed, Devansh R. Agrawal, Daniel M. Cherenson, Haejoon Lee, Alia Gilbert, Hardik Parwana, Vishnu S. Chipade, William Bentz, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.08601)  

**Abstract**: Ensuring safe autonomy is crucial for deploying aerial robots in real-world applications. However, safety is a multifaceted challenge that must be addressed from multiple perspectives, including navigation in dynamic environments, operation under resource constraints, and robustness against adversarial attacks and uncertainties. In this paper, we present the authors' recent work that tackles some of these challenges and highlights key aspects that must be considered to enhance the safety and performance of autonomous aerial systems. All presented approaches are validated through hardware experiments. 

**Abstract (ZH)**: 确保自主性的安全性对于在实际应用中部署无人机至关重要。然而，安全是一个多方面的问题，必须从导航动态环境、资源约束下的操作、以及对抗攻击和不确定性下的鲁棒性等多个角度来解决。在本文中，我们展示了作者最近的工作，解决了一些挑战，并强调了必须考虑的关键方面，以提高自主无人机系统的安全性和性能。所有提出的方法都通过硬件实验进行了验证。 

---
# Ready, Bid, Go! On-Demand Delivery Using Fleets of Drones with Unknown, Heterogeneous Energy Storage Constraints 

**Title (ZH)**: 随时准备，出价出发！基于未知且异构能量存储约束无人机队的按需交付 

**Authors**: Mohamed S. Talamali, Genki Miyauchi, Thomas Watteyne, Micael S. Couceiro, Roderich Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.08585)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are expected to transform logistics, reducing delivery time, costs, and emissions. This study addresses an on-demand delivery , in which fleets of UAVs are deployed to fulfil orders that arrive stochastically. Unlike previous work, it considers UAVs with heterogeneous, unknown energy storage capacities and assumes no knowledge of the energy consumption models. We propose a decentralised deployment strategy that combines auction-based task allocation with online learning. Each UAV independently decides whether to bid for orders based on its energy storage charge level, the parcel mass, and delivery distance. Over time, it refines its policy to bid only for orders within its capability. Simulations using realistic UAV energy models reveal that, counter-intuitively, assigning orders to the least confident bidders reduces delivery times and increases the number of successfully fulfilled orders. This strategy is shown to outperform threshold-based methods which require UAVs to exceed specific charge levels at deployment. We propose a variant of the strategy which uses learned policies for forecasting. This enables UAVs with insufficient charge levels to commit to fulfilling orders at specific future times, helping to prioritise early orders. Our work provides new insights into long-term deployment of UAV swarms, highlighting the advantages of decentralised energy-aware decision-making coupled with online learning in real-world dynamic environments. 

**Abstract (ZH)**: 无人飞机(UAVs)有望通过减少交付时间和成本、降低排放来变革物流。本文研究了按需送货问题，在该问题中，调度员将派遣具有异构且未知能量存储容量的无人机机群来履行随机到达的订单。不同于以往的研究，本研究假设没有无人机能量消耗模型的知识，并提出了一种结合拍卖式任务分配和在线学习的分散部署策略。每架无人机根据其能量存储电荷水平、包裹质量和送货距离独立决定是否竞标订单，并随着时间的推移不断优化其仅竞标自己能够胜任的订单的策略。通过使用实际的无人机能量模型进行仿真，结果显示，反直觉地，将订单分配给最不自信的竞标者可以减少交付时间并增加成功完成的订单数量。该策略被证明优于要求无人机在部署时达到特定电荷水平的阈值方法。我们提出了一种使用学习策略进行预测的策略变体，这使得电荷水平不足的无人机能够承诺在特定未来时间履行订单，有助于优先处理早期订单。本研究为无人机机群的长期部署提供了新的见解，强调了在现实动态环境中的分散能耗意识决策与在线学习相结合的优势。 

---
# CATCH-FORM-3D: Compliance-Aware Tactile Control and Hybrid Deformation Regulation for 3D Viscoelastic Object Manipulation 

**Title (ZH)**: CATCH-FORM-3D：弹性体合规控制与混合变形调节的触觉控制及3D黏弹性物体操作 

**Authors**: Hongjun Ma, Weichang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08238)  

**Abstract**: This paper investigates a framework (CATCH-FORM-3D) for the precise contact force control and surface deformation regulation in viscoelastic material manipulation. A partial differential equation (PDE) is proposed to model the spatiotemporal stress-strain dynamics, integrating 3D Kelvin-Voigt (stiffness-damping) and Maxwell (diffusion) effects to capture the material's viscoelastic behavior. Key mechanical parameters (stiffness, damping, diffusion coefficients) are estimated in real time via a PDE-driven observer. This observer fuses visual-tactile sensor data and experimentally validated forces to generate rich regressor signals. Then, an inner-outer loop control structure is built up. In the outer loop, the reference deformation is updated by a novel admittance control law, a proportional-derivative (PD) feedback law with contact force measurements, ensuring that the system responds adaptively to external interactions. In the inner loop, a reaction-diffusion PDE for the deformation tracking error is formulated and then exponentially stabilized by conforming the contact surface to analytical geometric configurations (i.e., defining Dirichlet boundary conditions). This dual-loop architecture enables the effective deformation regulation in dynamic contact environments. Experiments using a PaXini robotic hand demonstrate sub-millimeter deformation accuracy and stable force tracking. The framework advances compliant robotic interactions in applications like industrial assembly, polymer shaping, surgical treatment, and household service. 

**Abstract (ZH)**: 一种用于粘弹性材料操作的精确接触力控制和表面变形调节框架（CATCH-FORM-3D） 

---
# CATCH-FORM-ACTer: Compliance-Aware Tactile Control and Hybrid Deformation Regulation-Based Action Transformer for Viscoelastic Object Manipulation 

**Title (ZH)**: CATCH-FORM-ACTer: 遵从意识触觉控制与混合变形调节驱动的动作变换器在粘弹性物体操作中的应用 

**Authors**: Hongjun Ma, Weichang Li, Jingwei Zhang, Shenlai He, Xiaoyan Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08232)  

**Abstract**: Automating contact-rich manipulation of viscoelastic objects with rigid robots faces challenges including dynamic parameter mismatches, unstable contact oscillations, and spatiotemporal force-deformation coupling. In our prior work, a Compliance-Aware Tactile Control and Hybrid Deformation Regulation (CATCH-FORM-3D) strategy fulfills robust and effective manipulations of 3D viscoelastic objects, which combines a contact force-driven admittance outer loop and a PDE-stabilized inner loop, achieving sub-millimeter surface deformation accuracy. However, this strategy requires fine-tuning of object-specific parameters and task-specific calibrations, to bridge this gap, a CATCH-FORM-ACTer is proposed, by enhancing CATCH-FORM-3D with a framework of Action Chunking with Transformer (ACT). An intuitive teleoperation system performs Learning from Demonstration (LfD) to build up a long-horizon sensing, decision-making and execution sequences. Unlike conventional ACT methods focused solely on trajectory planning, our approach dynamically adjusts stiffness, damping, and diffusion parameters in real time during multi-phase manipulations, effectively imitating human-like force-deformation modulation. Experiments on single arm/bimanual robots in three tasks show better force fields patterns and thus 10%-20% higher success rates versus conventional methods, enabling precise, safe interactions for industrial, medical or household scenarios. 

**Abstract (ZH)**: 基于Transformer的动作切片增强的Compliance-Aware触觉控制和混合变形调节策略（CATCH-FORM-ACTer）：应用于粘弹物体的刚性机器人接触丰富操作自动化 

---
# Leveraging Passive Compliance of Soft Robotics for Physical Human-Robot Collaborative Manipulation 

**Title (ZH)**: 利用软机器人领域的被动合规性进行物理人机协作操作 

**Authors**: Dallin L. Cordon, Shaden Moss, Marc Killpack, John L. Salmon  

**Link**: [PDF](https://arxiv.org/pdf/2504.08184)  

**Abstract**: This work represents an initial benchmark of a large-scale soft robot performing physical, collaborative manipulation of a long, extended object with a human partner. The robot consists of a pneumatically-actuated, three-link continuum soft manipulator mounted to an omni-directional mobile base. The system level configuration of the robot and design of the collaborative manipulation (co-manipulation) study are presented. The initial results, both quantitative and qualitative, are directly compared to previous similar human-human co-manipulation studies. These initial results show promise in the ability for large-scale soft robots to perform comparably to human partners acting as non-visual followers in a co-manipulation task. Furthermore, these results challenge traditional soft robot strength limitations and indicate potential for applications requiring strength and adaptability. 

**Abstract (ZH)**: 这项工作代表了对一种大型软机器人进行物理协同操作的初步评估，该机器人与人类伙伴共同操作一个长伸展物体。该机器人由气动驱动的三连杆连续软 manipulator安装在全向移动基座上。介绍了机器人的系统级配置和协作操作（co-manipulation）研究的设计。初始结果，无论是定量的还是定性的，都直接与之前的类似人类-人类协作操作研究进行了比较。这些初始结果表明，大型软机器人有能力在协同操作任务中与作为非视觉跟随者的同人类搭档表现得相媲美。此外，这些结果挑战了传统软机器人的力量限制，表明了在需要力量和适应性的应用中潜在的应用价值。 

---
# External-Wrench Estimation for Aerial Robots Exploiting a Learned Model 

**Title (ZH)**: 基于学习模型的飞行机器人外部力矩估计 

**Authors**: Ayham Alharbat, Gabriele Ruscelli, Roberto Diversi, Abeje Mersha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08156)  

**Abstract**: This paper presents an external wrench estimator that uses a hybrid dynamics model consisting of a first-principles model and a neural network. This framework addresses one of the limitations of the state-of-the-art model-based wrench observers: the wrench estimation of these observers comprises the external wrench (e.g. collision, physical interaction, wind); in addition to residual wrench (e.g. model parameters uncertainty or unmodeled dynamics). This is a problem if these wrench estimations are to be used as wrench feedback to a force controller, for example. In the proposed framework, a neural network is combined with a first-principles model to estimate the residual dynamics arising from unmodeled dynamics and parameters uncertainties, then, the hybrid trained model is used to estimate the external wrench, leading to a wrench estimation that has smaller contributions from the residual dynamics, and affected more by the external wrench. This method is validated with numerical simulations of an aerial robot in different flying scenarios and different types of residual dynamics, and the statistical analysis of the results shows that the wrench estimation error has improved significantly compared to a model-based wrench observer using only a first-principles model. 

**Abstract (ZH)**: 本文提出了一种使用结合了先验模型和神经网络的混合动力学模型的外部力矩估计器。该框架解决了当前基于模型的力矩观察器的一项局限性：这些观察器的力矩估计不仅包括外部力矩（例如碰撞、物理交互、风），还包含残余力矩（例如模型参数不确定性或未建模动态）。如果这些力矩估计用于力控制器的力矩反馈，这将是一个问题。在所提出的框架中，结合使用先验模型和神经网络来估计由未建模动态和参数不确定性引起的部分运动学差异，然后使用混合训练模型来估计外部力矩，从而使得力矩估计的主要贡献来自于外部力矩，而非残余运动学差异。该方法通过在不同飞行场景和不同类型的残余运动学条件下对飞行机器人进行数值模拟进行了验证，并对结果的统计分析表明，与仅使用先验模型的基于模型的力矩观察器相比，力矩估计误差有了显著改善。 

---
# Threading the Needle: Test and Evaluation of Early Stage UAS Capabilities to Autonomously Navigate GPS-Denied Environments in the DARPA Fast Lightweight Autonomy (FLA) Program 

**Title (ZH)**: 针线活：测试与评估早期阶段自主飞行系统在GPS受限环境中的自主导航能力——DARPA快速轻量级自主（FLA）计划中的研究 

**Authors**: Adam Norton, Holly Yanco  

**Link**: [PDF](https://arxiv.org/pdf/2504.08122)  

**Abstract**: The DARPA Fast Lightweight Autonomy (FLA) program (2015 - 2018) served as a significant milestone in the development of UAS, particularly for autonomous navigation through unknown GPS-denied environments. Three performing teams developed UAS using a common hardware platform, focusing their contributions on autonomy algorithms and sensing. Several experiments were conducted that spanned indoor and outdoor environments, increasing in complexity over time. This paper reviews the testing methodology developed in order to benchmark and compare the performance of each team, each of the FLA Phase 1 experiments that were conducted, and a summary of the Phase 1 results. 

**Abstract (ZH)**: DARPA快速轻量级自主（FLA）计划（2015-2018）在无人飞行器自主导航技术发展中的里程碑作用：通过未知GPS受限环境的自主导航。本文回顾了用于评估和比较各参赛团队性能的测试方法，以及FLA第一阶段的所有实验和第一阶段结果的总结。 

---
# RL-based Control of UAS Subject to Significant Disturbance 

**Title (ZH)**: 基于RL的受显著干扰影响的UAS控制 

**Authors**: Kousheek Chakraborty, Thijs Hof, Ayham Alharbat, Abeje Mersha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08114)  

**Abstract**: This paper proposes a Reinforcement Learning (RL)-based control framework for position and attitude control of an Unmanned Aerial System (UAS) subjected to significant disturbance that can be associated with an uncertain trigger signal. The proposed method learns the relationship between the trigger signal and disturbance force, enabling the system to anticipate and counteract the impending disturbances before they occur. We train and evaluate three policies: a baseline policy trained without exposure to the disturbance, a reactive policy trained with the disturbance but without the trigger signal, and a predictive policy that incorporates the trigger signal as an observation and is exposed to the disturbance during training. Our simulation results show that the predictive policy outperforms the other policies by minimizing position deviations through a proactive correction maneuver. This work highlights the potential of integrating predictive cues into RL frameworks to improve UAS performance. 

**Abstract (ZH)**: 基于强化学习的预测控制框架在不确定触发信号关联显著干扰下无人空中系统的位姿控制 

---
# Interior Point Differential Dynamic Programming, Redux 

**Title (ZH)**: 重新审视内部点微分动态规划 

**Authors**: Ming Xu, Stephen Gould, Iman Shames  

**Link**: [PDF](https://arxiv.org/pdf/2504.08278)  

**Abstract**: We present IPDDP2, a structure-exploiting algorithm for solving discrete-time, finite horizon optimal control problems with nonlinear constraints. Inequality constraints are handled using a primal-dual interior point formulation and step acceptance for equality constraints follows a line-search filter approach. The iterates of the algorithm are derived under the Differential Dynamic Programming (DDP) framework. Our numerical experiments evaluate IPDDP2 on four robotic motion planning problems. IPDDP2 reliably converges to low optimality error and exhibits local quadratic and global convergence from remote starting points. Notably, we showcase the robustness of IPDDP2 by using it to solve a contact-implicit, joint limited acrobot swing-up problem involving complementarity constraints from a range of initial conditions. We provide a full implementation of IPDDP2 in the Julia programming language. 

**Abstract (ZH)**: IPDDP2：一种用于求解具有非线性约束的离散时间有限时段最优控制问题的结构利用算法 

---
