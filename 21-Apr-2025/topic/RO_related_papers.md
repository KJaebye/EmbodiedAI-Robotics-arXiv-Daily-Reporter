# Unified Manipulability and Compliance Analysis of Modular Soft-Rigid Hybrid Fingers 

**Title (ZH)**: 模块化软硬混合手指的统一操作与顺应性分析 

**Authors**: Jianshu Zhou, Boyuan Liang, Junda Huang, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2504.13800)  

**Abstract**: This paper presents a unified framework to analyze the manipulability and compliance of modular soft-rigid hybrid robotic fingers. The approach applies to both hydraulic and pneumatic actuation systems. A Jacobian-based formulation maps actuator inputs to joint and task-space responses. Hydraulic actuators are modeled under incompressible assumptions, while pneumatic actuators are described using nonlinear pressure-volume relations. The framework enables consistent evaluation of manipulability ellipsoids and compliance matrices across actuation modes. We validate the analysis using two representative hands: DexCo (hydraulic) and Edgy-2 (pneumatic). Results highlight actuation-dependent trade-offs in dexterity and passive stiffness. These findings provide insights for structure-aware design and actuator selection in soft-rigid robotic fingers. 

**Abstract (ZH)**: 本文提出了一种统一框架来分析模块化软硬混合机器人手指的可控性和顺应性。该方法适用于液压和气动驱动系统。基于雅可比的表示将驱动器输入映射到关节空间和任务空间的响应。假定不可压缩流体条件建模液压驱动器，而气动驱动器则使用非线性压力-体积关系进行描述。该框架使得可以在不同的驱动模式下一致地评估可控性椭球体和顺应性矩阵。我们使用两种代表性手部结构对分析进行了验证：DexCo（液压驱动）和Edgy-2（气动驱动）。结果强调了不同驱动方式对灵巧性和被动刚度之间的权衡。这些发现为软硬混合机器人手指的结构意识设计和驱动器选择提供了见解。 

---
# Self-Mixing Laser Interferometry: In Search of an Ambient Noise-Resilient Alternative to Acoustic Sensing 

**Title (ZH)**: 自混激光干涉测量：寻找一种抗环境噪声的 acoustic sensing 替代方案 

**Authors**: Remko Proesmans, Thomas Lips, Francis wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2504.13711)  

**Abstract**: Self-mixing interferometry (SMI) has been lauded for its sensitivity in detecting microvibrations, while requiring no physical contact with its target. Microvibrations, i.e., sounds, have recently been used as a salient indicator of extrinsic contact in robotic manipulation. In previous work, we presented a robotic fingertip using SMI for extrinsic contact sensing as an ambient-noise-resilient alternative to acoustic sensing. Here, we extend the validation experiments to the frequency domain. We find that for broadband ambient noise, SMI still outperforms acoustic sensing, but the difference is less pronounced than in time-domain analyses. For targeted noise disturbances, analogous to multiple robots simultaneously collecting data for the same task, SMI is still the clear winner. Lastly, we show how motor noise affects SMI sensing more so than acoustic sensing, and that a higher SMI readout frequency is important for future work. Design and data files are available at this https URL. 

**Abstract (ZH)**: 自混合干涉ometry (SMI) 由于其在检测微振动方面的高灵敏度而备受赞扬，且无需与目标物理接触。微振动，即声音，最近被用作机器人操作中外来接触的一个显著指标。在以往的工作中，我们提出了一种使用 SMI 的机器人指尖，用作对背景噪声具有抗性的替代声学感知方法。在此，我们将验证实验扩展到频域。我们发现，对于宽带背景噪声，SMI 仍然优于声学感知，但差异小于时域分析中的情况。对于针对噪声干扰，类似于多台机器人同时为同一任务收集数据的情况，SMI 仍然是明显的优胜者。最后，我们展示了电机噪声如何比声学噪声更影响 SMI 的感知，并表明未来工作中较高的 SMI 读数频率很重要。设计和数据文件可在以下链接获取。 

---
# Performance Analysis of a Mass-Spring-Damper Deformable Linear Object Model in Robotic Simulation Frameworks 

**Title (ZH)**: 基于机器人模拟框架的质点-弹簧-阻尼可变形线性物体模型性能分析 

**Authors**: Andrea Govoni, Nadia Zubair, Simone Soprani, Gianluca Palli  

**Link**: [PDF](https://arxiv.org/pdf/2504.13659)  

**Abstract**: The modelling of Deformable Linear Objects (DLOs) such as cables, wires, and strings presents significant challenges due to their flexible and deformable nature. In robotics, accurately simulating the dynamic behavior of DLOs is essential for automating tasks like wire handling and assembly. The presented study is a preliminary analysis aimed at force data collection through domain randomization (DR) for training a robot in simulation, using a Mass-Spring-Damper (MSD) system as the reference model. The study aims to assess the impact of model parameter variations on DLO dynamics, using Isaac Sim and Gazebo to validate the applicability of DR technique in these scenarios. 

**Abstract (ZH)**: 柔性线性物体（如电缆、导线和绳索）的建模因其柔性可变形的性质面临重大挑战。在机器人学中，准确模拟柔性线性物体的动力学行为对于自动化线材处理和组装任务至关重要。本研究是一项初步分析，旨在通过领域随机化（DR）收集力数据以在模拟中训练机器人，并使用质量-弹簧-阻尼（MSD）系统作为参考模型。本研究旨在评估模型参数变化对柔性线性物体动力学的影响，并使用Isaac Sim和Gazebo验证DR技术在这些场景中的适用性。 

---
# Robot Navigation in Dynamic Environments using Acceleration Obstacles 

**Title (ZH)**: 动态环境中国基于加速度障碍的机器人导航 

**Authors**: Asher Stern, Zvi Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2504.13637)  

**Abstract**: This paper addresses the issue of motion planning in dynamic environments by extending the concept of Velocity Obstacle and Nonlinear Velocity Obstacle to Acceleration Obstacle AO and Nonlinear Acceleration Obstacle NAO. Similarly to VO and NLVO, the AO and NAO represent the set of colliding constant accelerations of the maneuvering robot with obstacles moving along linear and nonlinear trajectories, respectively. Contrary to prior works, we derive analytically the exact boundaries of AO and NAO. To enhance an intuitive understanding of these representations, we first derive the AO in several steps: first extending the VO to the Basic Acceleration Obstacle BAO that consists of the set of constant accelerations of the robot that would collide with an obstacle moving at constant accelerations, while assuming zero initial velocities of the robot and obstacle. This is then extended to the AO while assuming arbitrary initial velocities of the robot and obstacle. And finally, we derive the NAO that in addition to the prior assumptions, accounts for obstacles moving along arbitrary trajectories. The introduction of NAO allows the generation of safe avoidance maneuvers that directly account for the robot's second-order dynamics, with acceleration as its control input. The AO and NAO are demonstrated in several examples of selecting avoidance maneuvers in challenging road traffic. It is shown that the use of NAO drastically reduces the adjustment rate of the maneuvering robot's acceleration while moving in complex road traffic scenarios. The presented approach enables reactive and efficient navigation for multiple robots, with potential application for autonomous vehicles operating in complex dynamic environments. 

**Abstract (ZH)**: 基于加速度障碍和非线性加速度障碍的动态环境下的运动规划 

---
# Multi-Sensor Fusion-Based Mobile Manipulator Remote Control for Intelligent Smart Home Assistance 

**Title (ZH)**: 基于多传感器融合的移动 manipulator 远程控制技术及其在智能智能家居辅助中的应用 

**Authors**: Xiao Jin, Bo Xiao, Huijiang Wang, Wendong Wang, Zhenhua Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13370)  

**Abstract**: This paper proposes a wearable-controlled mobile manipulator system for intelligent smart home assistance, integrating MEMS capacitive microphones, IMU sensors, vibration motors, and pressure feedback to enhance human-robot interaction. The wearable device captures forearm muscle activity and converts it into real-time control signals for mobile manipulation. The wearable device achieves an offline classification accuracy of 88.33\%\ across six distinct movement-force classes for hand gestures by using a CNN-LSTM model, while real-world experiments involving five participants yield a practical accuracy of 83.33\%\ with an average system response time of 1.2 seconds. In Human-Robot synergy in navigation and grasping tasks, the robot achieved a 98\%\ task success rate with an average trajectory deviation of only 3.6 cm. Finally, the wearable-controlled mobile manipulator system achieved a 93.3\%\ gripping success rate, a transfer success of 95.6\%\, and a full-task success rate of 91.1\%\ during object grasping and transfer tests, in which a total of 9 object-texture combinations were evaluated. These three experiments' results validate the effectiveness of MEMS-based wearable sensing combined with multi-sensor fusion for reliable and intuitive control of assistive robots in smart home scenarios. 

**Abstract (ZH)**: 基于MEMS传感器的可穿戴控制移动 manipulator系统智能智能家居辅助设计与实验研究 

---
# Physical Reservoir Computing in Hook-Shaped Rover Wheel Spokes for Real-Time Terrain Identification 

**Title (ZH)**: 钩形机器人车轮辐条上的物理储槽计算用于实时地形识别 

**Authors**: Xiao Jin, Zihan Wang, Zhenhua Yu, Changrak Choi, Kalind Carpenter, Thrishantha Nanayakkara  

**Link**: [PDF](https://arxiv.org/pdf/2504.13348)  

**Abstract**: Effective terrain detection in unknown environments is crucial for safe and efficient robotic navigation. Traditional methods often rely on computationally intensive data processing, requiring extensive onboard computational capacity and limiting real-time performance for rovers. This study presents a novel approach that combines physical reservoir computing with piezoelectric sensors embedded in rover wheel spokes for real-time terrain identification. By leveraging wheel dynamics, terrain-induced vibrations are transformed into high-dimensional features for machine learning-based classification. Experimental results show that strategically placing three sensors on the wheel spokes achieves 90$\%$ classification accuracy, which demonstrates the accuracy and feasibility of the proposed method. The experiment results also showed that the system can effectively distinguish known terrains and identify unknown terrains by analyzing their similarity to learned categories. This method provides a robust, low-power framework for real-time terrain classification and roughness estimation in unstructured environments, enhancing rover autonomy and adaptability. 

**Abstract (ZH)**: 在未知环境中的有效地形检测对于机器人导航的安全与效率至关重要。传统方法往往依赖于计算密集型的数据处理，需要大量的车载计算能力，从而限制了漫游车的实时性能。本研究提出了一种新颖的方法，结合物理蓄水池计算与嵌入漫游车轮辐中的压电传感器进行实时地形识别。通过利用轮动动力学，地形引起的振动被转换为高维度特征用于机器学习分类。实验结果表明，在轮辐上战略位置放置三个传感器可实现90%的分类准确性，证明了所提方法的准确性和可行性。实验结果还显示，该系统可以通过分析未知地形与已学习类别之间的相似性，有效地区分已知地形和未知地形。该方法为在无结构环境中提供了一种稳健且低功耗的实时地形分类和粗糙度估计框架，增强了漫游车的自主性和适应性。 

---
# Integration of a Graph-Based Path Planner and Mixed-Integer MPC for Robot Navigation in Cluttered Environments 

**Title (ZH)**: 基于图的路径规划器与混合整数MPC在杂乱环境中的机器人导航集成 

**Authors**: Joshua A. Robbins, Stephen J. Harnett, Andrew F. Thompson, Sean Brennan, Herschel C. Pangborn  

**Link**: [PDF](https://arxiv.org/pdf/2504.13372)  

**Abstract**: The ability to update a path plan is a required capability for autonomous mobile robots navigating through uncertain environments. This paper proposes a re-planning strategy using a multilayer planning and control framework for cases where the robot's environment is partially known. A medial axis graph-based planner defines a global path plan based on known obstacles where each edge in the graph corresponds to a unique corridor. A mixed-integer model predictive control (MPC) method detects if a terminal constraint derived from the global plan is infeasible, subject to a non-convex description of the local environment. Infeasibility detection is used to trigger efficient global re-planning via medial axis graph edge deletion. The proposed re-planning strategy is demonstrated experimentally. 

**Abstract (ZH)**: 自主移动机器人在部分已知环境中的路径更新策略研究 

---
# On the Definition of Robustness and Resilience of AI Agents for Real-time Congestion Management 

**Title (ZH)**: 关于AI代理在实时拥堵管理中鲁棒性和韧性定义的研究 

**Authors**: Timothy Tjhay, Ricardo J. Bessa, Jose Paulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.13314)  

**Abstract**: The European Union's Artificial Intelligence (AI) Act defines robustness, resilience, and security requirements for high-risk sectors but lacks detailed methodologies for assessment. This paper introduces a novel framework for quantitatively evaluating the robustness and resilience of reinforcement learning agents in congestion management. Using the AI-friendly digital environment Grid2Op, perturbation agents simulate natural and adversarial disruptions by perturbing the input of AI systems without altering the actual state of the environment, enabling the assessment of AI performance under various scenarios. Robustness is measured through stability and reward impact metrics, while resilience quantifies recovery from performance degradation. The results demonstrate the framework's effectiveness in identifying vulnerabilities and improving AI robustness and resilience for critical applications. 

**Abstract (ZH)**: 欧洲联盟的人工智能（AI）法案为高风险领域定义了稳健性、韧性和安全性要求，但缺乏详细的评估方法。本文提出了一种新型框架，用于定量评估强化学习代理在拥堵管理中的稳健性和韧性。利用AI友好的数字环境Grid2Op，扰动代理通过扰动输入而不改变实际环境状态来模拟自然和敌对的干扰，从而在不同场景下评估AI性能。稳健性通过稳定性和奖励影响指标来衡量，而韧性则量化了性能退化的恢复能力。结果表明，该框架在识别漏洞并提高关键应用中AI的稳健性和韧性方面具有有效性。 

---
