# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task 

**Title (ZH)**: BOSS: 长时间 horizon 任务中观测空间变化的基准 

**Authors**: Yue Yang, Linfeng Zhao, Mingyu Ding, Gedas Bertasius, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2502.15679)  

**Abstract**: Robotics has long sought to develop visual-servoing robots capable of completing previously unseen long-horizon tasks. Hierarchical approaches offer a pathway for achieving this goal by executing skill combinations arranged by a task planner, with each visuomotor skill pre-trained using a specific imitation learning (IL) algorithm. However, even in simple long-horizon tasks like skill chaining, hierarchical approaches often struggle due to a problem we identify as Observation Space Shift (OSS), where the sequential execution of preceding skills causes shifts in the observation space, disrupting the performance of subsequent individually trained skill policies. To validate OSS and evaluate its impact on long-horizon tasks, we introduce BOSS (a Benchmark for Observation Space Shift). BOSS comprises three distinct challenges: "Single Predicate Shift", "Accumulated Predicate Shift", and "Skill Chaining", each designed to assess a different aspect of OSS's negative effect. We evaluated several recent popular IL algorithms on BOSS, including three Behavioral Cloning methods and the Visual Language Action model OpenVLA. Even on the simplest challenge, we observed average performance drops of 67%, 35%, 34%, and 54%, respectively, when comparing skill performance with and without OSS. Additionally, we investigate a potential solution to OSS that scales up the training data for each skill with a larger and more visually diverse set of demonstrations, with our results showing it is not sufficient to resolve OSS. The project page is: this https URL 

**Abstract (ZH)**: 机器人学致力于开发能够完成未见过的长期任务的视觉伺服机器人。分层方法通过由任务规划器安排技能组合来提供实现这一目标的途径，每种视觉运动技能都使用特定的imitation learning (IL) 算法进行预先训练。然而，即便是在如技能链接这样简单的长期任务中，分层方法也常常因我们识别出的一种问题——观察空间偏移（OSS）——而挣扎，这种问题会导致后续技能执行时观察空间出现偏移，破坏每个单独训练的技能策略的表现。为了验证OSS并评估其对长期任务的影响，我们引入了BOSS（观察空间偏移基准）。BOSS包括三个不同的挑战：“单一谓词偏移”、“累积谓词偏移”和“技能链接”，旨在评估OSS负面影响的不同方面。我们对BOSS评估了几个最近流行的IL算法，包括三种行为克隆方法和视觉语言动作模型OpenVLA。即使在最简单的挑战中，我们观察到，在有和没有OSS的情况下，技能表现的平均性能分别下降了67%、35%、34%和54%。此外，我们研究了OSS的一种可能解决方法，即通过使用更大且视觉多样更多的示范数据来放大每种技能的训练数据，结果表明这不足以解决OSS问题。项目页面：https://github.com/alibabaqwen/BOSS。 

---
# A Simulation Pipeline to Facilitate Real-World Robotic Reinforcement Learning Applications 

**Title (ZH)**: 一种促进现实机器人强化学习应用的仿真管道 

**Authors**: Jefferson Silveira, Joshua A. Marshall, Sidney N. Givigi Jr  

**Link**: [PDF](https://arxiv.org/pdf/2502.15649)  

**Abstract**: Reinforcement learning (RL) has gained traction for its success in solving complex tasks for robotic applications. However, its deployment on physical robots remains challenging due to safety risks and the comparatively high costs of training. To avoid these problems, RL agents are often trained on simulators, which introduces a new problem related to the gap between simulation and reality. This paper presents an RL pipeline designed to help reduce the reality gap and facilitate developing and deploying RL policies for real-world robotic systems. The pipeline organizes the RL training process into an initial step for system identification and three training stages: core simulation training, high-fidelity simulation, and real-world deployment, each adding levels of realism to reduce the sim-to-real gap. Each training stage takes an input policy, improves it, and either passes the improved policy to the next stage or loops it back for further improvement. This iterative process continues until the policy achieves the desired performance. The pipeline's effectiveness is shown through a case study with the Boston Dynamics Spot mobile robot used in a surveillance application. The case study presents the steps taken at each pipeline stage to obtain an RL agent to control the robot's position and orientation. 

**Abstract (ZH)**: 强化学习（RL）因其在解决机器人应用中的复杂任务方面的成功而受到关注。然而，其在物理机器人上的部署仍然面临着安全风险和相对较高的训练成本的挑战。为避免这些问题，RL代理通常在模拟器中训练，这引入了模拟与现实之间差距的新问题。本文介绍了一种RL管道，旨在减少这种现实差距，并促进在实地机器人系统中开发和部署RL策略。该管道将RL训练过程组织为初始步骤进行系统识别和三个训练阶段：核心仿真训练、高保真仿真和实地部署，每个阶段都增加了现实感以减少仿真到现实的差距。每个训练阶段都以输入策略为输入，对其进行改进，并将改进后的策略传递到下一个阶段或反馈回进行进一步改进。这一迭代过程将持续进行，直到策略达到所需的性能。通过使用波士顿动力公司Spot移动机器人在监视应用中的案例研究，展示了该管道的有效性。该案例研究介绍了每个管道阶段所采取的步骤，以获得控制机器人位置和方向的RL代理。 

---
# Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion 

**Title (ZH)**: 基于降阶模型引导的接触显式模型预测控制的人形步行控制 

**Authors**: Sergio A. Esteban, Vince Kurtz, Adrian B. Ghansah, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2502.15630)  

**Abstract**: Humanoid robots have great potential for real-world applications due to their ability to operate in environments built for humans, but their deployment is hindered by the challenge of controlling their underlying high-dimensional nonlinear hybrid dynamics. While reduced-order models like the Hybrid Linear Inverted Pendulum (HLIP) are simple and computationally efficient, they lose whole-body expressiveness. Meanwhile, recent advances in Contact-Implicit Model Predictive Control (CI-MPC) enable robots to plan through multiple hybrid contact modes, but remain vulnerable to local minima and require significant tuning. We propose a control framework that combines the strengths of HLIP and CI-MPC. The reduced-order model generates a nominal gait, while CI-MPC manages the whole-body dynamics and modifies the contact schedule as needed. We demonstrate the effectiveness of this approach in simulation with a novel 24 degree-of-freedom humanoid robot: Achilles. Our proposed framework achieves rough terrain walking, disturbance recovery, robustness under model and state uncertainty, and allows the robot to interact with obstacles in the environment, all while running online in real-time at 50 Hz. 

**Abstract (ZH)**: 人形机器人由于能够在为人类构建的环境中操作而具有广泛的应用潜力，但其部署受到控制其潜在高维非线性混合动力学挑战的阻碍。虽然简化模型如混合线性倒摆（HLIP）简单且计算效率高，但会失去全身表达性。同时，最近在接触隐式模型预测控制（CI-MPC）方面的进展使得机器人能够计划通过多种混合接触模式，但仍容易陷入局部极值，并需要大量调整。我们提出了一种结合HLIP和CI-MPC优点的控制框架。简化模型生成名义步态，而CI-MPC管理全身动力学并在必要时修改接触表调度。我们通过一个新的24自由度人形机器人Achilles在仿真中展示了该方法的有效性，实现崎岖地形行走、干扰恢复、在模型和状态不确定性下的鲁棒性，并允许机器人与环境中的障碍物互动，同时以50 Hz的在线实时速度运行。 

---
# Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach 

**Title (ZH)**: 无需重新训练的手爪之间拾取放置操作：一种学习-优化扩散策略方法 

**Authors**: Xiangtong Yao, Yirui Zhou, Yuan Meng, Liangyu Dong, Lin Hong, Zitao Zhang, Zhenshan Bing, Kai Huang, Fuchun Sun, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2502.15613)  

**Abstract**: Current robotic pick-and-place policies typically require consistent gripper configurations across training and inference. This constraint imposes high retraining or fine-tuning costs, especially for imitation learning-based approaches, when adapting to new end-effectors. To mitigate this issue, we present a diffusion-based policy with a hybrid learning-optimization framework, enabling zero-shot adaptation to novel grippers without additional data collection for retraining policy. During training, the policy learns manipulation primitives from demonstrations collected using a base gripper. At inference, a diffusion-based optimization strategy dynamically enforces kinematic and safety constraints, ensuring that generated trajectories align with the physical properties of unseen grippers. This is achieved through a constrained denoising procedure that adapts trajectories to gripper-specific parameters (e.g., tool-center-point offsets, jaw widths) while preserving collision avoidance and task feasibility. We validate our method on a Franka Panda robot across six gripper configurations, including 3D-printed fingertips, flexible silicone gripper, and Robotiq 2F-85 gripper. Our approach achieves a 93.3% average task success rate across grippers (vs. 23.3-26.7% for diffusion policy baselines), supporting tool-center-point variations of 16-23.5 cm and jaw widths of 7.5-11.5 cm. The results demonstrate that constrained diffusion enables robust cross-gripper manipulation while maintaining the sample efficiency of imitation learning, eliminating the need for gripper-specific retraining. Video and code are available at this https URL. 

**Abstract (ZH)**: 基于扩散的零 shot 夹持器适应策略：混合学习-优化框架 

---
# Autonomous helicopter aerial refueling: controller design and performance guarantees 

**Title (ZH)**: 自主旋翼机空中加油：控制设计与性能保证 

**Authors**: Damsara Jayarathne, Santiago Paternain, Sandipan Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2502.15562)  

**Abstract**: In this paper, we present a control design methodology, stability criteria, and performance bounds for autonomous helicopter aerial refueling. Autonomous aerial refueling is particularly difficult due to the aerodynamic interaction between the wake of the tanker, the contact-sensitive nature of the maneuver, and the uncertainty in drogue motion. Since the probe tip is located significantly away from the helicopter's center-of-gravity, its position (and velocity) is strongly sensitive to the helicopter's attitude (and angular rates). In addition, the fact that the helicopter is operating at high speeds to match the velocity of the tanker forces it to maintain a particular orientation, making the docking maneuver especially challenging. In this paper, we propose a novel outer-loop position controller that incorporates the probe position and velocity into the feedback loop. The position and velocity of the probe tip depend both on the position (velocity) and on the attitude (angular rates) of the aircraft. We derive analytical guarantees for docking performance in terms of the uncertainty of the drogue motion and the angular acceleration of the helicopter, using the ultimate boundedness property of the closed-loop error dynamics. Simulations are performed on a high-fidelity UH60 helicopter model with a high-fidelity drogue motion under wind effects to validate the proposed approach for realistic refueling scenarios. These high-fidelity simulations reveal that the proposed control methodology yields an improvement of 36% in the 2-norm docking error compared to the existing standard controller. 

**Abstract (ZH)**: 基于探管位置和速度的自主直升机空中加油控制设计方法及性能分析 

---
# Enhanced Probabilistic Collision Detection for Motion Planning Under Sensing Uncertainty 

**Title (ZH)**: 增强的基于概率的碰撞检测方法以应对感知不确定性下的运动规划 

**Authors**: Xiaoli Wang, Sipu Ruan, Xin Meng, Gregory Chirikjian  

**Link**: [PDF](https://arxiv.org/pdf/2502.15525)  

**Abstract**: Probabilistic collision detection (PCD) is essential in motion planning for robots operating in unstructured environments, where considering sensing uncertainty helps prevent damage. Existing PCD methods mainly used simplified geometric models and addressed only position estimation errors. This paper presents an enhanced PCD method with two key advancements: (a) using superquadrics for more accurate shape approximation and (b) accounting for both position and orientation estimation errors to improve robustness under sensing uncertainty. Our method first computes an enlarged surface for each object that encapsulates its observed rotated copies, thereby addressing the orientation estimation errors. Then, the collision probability under the position estimation errors is formulated as a chance-constraint problem that is solved with a tight upper bound. Both the two steps leverage the recently developed normal parameterization of superquadric surfaces. Results show that our PCD method is twice as close to the Monte-Carlo sampled baseline as the best existing PCD method and reduces path length by 30% and planning time by 37%, respectively. A Real2Sim pipeline further validates the importance of considering orientation estimation errors, showing that the collision probability of executing the planned path in simulation is only 2%, compared to 9% and 29% when considering only position estimation errors or none at all. 

**Abstract (ZH)**: 概率碰撞检测（PCD）在机器人在未结构化环境中运动规划中的应用：考虑感知不确定性进行更准确形状 approx 的增强 PCD 方法 

---
# Robust 4D Radar-aided Inertial Navigation for Aerial Vehicles 

**Title (ZH)**: 稳健的雷达辅助四维惯性导航系统在空中车辆中的应用 

**Authors**: Jinwen Zhu, Jun Hu, Xudong Zhao, Xiaoming Lang, Yinian Mao, Guoquan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15452)  

**Abstract**: While LiDAR and cameras are becoming ubiquitous for unmanned aerial vehicles (UAVs) but can be ineffective in challenging environments, 4D millimeter-wave (MMW) radars that can provide robust 3D ranging and Doppler velocity measurements are less exploited for aerial navigation. In this paper, we develop an efficient and robust error-state Kalman filter (ESKF)-based radar-inertial navigation for UAVs. The key idea of the proposed approach is the point-to-distribution radar scan matching to provide motion constraints with proper uncertainty qualification, which are used to update the navigation states in a tightly coupled manner, along with the Doppler velocity measurements. Moreover, we propose a robust keyframe-based matching scheme against the prior map (if available) to bound the accumulated navigation errors and thus provide a radar-based global localization solution with high accuracy. Extensive real-world experimental validations have demonstrated that the proposed radar-aided inertial navigation outperforms state-of-the-art methods in both accuracy and robustness. 

**Abstract (ZH)**: 4D毫米波雷达辅助的高效鲁棒惯性导航ethod学研究 

---
# Learning Long-Horizon Robot Manipulation Skills via Privileged Action 

**Title (ZH)**: 通过优先级动作学习长时_horizon机器人操作技能 

**Authors**: Xiaofeng Mao, Yucheng Xu, Zhaole Sun, Elle Miller, Daniel Layeghi, Michael Mistry  

**Link**: [PDF](https://arxiv.org/pdf/2502.15442)  

**Abstract**: Long-horizon contact-rich tasks are challenging to learn with reinforcement learning, due to ineffective exploration of high-dimensional state spaces with sparse rewards. The learning process often gets stuck in local optimum and demands task-specific reward fine-tuning for complex scenarios. In this work, we propose a structured framework that leverages privileged actions with curriculum learning, enabling the policy to efficiently acquire long-horizon skills without relying on extensive reward engineering or reference trajectories. Specifically, we use privileged actions in simulation with a general training procedure that would be infeasible to implement in real-world scenarios. These privileges include relaxed constraints and virtual forces that enhance interaction and exploration with objects. Our results successfully achieve complex multi-stage long-horizon tasks that naturally combine non-prehensile manipulation with grasping to lift objects from non-graspable poses. We demonstrate generality by maintaining a parsimonious reward structure and showing convergence to diverse and robust behaviors across various environments. Additionally, real-world experiments further confirm that the skills acquired using our approach are transferable to real-world environments, exhibiting robust and intricate performance. Our approach outperforms state-of-the-art methods in these tasks, converging to solutions where others fail. 

**Abstract (ZH)**: 长时域高接触任务使用强化学习学习具有稀疏奖励的高维状态空间探索不足，往往容易陷入局部最优，并需要针对复杂场景进行特定的任务奖励微调。本文提出了一种结构化框架，结合先验动作与 curriculum learning，使策略能够高效地获得长时域技能，无需依赖广泛的奖励工程或参考轨迹。具体而言，我们利用模拟中的先验动作和通用训练程序，这些程序在实际场景中难以实施。这些先验特权包括宽松的约束和虚拟力，以增强与物体的交互和探索。我们的实验结果成功实现了结合非抓取操作与抓取的复杂多阶段长时域任务，将物体从不可抓取的姿态提起。通过保持简洁的奖励结构并展示多种环境下的收敛和稳健行为，我们展示了通用性。此外，现实世界实验进一步证实，使用我们方法获得的技能在实际环境中的可迁移性，表现出稳健和复杂的性能。我们的方法在这些任务中优于现有最佳方法，能够收敛到其他方法失败的解决方案。 

---
# Self-Mixing Laser Interferometry for Robotic Tactile Sensing 

**Title (ZH)**: 自混合激光干涉ometry在机器人触觉传感中的应用 

**Authors**: Remko Proesmans, Ward Goossens, Lowiek Van den Stockt, Lowie Christiaen, Francis wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2502.15390)  

**Abstract**: Self-mixing interferometry (SMI) has been lauded for its sensitivity in detecting microvibrations, while requiring no physical contact with its target. In robotics, microvibrations have traditionally been interpreted as a marker for object slip, and recently as a salient indicator of extrinsic contact. We present the first-ever robotic fingertip making use of SMI for slip and extrinsic contact sensing. The design is validated through measurement of controlled vibration sources, both before and after encasing the readout circuit in its fingertip package. Then, the SMI fingertip is compared to acoustic sensing through three experiments. The results are distilled into a technology decision map. SMI was found to be more sensitive to subtle slip events and significantly more robust against ambient noise. We conclude that the integration of SMI in robotic fingertips offers a new, promising branch of tactile sensing in robotics. 

**Abstract (ZH)**: 自混合干涉ometry (SMI) 由于其在检测微振动方面的高灵敏度而备受推崇，且无需与目标物理接触。在机器人学中，微振动 traditionally 被解读为物体滑动的标志，最近则被视为外在接触的显著指标。我们首次展示了使用 SMI 进行滑动和外在接触感知的机器人指尖设计。该设计通过在安装读出电路的指尖壳体内和外测量控制振动源来进行验证。然后，将 SMI 指尖与声学感知进行三次实验比较。结果被提炼成一项技术决策图。研究发现，SMI 对微妙的滑动事件更为敏感，并且在抵制环境噪声方面表现出显著的 robustness。我们得出结论，将 SMI 集成到机器人指尖中为机器人领域提供了新的、有前景的触觉感知分支。 

---
# Rapid Online Learning of Hip Exoskeleton Assistance Preferences 

**Title (ZH)**: 快速在线学习髋部外骨骼辅助偏好 

**Authors**: Giulia Ramella, Auke Ijspeert, Mohamed Bouri  

**Link**: [PDF](https://arxiv.org/pdf/2502.15366)  

**Abstract**: Hip exoskeletons are increasing in popularity due to their effectiveness across various scenarios and their ability to adapt to different users. However, personalizing the assistance often requires lengthy tuning procedures and computationally intensive algorithms, and most existing methods do not incorporate user feedback. In this work, we propose a novel approach for rapidly learning users' preferences for hip exoskeleton assistance. We perform pairwise comparisons of distinct randomly generated assistive profiles, and collect participants preferences through active querying. Users' feedback is integrated into a preference-learning algorithm that updates its belief, learns a user-dependent reward function, and changes the assistive torque profiles accordingly. Results from eight healthy subjects display distinct preferred torque profiles, and users' choices remain consistent when compared to a perturbed profile. A comprehensive evaluation of users' preferences reveals a close relationship with individual walking strategies. The tested torque profiles do not disrupt kinematic joint synergies, and participants favor assistive torques that are synchronized with their movements, resulting in lower negative power from the device. This straightforward approach enables the rapid learning of users preferences and rewards, grounding future studies on reward-based human-exoskeleton interaction. 

**Abstract (ZH)**: 基于髋部外骨骼的用户偏好快速学习方法：一种增强计算适应性的新途径 

---
# Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions 

**Title (ZH)**: 探索具身多模态大模型：发展、数据集及未来方向 

**Authors**: Shoubin Chen, Zehao Wu, Kai Zhang, Chunyu Li, Baiyang Zhang, Fei Ma, Fei Richard Yu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15336)  

**Abstract**: Embodied multimodal large models (EMLMs) have gained significant attention in recent years due to their potential to bridge the gap between perception, cognition, and action in complex, real-world environments. This comprehensive review explores the development of such models, including Large Language Models (LLMs), Large Vision Models (LVMs), and other models, while also examining other emerging architectures. We discuss the evolution of EMLMs, with a focus on embodied perception, navigation, interaction, and simulation. Furthermore, the review provides a detailed analysis of the datasets used for training and evaluating these models, highlighting the importance of diverse, high-quality data for effective learning. The paper also identifies key challenges faced by EMLMs, including issues of scalability, generalization, and real-time decision-making. Finally, we outline future directions, emphasizing the integration of multimodal sensing, reasoning, and action to advance the development of increasingly autonomous systems. By providing an in-depth analysis of state-of-the-art methods and identifying critical gaps, this paper aims to inspire future advancements in EMLMs and their applications across diverse domains. 

**Abstract (ZH)**: 具身多模态大型模型（EMLMs）由于其在复杂真实环境中的感知、认知和行动之间桥接的潜力，近年来引起了广泛关注。本文全面回顾了此类模型的发展，包括大型语言模型（LLMs）、大型视觉模型（LVMs）以及其他模型，同时探讨了其他新兴架构。我们讨论了EMLMs的发展演变，重点在于具身感知、导航、交互和模拟。此外，本文详细分析了用于训练和评估这些模型的_datasets_，突出了多样性和高质量数据对于有效学习的重要性。文章还指出了EMLMs面临的几个关键挑战，包括可扩展性、泛化能力和实时决策问题。最后，我们展望了未来的发展方向，强调了多模态感知、推理和行动的集成，以推进自主系统的发展。通过深入分析现有先进技术并指出关键空白，本文旨在激发EMLMs及其在各个领域的应用的未来进步。 

---
# DynamicGSG: Dynamic 3D Gaussian Scene Graphs for Environment Adaptation 

**Title (ZH)**: DynamicGSG: 动态3D高斯场景图及其在环境适应中的应用 

**Authors**: Luzhou Ge, Xiangyu Zhu, Zhuo Yang, Xuesong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15309)  

**Abstract**: In real-world scenarios, the environment changes caused by agents or human activities make it extremely challenging for robots to perform various long-term tasks. To effectively understand and adapt to dynamic environments, the perception system of a robot needs to extract instance-level semantic information, reconstruct the environment in a fine-grained manner, and update its environment representation in memory according to environment changes. To address these challenges, We propose \textbf{DynamicGSG}, a dynamic, high-fidelity, open-vocabulary scene graph generation system leveraging Gaussian splatting. Our system comprises three key components: (1) constructing hierarchical scene graphs using advanced vision foundation models to represent the spatial and semantic relationships of objects in the environment, (2) designing a joint feature loss to optimize the Gaussian map for incremental high-fidelity reconstruction, and (3) updating the Gaussian map and scene graph according to real environment changes for long-term environment adaptation. Experiments and ablation studies demonstrate the performance and efficacy of the proposed method in terms of semantic segmentation, language-guided object retrieval, and reconstruction quality. Furthermore, we have validated the dynamic updating capabilities of our system in real laboratory environments. The source code will be released at:~\href{this https URL}{this https URL}. 

**Abstract (ZH)**: 在现实场景中，由于代理或人类活动导致的环境变化使机器人完成各种长期任务极具挑战性。为了有效理解和适应动态环境，机器人的感知系统需要提取实例级语义信息，以精细的方式重建环境，并根据环境变化在记忆中更新环境表示。为了解决这些挑战，我们提出了一种基于高斯点扩散的动态、高保真、开放词汇场景图生成系统——DynamicGSG。该系统包含三个关键组件：（1）使用先进的视觉基础模型构建分层场景图以表示环境中的对象的空间和语义关系；（2）设计联合特征损失以优化高斯图，实现增量高保真重建；（3）根据实际环境变化更新高斯图和场景图，实现长期环境适应。实验和消融研究证明了该方法在语义分割、语言引导的物体检索和重建质量方面的性能和有效性。此外，我们还在真实的实验室环境中验证了系统动态更新的能力。源代码将在以下链接发布：this https URL。 

---
# Realm: Real-Time Line-of-Sight Maintenance in Multi-Robot Navigation with Unknown Obstacles 

**Title (ZH)**: Realm: 多机器人导航中未知障碍物下的实时视线保持 

**Authors**: Ruofei Bai, Shenghai Yuan, Kun Li, Hongliang Guo, Wei-Yun Yau, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.15162)  

**Abstract**: Multi-robot navigation in complex environments relies on inter-robot communication and mutual observations for coordination and situational awareness. This paper studies the multi-robot navigation problem in unknown environments with line-of-sight (LoS) connectivity constraints. While previous works are limited to known environment models to derive the LoS constraints, this paper eliminates such requirements by directly formulating the LoS constraints between robots from their real-time point cloud measurements, leveraging point cloud visibility analysis techniques. We propose a novel LoS-distance metric to quantify both the urgency and sensitivity of losing LoS between robots considering potential robot movements. Moreover, to address the imbalanced urgency of losing LoS between two robots, we design a fusion function to capture the overall urgency while generating gradients that facilitate robots' collaborative movement to maintain LoS. The LoS constraints are encoded into a potential function that preserves the positivity of the Fiedler eigenvalue of the robots' network graph to ensure connectivity. Finally, we establish a LoS-constrained exploration framework that integrates the proposed connectivity controller. We showcase its applications in multi-robot exploration in complex unknown environments, where robots can always maintain the LoS connectivity through distributed sensing and communication, while collaboratively mapping the unknown environment. The implementations are open-sourced at this https URL. 

**Abstract (ZH)**: 多机器人在复杂未知环境中的导航依赖于机器人间的通信和相互观测以实现协调和情境感知。本文研究了具有视线（LoS）连接约束的未知环境下的多机器人导航问题。以往工作受限于已知环境模型来推导LoS约束，而本文通过直接从机器人实时点云测量中制定LoS约束，利用点云可见性分析技术消除了此类要求。我们提出了一种新颖的LoS距离度量来量化机器人之间失去视线的紧迫性和敏感性，同时考虑潜在的机器人移动。此外，为了解决机器人之间失去视线紧迫性的不平衡性，我们设计了一种融合函数以捕捉总体紧迫性并生成促进机器人协同运动、保持LoS的梯度。将LoS约束编码成一个势能函数，以保持机器人网络图杨氏特征值的正值，确保连接性。最后，我们建立了一种包含所提连接控制器的LoS约束探索框架。在复杂未知环境下的多机器人探索中展示了该框架的应用，其中机器人可以通过分布式感知和通信始终保持LoS连接，协作绘制未知环境。实现已在以下链接开源：this https URL。 

---
# CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models 

**Title (ZH)**: CurricuVLM：通过个性化安全关键课程学习实现安全自动驾驶 

**Authors**: Zihao Sheng, Zilin Huang, Yansong Qu, Yue Leng, Sruthi Bhavanam, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15119)  

**Abstract**: Ensuring safety in autonomous driving systems remains a critical challenge, particularly in handling rare but potentially catastrophic safety-critical scenarios. While existing research has explored generating safety-critical scenarios for autonomous vehicle (AV) testing, there is limited work on effectively incorporating these scenarios into policy learning to enhance safety. Furthermore, developing training curricula that adapt to an AV's evolving behavioral patterns and performance bottlenecks remains largely unexplored. To address these challenges, we propose CurricuVLM, a novel framework that leverages Vision-Language Models (VLMs) to enable personalized curriculum learning for autonomous driving agents. Our approach uniquely exploits VLMs' multimodal understanding capabilities to analyze agent behavior, identify performance weaknesses, and dynamically generate tailored training scenarios for curriculum adaptation. Through comprehensive analysis of unsafe driving situations with narrative descriptions, CurricuVLM performs in-depth reasoning to evaluate the AV's capabilities and identify critical behavioral patterns. The framework then synthesizes customized training scenarios targeting these identified limitations, enabling effective and personalized curriculum learning. Extensive experiments on the Waymo Open Motion Dataset show that CurricuVLM outperforms state-of-the-art baselines across both regular and safety-critical scenarios, achieving superior performance in terms of navigation success, driving efficiency, and safety metrics. Further analysis reveals that CurricuVLM serves as a general approach that can be integrated with various RL algorithms to enhance autonomous driving systems. The code and demo video are available at: this https URL. 

**Abstract (ZH)**: 确保自动驾驶系统安全仍然是一个关键挑战，特别是在处理罕见但可能造成灾难性后果的安全关键场景方面。尽管现有研究探讨了为自动驾驶车辆（AV）测试生成安全关键场景的方法，但在有效将这些场景融入政策学习以增强安全性方面的工作仍然有限。此外，开发能够适应自动驾驶车辆不断演变的行为模式和性能瓶颈的培训课程框架也鲜有研究。为应对这些挑战，我们提出了CurricuVLM，一种新颖的框架，利用视觉-语言模型（VLMs）实现自动驾驶代理的个性化课程学习。我们的方法独特地利用了VLMs的多模态理解能力来分析代理行为、识别性能缺陷，并动态生成针对课程适应的定制化训练场景。通过对包含详述的不安全驾驶情况进行全面分析，CurricuVLM进行深入推理，评估AV的能力并识别关键行为模式。该框架随后合成针对这些识别出的限制条件的定制化训练场景，从而实现有效的个性化课程学习。在Waymo Open Motion数据集上的广泛实验结果显示，CurricuVLM在常规和安全关键场景中均优于最先进的基线方法，在导航成功率、驾驶效率和安全指标方面表现出更优异的性能。进一步分析表明，CurricuVLM可以作为一种通用方法，与各种强化学习算法集成，以增强自动驾驶系统。相关代码和演示视频可在以下链接获取：this https URL。 

---
# DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories 

**Title (ZH)**: DDAT: 扩散策略约束动态可接受的机器人轨迹 

**Authors**: Jean-Baptiste Bouvier, Kanghyun Ryu, Kartik Nagpal, Qiayuan Liao, Koushil Sreenath, Negar Mehr  

**Link**: [PDF](https://arxiv.org/pdf/2502.15043)  

**Abstract**: Diffusion models excel at creating images and videos thanks to their multimodal generative capabilities. These same capabilities have made diffusion models increasingly popular in robotics research, where they are used for generating robot motion. However, the stochastic nature of diffusion models is fundamentally at odds with the precise dynamical equations describing the feasible motion of robots. Hence, generating dynamically admissible robot trajectories is a challenge for diffusion models. To alleviate this issue, we introduce DDAT: Diffusion policies for Dynamically Admissible Trajectories to generate provably admissible trajectories of black-box robotic systems using diffusion models. A sequence of states is a dynamically admissible trajectory if each state of the sequence belongs to the reachable set of its predecessor by the robot's equations of motion. To generate such trajectories, our diffusion policies project their predictions onto a dynamically admissible manifold during both training and inference to align the objective of the denoiser neural network with the dynamical admissibility constraint. The auto-regressive nature of these projections along with the black-box nature of robot dynamics render these projections immensely challenging. We thus enforce admissibility by iteratively sampling a polytopic under-approximation of the reachable set of a state onto which we project its predicted successor, before iterating this process with the projected successor. By producing accurate trajectories, this projection eliminates the need for diffusion models to continually replan, enabling one-shot long-horizon trajectory planning. We demonstrate that our framework generates higher quality dynamically admissible robot trajectories through extensive simulations on a quadcopter and various MuJoCo environments, along with real-world experiments on a Unitree GO1 and GO2. 

**Abstract (ZH)**: Diffusion 模型通过其多模态生成能力在创建图像和视频方面表现出色。这些相同的生成能力使其在机器人研究中越来越受欢迎，其中扩散模型被用于生成机器人运动。然而，扩散模型的随机性质与描述机器人可行运动的精确动力学方程本质上存在冲突。因此，生成动态可接受的机器人轨迹是扩散模型的一个挑战。为了缓解这一问题，我们引入了 DDAT：扩散策略以生成黑盒机器人系统的可证明动态可接受轨迹。如果序列中的每个状态都由机器人动力学方程的可达集决定，则该状态序列是一个动态可接受轨迹。为了生成这样的轨迹，我们的扩散策略在训练和推断过程中将预测投影到动态可接受流形上，以使去噪神经网络的目标与动态可接受性约束一致。这些预测的自回归性质以及机器人动力学的黑盒性质使得这些投影极其具有挑战性。因此，我们通过迭代地将多面体下近似投影到一个状态的可达集上，并在其上面预测该状态的后继状态，然后迭代这个过程，以实现此过程中的投影后继状态，从而确保可接受性。通过生成准确的轨迹，这种投影消除了扩散模型不断重新规划的需要，从而实现一次性长时间轨迹规划。我们通过在四旋翼无人机和各种 MuJoCo 环境上的广泛仿真以及在 Unitree GO1 和 GO2 上的实地实验，证明了我们的框架能够生成更高质量的动态可接受机器人轨迹。 

---
# DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time 

**Title (ZH)**: DEFT: 可微分支离散弹性杆件方法用于实时建模分叉的DLO 

**Authors**: Yizhou Chen, Xiaoyue Wu, Yeheng Zong, Anran Li, Yuzhen Chen, Julie Wu, Bohao Zhang, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15037)  

**Abstract**: Autonomous wire harness assembly requires robots to manipulate complex branched cables with high precision and reliability. A key challenge in automating this process is predicting how these flexible and branched structures behave under manipulation. Without accurate predictions, it is difficult for robots to reliably plan or execute assembly operations. While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models. To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation. A comprehensive series of real-world experiments demonstrates DEFT's efficacy in terms of accuracy, computational speed, and generalizability compared to state-of-the-art alternatives. Project page:this https URL. 

**Abstract (ZH)**: 自主线束组装要求机器人以高精度和可靠性 manipulate 复杂分支电缆。自动化这一过程的关键挑战在于预测这些柔性分支结构在操纵过程中的行为。在没有准确预测的情况下，机器人难以可靠地规划或执行组装操作。尽管现有的研究在建模单线 Deformable Linear Objects (DLOs) 方面取得了进展，但将这些方法扩展到 Branched Deformable Linear Objects (BDLOs) 呈现了根本性的挑战。BDLOs 的连接点产生了复杂的力交互和应变传播模式，仅通过简单地连接多个单 DLO 模型是无法充分捕捉的。为了解决这些挑战，本文提出了一种名为 Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT) 的新型框架，该框架结合了可微分物理模型与学习框架，用于：1) 准确建模 BDLO 动力学，包括连接点的动态传播和 BDLO 中部的抓取，2) 实现实时推理的高效计算，3) 使规划能够演示灵巧的 BDLO 操纵。全面的实验证明了 DEFT 在准确度、计算速度和泛化能力方面优于当前最先进的替代方案。项目页面：this https URL。 

---
# Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions 

**Title (ZH)**: 超越地平线的安全性：基于 Neural 控制 barrier 函数的高效采样 MPC 

**Authors**: Ji Yin, Oswin So, Eric Yang Yu, Chuchu Fan, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2502.15006)  

**Abstract**: A common problem when using model predictive control (MPC) in practice is the satisfaction of safety specifications beyond the prediction horizon. While theoretical works have shown that safety can be guaranteed by enforcing a suitable terminal set constraint or a sufficiently long prediction horizon, these techniques are difficult to apply and thus are rarely used by practitioners, especially in the case of general nonlinear dynamics. To solve this problem, we impose a tradeoff between exact recursive feasibility, computational tractability, and applicability to ''black-box'' dynamics by learning an approximate discrete-time control barrier function and incorporating it into a variational inference MPC (VIMPC), a sampling-based MPC paradigm. To handle the resulting state constraints, we further propose a new sampling strategy that greatly reduces the variance of the estimated optimal control, improving the sample efficiency, and enabling real-time planning on a CPU. The resulting Neural Shield-VIMPC (NS-VIMPC) controller yields substantial safety improvements compared to existing sampling-based MPC controllers, even under badly designed cost functions. We validate our approach in both simulation and real-world hardware experiments. 

**Abstract (ZH)**: 使用模型预测控制（MPC）时一个常见问题是超出预测_horizon_的安全规范满足问题。虽然理论工作已经证明可以通过施加适当的终端集约束或延长足够的预测_horizon_来保证安全，但这些技术在实际应用中难以实施，因此很少被实践者使用，尤其是在一般非线性动力学情况下的应用。为了解决这一问题，我们通过学习近似离散时间控制障碍函数，并将其纳入一种基于采样的MPC（VIMPC）框架中，来在精确递归可行性、计算效率和对“黑盒”动力学的适用性之间寻求权衡。为了处理由此产生的状态约束，我们进一步提出了一种新的采样策略，该策略大大减少了估计最优控制的方差，提高了样本效率，并使基于CPU的实时规划成为可能。所得到的Neural Shield-VIMPC控制器在现有基于采样的MPC控制器中提供了显著的安全改进，即使在成本函数设计不佳的情况下也是如此。我们在仿真和实际硬件实验中验证了我们的方法。 

---
# Ultra-High-Frequency Harmony: mmWave Radar and Event Camera Orchestrate Accurate Drone Landing 

**Title (ZH)**: 超高频和谐：毫米波雷达与事件相机协同实现精准无人机着陆 

**Authors**: Haoyang Wang, Jingao Xu, Xinyu Luo, Xuecheng Chen, Ting Zhang, Ruiyang Duan, Yunhao Liu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14992)  

**Abstract**: For precise, efficient, and safe drone landings, ground platforms should real-time, accurately locate descending drones and guide them to designated spots. While mmWave sensing combined with cameras improves localization accuracy, the lower sampling frequency of traditional frame cameras compared to mmWave radar creates bottlenecks in system throughput. In this work, we replace the traditional frame camera with event camera, a novel sensor that harmonizes in sampling frequency with mmWave radar within the ground platform setup, and introduce mmE-Loc, a high-precision, low-latency ground localization system designed for drone landings. To fully leverage the \textit{temporal consistency} and \textit{spatial complementarity} between these modalities, we propose two innovative modules, \textit{consistency-instructed collaborative tracking} and \textit{graph-informed adaptive joint optimization}, for accurate drone measurement extraction and efficient sensor fusion. Extensive real-world experiments in landing scenarios from a leading drone delivery company demonstrate that mmE-Loc outperforms state-of-the-art methods in both localization accuracy and latency. 

**Abstract (ZH)**: 基于事件相机的毫米波雷达融合地面定位系统mmoE-Locdee for精准、高效、安全的无人机降落 

---
# A novel step-by-step procedure for the kinematic calibration of robots using a single draw-wire encoder 

**Title (ZH)**: 一种使用单根抽丝编码器进行机器人运动标定的新型逐点方法 

**Authors**: Giovanni Boschetti, Teresa Sinico  

**Link**: [PDF](https://arxiv.org/pdf/2502.14983)  

**Abstract**: Robot positioning accuracy is a key factory when performing high-precision manufacturing tasks. To effectively improve the accuracy of a manipulator, often up to a value close to its repeatability, calibration plays a crucial role. In the literature, various approaches to robot calibration have been proposed, and they range considerably in the type of measurement system and identification algorithm used. Our aim was to develop a novel step-by-step kinematic calibration procedure - where the parameters are subsequently estimated one at a time - that only uses 1D distance measurement data obtained through a draw-wire encoder. To pursue this objective, we derived an analytical approach to find, for each unknown parameter, a set of calibration points where the discrepancy between the measured and predicted distances only depends on that unknown parameter. This reduces the computational burden of the identification process while potentially improving its accuracy. Simulations and experimental tests were carried out on a 6 degrees-of-freedom robot arm: the results confirmed the validity of the proposed strategy. As a result, the proposed step-by-step calibration approach represents a practical, cost-effective and computationally less demanding alternative to standard calibration approaches, making robot calibration more accessible and easier to perform. 

**Abstract (ZH)**: 机器人定位精度是执行高精度制造任务的关键因素。为了有效提高 manipulator 的精度，使其接近重复性误差，标定起着至关重要的作用。文献中提出了多种机器人标定方法，使用的测量系统和识别算法存在较大差异。我们的目标是开发一种新的逐步适配步骤的运动学标定程序——逐步估计每个参数——仅通过钢丝编码器获得的一维距离测量数据。为实现这一目标，我们提出了一种分析方法，以找到一组校准点，在这些点上，测量距离和预测距离之间的偏差仅取决于该未知参数。这种方法减少了识别过程的计算负担，同时可能提高其准确性。在6自由度机器人臂上进行了仿真和实验测试：结果证实了所提策略的有效性。因此，所提出的逐步标定方法代表了一种实用、成本低且计算负担较小的标准标定方法的替代方案，使机器人标定更具可访问性和易操作性。 

---
# Design of a Visual Pose Estimation Algorithm for Moon Landing 

**Title (ZH)**: 月球着陆视觉姿态估计算法设计 

**Authors**: Atakan Süslü, Betül Rana Kuran, Halil Ersin Söken  

**Link**: [PDF](https://arxiv.org/pdf/2502.14942)  

**Abstract**: In order to make a pinpoint landing on the Moon, the spacecraft's navigation system must be accurate. To achieve the desired accuracy, navigational drift caused by the inertial sensors must be corrected. One way to correct this drift is to use absolute navigation solutions. In this study, a terrain absolute navigation method to estimate the spacecraft's position and attitude is proposed. This algorithm uses the position of the craters below the spacecraft for estimation. Craters seen by the camera onboard the spacecraft are detected and identified using a crater database known beforehand. In order to focus on estimation algorithms, image processing and crater matching steps are skipped. The accuracy of the algorithm and the effect of the crater number used for estimation are inspected by performing simulations. 

**Abstract (ZH)**: 基于地形的绝对导航方法用于月球着陆器的位置和姿态估计 

---
# Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting 

**Title (ZH)**: Hier-SLAM++: 基于层次分类高斯点云的神经符号语义SLAM 

**Authors**: Boying Li, Vuong Chi Hao, Peter J. Stuckey, Ian Reid, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14931)  

**Abstract**: We propose Hier-SLAM++, a comprehensive Neuro-Symbolic semantic 3D Gaussian Splatting SLAM method with both RGB-D and monocular input featuring an advanced hierarchical categorical representation, which enables accurate pose estimation as well as global 3D semantic mapping. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making scene understanding particularly challenging and costly. To address this problem, we introduce a novel and general hierarchical representation that encodes both semantic and geometric information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs) as well as the 3D generative model. By utilizing the proposed hierarchical tree structure, semantic information is symbolically represented and learned in an end-to-end manner. We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Additionally, we propose an improved SLAM system to support both RGB-D and monocular inputs using a feed-forward model. To the best of our knowledge, this is the first semantic monocular Gaussian Splatting SLAM system, significantly reducing sensor requirements for 3D semantic understanding and broadening the applicability of semantic Gaussian SLAM system. We conduct experiments on both synthetic and real-world datasets, demonstrating superior or on-par performance with state-of-the-art NeRF-based and Gaussian-based SLAM systems, while significantly reducing storage and training time requirements. 

**Abstract (ZH)**: Hier-SLAM++：一种结合RGB-D和单目输入的高级分类表示的综合神经符号语义三维高斯点云SLAM方法 

---
# VaViM and VaVAM: Autonomous Driving through Video Generative Modeling 

**Title (ZH)**: VaViM和VaVAM：通过视频生成模型实现自主驾驶 

**Authors**: Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, Renaud Marlet, Alexandre Boulch, Mickael Chen, Éloi Zablocki, Andrei Bursuc, Eduardo Valle, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2502.15672)  

**Abstract**: We explore the potential of large-scale generative video models for autonomous driving, introducing an open-source auto-regressive video model (VaViM) and its companion video-action model (VaVAM) to investigate how video pre-training transfers to real-world driving. VaViM is a simple auto-regressive video model that predicts frames using spatio-temporal token sequences. We show that it captures the semantics and dynamics of driving scenes. VaVAM, the video-action model, leverages the learned representations of VaViM to generate driving trajectories through imitation learning. Together, the models form a complete perception-to-action pipeline. We evaluate our models in open- and closed-loop driving scenarios, revealing that video-based pre-training holds promise for autonomous driving. Key insights include the semantic richness of the learned representations, the benefits of scaling for video synthesis, and the complex relationship between model size, data, and safety metrics in closed-loop evaluations. We release code and model weights at this https URL 

**Abstract (ZH)**: 我们探讨了大规模生成视频模型在自动驾驶领域的潜力，介绍了开源的自回归视频模型（VaViM）及其同伴视频动作模型（VaVAM），以研究视频预训练如何转移至真实世界的驾驶场景。VaViM 是一个简单的自回归视频模型，使用时空令牌序列预测帧，并展示了其捕捉驾驶场景的语义和动力学的能力。VaVAM 视频动作模型利用 VaViM 学习到的表示，通过模仿学习生成驾驶轨迹。这两个模型共同构成了从感知到行为的完整管道。我们在开放环和闭环驾驶场景中评估了我们的模型，结果显示基于视频的预训练在自动驾驶领域具有潜力。关键见解包括学习表示的语义丰富性、视频合成中的规模效益，以及在闭环评估中模型规模、数据与安全指标的复杂关系。我们在此 https://链接中发布了代码和模型权重。 

---
# Depth-aware Fusion Method based on Image and 4D Radar Spectrum for 3D Object Detection 

**Title (ZH)**: 基于图像和4D雷达谱的深度aware融合方法用于3D物体检测 

**Authors**: Yue Sun, Yeqiang Qian, Chunxiang Wang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15516)  

**Abstract**: Safety and reliability are crucial for the public acceptance of autonomous driving. To ensure accurate and reliable environmental perception, intelligent vehicles must exhibit accuracy and robustness in various environments. Millimeter-wave radar, known for its high penetration capability, can operate effectively in adverse weather conditions such as rain, snow, and fog. Traditional 3D millimeter-wave radars can only provide range, Doppler, and azimuth information for objects. Although the recent emergence of 4D millimeter-wave radars has added elevation resolution, the radar point clouds remain sparse due to Constant False Alarm Rate (CFAR) operations. In contrast, cameras offer rich semantic details but are sensitive to lighting and weather conditions. Hence, this paper leverages these two highly complementary and cost-effective sensors, 4D millimeter-wave radar and camera. By integrating 4D radar spectra with depth-aware camera images and employing attention mechanisms, we fuse texture-rich images with depth-rich radar data in the Bird's Eye View (BEV) perspective, enhancing 3D object detection. Additionally, we propose using GAN-based networks to generate depth images from radar spectra in the absence of depth sensors, further improving detection accuracy. 

**Abstract (ZH)**: 毫米波雷达和摄像头在提高自主驾驶安全性与可靠性中的融合研究：基于Attention机制的4D毫米波雷达与深度感知摄像头的3D目标检测优化 

---
# OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework 

**Title (ZH)**: OccProphet: 基于观察者-预测器-精炼器框架的摄像机片面元4D占位预测效率前沿推动 

**Authors**: Junliang Chen, Huaiyuan Xu, Yi Wang, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2502.15180)  

**Abstract**: Predicting variations in complex traffic environments is crucial for the safety of autonomous driving. Recent advancements in occupancy forecasting have enabled forecasting future 3D occupied status in driving environments by observing historical 2D images. However, high computational demands make occupancy forecasting less efficient during training and inference stages, hindering its feasibility for deployment on edge agents. In this paper, we propose a novel framework, i.e., OccProphet, to efficiently and effectively learn occupancy forecasting with significantly lower computational requirements while improving forecasting accuracy. OccProphet comprises three lightweight components: Observer, Forecaster, and Refiner. The Observer extracts spatio-temporal features from 3D multi-frame voxels using the proposed Efficient 4D Aggregation with Tripling-Attention Fusion, while the Forecaster and Refiner conditionally predict and refine future occupancy inferences. Experimental results on nuScenes, Lyft-Level5, and nuScenes-Occupancy datasets demonstrate that OccProphet is both training- and inference-friendly. OccProphet reduces 58\%$\sim$78\% of the computational cost with a 2.6$\times$ speedup compared with the state-of-the-art Cam4DOcc. Moreover, it achieves 4\%$\sim$18\% relatively higher forecasting accuracy. Code and models are publicly available at this https URL. 

**Abstract (ZH)**: 高效学习占用预测的OccProphet框架：在降低计算需求的同时提升预测准确性 

---
# Synth It Like KITTI: Synthetic Data Generation for Object Detection in Driving Scenarios 

**Title (ZH)**: KITTI风格合成：驾驶场景中的物体检测合成数据生成 

**Authors**: Richard Marcus, Christian Vogel, Inga Jatzkowski, Niklas Knoop, Marc Stamminger  

**Link**: [PDF](https://arxiv.org/pdf/2502.15076)  

**Abstract**: An important factor in advancing autonomous driving systems is simulation. Yet, there is rather small progress for transferability between the virtual and real world. We revisit this problem for 3D object detection on LiDAR point clouds and propose a dataset generation pipeline based on the CARLA simulator. Utilizing domain randomization strategies and careful modeling, we are able to train an object detector on the synthetic data and demonstrate strong generalization capabilities to the KITTI dataset. Furthermore, we compare different virtual sensor variants to gather insights, which sensor attributes can be responsible for the prevalent domain gap. Finally, fine-tuning with a small portion of real data almost matches the baseline and with the full training set slightly surpasses it. 

**Abstract (ZH)**: 改进自动驾驶系统的一个重要因素是仿真，但在虚拟世界和现实世界之间的迁移尚缺乏进展。我们重新审视了这一问题，在基于CARLA模拟器的3D物体检测于LiDAR点云上的问题上提出了一种数据集生成管道。通过使用领域随机化策略和精细建模，我们能够在合成数据上训练物体检测器，并展示其在KITTI数据集上的强泛化能力。此外，我们比较了不同的虚拟传感器变体以收集见解，探讨哪些传感器属性可能导致了显著的领域差距。最后，使用一小部分真实数据进行微调几乎能够匹配基线，并在完整的训练集中稍微超越它。 

---
# Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition 

**Title (ZH)**: 跨模态場景識別的多視點文本-視覺 Registration 方法 

**Authors**: Tianyi Shang, Zhenyu Li, Pengjie Xu, Jinwei Qiao, Gang Chen, Zihan Ruan, Weijun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14195)  

**Abstract**: Mobile robots necessitate advanced natural language understanding capabilities to accurately identify locations and perform tasks such as package delivery. However, traditional visual place recognition (VPR) methods rely solely on single-view visual information and cannot interpret human language descriptions. To overcome this challenge, we bridge text and vision by proposing a multiview (360° views of the surroundings) text-vision registration approach called Text4VPR for place recognition task, which is the first method that exclusively utilizes textual descriptions to match a database of images. Text4VPR employs the frozen T5 language model to extract global textual embeddings. Additionally, it utilizes the Sinkhorn algorithm with temperature coefficient to assign local tokens to their respective clusters, thereby aggregating visual descriptors from images. During the training stage, Text4VPR emphasizes the alignment between individual text-image pairs for precise textual description. In the inference stage, Text4VPR uses the Cascaded Cross-Attention Cosine Alignment (CCCA) to address the internal mismatch between text and image groups. Subsequently, Text4VPR performs precisely place match based on the descriptions of text-image groups. On Street360Loc, the first text to image VPR dataset we created, Text4VPR builds a robust baseline, achieving a leading top-1 accuracy of 57% and a leading top-10 accuracy of 92% within a 5-meter radius on the test set, which indicates that localization from textual descriptions to images is not only feasible but also holds significant potential for further advancement, as shown in Figure 1. 

**Abstract (ZH)**: 基于多视角文本-视觉注册的自然语言理解在移动机器人位置识别中的应用：Text4VPR方法 

---
# MambaPlace:Text-to-Point-Cloud Cross-Modal Place Recognition with Attention Mamba Mechanisms 

**Title (ZH)**: MambaPlace：基于注意力Mamba机制的跨模态文本到点云地方识别 

**Authors**: Tianyi Shang, Zhenyu Li, Pengjie Xu, Jinwei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2408.15740)  

**Abstract**: Vision Language Place Recognition (VLVPR) enhances robot localization performance by incorporating natural language descriptions from images. By utilizing language information, VLVPR directs robot place matching, overcoming the constraint of solely depending on vision. The essence of multimodal fusion lies in mining the complementary information between different modalities. However, general fusion methods rely on traditional neural architectures and are not well equipped to capture the dynamics of cross modal interactions, especially in the presence of complex intra modal and inter modal correlations. To this end, this paper proposes a novel coarse to fine and end to end connected cross modal place recognition framework, called MambaPlace. In the coarse localization stage, the text description and 3D point cloud are encoded by the pretrained T5 and instance encoder, respectively. They are then processed using Text Attention Mamba (TAM) and Point Clouds Mamba (PCM) for data enhancement and alignment. In the subsequent fine localization stage, the features of the text description and 3D point cloud are cross modally fused and further enhanced through cascaded Cross Attention Mamba (CCAM). Finally, we predict the positional offset from the fused text point cloud features, achieving the most accurate localization. Extensive experiments show that MambaPlace achieves improved localization accuracy on the KITTI360Pose dataset compared to the state of the art methods. 

**Abstract (ZH)**: Vision-Language-Powered Place Recognition (VLVPR) 提高了机器人定位性能通过结合图像中的自然语言描述。通过利用语言信息，VLVPR 引导机器人位置匹配，克服了仅仅依赖视觉的限制。多模态融合的本质在于挖掘不同模态之间的互补信息。然而，一般的融合方法依赖于传统的神经架构，并不擅长捕捉跨模态交互的动态特性，尤其是在存在复杂内模态和跨模态相关性的情况下。为此，本文提出了一种从粗到细且端到端连接的跨模态位置识别框架，称为 MambaPlace。在粗定位阶段，文本描述和 3D 点云分别由预训练的 T5 和实例编码器编码。然后使用文本注意力 Mamba (TAM) 和点云 Mamba (PCM) 对数据进行增强和对齐。在后续的精细定位阶段，文本描述和 3D 点云的特征通过级联跨注意力 Mamba (CCAM) 跨模态融合并进一步增强。最后，我们从融合的文本点云特征中预测位置偏移，实现最准确的定位。大量实验表明，MambaPlace 在 KITTI360Pose 数据集上实现了比现有方法更高的定位精度。 

---
