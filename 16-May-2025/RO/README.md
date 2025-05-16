# Loop closure grasping: Topological transformations enable strong, gentle, and versatile grasps 

**Title (ZH)**: 环状闭合抓取：拓扑变换实现稳健、温柔且多功能的抓取 

**Authors**: Kentaro Barhydt, O. Godson Osele, Sreela Kodali, Cosima du Pasquier, Chase M. Hartquist, H. Harry Asada, Allison M. Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.10552)  

**Abstract**: Grasping mechanisms must both create and subsequently hold grasps that permit safe and effective object manipulation. Existing mechanisms address the different functional requirements of grasp creation and grasp holding using a single morphology, but have yet to achieve the simultaneous strength, gentleness, and versatility needed for many applications. We present "loop closure grasping", a class of robotic grasping that addresses these different functional requirements through topological transformations between open-loop and closed-loop morphologies. We formalize these morphologies for grasping, formulate the loop closure grasping method, and present principles and a design architecture that we implement using soft growing inflated beams, winches, and clamps. The mechanisms' initial open-loop topology enables versatile grasp creation via unencumbered tip movement, and closing the loop enables strong and gentle holding with effectively infinite bending compliance. Loop closure grasping circumvents the tradeoffs of single-morphology designs, enabling grasps involving historically challenging objects, environments, and configurations. 

**Abstract (ZH)**: 基于拓扑变换的闭环抓取 

---
# Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning 

**Title (ZH)**: 基于多模态推理的实时离分布故障预防 

**Authors**: Milan Ganai, Rohan Sinha, Christopher Agia, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.10547)  

**Abstract**: Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. 

**Abstract (ZH)**: FORTRESS：一种实时生成和推理语义安全Fallback策略的框架 

---
# AORRTC: Almost-Surely Asymptotically Optimal Planning with RRT-Connect 

**Title (ZH)**: AORRTC：几乎 surely 趋近最优的 RRT-Connect 规划算法 

**Authors**: Tyler Wilson, Wil Thomason, Zachary Kingston, Jonathan Gammell  

**Link**: [PDF](https://arxiv.org/pdf/2505.10542)  

**Abstract**: Finding high-quality solutions quickly is an important objective in motion planning. This is especially true for high-degree-of-freedom robots. Satisficing planners have traditionally found feasible solutions quickly but provide no guarantees on their optimality, while almost-surely asymptotically optimal (a.s.a.o.) planners have probabilistic guarantees on their convergence towards an optimal solution but are more computationally expensive.
This paper uses the AO-x meta-algorithm to extend the satisficing RRT-Connect planner to optimal planning. The resulting Asymptotically Optimal RRT-Connect (AORRTC) finds initial solutions in similar times as RRT-Connect and uses any additional planning time to converge towards the optimal solution in an anytime manner. It is proven to be probabilistically complete and a.s.a.o.
AORRTC was tested with the Panda (7 DoF) and Fetch (8 DoF) robotic arms on the MotionBenchMaker dataset. These experiments show that AORRTC finds initial solutions as fast as RRT-Connect and faster than the tested state-of-the-art a.s.a.o. algorithms while converging to better solutions faster. AORRTC finds solutions to difficult high-DoF planning problems in milliseconds where the other a.s.a.o. planners could not consistently find solutions in seconds. This performance was demonstrated both with and without single instruction/multiple data (SIMD) acceleration. 

**Abstract (ZH)**: 快速找到高质量解决方案是运动规划中的一个重要目标，尤其对于高自由度机器人而言。机会型规划器可以快速找到可行解，但不保证解的最优化；几乎肯定渐近最优（a.s.a.o.）规划器在概率上可以向最优解收敛，但计算成本更高。

本文利用AO-x元算法将机会型RRT-Connect规划器扩展为最优规划器。由此产生的渐近最优RRT-Connect (AORRTC)可以在与RRT-Connect类似的时间内找到初始解，并利用额外的规划时间以任意时间的方式向最优解收敛。证明其具有概率完备性和几乎肯定渐近最优性。

AORRTC在MotionBenchMaker数据集上使用Panda（7自由度）和Fetch（8自由度）机器人臂进行了测试。这些实验表明，AORRTC可以像RRT-Connect一样快速找到初始解，并且在收敛到更好解时更快。AORRTC可以在毫秒内找到困难的高自由度规划问题的解，而其他几乎肯定渐近最优规划器在数秒内无法一致地找到解。无论是否有单指令多数据（SIMD）加速，这种性能都得到了验证。 

---
# Knowledge capture, adaptation and composition (KCAC): A framework for cross-task curriculum learning in robotic manipulation 

**Title (ZH)**: 知识获取、适应与组合（KCAC）：面向机器人操作跨任务课程学习的框架 

**Authors**: Xinrui Wang, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10522)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable potential in robotic manipulation but faces challenges in sample inefficiency and lack of interpretability, limiting its applicability in real world scenarios. Enabling the agent to gain a deeper understanding and adapt more efficiently to diverse working scenarios is crucial, and strategic knowledge utilization is a key factor in this process. This paper proposes a Knowledge Capture, Adaptation, and Composition (KCAC) framework to systematically integrate knowledge transfer into RL through cross-task curriculum learning. KCAC is evaluated using a two block stacking task in the CausalWorld benchmark, a complex robotic manipulation environment. To our knowledge, existing RL approaches fail to solve this task effectively, reflecting deficiencies in knowledge capture. In this work, we redesign the benchmark reward function by removing rigid constraints and strict ordering, allowing the agent to maximize total rewards concurrently and enabling flexible task completion. Furthermore, we define two self-designed sub-tasks and implement a structured cross-task curriculum to facilitate efficient learning. As a result, our KCAC approach achieves a 40 percent reduction in training time while improving task success rates by 10 percent compared to traditional RL methods. Through extensive evaluation, we identify key curriculum design parameters subtask selection, transition timing, and learning rate that optimize learning efficiency and provide conceptual guidance for curriculum based RL frameworks. This work offers valuable insights into curriculum design in RL and robotic learning. 

**Abstract (ZH)**: 基于知识捕获、适应与组合的强化学习框架：跨任务课程学习在机器人操作中的系统集成 

---
# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning 

**Title (ZH)**: 交错强化学习与模仿学习方法：策略微调 

**Authors**: Dechen Gao, Hang Wang, Hanchu Zhou, Nejib Ammar, Shatadal Mishra, Ahmadreza Moradipari, Iman Soltani, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10442)  

**Abstract**: Imitation learning (IL) and reinforcement learning (RL) each offer distinct advantages for robotics policy learning: IL provides stable learning from demonstrations, and RL promotes generalization through exploration. While existing robot learning approaches using IL-based pre-training followed by RL-based fine-tuning are promising, this two-step learning paradigm often suffers from instability and poor sample efficiency during the RL fine-tuning phase. In this work, we introduce IN-RIL, INterleaved Reinforcement learning and Imitation Learning, for policy fine-tuning, which periodically injects IL updates after multiple RL updates and hence can benefit from the stability of IL and the guidance of expert data for more efficient exploration throughout the entire fine-tuning process. Since IL and RL involve different optimization objectives, we develop gradient separation mechanisms to prevent destructive interference during \ABBR fine-tuning, by separating possibly conflicting gradient updates in orthogonal subspaces. Furthermore, we conduct rigorous analysis, and our findings shed light on why interleaving IL with RL stabilizes learning and improves sample-efficiency. Extensive experiments on 14 robot manipulation and locomotion tasks across 3 benchmarks, including FurnitureBench, OpenAI Gym, and Robomimic, demonstrate that \ABBR can significantly improve sample efficiency and mitigate performance collapse during online finetuning in both long- and short-horizon tasks with either sparse or dense rewards. IN-RIL, as a general plug-in compatible with various state-of-the-art RL algorithms, can significantly improve RL fine-tuning, e.g., from 12\% to 88\% with 6.3x improvement in the success rate on Robomimic Transport. Project page: this https URL. 

**Abstract (ZH)**: 交替强化学习和 imitation 学习（IN-RIL）：在策略微调中的交替强化学习和 imitation 学习 

---
# Internal State Estimation in Groups via Active Information Gathering 

**Title (ZH)**: 群体中的内部状态估计通过主动信息收集 

**Authors**: Xuebo Ji, Zherong Pan, Xifeng Gao, Lei Yang, Xinxin Du, Kaiyun Li, Yongjin Liu, Wenping Wang, Changhe Tu, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10415)  

**Abstract**: Accurately estimating human internal states, such as personality traits or behavioral patterns, is critical for enhancing the effectiveness of human-robot interaction, particularly in group settings. These insights are key in applications ranging from social navigation to autism diagnosis. However, prior methods are limited by scalability and passive observation, making real-time estimation in complex, multi-human settings difficult. In this work, we propose a practical method for active human personality estimation in groups, with a focus on applications related to Autism Spectrum Disorder (ASD). Our method combines a personality-conditioned behavior model, based on the Eysenck 3-Factor theory, with an active robot information gathering policy that triggers human behaviors through a receding-horizon planner. The robot's belief about human personality is then updated via Bayesian inference. We demonstrate the effectiveness of our approach through simulations, user studies with typical adults, and preliminary experiments involving participants with ASD. Our results show that our method can scale to tens of humans and reduce personality prediction error by 29.2% and uncertainty by 79.9% in simulation. User studies with typical adults confirm the method's ability to generalize across complex personality distributions. Additionally, we explore its application in autism-related scenarios, demonstrating that the method can identify the difference between neurotypical and autistic behavior, highlighting its potential for diagnosing ASD. The results suggest that our framework could serve as a foundation for future ASD-specific interventions. 

**Abstract (ZH)**: 准确估计人类内心状态，如个性特征或行为模式，对于增强人机交互的有效性，尤其是在群体设置中，至关重要。这些见解在从社会导航到自闭症诊断的应用中至关重要。然而，先前的方法由于可扩展性和被动观察的限制，使得在复杂多人类环境中实现实时估计变得困难。在本工作中，我们提出了一种实用的方法，用于群体中的人格主动估计，重点关注与自闭症谱系障碍（ASD）相关应用。我们的方法结合了基于耶森三因素理论的个性条件行为模型，以及通过回溯规划器触发人类行为的主动机器人信息收集策略。机器人的关于人类人格的信念通过贝叶斯推理进行更新。我们通过仿真、典型成人用户的实验以及自闭症谱系障碍参与者的初步实验，证明了该方法的有效性。结果显示，我们的方法可以扩展到数十人，并在仿真中将人格预测误差降低了29.2%，不确定性降低了79.9%。典型成人用户的实验确认了该方法在复杂人格分布中的泛化能力。此外，我们探讨了该方法在自闭症相关场景中的应用，证明该方法可以识别正常人和自闭症患者的行为差异，突显了其在诊断自闭症谱系障碍方面的潜力。研究结果表明，我们的框架可以为未来的自闭症特定干预措施提供基础。 

---
# AutoCam: Hierarchical Path Planning for an Autonomous Auxiliary Camera in Surgical Robotics 

**Title (ZH)**: AutoCam: 外科机器人中自主辅助相机的分层路径规划 

**Authors**: Alexandre Banks, Randy Moore, Sayem Nazmuz Zaman, Alaa Eldin Abdelaal, Septimiu E. Salcudean  

**Link**: [PDF](https://arxiv.org/pdf/2505.10398)  

**Abstract**: Incorporating an autonomous auxiliary camera into robot-assisted minimally invasive surgery (RAMIS) enhances spatial awareness and eliminates manual viewpoint control. Existing path planning methods for auxiliary cameras track two-dimensional surgical features but do not simultaneously account for camera orientation, workspace constraints, and robot joint limits. This study presents AutoCam: an automatic auxiliary camera placement method to improve visualization in RAMIS. Implemented on the da Vinci Research Kit, the system uses a priority-based, workspace-constrained control algorithm that combines heuristic geometric placement with nonlinear optimization to ensure robust camera tracking. A user study (N=6) demonstrated that the system maintained 99.84% visibility of a salient feature and achieved a pose error of 4.36 $\pm$ 2.11 degrees and 1.95 $\pm$ 5.66 mm. The controller was computationally efficient, with a loop time of 6.8 $\pm$ 12.8 ms. An additional pilot study (N=6), where novices completed a Fundamentals of Laparoscopic Surgery training task, suggests that users can teleoperate just as effectively from AutoCam's viewpoint as from the endoscope's while still benefiting from AutoCam's improved visual coverage of the scene. These results indicate that an auxiliary camera can be autonomously controlled using the da Vinci patient-side manipulators to track a salient feature, laying the groundwork for new multi-camera visualization methods in RAMIS. 

**Abstract (ZH)**: 将自主辅助相机集成到机器人辅助微创手术（RAMIS）中，增强空间感知并消除手动视角控制。现有的辅助相机路径规划方法追踪二维手术特征，但未同时考虑相机姿态、工作空间约束和机器人关节限制。本研究提出AutoCam：一种自动辅助相机定位方法，以提高RAMIS中的可视化效果。该系统基于达芬奇研究套件实现，使用基于优先级的工作空间约束控制算法，结合启发式几何定位和非线性优化，确保相机跟踪的鲁棒性。用户研究（N=6）表明，系统保持了99.84%的重要特征可视性，实现了姿态误差4.36±2.11度和1.95±5.66毫米。控制器计算效率高，循环时间为6.8±12.8毫秒。此外，初步试验（N=6）显示，新手可以在使用AutoCam视角进行腹腔镜手术基础培训任务时，依然能够有效地进行远程操作，同时受益于AutoCam改善的场景视觉覆盖。这些结果表明，可以通过达芬奇术野 manipulators 自主控制辅助相机跟踪重要特征，为RAMIS中的多相机可视化方法奠定了基础。 

---
# NVSPolicy: Adaptive Novel-View Synthesis for Generalizable Language-Conditioned Policy Learning 

**Title (ZH)**: NVSPolicy：自适应新颖视角合成的通用语言条件政策学习 

**Authors**: Le Shi, Yifei Shi, Xin Xu, Tenglong Liu, Junhua Xi, Chengyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10359)  

**Abstract**: Recent advances in deep generative models demonstrate unprecedented zero-shot generalization capabilities, offering great potential for robot manipulation in unstructured environments. Given a partial observation of a scene, deep generative models could generate the unseen regions and therefore provide more context, which enhances the capability of robots to generalize across unseen environments. However, due to the visual artifacts in generated images and inefficient integration of multi-modal features in policy learning, this direction remains an open challenge. We introduce NVSPolicy, a generalizable language-conditioned policy learning method that couples an adaptive novel-view synthesis module with a hierarchical policy network. Given an input image, NVSPolicy dynamically selects an informative viewpoint and synthesizes an adaptive novel-view image to enrich the visual context. To mitigate the impact of the imperfect synthesized images, we adopt a cycle-consistent VAE mechanism that disentangles the visual features into the semantic feature and the remaining feature. The two features are then fed into the hierarchical policy network respectively: the semantic feature informs the high-level meta-skill selection, and the remaining feature guides low-level action estimation. Moreover, we propose several practical mechanisms to make the proposed method efficient. Extensive experiments on CALVIN demonstrate the state-of-the-art performance of our method. Specifically, it achieves an average success rate of 90.4\% across all tasks, greatly outperforming the recent methods. Ablation studies confirm the significance of our adaptive novel-view synthesis paradigm. In addition, we evaluate NVSPolicy on a real-world robotic platform to demonstrate its practical applicability. 

**Abstract (ZH)**: Recent advances in深生成模型展示了前所未有的零样本泛化能力，为机器人在非结构化环境中的操作提供了巨大潜力。给定场景的部分观测，深度生成模型可以生成未观测到的区域，从而提供更多的上下文信息，增强机器人在未见过的环境中泛化的能力。然而，由于生成图像中的视觉伪影以及多模态特征在策略学习中的低效整合，这一方向仍是一项开放性挑战。我们提出了NVSPolicy，这是一种通用的语言条件策略学习方法，结合了自适应新颖视图合成模块和分层策略网络。给定输入图像，NVSPolicy动态选择一个信息丰富的视角并合成自适应新颖视图图像以丰富视觉上下文。为减轻合成图像不完美的影响，我们采用了循环一致的VAE机制，将视觉特征分解为语义特征和剩余特征。这两个特征分别输入到分层策略网络中：语义特征指导高层元技能的选择，而剩余特征指导低层面动作的估计。此外，我们还提出了一些实用机制以提高所提方法的效率。在CALVIN上的广泛实验表明，我们的方法实现了最先进的性能。具体而言，该方法在所有任务中的平均成功率达到了90.4%，显著优于最近的方法。消融研究证实了我们自适应新颖视图合成框架的重要性。此外，我们在实际的机器人平台上评估了NVSPolicy，以证明其实际应用性。 

---
# pc-dbCBS: Kinodynamic Motion Planning of Physically-Coupled Robot Teams 

**Title (ZH)**: 物理耦合机器人团队的 kinodynamic 运动规划 

**Authors**: Khaled Wahba, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2505.10355)  

**Abstract**: Motion planning problems for physically-coupled multi-robot systems in cluttered environments are challenging due to their high dimensionality. Existing methods combining sampling-based planners with trajectory optimization produce suboptimal results and lack theoretical guarantees. We propose Physically-coupled discontinuity-bounded Conflict-Based Search (pc-dbCBS), an anytime kinodynamic motion planner, that extends discontinuity-bounded CBS to rigidly-coupled systems. Our approach proposes a tri-level conflict detection and resolution framework that includes the physical coupling between the robots. Moreover, pc-dbCBS alternates iteratively between state space representations, thereby preserving probabilistic completeness and asymptotic optimality while relying only on single-robot motion primitives. Across 25 simulated and six real-world problems involving multirotors carrying a cable-suspended payload and differential-drive robots linked by rigid rods, pc-dbCBS solves up to 92% more instances than a state-of-the-art baseline and plans trajectories that are 50-60% faster while reducing planning time by an order of magnitude. 

**Abstract (ZH)**: 物理耦合多机器人系统在复杂环境中的运动规划问题由于其高维性具有挑战性。现有方法结合基于采样的规划器与轨迹优化会产生次优化的结果并缺乏理论保证。我们提出了物理耦合的断点受限冲突基于搜索（pc-dbCBS），这是一种可随时使用的动力学运动规划器，将断点受限冲突基于搜索扩展到刚性耦合系统。我们的方法提出了一种包含机器人之间物理耦合的三级冲突检测与解决框架。此外，pc-dbCBS 通过迭代地交替使用状态空间表示，从而保持概率完备性和渐近优化性，同时仅依赖单机器人运动基元。在包括携带悬挂载荷的多旋翼和通过刚性杆链接的差分驱动机器人在内的25个模拟问题和6个真实世界问题中，pc-dbCBS 比最先进的基线解决了多出92%的实例，并规划了快50-60%的路径，同时将规划时间缩短了一个数量级。 

---
# SRT-H: A Hierarchical Framework for Autonomous Surgery via Language Conditioned Imitation Learning 

**Title (ZH)**: SRT-H：一种基于语言条件化imitation learning的分级自主手术框架 

**Authors**: Ji Woong Kim, Juo-Tung Chen, Pascal Hansen, Lucy X. Shi, Antony Goldenberg, Samuel Schmidgall, Paul Maria Scheikl, Anton Deguet, Brandon M. White, De Ru Tsai, Richard Cha, Jeffrey Jopling, Chelsea Finn, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2505.10251)  

**Abstract**: Research on autonomous robotic surgery has largely focused on simple task automation in controlled environments. However, real-world surgical applications require dexterous manipulation over extended time scales while demanding generalization across diverse variations in human tissue. These challenges remain difficult to address using existing logic-based or conventional end-to-end learning strategies. To bridge this gap, we propose a hierarchical framework for dexterous, long-horizon surgical tasks. Our method employs a high-level policy for task planning and a low-level policy for generating task-space controls for the surgical robot. The high-level planner plans tasks using language, producing task-specific or corrective instructions that guide the robot at a coarse level. Leveraging language as a planning modality offers an intuitive and generalizable interface, mirroring how experienced surgeons instruct traineers during procedures. We validate our framework in ex-vivo experiments on a complex minimally invasive procedure, cholecystectomy, and conduct ablative studies to assess key design choices. Our approach achieves a 100% success rate across n=8 different ex-vivo gallbladders, operating fully autonomously without human intervention. The hierarchical approach greatly improves the policy's ability to recover from suboptimal states that are inevitable in the highly dynamic environment of realistic surgical applications. This work represents the first demonstration of step-level autonomy, marking a critical milestone toward autonomous surgical systems for clinical studies. By advancing generalizable autonomy in surgical robotics, our approach brings the field closer to real-world deployment. 

**Abstract (ZH)**: 基于语言的层次化框架用于长时间尺度的灵巧手术任务研究 

---
# Context-aware collaborative pushing of heavy objects using skeleton-based intention prediction 

**Title (ZH)**: 基于骨架意图预测的上下文感知重物协作推拿 

**Authors**: Gokhan Solak, Gustavo J. G. Lahr, Idil Ozdamar, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2505.10239)  

**Abstract**: In physical human-robot interaction, force feedback has been the most common sensing modality to convey the human intention to the robot. It is widely used in admittance control to allow the human to direct the robot. However, it cannot be used in scenarios where direct force feedback is not available since manipulated objects are not always equipped with a force sensor. In this work, we study one such scenario: the collaborative pushing and pulling of heavy objects on frictional surfaces, a prevalent task in industrial settings. When humans do it, they communicate through verbal and non-verbal cues, where body poses, and movements often convey more than words. We propose a novel context-aware approach using Directed Graph Neural Networks to analyze spatio-temporal human posture data to predict human motion intention for non-verbal collaborative physical manipulation. Our experiments demonstrate that robot assistance significantly reduces human effort and improves task efficiency. The results indicate that incorporating posture-based context recognition, either together with or as an alternative to force sensing, enhances robot decision-making and control efficiency. 

**Abstract (ZH)**: 基于物理的人机交互中，力反馈一直是最常用的传感模态以传达人类意图给机器人。它广泛应用于顺应控制，允许人类引导机器人。然而，在无法直接提供力反馈的场景中，由于操纵的对象并不总是配备有力传感器，这一方法无法使用。在本工作中，我们研究了这样一个场景：在摩擦表面上协作推拉重物，这是一个广泛存在于工业环境中的任务。当人类进行这项任务时，他们通过口头和非口头的线索进行沟通，肢体姿势和动作往往传达了更多意义。我们提出了一种新颖的基于上下文的定向图神经网络方法，以分析时空人类姿势数据来预测人类运动意图，以实现非言语协作物理操控。实验结果表明，机器人的协助显著减少了人类的努力，并提高了任务的效率。结果表明，结合或替代基于姿态的上下文识别，可以增强机器人的决策能力和控制效率。 

---
# Quad-LCD: Layered Control Decomposition Enables Actuator-Feasible Quadrotor Trajectory Planning 

**Title (ZH)**: Quad-LCD：分层控制分解实现可行的四旋翼飞行器轨迹规划 

**Authors**: Anusha Srikanthan, Hanli Zhang, Spencer Folk, Vijay Kumar, Nikolai Matni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10228)  

**Abstract**: In this work, we specialize contributions from prior work on data-driven trajectory generation for a quadrotor system with motor saturation constraints. When motors saturate in quadrotor systems, there is an ``uncontrolled drift" of the vehicle that results in a crash. To tackle saturation, we apply a control decomposition and learn a tracking penalty from simulation data consisting of low, medium and high-cost reference trajectories. Our approach reduces crash rates by around $49\%$ compared to baselines on aggressive maneuvers in simulation. On the Crazyflie hardware platform, we demonstrate feasibility through experiments that lead to successful flights. Motivated by the growing interest in data-driven methods to quadrotor planning, we provide open-source lightweight code with an easy-to-use abstraction of hardware platforms. 

**Abstract (ZH)**: 本研究专注于数据驱动的轨迹生成对于四旋翼系统在电机饱和约束下的贡献，特别是处理电机饱和问题。通过控制分解和从包含低成本、中成本和高成本参考轨迹的模拟数据中学习跟踪惩罚，我们降低了四旋翼系统在模拟中激进操作下的相撞率约49%。在疯狂flie硬件平台上，通过实验展示了其实现可行性并成功飞行。为推动四旋翼飞行器规划中数据驱动方法的兴趣增长，我们提供了开源轻量级代码，并提供了易于使用的硬件平台抽象。 

---
# Force-Driven Validation for Collaborative Robotics in Automated Avionics Testing 

**Title (ZH)**: 基于力驱动验证的协作机器人在自动化航空电子测试中的应用 

**Authors**: Pietro Dardano, Paolo Rocco, David Frisini  

**Link**: [PDF](https://arxiv.org/pdf/2505.10224)  

**Abstract**: ARTO is a project combining collaborative robots (cobots) and Artificial Intelligence (AI) to automate functional test procedures for civilian and military aircraft certification. This paper proposes a Deep Learning (DL) and eXplainable AI (XAI) approach, equipping ARTO with interaction analysis capabilities to verify and validate the operations on cockpit components. During these interactions, forces, torques, and end effector poses are recorded and preprocessed to filter disturbances caused by low performance force controllers and embedded Force Torque Sensors (FTS). Convolutional Neural Networks (CNNs) then classify the cobot actions as Success or Fail, while also identifying and reporting the causes of failure. To improve interpretability, Grad CAM, an XAI technique for visual explanations, is integrated to provide insights into the models decision making process. This approach enhances the reliability and trustworthiness of the automated testing system, facilitating the diagnosis and rectification of errors that may arise during testing. 

**Abstract (ZH)**: ARTO是一个结合协作机器人（cobot）和人工智能（AI）的项目，旨在自动化民用和军事航空器认证的功能测试程序。本文提出了一种深度学习（DL）和可解释人工智能（XAI）的方法，使ARTO具备交互分析能力，以验证和验证对驾驶舱组件操作的正确性。在这些交互过程中，记录并预处理力、扭矩和末端执行器姿态，以过滤由低性能力控制器和嵌入式力矩传感器（FTS）引起的干扰。卷积神经网络（CNN）对cobot的动作进行分类，区分成功和失败，并识别和报告失败的原因。为了提高可解释性，整合了Grad CAM这一XAI技术，用于视觉解释，提供对模型决策过程的洞察。该方法增强了自动化测试系统的可靠性和可信度，有助于诊断和纠正测试中可能出现的错误。 

---
# Towards Safe Robot Foundation Models Using Inductive Biases 

**Title (ZH)**: 基于归纳偏置实现安全的机器人基础模型 

**Authors**: Maximilian Tölle, Theo Gruner, Daniel Palenicek, Tim Schneider, Jonas Günster, Joe Watson, Davide Tateo, Puze Liu, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2505.10219)  

**Abstract**: Safety is a critical requirement for the real-world deployment of robotic systems. Unfortunately, while current robot foundation models show promising generalization capabilities across a wide variety of tasks, they fail to address safety, an important aspect for ensuring long-term operation. Current robot foundation models assume that safe behavior should emerge by learning from a sufficiently large dataset of demonstrations. However, this approach has two clear major drawbacks. Firstly, there are no formal safety guarantees for a behavior cloning policy trained using supervised learning. Secondly, without explicit knowledge of any safety constraints, the policy may require an unreasonable number of additional demonstrations to even approximate the desired constrained behavior. To solve these key issues, we show how we can instead combine robot foundation models with geometric inductive biases using ATACOM, a safety layer placed after the foundation policy that ensures safe state transitions by enforcing action constraints. With this approach, we can ensure formal safety guarantees for generalist policies without providing extensive demonstrations of safe behavior, and without requiring any specific fine-tuning for safety. Our experiments show that our approach can be beneficial both for classical manipulation tasks, where we avoid unwanted collisions with irrelevant objects, and for dynamic tasks, such as the robot air hockey environment, where we can generate fast trajectories respecting complex tasks and joint space constraints. 

**Abstract (ZH)**: 机器人系统中安全性是实际部署的关键要求。尽管当前的机器人基础模型在广泛的任务中展现了有希望的泛化能力，但它们未能解决安全性问题，这是确保长期运行的重要方面。当前的机器人基础模型假设安全行为可以通过从足够大的示例数据集中学习而自然地产生。然而，这种做法有两个明显的重大缺点。首先，通过监督学习训练的行为克隆策略无法提供任何形式的安全保证。其次，在没有明确的安全约束知识的情况下，策略可能需要不合理的大量额外示例来近似所需的约束行为。为解决这些关键问题，我们展示了如何通过ATACOM安全层将机器人基础模型与几何归纳偏置相结合，在基础策略之后放置一个确保安全状态转换的安全层，通过强制执行动作约束来实现。借助此方法，我们可以在不提供大量安全行为示例的情况下，确保通用策略的形式安全保证，并且不需要任何特定的安全微调。我们的实验表明，该方法在传统操作任务中（如避免不相关物体的不必要的碰撞）和动态任务（如机器人桌上冰壶环境）中均可带来益处，可以生成尊重复杂任务和关节空间约束的快速轨迹。 

---
# Training People to Reward Robots 

**Title (ZH)**: 训练人类奖励机器人 

**Authors**: Endong Sun, Yuqing Zhu, Matthew Howard  

**Link**: [PDF](https://arxiv.org/pdf/2505.10151)  

**Abstract**: Learning from demonstration (LfD) is a technique that allows expert teachers to teach task-oriented skills to robotic systems. However, the most effective way of guiding novice teachers to approach expert-level demonstrations quantitatively for specific teaching tasks remains an open question. To this end, this paper investigates the use of machine teaching (MT) to guide novice teachers to improve their teaching skills based on reinforcement learning from demonstration (RLfD). The paper reports an experiment in which novices receive MT-derived guidance to train their ability to teach a given motor skill with only 8 demonstrations and generalise this to previously unseen ones. Results indicate that the MT-guidance not only enhances robot learning performance by 89% on the training skill but also causes a 70% improvement in robot learning performance on skills not seen by subjects during training. These findings highlight the effectiveness of MT-guidance in upskilling human teaching behaviours, ultimately improving demonstration quality in RLfD. 

**Abstract (ZH)**: 基于机器教学的示谱方法提升新老师的教学技能：强化学习从演示中指导初学者教师的行为分析 

---
# EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation 

**Title (ZH)**: 具身MAE：用于机器人操作的统一3D多模态表示 

**Authors**: Zibin Dong, Fei Ni, Yifu Yuan, Yinchuan Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10105)  

**Abstract**: We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical. 

**Abstract (ZH)**: EmbodiedMAE: 统一的3D多模态表示框架以促进机器人 manipulation 

---
# FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation 

**Title (ZH)**: FlowDreamer：一种基于流的运动表示的RGB-D世界模型在机器人操作中的应用 

**Authors**: Jun Guo, Xiaojian Ma, Yikai Wang, Min Yang, Huaping Liu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10075)  

**Abstract**: This paper investigates training better visual world models for robot manipulation, i.e., models that can predict future visual observations by conditioning on past frames and robot actions. Specifically, we consider world models that operate on RGB-D frames (RGB-D world models). As opposed to canonical approaches that handle dynamics prediction mostly implicitly and reconcile it with visual rendering in a single model, we introduce FlowDreamer, which adopts 3D scene flow as explicit motion representations. FlowDreamer first predicts 3D scene flow from past frame and action conditions with a U-Net, and then a diffusion model will predict the future frame utilizing the scene flow. FlowDreamer is trained end-to-end despite its modularized nature. We conduct experiments on 4 different benchmarks, covering both video prediction and visual planning tasks. The results demonstrate that FlowDreamer achieves better performance compared to other baseline RGB-D world models by 7% on semantic similarity, 11% on pixel quality, and 6% on success rate in various robot manipulation domains. 

**Abstract (ZH)**: 本文研究了更好的视觉世界模型在机器人操作中的训练，即能够在过去帧和机器人动作的条件下预测未来视觉观察的模型。具体而言，我们考虑基于RGB-D帧的视觉世界模型（RGB-D视觉世界模型）。与大多数经典方法主要通过隐式处理动力学预测并将其与视觉渲染统一在一个模型中不同，我们提出了FlowDreamer，它采用三维场景流作为显式的运动表示。FlowDreamer 首先使用U-Net预测三维场景流，并利用场景流预测未来帧，然后通过扩散模型完成预测。尽管FlowDreamer 具有模块化结构，但它是端到端训练的。我们在4个不同的基准上进行了实验，涵盖了视频预测和视觉规划任务。实验结果表明，FlowDreamer 在语义相似性、像素质量以及不同机器人操作领域的成功率上分别优于其他基线RGB-D视觉世界模型7%、11%和6%。 

---
# Multi-Robot Task Allocation for Homogeneous Tasks with Collision Avoidance via Spatial Clustering 

**Title (ZH)**: 基于空间聚类的同质任务多机器人任务分配与避碰 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10073)  

**Abstract**: In this paper, a novel framework is presented that achieves a combined solution based on Multi-Robot Task Allocation (MRTA) and collision avoidance with respect to homogeneous measurement tasks taking place in industrial environments. The spatial clustering we propose offers to simultaneously solve the task allocation problem and deal with collision risks by cutting the workspace into distinguishable operational zones for each robot. To divide task sites and to schedule robot routes within corresponding clusters, we use K-means clustering and the 2-Opt algorithm. The presented framework shows satisfactory performance, where up to 93\% time reduction (1.24s against 17.62s) with a solution quality improvement of up to 7\% compared to the best performing method is demonstrated. Our method also completely eliminates collision points that persist in comparative methods in a most significant sense. Theoretical analysis agrees with the claim that spatial partitioning unifies the apparently disjoint tasks allocation and collision avoidance problems under conditions of many identical tasks to be distributed over sparse geographical areas. Ultimately, the findings in this work are of substantial importance for real world applications where both computational efficiency and operation free from collisions is of paramount importance. 

**Abstract (ZH)**: 基于工业环境中同质测量任务的多机器人任务分配与避碰新型框架 

---
# Evaluating Robustness of Deep Reinforcement Learning for Autonomous Surface Vehicle Control in Field Tests 

**Title (ZH)**: 评估深度强化学习在田间试验中对自主水面车辆控制健壮性的性能 

**Authors**: Luis F. W. Batista, Stéphanie Aravecchia, Seth Hutchinson, Cédric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2505.10033)  

**Abstract**: Despite significant advancements in Deep Reinforcement Learning (DRL) for Autonomous Surface Vehicles (ASVs), their robustness in real-world conditions, particularly under external disturbances, remains insufficiently explored. In this paper, we evaluate the resilience of a DRL-based agent designed to capture floating waste under various perturbations. We train the agent using domain randomization and evaluate its performance in real-world field tests, assessing its ability to handle unexpected disturbances such as asymmetric drag and an off-center payload. We assess the agent's performance under these perturbations in both simulation and real-world experiments, quantifying performance degradation and benchmarking it against an MPC baseline. Results indicate that the DRL agent performs reliably despite significant disturbances. Along with the open-source release of our implementation, we provide insights into effective training strategies, real-world challenges, and practical considerations for deploying DRLbased ASV controllers. 

**Abstract (ZH)**: 尽管在自主水面车辆（ASVs）的深度强化学习（DRL）方面取得了显著进展，但它们在现实世界条件下的稳健性，尤其是在外来干扰下的表现，仍缺乏充分探索。本文评估了一种基于DRL的代理在各种干扰下的恢复能力，该代理旨在捕捉漂浮废弃物。我们使用领域随机化进行训练，并在实地测试中评估其性能，评估其处理非对称阻力和偏心载荷等意外干扰的能力。我们在模拟和实际实验中评估代理在这些干扰下的性能，量化性能退化，并将其基准与MPC基线进行比较。结果表明，即使在显著干扰下，DRL代理也能可靠地工作。除了开源发布我们的实现外，我们还提供了有效训练策略、现实挑战和部署基于DRL的ASV控制器的实用考虑的见解。 

---
# Fast Heuristic Scheduling and Trajectory Planning for Robotic Fruit Harvesters with Multiple Cartesian Arms 

**Title (ZH)**: 多 Cartesian 腕机器人水果收获机的快速启发式调度与轨迹规划 

**Authors**: Yuankai Zhu, Stavros Vougioukas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10028)  

**Abstract**: This work proposes a fast heuristic algorithm for the coupled scheduling and trajectory planning of multiple Cartesian robotic arms harvesting fruits. Our method partitions the workspace, assigns fruit-picking sequences to arms, determines tight and feasible fruit-picking schedules and vehicle travel speed, and generates smooth, collision-free arm trajectories. The fruit-picking throughput achieved by the algorithm was assessed using synthetically generated fruit coordinates and a harvester design featuring up to 12 arms. The throughput increased monotonically as more arms were added. Adding more arms when fruit densities were low resulted in diminishing gains because it took longer to travel from one fruit to another. However, when there were enough fruits, the proposed algorithm achieved a linear speedup as the number of arms increased. 

**Abstract (ZH)**: 本研究提出了一种快速启发式算法，用于多笛卡尔机器人手臂联合调度和轨迹规划的果实采摘。该方法将工作空间划分为多个区域，分配果实采摘顺序给各手臂，确定紧致且可行的果实采摘时间表和车辆行驶速度，并生成平滑且无碰撞的手臂轨迹。通过合成生成的果实坐标和最多配备12个手臂的采摘器设计，评估了算法实现的果实采摘 throughput。随着手臂数量的增加， throughput 呈单调增加趋势。在果实密度较低时，增加更多手臂会导致效益递减，因为从一个果实到另一个果实的行驶时间变长。然而，当果实数量足够时，所提出的算法在手臂数量增加时实现了线性加速。 

---
# APEX: Action Priors Enable Efficient Exploration for Skill Imitation on Articulated Robots 

**Title (ZH)**: APEX: 动作先验使有装配限制的机器人技能模仿探索更高效 

**Authors**: Shivam Sood, Laukik B Nakhwa, Yuhong Cao, Sun Ge, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2505.10022)  

**Abstract**: Learning by imitation provides an effective way for robots to develop well-regulated complex behaviors and directly benefit from natural demonstrations. State-of-the-art imitation learning (IL) approaches typically leverage Adversarial Motion Priors (AMP), which, despite their impressive results, suffer from two key limitations. They are prone to mode collapse, which often leads to overfitting to the simulation environment and thus increased sim-to-real gap, and they struggle to learn diverse behaviors effectively. To overcome these limitations, we introduce APEX (Action Priors enable Efficient eXploration): a simple yet versatile imitation learning framework that integrates demonstrations directly into reinforcement learning (RL), maintaining high exploration while grounding behavior with expert-informed priors. We achieve this through a combination of decaying action priors, which initially bias exploration toward expert demonstrations but gradually allow the policy to explore independently. This is complemented by a multi-critic RL framework that effectively balances stylistic consistency with task performance. Our approach achieves sample-efficient imitation learning and enables the acquisition of diverse skills within a single policy. APEX generalizes to varying velocities and preserves reference-like styles across complex tasks such as navigating rough terrain and climbing stairs, utilizing only flat-terrain kinematic motion data as a prior. We validate our framework through extensive hardware experiments on the Unitree Go2 quadruped. There, APEX yields diverse and agile locomotion gaits, inherent gait transitions, and the highest reported speed for the platform to the best of our knowledge (peak velocity of ~3.3 m/s on hardware). Our results establish APEX as a compelling alternative to existing IL methods, offering better efficiency, adaptability, and real-world performance. 

**Abstract (ZH)**: 通过模仿学习使机器人能够发展出受良好调控的复杂行为并直接从中受益于自然示范提供了有效的方法。最先进的模仿学习（IL）方法通常利用对抗运动先验（AMP），尽管它们取得了令人印象深刻的成果，但仍面临两个关键限制。它们容易发生模式崩溃，这通常导致过度拟合模拟环境，从而增加了模拟到现实的差距，并且难以有效学习多种行为。为克服这些限制，我们提出了APEX（动作先验促进高效探索）：一种简单但功能强大的模仿学习框架，将示范直接集成到强化学习（RL）中，同时保持高探索性并用专家指导的先验知识为基础行为。我们通过衰减动作先验实现这一点，这些先验最初偏向于专家示范，但逐渐允许策略独立探索。这与多批评家RL框架相辅相成，该框架有效地平衡了风格一致性与任务性能。我们的方法实现了高效的模仿学习，并能够在一个策略中获得多种技能。APEX能够泛化到不同的速度，并在复杂的任务如穿越崎岖地形和上下楼梯中保留参考样式的风格，仅使用平坦地形的运动学运动数据作为先验。我们通过在Unitree Go2四足机器人的广泛硬件实验验证了该框架。在那里，APEX产生了多样且灵活的运动模式，内在的步态转换，并且据我们所知，该平台最高的工作效率（峰值速度约为3.3 m/s）。我们的研究成果确立了APEX作为一种比现有IL方法更具吸引力的替代方案的地位，提供了更好的效率、适应性和实际性能。 

---
# LEMON-Mapping: Loop-Enhanced Large-Scale Multi-Session Point Cloud Merging and Optimization for Globally Consistent Mapping 

**Title (ZH)**: LEMON-Mapping：循环增强的大规模多会话点云合并与优化以实现全局一致的地图构建 

**Authors**: Lijie Wang, Xiaoyi Zhong, Ziyi Xu, Kaixin Chai, Anke Zhao, Tianyu Zhao, Qianhao Wang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10018)  

**Abstract**: With the rapid development of robotics, multi-robot collaboration has become critical and challenging. One key problem is integrating data from multiple robots to build a globally consistent and accurate map for robust cooperation and precise localization. While traditional multi-robot pose graph optimization (PGO) maintains basic global consistency, it focuses primarily on pose optimization and ignores the geometric structure of the map. Moreover, PGO only uses loop closure as a constraint between two nodes, failing to fully exploit its capability to maintaining local consistency of multi-robot maps. Therefore, PGO-based multi-robot mapping methods often suffer from serious map divergence and blur, especially in regions with overlapping submaps. To address this issue, we propose Lemon-Mapping, a loop-enhanced framework for large-scale multi-session point cloud map fusion and optimization, which reasonably utilizes loop closure and improves the geometric quality of the map. We re-examine the role of loops for multi-robot mapping and introduce three key innovations. First, we develop a robust loop processing mechanism that effectively rejects outliers and a novel loop recall strategy to recover mistakenly removed loops. Second, we introduce a spatial bundle adjustment method for multi-robot maps that significantly reduces the divergence in overlapping regions and eliminates map blur. Third, we design a PGO strategy that leverages the refined constraints of bundle adjustment to extend the local accuracy to the global map. We validate our framework on several public datasets and a self-collected dataset. Experimental results demonstrate that our method outperforms traditional map merging approaches in terms of mapping accuracy and reduction of map divergence. Scalability experiments also demonstrate the strong capability of our framework to handle scenarios involving numerous robots. 

**Abstract (ZH)**: 基于循环增强的大规模多会话点云地图融合与优化框架 

---
# Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots 

**Title (ZH)**: 增强四足机器人敏捷性的多样化自然行为学习 

**Authors**: Huiqiao Fu, Haoyu Dong, Wentao Xu, Zhehao Zhou, Guizhou Deng, Kaiqiang Tang, Daoyi Dong, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09979)  

**Abstract**: Achieving animal-like agility is a longstanding goal in quadrupedal robotics. While recent studies have successfully demonstrated imitation of specific behaviors, enabling robots to replicate a broader range of natural behaviors in real-world environments remains an open challenge. Here we propose an integrated controller comprising a Basic Behavior Controller (BBC) and a Task-Specific Controller (TSC) which can effectively learn diverse natural quadrupedal behaviors in an enhanced simulator and efficiently transfer them to the real world. Specifically, the BBC is trained using a novel semi-supervised generative adversarial imitation learning algorithm to extract diverse behavioral styles from raw motion capture data of real dogs, enabling smooth behavior transitions by adjusting discrete and continuous latent variable inputs. The TSC, trained via privileged learning with depth images as input, coordinates the BBC to efficiently perform various tasks. Additionally, we employ evolutionary adversarial simulator identification to optimize the simulator, aligning it closely with reality. After training, the robot exhibits diverse natural behaviors, successfully completing the quadrupedal agility challenge at an average speed of 1.1 m/s and achieving a peak speed of 3.2 m/s during hurdling. This work represents a substantial step toward animal-like agility in quadrupedal robots, opening avenues for their deployment in increasingly complex real-world environments. 

**Abstract (ZH)**: 实现类似动物的敏捷性一直是 quadruped 机器人领域一个长期的目标。虽然近期的研究成功地展示了特定行为的模仿，但在真实环境中超范围复制自然行为依然是一个开放的挑战。我们提出了一种综合控制器，包括基本行为控制器（BBC）和任务特定控制器（TSC），它可以有效地在增强的模拟器中学习多种自然的 quadruped 行为，并高效地将其转移到真实世界。具体而言，BBC 使用一种新的半监督生成对抗模仿学习算法进行训练，从真实的狗的原始运动捕捉数据中提取多样的行为风格，通过调整离散和连续的潜在变量输入实现平滑的行为过渡。TSC 通过特权学习训练，并以深度图像作为输入，协调 BBC 高效地执行各种任务。此外，我们采用了进化对抗模拟器识别来优化模拟器，使其与现实更为贴近。经过训练后的机器人表现出多样化的自然行为，在 quadruped 敏捷性挑战中以平均每秒 1.1 米的速度成功完成任务，并在跳跃过程中达到每秒 3.2 米的最高速度。这项工作代表了向 quadruped 机器人实现类似动物的敏捷性迈出的重要一步，为它们在日益复杂的现实环境中的部署提供了可能。 

---
# Hyper Yoshimura: How a slight tweak on a classical folding pattern unleashes meta-stability for deployable robots 

**Title (ZH)**: Hyper Yoshimura: 一个经典折叠图案的微小调整如何释放可部署机器人的亚稳态 

**Authors**: Ziyang Zhou, Yogesh Phalak, Vishrut Deshpande, Ian Walker, Suyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.09919)  

**Abstract**: Deployable structures inspired by origami offer lightweight, compact, and reconfigurable solutions for robotic and architectural applications. We present a geometric and mechanical framework for Yoshimura-Ori modules that supports a diverse set of metastable states, including newly identified asymmetric "pop-out" and "hyperfolded" configurations. These states are governed by three parameters -- tilt angle, phase shift, and slant height -- and enable discrete, programmable transformations. Using this model, we develop forward and inverse kinematic strategies to stack modules into deployable booms that approximate complex 3D shapes. We validate our approach through mechanical tests and demonstrate a tendon- and pneumatically-actuated Yoshimura Space Crane capable of object manipulation, solar tracking, and high load-bearing performance. A meter-scale solar charging station further illustrates the design's scalability. These results establish Yoshimura-Ori structures as a promising platform for adaptable, multifunctional deployable systems in both terrestrial and space environments. 

**Abstract (ZH)**: 仿 Origami 的可变形结构因其轻量化、紧凑化和可重构特性，在机器人和建筑应用中提供了解决方案。我们提出了一个几何和力学框架，支持 Yoshimura-Ori 模块的多种亚稳态配置，包括新发现的不对称“弹出”和“超折叠”配置。这些状态由三个参数——倾角、相位偏移和斜高——控制，并能使模块实现离散的、可编程的变换。使用此模型，我们开发了前向和逆向运动策略，将模块堆叠成可变形的臂架，以近似复杂的三维形状。我们通过机械测试验证了这种方法，并展示了可实现物体操作、太阳跟踪和高承载性能的腱驱动和气动驱动的 Yoshimura 空间起重机。进一步，一米规模的太阳能充电站演示了该设计的可扩展性。这些结果确立了 Yoshimura-Ori 结构作为适应性强、多用途可变形系统平台，在陆地和太空环境中的潜力。 

---
# Diffusion-SAFE: Shared Autonomy Framework with Diffusion for Safe Human-to-Robot Driving Handover 

**Title (ZH)**: 扩散-SAFE：包含扩散的共享自主框架以实现安全的人机驾驶权交接 

**Authors**: Yunxin Fan, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2505.09889)  

**Abstract**: Safe handover in shared autonomy for vehicle control is well-established in modern vehicles. However, avoiding accidents often requires action several seconds in advance. This necessitates understanding human driver behavior and an expert control strategy for seamless intervention when a collision or unsafe state is predicted. We propose Diffusion-SAFE, a closed-loop shared autonomy framework leveraging diffusion models to: (1) predict human driving behavior for detection of potential risks, (2) generate safe expert trajectories, and (3) enable smooth handovers by blending human and expert policies over a short time horizon. Unlike prior works which use engineered score functions to rate driving performance, our approach enables both performance evaluation and optimal action sequence generation from demonstrations. By adjusting the forward and reverse processes of the diffusion-based copilot, our method ensures a gradual transition of control authority, by mimicking the drivers' behavior before intervention, which mitigates abrupt takeovers, leading to smooth transitions. We evaluated Diffusion-SAFE in both simulation (CarRacing-v0) and real-world (ROS-based race car), measuring human-driving similarity, safety, and computational efficiency. Results demonstrate a 98.5\% successful handover rate, highlighting the framework's effectiveness in progressively correcting human actions and continuously sampling optimal robot actions. 

**Abstract (ZH)**: 基于扩散模型的Safe手递在共享自主车辆控制中的安全递归研究 

---
# Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior 

**Title (ZH)**: 基于任意LiDAR引导扩散先验的无监督雷达点云增强 

**Authors**: Yanlong Yang, Jianan Liu, Guanxiong Luo, Hao Li, Euijoon Ahn, Mostafa Rahimi Azghadi, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09887)  

**Abstract**: In industrial automation, radar is a critical sensor in machine perception. However, the angular resolution of radar is inherently limited by the Rayleigh criterion, which depends on both the radar's operating wavelength and the effective aperture of its antenna this http URL overcome these hardware-imposed limitations, recent neural network-based methods have leveraged high-resolution LiDAR data, paired with radar measurements, during training to enhance radar point cloud resolution. While effective, these approaches require extensive paired datasets, which are costly to acquire and prone to calibration error. These challenges motivate the need for methods that can improve radar resolution without relying on paired high-resolution ground-truth data. Here, we introduce an unsupervised radar points enhancement algorithm that employs an arbitrary LiDAR-guided diffusion model as a prior without the need for paired training data. Specifically, our approach formulates radar angle estimation recovery as an inverse problem and incorporates prior knowledge through a diffusion model with arbitrary LiDAR domain knowledge. Experimental results demonstrate that our method attains high fidelity and low noise performance compared to traditional regularization techniques. Additionally, compared to paired training methods, it not only achieves comparable performance but also offers improved generalization capability. To our knowledge, this is the first approach that enhances radar points output by integrating prior knowledge via a diffusion model rather than relying on paired training data. Our code is available at this https URL. 

**Abstract (ZH)**: 工业自动化中，雷达是机器感知中的关键传感器。然而，雷达的角分辨率受瑞利准则的固有限制，取决于雷达的工作波长和天线的有效孔径。为了克服这些由硬件引起的限制，近年来基于神经网络的方法在训练中结合了高分辨率LiDAR数据和雷达测量数据，以提升雷达点云分辨率。虽然这些方法有效，但它们需要大量的配对数据集，这些数据集获取成本高且容易出现校准误差。这些问题推动了需要不依赖配对高分辨率真实数据的方法来提高雷达分辨率的需求。在这里，我们引入了一种无需配对训练数据的无监督雷达点增强算法，该算法采用任意LiDAR引导的扩散模型作为先验。具体而言，我们的方法将雷达角度估计恢复视为一个逆问题，并通过具有任意LiDAR领域知识的扩散模型融入先验知识。实验结果表明，与传统的正则化技术相比，我们的方法具有更高的保真度和更低的噪声性能。此外，与配对训练方法相比，它不仅实现了可比的性能，还提高了泛化能力。据我们所知，这是第一个通过扩散模型整合先验知识来提升雷达点输出的方法，而不是依赖配对训练数据。我们的代码可在以下链接获取：this https URL。 

---
# EdgeAI Drone for Autonomous Construction Site Demonstrator 

**Title (ZH)**: 边缘AI无人机自主建筑工地演示器 

**Authors**: Emre Girgin, Arda Taha Candan, Coşkun Anıl Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2505.09837)  

**Abstract**: The fields of autonomous systems and robotics are receiving considerable attention in civil applications such as construction, logistics, and firefighting. Nevertheless, the widespread adoption of these technologies is hindered by the necessity for robust processing units to run AI models. Edge-AI solutions offer considerable promise, enabling low-power, cost-effective robotics that can automate civil services, improve safety, and enhance sustainability. This paper presents a novel Edge-AI-enabled drone-based surveillance system for autonomous multi-robot operations at construction sites. Our system integrates a lightweight MCU-based object detection model within a custom-built UAV platform and a 5G-enabled multi-agent coordination infrastructure. We specifically target the real-time obstacle detection and dynamic path planning problem in construction environments, providing a comprehensive dataset specifically created for MCU-based edge applications. Field experiments demonstrate practical viability and identify optimal operational parameters, highlighting our approach's scalability and computational efficiency advantages compared to existing UAV solutions. The present and future roles of autonomous vehicles on construction sites are also discussed, as well as the effectiveness of edge-AI solutions. We share our dataset publicly at this http URL 

**Abstract (ZH)**: 自主系统和机器人技术在建筑、物流和消防等民用应用领域受到广泛关注。然而，这些技术的广泛采用受限于运行AI模型所需的 robust 处理单元。边缘AI解决方案显示出巨大潜力，能够实现低功耗、低成本的机器人，从而自动化公共服务、提高安全性和增强可持续性。本文介绍了用于建筑工地自主多机器人操作的新型边缘AI赋能无人机监视系统。我们的系统在自定义构建的无人机平台和5G使能的多智能体协调基础设施中集成了一个轻量级MCU基础的对象检测模型。我们特别针对建筑环境中的实时障碍检测和动态路径规划问题，提供了专门为MCU基础边缘应用创建的全面数据集。实地实验展示了其实用可行性，确定了最佳操作参数，并突出了与现有无人机解决方案相比，我们的方法在规模性和计算效率方面的优势。本文还讨论了自主车辆在建筑工地当前和未来的作用，以及边缘AI解决方案的有效性。我们已将数据集在此网址公开：[http://example.com]。 

---
# Learning Rock Pushability on Rough Planetary Terrain 

**Title (ZH)**: 在粗糙行星地形中学习岩石推移性 

**Authors**: Tuba Girgin, Emre Girgin, Cagri Kilic  

**Link**: [PDF](https://arxiv.org/pdf/2505.09833)  

**Abstract**: In the context of mobile navigation in unstructured environments, the predominant approach entails the avoidance of obstacles. The prevailing path planning algorithms are contingent upon deviating from the intended path for an indefinite duration and returning to the closest point on the route after the obstacle is left behind spatially. However, avoiding an obstacle on a path that will be used repeatedly by multiple agents can hinder long-term efficiency and lead to a lasting reliance on an active path planning system. In this study, we propose an alternative approach to mobile navigation in unstructured environments by leveraging the manipulation capabilities of a robotic manipulator mounted on top of a mobile robot. Our proposed framework integrates exteroceptive and proprioceptive feedback to assess the push affordance of obstacles, facilitating their repositioning rather than avoidance. While our preliminary visual estimation takes into account the characteristics of both the obstacle and the surface it relies on, the push affordance estimation module exploits the force feedback obtained by interacting with the obstacle via a robotic manipulator as the guidance signal. The objective of our navigation approach is to enhance the efficiency of routes utilized by multiple agents over extended periods by reducing the overall time spent by a fleet in environments where autonomous infrastructure development is imperative, such as lunar or Martian surfaces. 

**Abstract (ZH)**: 在非结构化环境中的移动导航中，主流的方法是避免障碍物。现有的路径规划算法依赖于偏离预定路径一段时间，并在离开障碍物后返回路径上的最近点。然而，对于将被多个代理反复使用的路径上的障碍物避障，可能会阻碍长期效率并导致对活跃路径规划系统的依赖。在本研究中，我们提出了一种利用安装在移动机器人顶部的机器人操作器操作能力的替代移动导航方法。我们提出的框架结合外部和内部反馈来评估障碍物的推搡可行性，从而促进障碍物的重新定位而非避免。虽然我们初步的视觉估计考虑了障碍物及其支撑表面的特性，但推搡可行性估计模块则利用与障碍物交互时通过机器人操作器获得的力反馈作为导向信号。我们的导航方法旨在通过减少在诸如月球或火星表面等需要自主基础设施发展的环境中，车队所花费的总体时间，从而提高多代理长期使用的路径的效率。 

---
# Neural Inertial Odometry from Lie Events 

**Title (ZH)**: 基于李事件的神经惯性里程计 

**Authors**: Royina Karegoudra Jayanth, Yinshuang Xu, Evangelos Chatzipantazis, Kostas Daniilidis, Daniel Gehrig  

**Link**: [PDF](https://arxiv.org/pdf/2505.09780)  

**Abstract**: Neural displacement priors (NDP) can reduce the drift in inertial odometry and provide uncertainty estimates that can be readily fused with off-the-shelf filters. However, they fail to generalize to different IMU sampling rates and trajectory profiles, which limits their robustness in diverse settings. To address this challenge, we replace the traditional NDP inputs comprising raw IMU data with Lie events that are robust to input rate changes and have favorable invariances when observed under different trajectory profiles. Unlike raw IMU data sampled at fixed rates, Lie events are sampled whenever the norm of the IMU pre-integration change, mapped to the Lie algebra of the SE(3) group, exceeds a threshold. Inspired by event-based vision, we generalize the notion of level-crossing on 1D signals to level-crossings on the Lie algebra and generalize binary polarities to normalized Lie polarities within this algebra. We show that training NDPs on Lie events incorporating these polarities reduces the trajectory error of off-the-shelf downstream inertial odometry methods by up to 21% with only minimal preprocessing. We conjecture that many more sensors than IMUs or cameras can benefit from an event-based sampling paradigm and that this work makes an important first step in this direction. 

**Abstract (ZH)**: 基于李事件的神经位移先验（NDP）可以减少惯性 odometry 的漂移，并提供可以与商用滤波器轻松融合的不确定性估计。然而，它们无法在不同的 IMU 采样率和轨迹特征下泛化，这限制了它们在多样环境中的鲁棒性。为了解决这一挑战，我们用对输入率变化具有鲁棒性的李事件替代传统的 NDP 输入，这些李事件在 SE(3) 群的李代数中 IMU 预积分变化的范数超过阈值时进行采样。受事件驱动视觉的启发，我们将 1D 信号上的阈值穿越推广到李代数上的阈值穿越，并在该代数中将二进制极性推广为归一化的李极性。实验表明，通过结合这些极性对李事件进行训练，可以将商用的时间下沉惯性 odometry 方法的轨迹误差最多减少 21%，且仅需少量预处理。我们认为，除了 IMU 或相机之外，还有许多其他传感器可以从基于事件的采样范式中受益，而这项工作是朝着这个方向迈出的重要一步。 

---
# Grasp EveryThing (GET): 1-DoF, 3-Fingered Gripper with Tactile Sensing for Robust Grasping 

**Title (ZH)**: 全方位抓取（GET）：具备触觉感知的1-DoF三指 gripper 及其稳健抓取技术 

**Authors**: Michael Burgess, Edward H. Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2505.09771)  

**Abstract**: We introduce the Grasp EveryThing (GET) gripper, a novel 1-DoF, 3-finger design for securely grasping objects of many shapes and sizes. Mounted on a standard parallel jaw actuator, the design features three narrow, tapered fingers arranged in a two-against-one configuration, where the two fingers converge into a V-shape. The GET gripper is more capable of conforming to object geometries and forming secure grasps than traditional designs with two flat fingers. Inspired by the principle of self-similarity, these V-shaped fingers enable secure grasping across a wide range of object sizes. Further to this end, fingers are parametrically designed for convenient resizing and interchangeability across robotic embodiments with a parallel jaw gripper. Additionally, we incorporate a rigid fingernail to enhance small object manipulation. Tactile sensing can be integrated into the standalone finger via an externally-mounted camera. A neural network was trained to estimate normal force from tactile images with an average validation error of 1.3~N across a diverse set of geometries. In grasping 15 objects and performing 3 tasks via teleoperation, the GET fingers consistently outperformed standard flat fingers. Finger designs for use with multiple robotic embodiments are available on GitHub. 

**Abstract (ZH)**: 一种新颖的单自由度三指夹持器：Grasp EveryThing (GET) 夹持器的设计与应用 

---
# Neural Associative Skill Memories for safer robotics and modelling human sensorimotor repertoires 

**Title (ZH)**: 基于神经关联技能记忆的安全机器人和模拟人类感觉运动 repertoire 方法 

**Authors**: Pranav Mahajan, Mufeng Tang, T. Ed Li, Ioannis Havoutis, Ben Seymour  

**Link**: [PDF](https://arxiv.org/pdf/2505.09760)  

**Abstract**: Modern robots face challenges shared by humans, where machines must learn multiple sensorimotor skills and express them adaptively. Equipping robots with a human-like memory of how it feels to do multiple stereotypical movements can make robots more aware of normal operational states and help develop self-preserving safer robots. Associative Skill Memories (ASMs) aim to address this by linking movement primitives to sensory feedback, but existing implementations rely on hard-coded libraries of individual skills. A key unresolved problem is how a single neural network can learn a repertoire of skills while enabling fault detection and context-aware execution. Here we introduce Neural Associative Skill Memories (ASMs), a framework that utilises self-supervised predictive coding for temporal prediction to unify skill learning and expression, using biologically plausible learning rules. Unlike traditional ASMs which require explicit skill selection, Neural ASMs implicitly recognize and express skills through contextual inference, enabling fault detection across learned behaviours without an explicit skill selection mechanism. Compared to recurrent neural networks trained via backpropagation through time, our model achieves comparable qualitative performance in skill memory expression while using local learning rules and predicts a biologically relevant speed-accuracy trade-off during skill memory expression. This work advances the field of neurorobotics by demonstrating how predictive coding principles can model adaptive robot control and human motor preparation. By unifying fault detection, reactive control, skill memorisation and expression into a single energy-based architecture, Neural ASMs contribute to safer robotics and provide a computational lens to study biological sensorimotor learning. 

**Abstract (ZH)**: 基于预测编码的神经关联技能记忆：统一技能学习与表达以实现自适应机器人控制和生物传感器运动学习 

---
# Trailblazer: Learning offroad costmaps for long range planning 

**Title (ZH)**: Trailblazer: 学习用于长距离规划的离路成本图 

**Authors**: Kasi Viswanath, Felix Sanchez, Timothy Overbye, Jason M. Gregory, Srikanth Saripalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.09739)  

**Abstract**: Autonomous navigation in off-road environments remains a significant challenge in field robotics, particularly for Unmanned Ground Vehicles (UGVs) tasked with search and rescue, exploration, and surveillance. Effective long-range planning relies on the integration of onboard perception systems with prior environmental knowledge, such as satellite imagery and LiDAR data. This work introduces Trailblazer, a novel framework that automates the conversion of multi-modal sensor data into costmaps, enabling efficient path planning without manual tuning. Unlike traditional approaches, Trailblazer leverages imitation learning and a differentiable A* planner to learn costmaps directly from expert demonstrations, enhancing adaptability across diverse terrains. The proposed methodology was validated through extensive real-world testing, achieving robust performance in dynamic and complex environments, demonstrating Trailblazer's potential for scalable, efficient autonomous navigation. 

**Abstract (ZH)**: 自主导航在非道路环境中的实现仍然是领域机器人领域的重大挑战，特别是在无人驾驶地面车辆（UGVs）进行搜索与救援、探索和监控任务时。有效的长距离规划依赖于车载感知系统与先验环境知识的集成，如卫星影像和LiDAR数据。本文介绍了Trailblazer，这是一种新型框架，能够自动化多模态传感器数据到成本地图的转换，从而实现高效的路径规划而无需手动调参。与传统方法不同，Trailblazer利用模仿学习和可微分A*规划器直接从专家演示中学习成本地图，增强了其在多样化地形上的适应性。所提出的方法通过广泛的实地测试得到了验证，表现出在动态和复杂环境中的稳健性能，证明了Trailblazer在可扩展和高效自主导航方面的潜力。 

---
# Unfettered Forceful Skill Acquisition with Physical Reasoning and Coordinate Frame Labeling 

**Title (ZH)**: 无约束的物理推理与坐标系标签驱动的技能习得 

**Authors**: William Xie, Max Conway, Yutong Zhang, Nikolaus Correll  

**Link**: [PDF](https://arxiv.org/pdf/2505.09731)  

**Abstract**: Vision language models (VLMs) exhibit vast knowledge of the physical world, including intuition of physical and spatial properties, affordances, and motion. With fine-tuning, VLMs can also natively produce robot trajectories. We demonstrate that eliciting wrenches, not trajectories, allows VLMs to explicitly reason about forces and leads to zero-shot generalization in a series of manipulation tasks without pretraining. We achieve this by overlaying a consistent visual representation of relevant coordinate frames on robot-attached camera images to augment our query. First, we show how this addition enables a versatile motion control framework evaluated across four tasks (opening and closing a lid, pushing a cup or chair) spanning prismatic and rotational motion, an order of force and position magnitude, different camera perspectives, annotation schemes, and two robot platforms over 220 experiments, resulting in 51% success across the four tasks. Then, we demonstrate that the proposed framework enables VLMs to continually reason about interaction feedback to recover from task failure or incompletion, with and without human supervision. Finally, we observe that prompting schemes with visual annotation and embodied reasoning can bypass VLM safeguards. We characterize prompt component contribution to harmful behavior elicitation and discuss its implications for developing embodied reasoning. Our code, videos, and data are available at: this https URL. 

**Abstract (ZH)**: 视觉语言模型通过在机器人附着的相机图像上叠加一致的视觉表示相关坐标框架，激发力而不是轨迹，展示了在一系列 manipulation 任务中零样本泛化的潜力，无需预训练。我们的代码、视频和数据可在以下网址获取：this https URL。 

---
# EnerVerse-AC: Envisioning Embodied Environments with Action Condition 

**Title (ZH)**: EnerVerse-AC: 融入动作条件的体现环境憧憬 

**Authors**: Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.09723)  

**Abstract**: Robotic imitation learning has advanced from solving static tasks to addressing dynamic interaction scenarios, but testing and evaluation remain costly and challenging due to the need for real-time interaction with dynamic environments. We propose EnerVerse-AC (EVAC), an action-conditional world model that generates future visual observations based on an agent's predicted actions, enabling realistic and controllable robotic inference. Building on prior architectures, EVAC introduces a multi-level action-conditioning mechanism and ray map encoding for dynamic multi-view image generation while expanding training data with diverse failure trajectories to improve generalization. As both a data engine and evaluator, EVAC augments human-collected trajectories into diverse datasets and generates realistic, action-conditioned video observations for policy testing, eliminating the need for physical robots or complex simulations. This approach significantly reduces costs while maintaining high fidelity in robotic manipulation evaluation. Extensive experiments validate the effectiveness of our method. Code, checkpoints, and datasets can be found at <this https URL. 

**Abstract (ZH)**: 基于动作条件的世界模型EnerVerse-AC (EVAC):实现真实且可控的机器人推理 

---
# ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation 

**Title (ZH)**: ManipBench: 用于低层次机器人操作的视觉-语言模型基准测试 

**Authors**: Enyu Zhao, Vedant Raval, Hejia Zhang, Jiageng Mao, Zeyu Shangguan, Stefanos Nikolaidis, Yue Wang, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.09698)  

**Abstract**: Vision-Language Models (VLMs) have revolutionized artificial intelligence and robotics due to their commonsense reasoning capabilities. In robotic manipulation, VLMs are used primarily as high-level planners, but recent work has also studied their lower-level reasoning ability, which refers to making decisions about precise robot movements. However, the community currently lacks a clear and common benchmark that can evaluate how well VLMs can aid low-level reasoning in robotics. Consequently, we propose a novel benchmark, ManipBench, to evaluate the low-level robot manipulation reasoning capabilities of VLMs across various dimensions, including how well they understand object-object interactions and deformable object manipulation. We extensively test 33 representative VLMs across 10 model families on our benchmark, including variants to test different model sizes. Our evaluation shows that the performance of VLMs significantly varies across tasks, and there is a strong correlation between this performance and trends in our real-world manipulation tasks. It also shows that there remains a significant gap between these models and human-level understanding. See our website at: this https URL. 

**Abstract (ZH)**: Vision-Language模型（VLMs）由于其常识推理能力，已经革新了人工智能和机器人技术。在机器人操控中，VLMs 主要用作高级规划者，但最近的研究也开始探索它们的低级推理能力，即关于精准机器人动作的决策。然而，目前机器人社区缺乏一个明确且通用的基准，用于评估VLMs在机器人低级推理方面的协助效果。因此，我们提出了一种新的基准——ManipBench，以从多个维度评估VLMs在机器人低级操控推理能力方面的能力，包括它们理解物体间交互和柔性物体操控的能力。我们在该基准上对10个模型家族中的33个代表性VLMs进行了广泛的测试，包括不同模型大小的变体。我们的评估显示，VLMs在不同任务上的性能存在显著差异，并且这种性能与我们在真实世界操控任务中的趋势之间存在密切联系。此外，我们的评估还显示出这些模型与人类理解水平之间仍存在显著差距。更多详情请参见我们的网站：this https URL。 

---
# EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models 

**Title (ZH)**: EWMBench: 评估体态世界模型中的场景、运动和语义质量 

**Authors**: Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.09694)  

**Abstract**: Recent advances in creative AI have enabled the synthesis of high-fidelity images and videos conditioned on language instructions. Building on these developments, text-to-video diffusion models have evolved into embodied world models (EWMs) capable of generating physically plausible scenes from language commands, effectively bridging vision and action in embodied AI applications. This work addresses the critical challenge of evaluating EWMs beyond general perceptual metrics to ensure the generation of physically grounded and action-consistent behaviors. We propose the Embodied World Model Benchmark (EWMBench), a dedicated framework designed to evaluate EWMs based on three key aspects: visual scene consistency, motion correctness, and semantic alignment. Our approach leverages a meticulously curated dataset encompassing diverse scenes and motion patterns, alongside a comprehensive multi-dimensional evaluation toolkit, to assess and compare candidate models. The proposed benchmark not only identifies the limitations of existing video generation models in meeting the unique requirements of embodied tasks but also provides valuable insights to guide future advancements in the field. The dataset and evaluation tools are publicly available at this https URL. 

**Abstract (ZH)**: Recent Advances in Creative AI Have Enabled the Synthesis of High-Fidelity Images and Videos Conditioned on Language Instructions: Building Embodied World Models (EWMs) for Physically Plausible Scene Generation from Language Commands 

---
# Inferring Driving Maps by Deep Learning-based Trail Map Extraction 

**Title (ZH)**: 基于深度学习的轨迹地图提取驱动地图推断 

**Authors**: Michael Hubbertz, Pascal Colling, Qi Han, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10258)  

**Abstract**: High-definition (HD) maps offer extensive and accurate environmental information about the driving scene, making them a crucial and essential element for planning within autonomous driving systems. To avoid extensive efforts from manual labeling, methods for automating the map creation have emerged. Recent trends have moved from offline mapping to online mapping, ensuring availability and actuality of the utilized maps. While the performance has increased in recent years, online mapping still faces challenges regarding temporal consistency, sensor occlusion, runtime, and generalization. We propose a novel offline mapping approach that integrates trails - informal routes used by drivers - into the map creation process. Our method aggregates trail data from the ego vehicle and other traffic participants to construct a comprehensive global map using transformer-based deep learning models. Unlike traditional offline mapping, our approach enables continuous updates while remaining sensor-agnostic, facilitating efficient data transfer. Our method demonstrates superior performance compared to state-of-the-art online mapping approaches, achieving improved generalization to previously unseen environments and sensor configurations. We validate our approach on two benchmark datasets, highlighting its robustness and applicability in autonomous driving systems. 

**Abstract (ZH)**: 高分辨率（HD）地图提供了 Driving 场景的广泛而准确的环境信息，是自主驾驶系统规划中至关重要的元素。为避免手动标注的大量努力，出现了自动化地图创建的方法。近年来的趋势从离线制图转向在线制图，确保使用的地图的可用性和时效性。尽管近年来性能有所提高，但在线制图仍然面临着时间一致性、传感器遮挡、运行时间和泛化等挑战。我们提出了一种新的离线制图方法，将驾驶员使用的随机路线（trails）整合到地图创建过程中。我们的方法使用基于变压器的深度学习模型聚合来自 ego 车辆和其他交通参与者的轨迹数据，构建全面的全局地图。与传统离线制图不同，我们的方法能够持续更新，同时保持传感器无感知，便于高效数据传输。我们的方法在最先进的在线制图方法中表现出优越性能，实现了对以前未见过的环境和传感器配置的更好泛化。我们通过两个基准数据集验证了该方法，强调了其在自主驾驶系统中的鲁棒性和适用性。 

---
# Threshold Strategy for Leaking Corner-Free Hamilton-Jacobi Reachability with Decomposed Computations 

**Title (ZH)**: 泄漏角-free哈密尔顿-雅可比可达性分解计算的门槛策略 

**Authors**: Chong He, Mugilan Mariappan, Keval Vora, Mo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10020)  

**Abstract**: Hamilton-Jacobi (HJ) Reachability is widely used to compute value functions for states satisfying specific control objectives. However, it becomes intractable for high-dimensional problems due to the curse of dimensionality. Dimensionality reduction approaches are essential for mitigating this challenge, whereas they could introduce the ``leaking corner issue", leading to inaccuracies in the results. In this paper, we define the ``leaking corner issue" in terms of value functions, propose and prove a necessary condition for its occurrence. We then use these theoretical contributions to introduce a new local updating method that efficiently corrects inaccurate value functions while maintaining the computational efficiency of the dimensionality reduction approaches. We demonstrate the effectiveness of our method through numerical simulations. Although we validate our method with the self-contained subsystem decomposition (SCSD), our approach is applicable to other dimensionality reduction techniques that introduce the ``leaking corners". 

**Abstract (ZH)**: Hamilton-Jacobi (HJ)可达性广泛用于计算满足特定控制目标的状态的价值函数。然而，由于维数灾难，它在高维问题上变得不可行。降维方法对于缓解这一挑战至关重要，但可能会引入“泄露角问题”，导致结果不准确。在本文中，我们从价值函数的角度定义了“泄露角问题”，提出了其发生的一个必要条件，并进行了证明。然后，我们利用这些理论贡献引入了一种新的局部更新方法，该方法能够高效地修正不准确的价值函数，同时保持降维方法的计算效率。我们通过数值模拟验证了方法的有效性。尽管我们使用自包含子系统分解（SCSD）验证了该方法，但我们的方法适用于其他引入“泄露角问题”的降维技术。 

---
# Provably safe and human-like car-following behaviors: Part 2. A parsimonious multi-phase model with projected braking 

**Title (ZH)**: 可验证安全且类人的跟随行为：第2部分——一种精简的多阶段模型及其制动力投影方法 

**Authors**: Wen-Long Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09988)  

**Abstract**: Ensuring safe and human-like trajectory planning for automated vehicles amidst real-world uncertainties remains a critical challenge. While existing car-following models often struggle to consistently provide rigorous safety proofs alongside human-like acceleration and deceleration patterns, we introduce a novel multi-phase projection-based car-following model. This model is designed to balance safety and performance by incorporating bounded acceleration and deceleration rates while emulating key human driving principles. Building upon a foundation of fundamental driving principles and a multi-phase dynamical systems analysis (detailed in Part 1 of this study \citep{jin2025WA20-02_Part1}), we first highlight the limitations of extending standard models like Newell's with simple bounded deceleration. Inspired by human drivers' anticipatory behavior, we mathematically define and analyze projected braking profiles for both leader and follower vehicles, establishing safety criteria and new phase definitions based on the projected braking lead-vehicle problem. The proposed parsimonious model combines an extended Newell's model for nominal driving with a new control law for scenarios requiring projected braking. Using speed-spacing phase plane analysis, we provide rigorous mathematical proofs of the model's adherence to defined safe and human-like driving principles, including collision-free operation, bounded deceleration, and acceptable safe stopping distance, under reasonable initial conditions. Numerical simulations validate the model's superior performance in achieving both safety and human-like braking profiles for the stationary lead-vehicle problem. Finally, we discuss the model's implications and future research directions. 

**Abstract (ZH)**: 确保在现实世界不确定性中的自动驾驶车辆安全且拟人化的轨迹规划仍然是一个关键挑战。尽管现有的车跟随模型往往难以一致地提供严格的安全证明同时保持拟人化的加减速模式，我们介绍了基于多阶段投影的新型车跟随模型。该模型旨在通过引入有界加减速率并模拟关键的人类驾驶原则来平衡安全与性能。基于基本驾驶原则和多阶段动力学系统分析（详见本研究第1部分 \citep{jin2025WA20-02_Part1}），我们首先指出将标准模型如Newell模型简单地扩展到带有限制的减速行为时的局限性。借鉴人类驾驶员的预见性行为，我们从数学上定义和分析了领导者和跟随者车辆的投影制动廓线，并基于投影制动前车问题建立了新的安全准则和阶段定义。提出的简约模型结合了扩展的Newell模型用于常规驾驶，并引入了新的控制律以应对需要投影制动的场景。利用速度-间距相平面分析，我们提供了关于该模型如何在合理初始条件下严格遵守定义的安全和拟人化驾驶原则（包括无碰撞运行、有界的减速和可接受的安全停车距离）的数学证明。数值仿真实验证了该模型在解决静止前车问题时在安全性和拟人化制动轮廓方面的优越性能。最后，我们讨论了模型的含义及其未来的研究方向。 

---
# Provably safe and human-like car-following behaviors: Part 1. Analysis of phases and dynamics in standard models 

**Title (ZH)**: 可验证的安全且类人的跟随行为：标准模型中的相位与动力学分析（第1部分） 

**Authors**: Wen-Long Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09987)  

**Abstract**: Trajectory planning is essential for ensuring safe driving in the face of uncertainties related to communication, sensing, and dynamic factors such as weather, road conditions, policies, and other road users. Existing car-following models often lack rigorous safety proofs and the ability to replicate human-like driving behaviors consistently. This article applies multi-phase dynamical systems analysis to well-known car-following models to highlight the characteristics and limitations of existing approaches. We begin by formulating fundamental principles for safe and human-like car-following behaviors, which include zeroth-order principles for comfort and minimum jam spacings, first-order principles for speeds and time gaps, and second-order principles for comfort acceleration/deceleration bounds as well as braking profiles. From a set of these zeroth- and first-order principles, we derive Newell's simplified car-following model. Subsequently, we analyze phases within the speed-spacing plane for the stationary lead-vehicle problem in Newell's model and its extensions, which incorporate both bounded acceleration and deceleration. We then analyze the performance of the Intelligent Driver Model and the Gipps model. Through this analysis, we highlight the limitations of these models with respect to some of the aforementioned principles. Numerical simulations and empirical observations validate the theoretical insights. Finally, we discuss future research directions to further integrate safety, human-like behaviors, and vehicular automation in car-following models, which are addressed in Part 2 of this study \citep{jin2025WA20-02_Part2}, where we develop a novel multi-phase projection-based car-following model that addresses the limitations identified here. 

**Abstract (ZH)**: 轨迹规划对于在通信、感知以及天气、道路条件、政策及其他道路使用者等动态因素带来的不确定性中确保安全驾驶至关重要。现有的跟随车辆模型往往缺乏严格的 safety 证明，并且不能一致地复制人类驾驶行为。本文应用多阶段动力系统分析方法对现有的跟随车辆模型进行研究，以突显现有方法的特点和局限性。我们首先提出了安全和类似人类的跟随车辆行为的基本原则，包括舒适性零阶原则和最小堵塞距离，速度和时间间隔的一阶原则，以及舒适加速度/减速度的边界和制动轨迹的二阶原则。基于这些零阶和一阶原则，我们推导出简化的新ell跟随车辆模型。随后，我们分析了新ell模型及其扩展版本在速度-间距平面上的各个阶段，这些扩展版本同时包含了有界加速度和减速度。接着，我们分析了智能驾驶模型和Gipps模型的性能。通过这些分析，我们指出了这些模型在某些基本原则方面的局限性。数值仿真和实证观察验证了理论洞见。最后，我们讨论了未来的研究方向以进一步将安全、类似人类的行为和车辆自动化整合到跟随车辆模型中，这些问题将在本文的第二部分［1］中详细讨论，在那里我们开发了一种新型多阶段投影跟随车辆模型，以解决本文中指出的限制。 

---
# Large-Scale Gaussian Splatting SLAM 

**Title (ZH)**: 大规模高斯插值 SLAM 

**Authors**: Zhe Xin, Chenyang Wu, Penghui Huang, Yanyong Zhang, Yinian Mao, Guoquan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09915)  

**Abstract**: The recently developed Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown encouraging and impressive results for visual SLAM. However, most representative methods require RGBD sensors and are only available for indoor environments. The robustness of reconstruction in large-scale outdoor scenarios remains unexplored. This paper introduces a large-scale 3DGS-based visual SLAM with stereo cameras, termed LSG-SLAM. The proposed LSG-SLAM employs a multi-modality strategy to estimate prior poses under large view changes. In tracking, we introduce feature-alignment warping constraints to alleviate the adverse effects of appearance similarity in rendering losses. For the scalability of large-scale scenarios, we introduce continuous Gaussian Splatting submaps to tackle unbounded scenes with limited memory. Loops are detected between GS submaps by place recognition and the relative pose between looped keyframes is optimized utilizing rendering and feature warping losses. After the global optimization of camera poses and Gaussian points, a structure refinement module enhances the reconstruction quality. With extensive evaluations on the EuRoc and KITTI datasets, LSG-SLAM achieves superior performance over existing Neural, 3DGS-based, and even traditional approaches. Project page: this https URL. 

**Abstract (ZH)**: 大规模3D高斯布判lor方法的立体视觉SLAM（LSG-SLAM） 

---
# General Dynamic Goal Recognition 

**Title (ZH)**: 通用动态目标识别 

**Authors**: Osher Elhadad, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.09737)  

**Abstract**: Understanding an agent's intent through its behavior is essential in human-robot interaction, interactive AI systems, and multi-agent collaborations. This task, known as Goal Recognition (GR), poses significant challenges in dynamic environments where goals are numerous and constantly evolving. Traditional GR methods, designed for a predefined set of goals, often struggle to adapt to these dynamic scenarios. To address this limitation, we introduce the General Dynamic GR problem - a broader definition of GR - aimed at enabling real-time GR systems and fostering further research in this area. Expanding on this foundation, this paper employs a model-free goal-conditioned RL approach to enable fast adaptation for GR across various changing tasks. 

**Abstract (ZH)**: 通过行为理解代理意图在人机交互、交互式AI系统和多代理协作中至关重要。这一任务，称为目标识别（Goal Recognition, GR），在目标众多且不断演变的动态环境中面临着重大挑战。传统的GR方法针对预定义的目标集，往往难以适应这些动态场景。为了解决这一局限性，我们提出了广泛定义的动态GR问题——旨在使实时GR系统成为可能，并推动该领域的进一步研究。在此基础上，本文采用模型自由的目标条件强化学习方法，以实现跨各种变化任务的快速适应性目标识别。 

---
# Risk-Aware Safe Reinforcement Learning for Control of Stochastic Linear Systems 

**Title (ZH)**: 风险意识的安全强化学习在随机线性系统控制中的应用 

**Authors**: Babak Esmaeili, Nariman Niknejad, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2505.09734)  

**Abstract**: This paper presents a risk-aware safe reinforcement learning (RL) control design for stochastic discrete-time linear systems. Rather than using a safety certifier to myopically intervene with the RL controller, a risk-informed safe controller is also learned besides the RL controller, and the RL and safe controllers are combined together. Several advantages come along with this approach: 1) High-confidence safety can be certified without relying on a high-fidelity system model and using limited data available, 2) Myopic interventions and convergence to an undesired equilibrium can be avoided by deciding on the contribution of two stabilizing controllers, and 3) highly efficient and computationally tractable solutions can be provided by optimizing over a scalar decision variable and linear programming polyhedral sets. To learn safe controllers with a large invariant set, piecewise affine controllers are learned instead of linear controllers. To this end, the closed-loop system is first represented using collected data, a decision variable, and noise. The effect of the decision variable on the variance of the safe violation of the closed-loop system is formalized. The decision variable is then designed such that the probability of safety violation for the learned closed-loop system is minimized. It is shown that this control-oriented approach reduces the data requirements and can also reduce the variance of safety violations. Finally, to integrate the safe and RL controllers, a new data-driven interpolation technique is introduced. This method aims to maintain the RL agent's optimal implementation while ensuring its safety within environments characterized by noise. The study concludes with a simulation example that serves to validate the theoretical results. 

**Abstract (ZH)**: 基于风险意识的鲁棒强化学习控制设计：适用于随机离散时间线性系统的安全控制器学习 

---
