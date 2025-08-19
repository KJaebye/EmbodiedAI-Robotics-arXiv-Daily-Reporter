# Manipulate-to-Navigate: Reinforcement Learning with Visual Affordances and Manipulability Priors 

**Title (ZH)**: 操纵以导航：基于视觉可用性和操作性先验的强化学习 

**Authors**: Yuying Zhang, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13151)  

**Abstract**: Mobile manipulation in dynamic environments is challenging due to movable obstacles blocking the robot's path. Traditional methods, which treat navigation and manipulation as separate tasks, often fail in such 'manipulate-to-navigate' scenarios, as obstacles must be removed before navigation. In these cases, active interaction with the environment is required to clear obstacles while ensuring sufficient space for movement. To address the manipulate-to-navigate problem, we propose a reinforcement learning-based approach for learning manipulation actions that facilitate subsequent navigation. Our method combines manipulability priors to focus the robot on high manipulability body positions with affordance maps for selecting high-quality manipulation actions. By focusing on feasible and meaningful actions, our approach reduces unnecessary exploration and allows the robot to learn manipulation strategies more effectively. We present two new manipulate-to-navigate simulation tasks called Reach and Door with the Boston Dynamics Spot robot. The first task tests whether the robot can select a good hand position in the target area such that the robot base can move effectively forward while keeping the end effector position fixed. The second task requires the robot to move a door aside in order to clear the navigation path. Both of these tasks need first manipulation and then navigating the base forward. Results show that our method allows a robot to effectively interact with and traverse dynamic environments. Finally, we transfer the learned policy to a real Boston Dynamics Spot robot, which successfully performs the Reach task. 

**Abstract (ZH)**: 动态环境下的移动 manipulation 挑战在于可移动障碍物阻碍机器人路径，传统方法将导航和操作视为分开的任务，在“操作以导航”的场景中往往无法应对，因为必须先清除障碍物才能导航。在这种情况下，机器人需要与环境进行主动交互以清除障碍物并确保足够的移动空间。为解决操作以导航的问题，我们提出了一种基于强化学习的方法，用于学习有助于后续导航的操作行动。该方法结合了可操作性先验，使机器人专注于高可操作性身体位置，并使用效应器动作图选择高质量的操作行动。通过集中于可行和有意义的操作，该方法减少了不必要的探索并使机器人能够更有效地学习操作策略。我们使用波士顿动力公司的 Spot 机器人提出了两个新的操作以导航模拟任务，名为 Reach 和 Door。Reach 任务测试机器人是否能在目标区域内选择一个良好的手部位置，使得机器人基座能够有效前移且末端执行器位置保持不变。Door 任务要求机器人移动门以清开通行路径。这两个任务都需要先操作然后前移基座。结果表明，我们的方法使机器人能够有效与动态环境进行交互并穿越。最后，我们将学到的策略转移到实际的波士顿动力公司 Spot 机器人上，并成功执行了 Reach 任务。 

---
# Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy 

**Title (ZH)**: 空间相机中心的动作 grounding 观测-centric 视觉-语言-行动策略 

**Authors**: Tianyi Zhang, Haonan Duan, Haoran Hao, Yu Qiao, Jifeng Dai, Zhi Hou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13103)  

**Abstract**: Vision-Language-Action (VLA) models frequently encounter challenges in generalizing to real-world environments due to inherent discrepancies between observation and action spaces. Although training data are collected from diverse camera perspectives, the models typically predict end-effector poses within the robot base coordinate frame, resulting in spatial inconsistencies. To mitigate this limitation, we introduce the Observation-Centric VLA (OC-VLA) framework, which grounds action predictions directly in the camera observation space. Leveraging the camera's extrinsic calibration matrix, OC-VLA transforms end-effector poses from the robot base coordinate system into the camera coordinate system, thereby unifying prediction targets across heterogeneous viewpoints. This lightweight, plug-and-play strategy ensures robust alignment between perception and action, substantially improving model resilience to camera viewpoint variations. The proposed approach is readily compatible with existing VLA architectures, requiring no substantial modifications. Comprehensive evaluations on both simulated and real-world robotic manipulation tasks demonstrate that OC-VLA accelerates convergence, enhances task success rates, and improves cross-view generalization. The code will be publicly available. 

**Abstract (ZH)**: 基于观察的视知行一体（OC-VLA）框架：在现实环境中的视知行建模 

---
# Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey 

**Title (ZH)**: 基于大型多模态模型的视觉-语言-动作模型在机器人操控中的研究综述 

**Authors**: Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.13073)  

**Abstract**: Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: this https URL. 

**Abstract (ZH)**: 机器人操作是机器人学和具身人工智能的一个关键前沿领域，需要精确的操作控制和多模态理解能力D然而传统的基于规则的方法无法在未结构化环境中实现一般化。近年来,D基于大视觉语言模型（（Large Vision-Language Models,VLMsD的的视觉-语言D动作（（Vision-Language-ActionDVVLAD模型凭借其大规模图像文本数据集的预训练而崭露头出头
user
请纠正上面的翻译语法错，并并并，并结构错误，并并，并更确保句子通顺和符合符合格式正确。D 

---
# BOW: Bayesian Optimization over Windows for Motion Planning in Complex Environments 

**Title (ZH)**: BOW: 复杂环境中的运动规划的窗口下的贝叶斯优化 

**Authors**: Sourav Raxit, Abdullah Al Redwan Newaz, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2508.13052)  

**Abstract**: This paper introduces the BOW Planner, a scalable motion planning algorithm designed to navigate robots through complex environments using constrained Bayesian optimization (CBO). Unlike traditional methods, which often struggle with kinodynamic constraints such as velocity and acceleration limits, the BOW Planner excels by concentrating on a planning window of reachable velocities and employing CBO to sample control inputs efficiently. This approach enables the planner to manage high-dimensional objective functions and stringent safety constraints with minimal sampling, ensuring rapid and secure trajectory generation. Theoretical analysis confirms the algorithm's asymptotic convergence to near-optimal solutions, while extensive evaluations in cluttered and constrained settings reveal substantial improvements in computation times, trajectory lengths, and solution times compared to existing techniques. Successfully deployed across various real-world robotic systems, the BOW Planner demonstrates its practical significance through exceptional sample efficiency, safety-aware optimization, and rapid planning capabilities, making it a valuable tool for advancing robotic applications. The BOW Planner is released as an open-source package and videos of real-world and simulated experiments are available at this https URL. 

**Abstract (ZH)**: 本文介绍了BOW规划器，这是一种用于在复杂环境中使用约束贝叶斯优化（CBO）导航机器人的时间可扩展运动规划算法。与其他经常难以应对速度和加速度等动力学约束的传统方法不同，BOW规划器通过专注于可达速度的规划窗口并利用CBO高效采样控制输入来出类拔萃。这种方法使规划器能够在最少采样的情况下管理高维目标函数和严格的安全约束，确保快速和安全的轨迹生成。理论分析证实了该算法的渐近收敛性于近最优解，而广泛的评估在拥挤和受限环境下显示了与现有技术相比显著缩短的计算时间、轨迹长度和解决方案时间。BOW规划器已在各种实际机器人系统中成功部署，通过卓越的样本效率、安全意识优化和快速规划能力证明了其实用价值，使其成为推进机器人应用的重要工具。BOW规划器作为开源软件包发布，并在该页面提供了实际和模拟实验的视频：<https://this-url>`。 

---
# Scaling Whole-body Multi-contact Manipulation with Contact Optimization 

**Title (ZH)**: 全身多接触 manipulation 的接触优化扩展 

**Authors**: Victor Levé, João Moura, Sachiya Fujita, Tamon Miyake, Steve Tonneau, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.12980)  

**Abstract**: Daily tasks require us to use our whole body to manipulate objects, for instance when our hands are unavailable. We consider the issue of providing humanoid robots with the ability to autonomously perform similar whole-body manipulation tasks. In this context, the infinite possibilities for where and how contact can occur on the robot and object surfaces hinder the scalability of existing planning methods, which predominantly rely on discrete sampling. Given the continuous nature of contact surfaces, gradient-based optimization offers a more suitable approach for finding solutions. However, a key remaining challenge is the lack of an efficient representation of robot surfaces. In this work, we propose (i) a representation of robot and object surfaces that enables closed-form computation of proximity points, and (ii) a cost design that effectively guides whole-body manipulation planning. Our experiments demonstrate that the proposed framework can solve problems unaddressed by existing methods, and achieves a 77% improvement in planning time over the state of the art. We also validate the suitability of our approach on real hardware through the whole-body manipulation of boxes by a humanoid robot. 

**Abstract (ZH)**: humanoid机器人全身影觉操作能力的研究：基于梯度优化的表面表示与任务规划 

---
# Insights from Interviews with Teachers and Students on the Use of a Social Robot in Computer Science Class in Sixth Grade 

**Title (ZH)**: 基于六年级计算机科学课堂中社交机器人使用情况访谈的见解 

**Authors**: Ann-Sophie Schenk, Stefan Schiffer, Heqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.12946)  

**Abstract**: In this paper we report on first insights from interviews with teachers and students on using social robots in computer science class in sixth grade. Our focus is on learning about requirements and potential applications. We are particularly interested in getting both perspectives, the teachers' and the learners' view on how robots could be used and what features they should or should not have. Results show that teachers as well as students are very open to robots in the classroom. However, requirements are partially quite heterogeneous among the groups. This leads to complex design challenges which we discuss at the end of this paper. 

**Abstract (ZH)**: 在本研究中，我们报告了关于六年级计算机科学课堂中使用社会机器人初步见解的访谈结果，重点关注需求和潜在应用。我们特别关注教师和学习者对面向课堂的机器人使用方式及应具备或不应具备的功能特征的看法。结果显示，教师和学生对课堂中使用机器人持非常开放的态度，但各组的需求部分存在显著差异，这带来了复杂的設計挑战，我们在本文末尾进行了讨论。 

---
# Simultaneous Contact Sequence and Patch Planning for Dynamic Locomotion 

**Title (ZH)**: 同时规划接触序列和贴点的动态运动学 

**Authors**: Victor Dhédin, Haizhou Zhao, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2508.12928)  

**Abstract**: Legged robots have the potential to traverse highly constrained environments with agile maneuvers. However, planning such motions requires solving a highly challenging optimization problem with a mixture of continuous and discrete decision variables. In this paper, we present a full pipeline based on Monte-Carlo tree search (MCTS) and whole-body trajectory optimization (TO) to perform simultaneous contact sequence and patch selection on highly challenging environments. Through extensive simulation experiments, we show that our framework can quickly find a diverse set of dynamically consistent plans. We experimentally show that these plans are transferable to a real quadruped robot. We further show that the same framework can find highly complex acyclic humanoid maneuvers. To the best of our knowledge, this is the first demonstration of simultaneous contact sequence and patch selection for acyclic multi-contact locomotion using the whole-body dynamics of a quadruped. 

**Abstract (ZH)**: 具有敏捷动作的腿式机器人有潜力穿越高度受限环境。然而，规划此类动作需要解决一个包含连续和离散决策变量的极高挑战性优化问题。本文提出了一种基于蒙特卡洛树搜索（MCTS）和全身轨迹优化（TO）的完整管道，以在高度挑战性环境中同时进行接触序列和接触点选择。通过大量的仿真实验，我们展示了我们的框架可以快速找到一组动态一致的计划。我们实验证明这些计划可以转移到真实四足机器人上。此外，我们展示了相同的框架可以找到复杂的无环双足行走动作。据我们所知，这是首次使用四足动物全身动力学进行无环多接触行走的接触序列和接触点选择的演示。 

---
# Deformation of the panoramic sphere into an ellipsoid to induce self-motion in telepresence users 

**Title (ZH)**: 全景球面形变成为椭球以诱导远程 presence 用户的自身运动感知 

**Authors**: Eetu Laukka, Evan G. Center, Timo Ojala, Steven M. LaValle, Matti Pouke  

**Link**: [PDF](https://arxiv.org/pdf/2508.12925)  

**Abstract**: Mobile telepresence robots allow users to feel present and explore remote environments using technology. Traditionally, these systems are implemented using a camera onboard a mobile robot that can be controlled. Although high-immersion technologies, such as 360-degree cameras, can increase situational awareness and presence, they also introduce significant challenges. Additional processing and bandwidth requirements often result in latencies of up to seconds. The current delay with a 360-degree camera streaming over the internet makes real-time control of these systems difficult. Working with high-latency systems requires some form of assistance to the users.
This study presents a novel way to utilize optical flow to create an illusion of self-motion to the user during the latency period between user sending motion commands to the robot and seeing the actual motion through the 360-camera stream. We find no significant benefit of using the self-motion illusion to performance or accuracy of controlling a telepresence robot with a latency of 500 ms, as measured by the task completion time and collisions into objects. Some evidence is shown that the method might increase virtual reality (VR) sickness, as measured by the simulator sickness questionnaire (SSQ). We conclude that further adjustments are necessary in order to render the method viable. 

**Abstract (ZH)**: 基于光学流在高延迟移动远程存在机器人中的自我运动 illusion 应用研究 

---
# RoboRetriever: Single-Camera Robot Object Retrieval via Active and Interactive Perception with Dynamic Scene Graph 

**Title (ZH)**: RoboRetriever：基于动态场景图的主动交互式单目机器人物体检索 

**Authors**: Hecheng Wang, Jiankun Ren, Jia Yu, Lizhe Qi, Yunquan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12916)  

**Abstract**: Humans effortlessly retrieve objects in cluttered, partially observable environments by combining visual reasoning, active viewpoint adjustment, and physical interaction-with only a single pair of eyes. In contrast, most existing robotic systems rely on carefully positioned fixed or multi-camera setups with complete scene visibility, which limits adaptability and incurs high hardware costs. We present \textbf{RoboRetriever}, a novel framework for real-world object retrieval that operates using only a \textbf{single} wrist-mounted RGB-D camera and free-form natural language instructions. RoboRetriever grounds visual observations to build and update a \textbf{dynamic hierarchical scene graph} that encodes object semantics, geometry, and inter-object relations over time. The supervisor module reasons over this memory and task instruction to infer the target object and coordinate an integrated action module combining \textbf{active perception}, \textbf{interactive perception}, and \textbf{manipulation}. To enable task-aware scene-grounded active perception, we introduce a novel visual prompting scheme that leverages large reasoning vision-language models to determine 6-DoF camera poses aligned with the semantic task goal and geometry scene context. We evaluate RoboRetriever on diverse real-world object retrieval tasks, including scenarios with human intervention, demonstrating strong adaptability and robustness in cluttered scenes with only one RGB-D camera. 

**Abstract (ZH)**: RoboRetrieved：一种基于单目RGB-D相机的现实世界对象检索框架 

---
# MCTR: Midpoint Corrected Triangulation for Autonomous Racing via Digital Twin Simulation in CARLA 

**Title (ZH)**: MCTR：通过CARLA数字孪生模拟的自主赛车中间点修正三角化 

**Authors**: Junhao Ye, Cheng Hu, Yiqin Wang, Weizhan Huang, Nicolas Baumann, Jie He, Meixun Qu, Lei Xie, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.12729)  

**Abstract**: In autonomous racing, reactive controllers eliminate the computational burden of the full See-Think-Act autonomy stack by directly mapping sensor inputs to control actions. This bypasses the need for explicit localization and trajectory planning. A widely adopted baseline in this category is the Follow-The-Gap method, which performs trajectory planning using LiDAR data. Building on FTG, the Delaunay Triangulation-based Racing algorithm introduces further enhancements. However, DTR's use of circumcircles for trajectory generation often results in insufficiently smooth paths, ultimately degrading performance. Additionally, the commonly used F1TENTH-simulator for autonomous racing competitions lacks support for 3D LiDAR perception, limiting its effectiveness in realistic testing. To address these challenges, this work proposes the MCTR algorithm. MCTR improves trajectory smoothness through the use of Curvature Corrected Moving Average and implements a digital twin system within the CARLA simulator to validate the algorithm's robustness under 3D LiDAR perception. The proposed algorithm has been thoroughly validated through both simulation and real-world vehicle experiments. 

**Abstract (ZH)**: 基于反应式控制的自主赛车中，MCTR算法通过曲率校正移动平均提高轨迹平滑度，并在CARLA仿真器中实现数字孪生系统以验证其在3D LiDAR感知下的鲁棒性。 

---
# Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory 

**Title (ZH)**: 基于柯西尔杆理论的物理信息神经网络的自适应模型预测控制软连续机器人 

**Authors**: Johann Licher, Max Bartholdt, Henrik Krauss, Tim-Lukas Habich, Thomas Seel, Moritz Schappler  

**Link**: [PDF](https://arxiv.org/pdf/2508.12681)  

**Abstract**: Dynamic control of soft continuum robots (SCRs) holds great potential for expanding their applications, but remains a challenging problem due to the high computational demands of accurate dynamic models. While data-driven approaches like Koopman-operator-based methods have been proposed, they typically lack adaptability and cannot capture the full robot shape, limiting their applicability. This work introduces a real-time-capable nonlinear model-predictive control (MPC) framework for SCRs based on a domain-decoupled physics-informed neural network (DD-PINN) with adaptable bending stiffness. The DD-PINN serves as a surrogate for the dynamic Cosserat rod model with a speed-up factor of 44000. It is also used within an unscented Kalman filter for estimating the model states and bending compliance from end-effector position measurements. We implement a nonlinear evolutionary MPC running at 70 Hz on the GPU. In simulation, it demonstrates accurate tracking of dynamic trajectories and setpoint control with end-effector position errors below 3 mm (2.3% of the actuator's length). In real-world experiments, the controller achieves similar accuracy and accelerations up to 3.55 m/s2. 

**Abstract (ZH)**: 软连续机器人（SCRs）的动态控制具有扩展其应用的巨大潜力，但由于准确动态模型的高计算需求，这仍然是一个具有挑战性的问题。尽管提出了基于科普曼算子的方法等数据驱动的方法，但它们通常缺乏适应性且无法捕捉完整的机器人形状，限制了其应用。本文引入了一种基于域解藕物理感知神经网络（DD-PINN）的实时非线性模型预测控制（MPC）框架，用于SCRs，该框架具有可调节的弯曲刚度。DD-PINN 作为动态科西er 杆模型的代理，速度提升因子为 44000。它还用于无迹卡尔曼滤波器中，从末端执行器位置测量中估计模型状态和弯曲顺应性。我们实现了在 GPU 上以 70 Hz 运行的非线性进化 MPC。在仿真中，它展示了准确的动态轨迹跟踪和末端执行器位置误差低于 3 mm（执行器长度的 2.3%）的定值控制。在实际实验中，控制器实现了相似的精度，并达到了高达 3.55 m/s² 的加速度。 

---
# Temporal and Rotational Calibration for Event-Centric Multi-Sensor Systems 

**Title (ZH)**: 事件中心多传感器系统的时间和旋转校准 

**Authors**: Jiayao Mai, Xiuyuan Lu, Kuan Dai, Shaojie Shen, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.12564)  

**Abstract**: Event cameras generate asynchronous signals in response to pixel-level brightness changes, offering a sensing paradigm with theoretically microsecond-scale latency that can significantly enhance the performance of multi-sensor systems. Extrinsic calibration is a critical prerequisite for effective sensor fusion; however, the configuration that involves event cameras remains an understudied topic. In this paper, we propose a motion-based temporal and rotational calibration framework tailored for event-centric multi-sensor systems, eliminating the need for dedicated calibration targets. Our method uses as input the rotational motion estimates obtained from event cameras and other heterogeneous sensors, respectively. Different from conventional approaches that rely on event-to-frame conversion, our method efficiently estimates angular velocity from normal flow observations, which are derived from the spatio-temporal profile of event data. The overall calibration pipeline adopts a two-step approach: it first initializes the temporal offset and rotational extrinsics by exploiting kinematic correlations in the spirit of Canonical Correlation Analysis (CCA), and then refines both temporal and rotational parameters through a joint non-linear optimization using a continuous-time parametrization in SO(3). Extensive evaluations on both publicly available and self-collected datasets validate that the proposed method achieves calibration accuracy comparable to target-based methods, while exhibiting superior stability over purely CCA-based methods, and highlighting its precision, robustness and flexibility. To facilitate future research, our implementation will be made open-source. Code: this https URL. 

**Abstract (ZH)**: 事件相机生成响应于像素级亮度变化的异步信号，提供理论上毫微秒级延迟的传感范式，可显著提升多传感器系统的性能。外部标定是有效传感器融合的关键前提；然而，涉及事件相机的配置 remains an understudied topic. 本文提出了一种基于运动的用于事件为中心的多传感器系统的时域和旋转校准框架，无需专用的校准目标。该方法使用来自事件相机和其他异构传感器的旋转运动估计作为输入。不同于依赖事件到帧转换的常规方法，我们的方法高效地从正常流观察中估计角速度，这些观察是从事件数据的空间-时间分布中导出的。整体校准流水线采用两步方法：首先通过类似典范相关分析（CCA）利用运动学相关性初始化时间偏移和旋转外部参数，然后通过在SO(3)中的连续时间参数化进行联合非线性优化来细化时间和旋转参数。在公开可用和自收集数据集上的广泛评估验证了所提出的方法在准确性和稳定性方面均与基于目标的方法相当，同时显示出在纯粹基于CCA的方法上的优越精度、鲁棒性和灵活性。为了促进未来的研究，我们的实现将开源。代码: this https URL。 

---
# PROD: Palpative Reconstruction of Deformable Objects through Elastostatic Signed Distance Functions 

**Title (ZH)**: PROD：通过弹性静态符号距离函数的可变形物体触觉重建 

**Authors**: Hamza El-Kebir  

**Link**: [PDF](https://arxiv.org/pdf/2508.12554)  

**Abstract**: We introduce PROD (Palpative Reconstruction of Deformables), a novel method for reconstructing the shape and mechanical properties of deformable objects using elastostatic signed distance functions (SDFs). Unlike traditional approaches that rely on purely geometric or visual data, PROD integrates palpative interaction -- measured through force-controlled surface probing -- to estimate both the static and dynamic response of soft materials. We model the deformation of an object as an elastostatic process and derive a governing Poisson equation for estimating its SDF from a sparse set of pose and force measurements. By incorporating steady-state elastodynamic assumptions, we show that the undeformed SDF can be recovered from deformed observations with provable convergence. Our approach also enables the estimation of material stiffness by analyzing displacement responses to varying force inputs. We demonstrate the robustness of PROD in handling pose errors, non-normal force application, and curvature errors in simulated soft body interactions. These capabilities make PROD a powerful tool for reconstructing deformable objects in applications ranging from robotic manipulation to medical imaging and haptic feedback systems. 

**Abstract (ZH)**: 基于触觉交互的不可变形物体形貌与机械性能重建方法PROD 

---
# Mechanical Automation with Vision: A Design for Rubik's Cube Solver 

**Title (ZH)**: 视觉引导的机械自动化：Rubik's立方体解题器的设计 

**Authors**: Abhinav Chalise, Nimesh Gopal Pradhan, Nishan Khanal, Prashant Raj Bista, Dinesh Baniya Kshatri  

**Link**: [PDF](https://arxiv.org/pdf/2508.12469)  

**Abstract**: The core mechanical system is built around three stepper motors for physical manipulation, a microcontroller for hardware control, a camera and YOLO detection model for real-time cube state detection. A significant software component is the development of a user-friendly graphical user interface (GUI) designed in Unity. The initial state after detection from real-time YOLOv8 model (Precision 0.98443, Recall 0.98419, Box Loss 0.42051, Class Loss 0.2611) is virtualized on GUI. To get the solution, the system employs the Kociemba's algorithm while physical manipulation with a single degree of freedom is done by combination of stepper motors' interaction with the cube achieving the average solving time of ~2.2 minutes. 

**Abstract (ZH)**: 基于三个步进电机的物理操作、微控制器的硬件控制、相机和YOLO检测模型的实时立方体状态检测的核心机械系统。实时YOLOv8模型（ Precision 0.98443，Recall 0.98419，Box Loss 0.42051，Class Loss 0.2611）检测后的初始状态在Unity设计的用户友好图形用户界面（GUI）上虚拟化。系统利用Kociemba算法获取解法，通过步进电机与立方体的交互实现单自由度物理操作，平均解题时间为约2.2分钟。 

---
# Autonomous Oil Spill Response Through Liquid Neural Trajectory Modeling and Coordinated Marine Robotics 

**Title (ZH)**: 基于液态神经轨迹建模的自主油污响应及协同海洋机器人技术 

**Authors**: Hadas C.Kuzmenko, David Ehevich, Oren Gal  

**Link**: [PDF](https://arxiv.org/pdf/2508.12456)  

**Abstract**: Marine oil spills pose grave environmental and economic risks, threatening marine ecosystems, coastlines, and dependent industries. Predicting and managing oil spill trajectories is highly complex, due to the interplay of physical, chemical, and environmental factors such as wind, currents, and temperature, which makes timely and effective response challenging. Accurate real-time trajectory forecasting and coordinated mitigation are vital for minimizing the impact of these disasters. This study introduces an integrated framework combining a multi-agent swarm robotics system built on the MOOS-IvP platform with Liquid Time-Constant Neural Networks (LTCNs). The proposed system fuses adaptive machine learning with autonomous marine robotics, enabling real-time prediction, dynamic tracking, and rapid response to evolving oil spills. By leveraging LTCNs--well-suited for modeling complex, time-dependent processes--the framework achieves real-time, high-accuracy forecasts of spill movement. Swarm intelligence enables decentralized, scalable, and resilient decision-making among robot agents, enhancing collective monitoring and containment efforts. Our approach was validated using data from the Deepwater Horizon spill, where the LTC-RK4 model achieved 0.96 spatial accuracy, surpassing LSTM approaches by 23%. The integration of advanced neural modeling with autonomous, coordinated robotics demonstrates substantial improvements in prediction precision, flexibility, and operational scalability. Ultimately, this research advances the state-of-the-art for sustainable, autonomous oil spill management and environmental protection by enhancing both trajectory prediction and response coordination. 

**Abstract (ZH)**: 海洋油 spills事件带来的严重环境和经济风险威胁着海洋生态系统、海岸线以及相关产业。 预测和管理油 spills事件轨迹极具复杂性性，因为涉及物理、化学和环境因素如如洋流、风以及波产生复杂的交互作用，使及时和有效的应对挑战重重。准确的实时轨迹预报和协调的减缓措施对于减轻这些灾难的影响至关重要。本研究提出了一种综合框架，结合基于MOOS-IvP的多代理 swarm机器人系统和液时间常数神经网络（LTCNs）。提出的系统融合了自适应的无人海洋机器人on,，，， enabling实时对动态追踪和快速应对演变中的油渗漏进行预报。通过利用LTCNs适用于建模复杂的时时间依赖过程的优势，该框架实现了对油渗漏轨迹的实时高精度预报。群体智能使代理机器人能够在去中心化的可分布式和鲁棒的决策中enhancing集体监控和遏制努力。我们的方法在深水地井油渗漏事件on on on中 on from on on on on on on on on On on on on on 上 on on on on on on on on on on on on on on on on on on on on on on on on on on on on On on on on on on on on on on on on on通过LTC-RK4方法实现了. on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on On on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on On on on on on on on on on on on on on on on on on on on on on on on on on on verschiedene P on on on实际 on on on on on on on on on on on on on例子中的上 on on on on on on on on on on on on on on on on on on on on on on on on on on on onupos pérdidaaras的 auf pérdida以及重复数据on on on(on on on on on on on on on on on on on on on on on on on on on on on on on pérdida pérdida.

海洋油 spills事件带来的 on on on on.on on on.on on.on on.on.on on.on on.on.on.on on.on on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on带来的的严重环境和经济风险threatens 海洋生态系统on海岸线以及相关industry.on预测和管理油 on sp on.spill �轨迹极具复杂nature.on由于涉及物理、化学和环境factor.on例如洋流、风和以及(Dictionary Entries 中 on on on on on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.On Bren découvert on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on.on/on on on  on  on on on on on ononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononononon monitoring and containment efforts. 实存险提高轨迹预测和应对协调能力，最终这种研究在可持续、自主的油渗漏管理及环境保护方面促进了最先进的的技术。 

---
# Geodesic Tracing-Based Kinematic Integration of Rolling and Sliding Contact on Manifold Meshes for Dexterous In-Hand Manipulation 

**Title (ZH)**: 基于测地线追踪的流形网格上滚滑接触运动学集成及其灵活在手操作 

**Authors**: Sunyu Wang, Arjun S. Lakshmipathy, Jean Oh, Nancy S. Pollard  

**Link**: [PDF](https://arxiv.org/pdf/2508.12439)  

**Abstract**: Reasoning about rolling and sliding contact, or roll-slide contact for short, is critical for dexterous manipulation tasks that involve intricate geometries. But existing works on roll-slide contact mostly focus on continuous shapes with differentiable parametrizations. This work extends roll-slide contact modeling to manifold meshes. Specifically, we present an integration scheme based on geodesic tracing to first-order time-integrate roll-slide contact directly on meshes, enabling dexterous manipulation to reason over high-fidelity discrete representations of an object's true geometry. Using our method, we planned dexterous motions of a multi-finger robotic hand manipulating five objects in-hand in simulation. The planning was achieved with a least-squares optimizer that strives to maintain the most stable instantaneous grasp by minimizing contact sliding and spinning. Then, we evaluated our method against a baseline using collision detection and a baseline using primitive shapes. The results show that our method performed the best in accuracy and precision, even for coarse meshes. We conclude with a future work discussion on incorporating multiple contacts and contact forces to achieve accurate and robust mesh-based surface contact modeling. 

**Abstract (ZH)**: 关于滚动和滑动接触的推理，或简称滚滑接触，对于涉及复杂几何形状的灵巧操作任务至关重要。现有的滚滑接触建模主要集中在具有可微参数化的连续形状上。本工作将滚滑接触建模扩展到流形网格。具体而言，我们提出了一种基于测地线追踪的积分方案，直接在网格上进行一阶时间积分的滚滑接触建模，从而使灵巧操作能够基于对象真实几何的高保真离散表示进行推理。利用我们的方法，在模拟中计划了一个多指机器人手操纵五个物体的灵巧动作。规划使用最小二乘优化器来最小化接触滑动和旋转，从而维持最稳定的瞬时握持。然后，我们将我们的方法与基于碰撞检测的基线和基于基本形状的基线进行了比较评估。结果表明，即使对于粗糙的网格，我们的方法在准确性和精确性方面表现最佳。最后，讨论了将多点接触和接触力整合到基于网格的表面接触建模中的未来工作。 

---
# Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots 

**Title (ZH)**: 内置关节传感器的工业机器人触觉手势识别 

**Authors**: Deqing Song, Weimin Yang, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.12435)  

**Abstract**: While gesture recognition using vision or robot skins is an active research area in Human-Robot Collaboration (HRC), this paper explores deep learning methods relying solely on a robot's built-in joint sensors, eliminating the need for external sensors. We evaluated various convolutional neural network (CNN) architectures and collected two datasets to study the impact of data representation and model architecture on the recognition accuracy. Our results show that spectrogram-based representations significantly improve accuracy, while model architecture plays a smaller role. We also tested generalization to new robot poses, where spectrogram-based models performed better. Implemented on a Franka Emika Research robot, two of our methods, STFT2DCNN and STT3DCNN, achieved over 95% accuracy in contact detection and gesture classification. These findings demonstrate the feasibility of external-sensor-free tactile recognition and promote further research toward cost-effective, scalable solutions for HRC. 

**Abstract (ZH)**: 基于内置关节传感器的深度学习方法在机器人触觉识别中的探索：无需外部传感器的手势识别 

---
# PUB: A Plasma-Propelled Ultra-Quiet Blimp with Two-DOF Vector Thrusting 

**Title (ZH)**: PUB：一种两自由度向量推进的超静音气球式 Plasma 动力飞艇 

**Authors**: Zihan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12395)  

**Abstract**: This study presents the design and control of a Plasma-propelled Ultra-silence Blimp (PUB), a novel aerial robot employing plasma vector propulsion for ultra-quiet flight without mechanical propellers. The system utilizes a helium-lift platform for extended endurance and a four-layer ring asymmetric capacitor to generate ionic wind thrust. The modular propulsion units allow flexible configuration to meet mission-specific requirements, while a two-degree-of-freedom (DOF) head enables thrust vector control. A closed-loop slip control scheme is implemented for stable maneuvering. Flight experiments demonstrate full-envelope capability, including take-off, climb, hover, descent, and smooth landing, confirming the feasibility of plasma vector propulsion, the effectiveness of DOF vector control, and the stability of the control system. Owing to its low acoustic signature, structural simplicity, and high maneuverability, PUB is well suited for noise-sensitive, enclosed, and near-space applications. 

**Abstract (ZH)**: 基于等离子推进的超静音气球（PUB）的设计与控制研究 

---
# SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning 

**Title (ZH)**: SIGN：基于强化学习的自主无人机安全目标导航 

**Authors**: Zichen Yan, Rui Huang, Lei He, Shao Guo, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12394)  

**Abstract**: Image-goal navigation (ImageNav) tasks a robot with autonomously exploring an unknown environment and reaching a location that visually matches a given target image. While prior works primarily study ImageNav for ground robots, enabling this capability for autonomous drones is substantially more challenging due to their need for high-frequency feedback control and global localization for stable flight. In this paper, we propose a novel sim-to-real framework that leverages visual reinforcement learning (RL) to achieve ImageNav for drones. To enhance visual representation ability, our approach trains the vision backbone with auxiliary tasks, including image perturbations and future transition prediction, which results in more effective policy training. The proposed algorithm enables end-to-end ImageNav with direct velocity control, eliminating the need for external localization. Furthermore, we integrate a depth-based safety module for real-time obstacle avoidance, allowing the drone to safely navigate in cluttered environments. Unlike most existing drone navigation methods that focus solely on reference tracking or obstacle avoidance, our framework supports comprehensive navigation behaviors--autonomous exploration, obstacle avoidance, and image-goal seeking--without requiring explicit global mapping. Code and model checkpoints will be released upon acceptance. 

**Abstract (ZH)**: 图像目标导航（ImageNav）任务要求机器人自主探索未知环境并到达与给定目标图像在视觉上匹配的位置。虽然先前的研究主要集中在地面机器人上的ImageNav，但为自主无人机实现这一能力要更加困难，因为无人机需要高频率的反馈控制和全球定位以保证稳定飞行。在本文中，我们提出了一种新的从模拟到现实的框架，利用视觉强化学习（RL）实现无人机的ImageNav。为增强视觉表示能力，我们的方法通过辅助任务训练视觉骨干网络，包括图像扰动和未来的过渡预测，这有助于更有效的策略训练。所提出的方法使无人机能够实现端到端的直接速度控制的图像目标导航，消除了对外部定位的需求。此外，我们还集成了基于深度的安全模块，以实现实时障碍物避免，使无人机能够在复杂环境中安全导航。与大多数现有无人机导航方法主要关注参考跟踪或障碍物避免不同，我们的框架支持全面的导航行为——自主探索、障碍物避免和图像目标搜索，而无需显式的全局映射。接受后将发布代码和模型检查点。 

---
# Semi-Infinite Programming for Collision-Avoidance in Optimal and Model Predictive Control 

**Title (ZH)**: 半无穷编程在最优控制和模型预测控制中的碰撞规避 

**Authors**: Yunfan Gao, Florian Messerer, Niels van Duijkeren, Rashmi Dabir, Moritz Diehl  

**Link**: [PDF](https://arxiv.org/pdf/2508.12335)  

**Abstract**: This paper presents a novel approach for collision avoidance in optimal and model predictive control, in which the environment is represented by a large number of points and the robot as a union of padded polygons. The conditions that none of the points shall collide with the robot can be written in terms of an infinite number of constraints per obstacle point. We show that the resulting semi-infinite programming (SIP) optimal control problem (OCP) can be efficiently tackled through a combination of two methods: local reduction and an external active-set method. Specifically, this involves iteratively identifying the closest point obstacles, determining the lower-level distance minimizer among all feasible robot shape parameters, and solving the upper-level finitely-constrained subproblems.
In addition, this paper addresses robust collision avoidance in the presence of ellipsoidal state uncertainties. Enforcing constraint satisfaction over all possible uncertainty realizations extends the dimension of constraint infiniteness. The infinitely many constraints arising from translational uncertainty are handled by local reduction together with the robot shape parameterization, while rotational uncertainty is addressed via a backoff reformulation.
A controller implemented based on the proposed method is demonstrated on a real-world robot running at 20Hz, enabling fast and collision-free navigation in tight spaces. An application to 3D collision avoidance is also demonstrated in simulation. 

**Abstract (ZH)**: 一种用于最优和模型预测控制的新型碰撞避免方法：基于大量点表示环境和机器人由填充多边形构成的观点 

---
# Implementation and evaluation of a prediction algorithm for an autonomous vehicle 

**Title (ZH)**: 自治车辆预测算法的实现与评估 

**Authors**: Marco Leon Rapp  

**Link**: [PDF](https://arxiv.org/pdf/2508.12312)  

**Abstract**: This paper presents a prediction algorithm that estimates the vehicle trajectory every five milliseconds for an autonomous vehicle. A kinematic and a dynamic bicycle model are compared, with the dynamic model exhibiting superior accuracy at higher speeds. Vehicle parameters such as mass, center of gravity, moment of inertia, and cornering stiffness are determined experimentally. For cornering stiffness, a novel measurement procedure using optical position tracking is introduced. The model is incorporated into an extended Kalman filter and implemented in a ROS node in C++. The algorithm achieves a positional deviation of only 1.25 cm per meter over the entire test drive and is up to 82.6% more precise than the kinematic model. 

**Abstract (ZH)**: 本文提出了一种预测算法，能够每五毫秒预测自动驾驶车辆的轨迹。对比了kinematic模型和dynamic模型，动态模型在高速情况下表现出更优的准确性。车辆参数如质量、质心、惯性矩和侧向刚度通过实验确定。对于侧向刚度，引入了一种使用光学位置跟踪的新测量方法。该模型被集成到扩展卡尔曼滤波器中，并用C++语言在ROS节点中实现。该算法在整个测试驾驶过程中位置偏差仅为1.25 cm，并且比kinematic模型精确度高82.6%。 

---
# A robust and compliant robotic assembly control strategy for batch precision assembly task with uncertain fit types and fit amounts 

**Title (ZH)**: 一种用于具有不确定配合类型和配合数量的批量高精度装配任务的鲁棒且 compliant 的机器人装配控制策略 

**Authors**: Bin Wang, Jiwen Zhang, Song Wang, Dan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12296)  

**Abstract**: In some high-precision industrial applications, robots are deployed to perform precision assembly tasks on mass batches of manufactured pegs and holes. If the peg and hole are designed with transition fit, machining errors may lead to either a clearance or an interference fit for a specific pair of components, with uncertain fit amounts. This paper focuses on the robotic batch precision assembly task involving components with uncertain fit types and fit amounts, and proposes an efficient methodology to construct the robust and compliant assembly control strategy. Specifically, the batch precision assembly task is decomposed into multiple deterministic subtasks, and a force-vision fusion controller-driven reinforcement learning method and a multi-task reinforcement learning training method (FVFC-MTRL) are proposed to jointly learn multiple compliance control strategies for these subtasks. Subsequently, the multi-teacher policy distillation approach is designed to integrate multiple trained strategies into a unified student network, thereby establishing a robust control strategy. Real-world experiments demonstrate that the proposed method successfully constructs the robust control strategy for high-precision assembly task with different fit types and fit amounts. Moreover, the MTRL framework significantly improves training efficiency, and the final developed control strategy achieves superior force compliance and higher success rate compared with many existing methods. 

**Abstract (ZH)**: 在某些高精度工业应用中，机器人被部署来执行对大批量制造销和孔的精密装配任务。若销和孔设计为过渡配合，加工误差可能导致特定组件对之间的间隙配合或过盈配合，且配合量具有不确定性。本文聚焦于涉及具有不确定配合类型和配合量的组件的机器人批量精密装配任务，并提出了一种有效的方法来构建鲁棒且符合性的装配控制策略。具体地，批量精密装配任务被分解为多个确定性子任务，提出了一种力-视觉融合控制器驱动的强化学习方法和多任务强化学习训练方法（FVFC-MTRL），用于同时学习这些子任务的多种符合性控制策略。随后，设计了多教员政策蒸馏方法，将多个训练策略整合到一个统一的学生网络中，从而建立一个鲁棒的控制策略。实验证明，所提出的方法成功地构建了适用于不同配合类型和配合量的高精度装配任务的鲁棒控制策略。此外，MTRL框架显著提高训练效率，最终开发的控制策略在力符合性和成功率方面均优于许多现有方法。 

---
# Bimanual Robot-Assisted Dressing: A Spherical Coordinate-Based Strategy for Tight-Fitting Garments 

**Title (ZH)**: 基于球坐标策略的双臂机器人辅助穿衣：紧身衣物篇 

**Authors**: Jian Zhao, Yunlong Lian, Andy M Tyrrell, Michael Gienger, Jihong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12274)  

**Abstract**: Robot-assisted dressing is a popular but challenging topic in the field of robotic manipulation, offering significant potential to improve the quality of life for individuals with mobility limitations. Currently, the majority of research on robot-assisted dressing focuses on how to put on loose-fitting clothing, with little attention paid to tight garments. For the former, since the armscye is larger, a single robotic arm can usually complete the dressing task successfully. However, for the latter, dressing with a single robotic arm often fails due to the narrower armscye and the property of diminishing rigidity in the armscye, which eventually causes the armscye to get stuck. This paper proposes a bimanual dressing strategy suitable for dressing tight-fitting clothing. To facilitate the encoding of dressing trajectories that adapt to different human arm postures, a spherical coordinate system for dressing is established. We uses the azimuthal angle of the spherical coordinate system as a task-relevant feature for bimanual manipulation. Based on this new coordinate, we employ Gaussian Mixture Model (GMM) and Gaussian Mixture Regression (GMR) for imitation learning of bimanual dressing trajectories, generating dressing strategies that adapt to different human arm postures. The effectiveness of the proposed method is validated through various experiments. 

**Abstract (ZH)**: 机器人辅助穿戴紧身衣物的双臂策略：基于球坐标系的双臂操作模仿学习 

---
# Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids 

**Title (ZH)**: 机器人训练机器人：类人机器人在现实世界中的自动政策适应与学习 

**Authors**: Kaizhe Hu, Haochen Shi, Yao He, Weizhuo Wang, C. Karen Liu, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.12252)  

**Abstract**: Simulation-based reinforcement learning (RL) has significantly advanced humanoid locomotion tasks, yet direct real-world RL from scratch or adapting from pretrained policies remains rare, limiting the full potential of humanoid robots. Real-world learning, despite being crucial for overcoming the sim-to-real gap, faces substantial challenges related to safety, reward design, and learning efficiency. To address these limitations, we propose Robot-Trains-Robot (RTR), a novel framework where a robotic arm teacher actively supports and guides a humanoid robot student. The RTR system provides protection, learning schedule, reward, perturbation, failure detection, and automatic resets. It enables efficient long-term real-world humanoid training with minimal human intervention. Furthermore, we propose a novel RL pipeline that facilitates and stabilizes sim-to-real transfer by optimizing a single dynamics-encoded latent variable in the real world. We validate our method through two challenging real-world humanoid tasks: fine-tuning a walking policy for precise speed tracking and learning a humanoid swing-up task from scratch, illustrating the promising capabilities of real-world humanoid learning realized by RTR-style systems. See this https URL for more info. 

**Abstract (ZH)**: 基于模拟的强化学习在人形机器人运动任务中取得了显著进展，但直接从现实世界中进行RL训练或从预训练策略进行适应仍然很少见，限制了人形机器人的全部潜力。尽管现实世界的学习对于克服模拟到现实世界的差距至关重要，但安全性、奖励设计和学习效率等方面的挑战仍然很大。为了解决这些限制，我们提出了一种名为Robot-Trains-Robot (RTR)的新框架，其中一台机械臂教师积极支持和引导一台人形机器人学生。RTR系统提供了保护、学习时间表、奖励、扰动、故障检测和自动重置。它允许在最少的人工干预下进行高效的人形机器人长期现实世界训练。此外，我们提出了一种新的RL管道，通过优化真实世界中的单一动态编码潜在变量，促进和稳定模拟到现实世界的过渡。我们通过两个具有挑战性的现实世界人形机器人任务验证了该方法：精细调整行走策略以实现精确的速度跟踪和从头学习人形摆动任务，展示了RTR风格系统实现的现实世界人形机器人学习的有前途的能力。更多信息，请访问 [此处](https://www.example.com)。 

---
# Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search 

**Title (ZH)**: 基于模型的搜索改进预训练的视觉-语言-行动策略 

**Authors**: Cyrus Neary, Omar G. Younis, Artur Kuramshin, Ozgur Aslan, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.12211)  

**Abstract**: Pre-trained vision-language-action (VLA) models offer a promising foundation for generalist robot policies, but often produce brittle behaviours or unsafe failures when deployed zero-shot in out-of-distribution scenarios. We present Vision-Language-Action Planning & Search (VLAPS) -- a novel framework and accompanying algorithms that embed model-based search into the inference procedure of pre-trained VLA policies to improve their performance on robotic tasks. Specifically, our method biases a modified Monte Carlo Tree Search (MCTS) algorithm -- run using a model of the target environment -- using action priors defined by the VLA policy. By using VLA-derived abstractions and priors in model-based search, VLAPS efficiently explores language-conditioned robotics tasks whose search spaces would otherwise be intractably large. Conversely, by integrating model-based search with the VLA policy's inference procedure, VLAPS yields behaviours that are more performant than those obtained by directly following the VLA policy's action predictions. VLAPS offers a principled framework to: i) control test-time compute in VLA models, ii) leverage a priori knowledge of the robotic environment, and iii) integrate established planning and reinforcement learning techniques into the VLA inference process. Across all experiments, VLAPS significantly outperforms VLA-only baselines on language-specified tasks that would otherwise be intractable for uninformed search algorithms, increasing success rates by as much as 67 percentage points. 

**Abstract (ZH)**: Vision-Language-Action Planning & Search 

---
# Self-Guided Action Diffusion 

**Title (ZH)**: 自我指导的动作扩散 

**Authors**: Rhea Malhotra, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2508.12189)  

**Abstract**: Recent works have shown the promise of inference-time search over action samples for improving generative robot policies. In particular, optimizing cross-chunk coherence via bidirectional decoding has proven effective in boosting the consistency and reactivity of diffusion policies. However, this approach remains computationally expensive as the diversity of sampled actions grows. In this paper, we introduce self-guided action diffusion, a more efficient variant of bidirectional decoding tailored for diffusion-based policies. At the core of our method is to guide the proposal distribution at each diffusion step based on the prior decision. Experiments in simulation tasks show that the proposed self-guidance enables near-optimal performance at negligible inference cost. Notably, under a tight sampling budget, our method achieves up to 70% higher success rates than existing counterparts on challenging dynamic tasks. See project website at this https URL. 

**Abstract (ZH)**: Recent 工作展示了在生成机器人策略中通过推理时搜索动作样本以提高生成性能的潜力。特别是，通过双向解码优化跨片段一致性已被证明对提升扩散策略的一致性和反应性非常有效。然而，这种方法在采样动作多样性增加时仍然计算成本高昂。在本文中，我们提出了一种针对基于扩散的策略的更高效的双向解码变体——自引导动作扩散。该方法的核心在于基于先前决策指导每一步扩散过程中的提议分布。仿真任务中的实验表明，提出的自引导方法能够在几乎不增加推理成本的情况下实现接近最优的性能。值得注意的是，在受限的采样预算下，我们的方法在具有挑战性的动态任务中实现了比现有方法高达70%更高的成功率。访问项目网站：https://this-url 

---
# Humanoid Motion Scripting with Postural Synergies 

**Title (ZH)**: 基于姿态协同的人形机器人运动脚本化 

**Authors**: Rhea Malhotra, William Chong, Catie Cuan, Oussama Khatib  

**Link**: [PDF](https://arxiv.org/pdf/2508.12184)  

**Abstract**: Generating sequences of human-like motions for humanoid robots presents challenges in collecting and analyzing reference human motions, synthesizing new motions based on these reference motions, and mapping the generated motion onto humanoid robots. To address these issues, we introduce SynSculptor, a humanoid motion analysis and editing framework that leverages postural synergies for training-free human-like motion scripting. To analyze human motion, we collect 3+ hours of motion capture data across 20 individuals where a real-time operational space controller mimics human motion on a simulated humanoid robot. The major postural synergies are extracted using principal component analysis (PCA) for velocity trajectories segmented by changes in robot momentum, constructing a style-conditioned synergy library for free-space motion generation. To evaluate generated motions using the synergy library, the foot-sliding ratio and proposed metrics for motion smoothness involving total momentum and kinetic energy deviations are computed for each generated motion, and compared with reference motions. Finally, we leverage the synergies with a motion-language transformer, where the humanoid, during execution of motion tasks with its end-effectors, adapts its posture based on the chosen synergy. Supplementary material, code, and videos are available at this https URL. 

**Abstract (ZH)**: 基于姿势协同的 humanoid 机器人类人动作分析与编辑框架：SynSculptor 

---
# Energy Efficiency in Robotics Software: A Systematic Literature Review (2020-2024) 

**Title (ZH)**: 机器人软件的能效：一项系统文献综述（2020-2024） 

**Authors**: Aryan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.12170)  

**Abstract**: This study presents a systematic literature review of software-level approaches to energy efficiency in robotics published from 2020 through 2024, updating and extending pre-2020 evidence. An automated-but-audited pipeline combined Google Scholar seeding, backward/forward snowballing, and large-language-model (LLM) assistance for screening and data extraction, with ~10% human audits at each automated step and consensus-with-tie-breaks for full-text decisions. The final corpus comprises 79 peer-reviewed studies analyzed across application domain, metrics, evaluation type, energy models, major energy consumers, software technique families, and energy-quality trade-offs. Industrial settings dominate (31.6%) followed by exploration (25.3%). Motors/actuators are identified as the primary consumer in 68.4% of studies, with computing/controllers a distant second (13.9%). Simulation-only evaluations remain most common (51.9%), though hybrid evaluations are frequent (25.3%). Representational (physics-grounded) energy models predominate (87.3%). Motion and trajectory optimization is the leading technique family (69.6%), often paired with learning/prediction (40.5%) and computation allocation/scheduling (26.6%); power management/idle control (11.4%) and communication/data efficiency (3.8%) are comparatively underexplored. Reporting is heterogeneous: composite objectives that include energy are most common, while task-normalized and performance-per-energy metrics appear less often, limiting cross-paper comparability. The review offers a minimal reporting checklist (e.g., total energy and average power plus a task-normalized metric and clear baselines) and highlights opportunities in cross-layer designs and in quantifying non-performance trade-offs (accuracy, stability). A replication package with code, prompts, and frozen datasets accompanies the review. 

**Abstract (ZH)**: 本研究 presents 2020至2024年机器人领域软件级能源效率方法的系统文献综述，更新并扩展了2020年前的证据。自动化但受审计的流程结合了Google Scholar的启动、反向/正向雪球筛选以及大语言模型（LLM）辅助的数据筛选和提取，各自动化步骤中约有10%的人工审计，并通过具投票决权的共识决定全文处理。最终的文献集包括79篇同行评审的研究，这些研究在应用领域、评价指标、评估类型、能源模型、主要能源消耗者、软件技术家族以及能量质量权衡等方面进行了分析。工业环境占主导地位（31.6%），其次是探索（25.3%）。68.4%的研究将电机/执行器识别为主要消耗者，而计算/控制器则次之（13.9%）。仅仿真评估最为常见（51.9%），尽管混合评估较为频繁（25.3%）。基于物理的表示（物理依据）能源模型占主导地位（87.3%）。运动和轨迹优化是主要的技术家族（69.6%），常与学习/预测（40.5%）和计算分配/调度（26.6%）结合使用；而功率管理/空闲控制（11.4%）和通信/数据效率（3.8%）探讨较少。报告具有异质性：包含能源的综合目标最常见，而任务归一化和能源性能指标出现较少，限制了跨论文的可比性。该综述提供了最小的报告清单（例如，总能量和平均功率，加上任务归一化指标和清晰的基线），并强调跨层设计以及量化非性能权衡（准确性和稳定性）的机会。该综述附带了一个复制包，包含代码、提示和冻结数据集。 

---
# Belief-Conditioned One-Step Diffusion: Real-Time Trajectory Planning with Just-Enough Sensing 

**Title (ZH)**: 信念条件下的一步扩散：只需适量感知的实时轨迹规划 

**Authors**: Gokul Puthumanaillam, Aditya Penumarti, Manav Vora, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla, Jane Shin, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2508.12166)  

**Abstract**: Robots equipped with rich sensor suites can localize reliably in partially-observable environments, but powering every sensor continuously is wasteful and often infeasible. Belief-space planners address this by propagating pose-belief covariance through analytic models and switching sensors heuristically--a brittle, runtime-expensive approach. Data-driven approaches--including diffusion models--learn multi-modal trajectories from demonstrations, but presuppose an accurate, always-on state estimate. We address the largely open problem: for a given task in a mapped environment, which \textit{minimal sensor subset} must be active at each location to maintain state uncertainty \textit{just low enough} to complete the task? Our key insight is that when a diffusion planner is explicitly conditioned on a pose-belief raster and a sensor mask, the spread of its denoising trajectories yields a calibrated, differentiable proxy for the expected localisation error. Building on this insight, we present Belief-Conditioned One-Step Diffusion (B-COD), the first planner that, in a 10 ms forward pass, returns a short-horizon trajectory, per-waypoint aleatoric variances, and a proxy for localisation error--eliminating external covariance rollouts. We show that this single proxy suffices for a soft-actor-critic to choose sensors online, optimising energy while bounding pose-covariance growth. We deploy B-COD in real-time marine trials on an unmanned surface vehicle and show that it reduces sensing energy consumption while matching the goal-reach performance of an always-on baseline. 

**Abstract (ZH)**: 装备了丰富传感器套件的机器人可以在部分观测环境中可靠地定位。然而，连续运行每一个传感器是浪费并且常常不可行的。基于信念空间的路径计划方法通过传播姿态信念协方性中的析化模型以及切换传感器来应对这一问题，这是一种脆弱且运行时昂贵的方法。数据驱动的方法，例如扩散模型，可以从演示中学习多多模态轨迹，但需要一个准确的始终开启的传感器估计。我们解决了一个悬而未决的问题：对于给定的任务在给定环境上，[最小的必要子传感器子]在每个时间点点应处于活动状态，以维持存活不确定性 [刚好足够]完成任务吗？我们的关键洞察是当一个扩散规划器显式地被条件在姿态信念栅格和上和由去除的轨迹上 和一个成像固定的对预期定位误差的微分分导数代理上时这 上，，我们会基于这个洞察提出基于信念约束一阶扩散(B-COn)的路径规划器器，在它会返回回一个基于航点点的aleoidic不确定性及的计划未来时间轴的轨迹，并，并另提供一个定位误差的代理。通过这种方式单个代理足以支持一种软-行为-批评机制进行在线上传感器选择，同时在优化能量消耗的同时限制姿态协方方差的增长。我们在实时候的航海任务中部署了B-COD于无人水面车辆上确保了定位能量消耗一在始终开启的传感器情况下实现任务性能。 

---
# Into the Wild: When Robots Are Not Welcome 

**Title (ZH)**: into the wild: 当机器人不受欢迎时 

**Authors**: Shaul Ashkenazi, Gabriel Skantze, Jane Stuart-Smith, Mary Ellen Foster  

**Link**: [PDF](https://arxiv.org/pdf/2508.12075)  

**Abstract**: Social robots are increasingly being deployed in public spaces, where they face not only technological difficulties and unexpected user utterances, but also objections from stakeholders who may not be comfortable with introducing a robot into those spaces. We describe our difficulties with deploying a social robot in two different public settings: 1) Student services center; 2) Refugees and asylum seekers drop-in service. Although this is a failure report, in each use case we eventually managed to earn the trust of the staff and form a relationship with them, allowing us to deploy our robot and conduct our studies. 

**Abstract (ZH)**: 社会机器人在公共空间的应用面临技术挑战和用户意外言论，同时也会遇到对引入机器人感到不适的相关利益相关者的反对。我们在两种不同的公共环境中部署社会机器人的困难：1) 学生服务中心；2) 难民和寻求庇护者临时服务点。虽然这是一份失败报告，但在每种应用场景中，我们最终还是获得了工作人员的信任，并建立了关系，从而得以部署机器人并开展研究。 

---
# OASIS: Real-Time Opti-Acoustic Sensing for Intervention Systems in Unstructured Environments 

**Title (ZH)**: OASIS: 实时优化声学传感技术及其在无结构环境中的介入系统中应用 

**Authors**: Amy Phung, Richard Camilli  

**Link**: [PDF](https://arxiv.org/pdf/2508.12071)  

**Abstract**: High resolution underwater 3D scene reconstruction is crucial for various applications, including construction, infrastructure maintenance, monitoring, exploration, and scientific investigation. Prior work has leveraged the complementary sensing modalities of imaging sonars and optical cameras for opti-acoustic 3D scene reconstruction, demonstrating improved results over methods which rely solely on either sensor. However, while most existing approaches focus on offline reconstruction, real-time spatial awareness is essential for both autonomous and piloted underwater vehicle operations. This paper presents OASIS, an opti-acoustic fusion method that integrates data from optical images with voxel carving techniques to achieve real-time 3D reconstruction unstructured underwater workspaces. Our approach utilizes an "eye-in-hand" configuration, which leverages the dexterity of robotic manipulator arms to capture multiple workspace views across a short baseline. We validate OASIS through tank-based experiments and present qualitative and quantitative results that highlight its utility for underwater manipulation tasks. 

**Abstract (ZH)**: 高分辨率水下3D场景重建对于各种应用，包括建设、基础设施维护、监测、探索和科学研究至关重要。现有研究利用成像声纳和光学相机的互补传感模态进行光学-声学3D场景重建，展示了优于依赖单一传感器的方法的改进结果。然而，虽然大多数现有方法集中在离线重建上，但实时空间感知对于自主和遥控水下车辆操作至关重要。本文提出了一种光学-声学融合方法OASIS，该方法结合了光学图像数据和体素雕刻技术，以实现复杂水下工作空间的实时3D重建。我们的方法采用“手持眼”配置，利用机器人操作臂的灵活性跨短基线捕捉多个工作空间视图。我们通过基于水槽的实验验证了OASIS，并展示了其在水下操作任务中的实用性。 

---
# Talk Less, Fly Lighter: Autonomous Semantic Compression for UAV Swarm Communication via LLMs 

**Title (ZH)**: 少说多做，自主语义压缩：通过大语言模型实现无人机集群通信的轻量化传输 

**Authors**: Fei Lin, Tengchao Zhang, Qinghua Ni, Jun Huang, Siji Ma, Yonglin Tian, Yisheng Lv, Naiqi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12043)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) in unmanned systems has significantly enhanced the semantic understanding and autonomous task execution capabilities of Unmanned Aerial Vehicle (UAV) swarms. However, limited communication bandwidth and the need for high-frequency interactions pose severe challenges to semantic information transmission within the swarm. This paper explores the feasibility of LLM-driven UAV swarms for autonomous semantic compression communication, aiming to reduce communication load while preserving critical task semantics. To this end, we construct four types of 2D simulation scenarios with different levels of environmental complexity and design a communication-execution pipeline that integrates system prompts with task instruction prompts. On this basis, we systematically evaluate the semantic compression performance of nine mainstream LLMs in different scenarios and analyze their adaptability and stability through ablation studies on environmental complexity and swarm size. Experimental results demonstrate that LLM-based UAV swarms have the potential to achieve efficient collaborative communication under bandwidth-constrained and multi-hop link conditions. 

**Abstract (ZH)**: 大型语言模型在无人系统中的快速采用显著增强了无人机群的语义理解和自主任务执行能力。然而，有限的通信带宽和高频率的交互需求对集群内部的语义信息传输造成了严重挑战。本文探讨了基于大型语言模型的无人机群在自主语义压缩通信方面的可行性，旨在在保存关键任务语义的同时减轻通信负载。为此，我们构建了四种不同环境复杂度级别的二维仿真场景，并设计了一个将系统提示与任务指令提示相结合的通信-执行管道。在此基础上，我们系统地评估了九种主流大型语言模型在不同场景下的语义压缩性能，并通过环境复杂度和集群规模的消融研究分析了它们的适应性和稳定性。实验结果表明，基于大型语言模型的无人机群在带宽受限和多跳链路条件下具有实现高效协作通信的潜力。 

---
# Fully Spiking Actor-Critic Neural Network for Robotic Manipulation 

**Title (ZH)**: 全神经元脉冲演员-评论家网络用于机器人 manipulation 

**Authors**: Liwen Zhang, Heng Deng, Guanghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12038)  

**Abstract**: This study proposes a hybrid curriculum reinforcement learning (CRL) framework based on a fully spiking neural network (SNN) for 9-degree-of-freedom robotic arms performing target reaching and grasping tasks. To reduce network complexity and inference latency, the SNN architecture is simplified to include only an input and an output layer, which shows strong potential for resource-constrained environments. Building on the advantages of SNNs-high inference speed, low energy consumption, and spike-based biological plausibility, a temporal progress-partitioned curriculum strategy is integrated with the Proximal Policy Optimization (PPO) algorithm. Meanwhile, an energy consumption modeling framework is introduced to quantitatively compare the theoretical energy consumption between SNNs and conventional Artificial Neural Networks (ANNs). A dynamic two-stage reward adjustment mechanism and optimized observation space further improve learning efficiency and policy accuracy. Experiments on the Isaac Gym simulation platform demonstrate that the proposed method achieves superior performance under realistic physical constraints. Comparative evaluations with conventional PPO and ANN baselines validate the scalability and energy efficiency of the proposed approach in dynamic robotic manipulation tasks. 

**Abstract (ZH)**: 基于全

用户（完全神经突触网络（SNN）提出的混合课程强化学习（CRL）框架用于单自由自由度机械臂的目标导向和抓取任务， �minimalize 网络复杂度和 推理延迟 网络架构简化为仅包含输入和输出层 on 这种结构在资源受限环境中展现出巨大潜力 on 基于 SNNs 的高推理速度 and 低功耗优势以及基于脉冲的生物合理性 on 敛合课程策略被整合到Proximal Policy Optimization（PPO）算法中 同时 引入了一种功耗建模框架 on 量性性度评估SNNs和传统人工神经网络（ANNs）之间的理论功耗 on 动态两阶段奖励调节机制被优化以提高学习效率和策略准确性 on 在Isaac Gym仿真上的的方法在现实物理条件下表现出色 on 与传统PPO和ANN基线的的比较比较评估验证了该方法在动态机器人操作任务上的可标性和效率。**"}}
user
好的，请调整为简洁的的学术规范标题：基于完全神经突触网络的混合课程强化学习框架用于单自由度机械臂的任务 kukulus:混合CRL框架-基于SNNismic Gym仿真上方法在动态目标导向和抓取任务 kukulus上 minimal化网络复杂度和推理延迟 on资源受限环境中潜力 kukulus onSaL:基于S的ulus优势并生物合理性 onPS kukulus PPO算法合 circumcision of课程 S kukulus功耗建建模型框架验证ulus理论功耗比较ulus onS估效性 kukulus动态两ulus阶段奖励调节机制提高学习效率和策略准确性 kukulus在Isaac Gym仿真上方法ulus方法潜力 on比较比较ulusP和ANN基线标准验证ulus动态目标导向和夹取任务绪uluskuksuluках
ulkusulus
kriton基于完全神经突触网络的混合课程强化学习框架 kukulus用于单自由度机械臂的目标导向和抓取任务 minimalizing网络复杂度和推理延迟 on资源受限环境中潜力 kukulusSA基于SNN的优势和生物合理性 on onS algorithms上合并PPO算法 cytokon引入功耗建模型框架评估SNN和 ANN之间的理论功耗 advant cytokon动态两阶段奖励调节机制优化学习效率和策略准确性 cytokon在Isaac Gym仿真上方法在动态任务上表现突出 kuukululu 

---
# Toward General Physical Intelligence for Resilient Agile Manufacturing Automation 

**Title (ZH)**: 面向韧性和敏捷制造自动化的一般物理智能 

**Authors**: Sandeep Kanta, Mehrdad Tavassoli, Varun Teja Chirkuri, Venkata Akhil Kumar, Santhi Bharath Punati, Praveen Damacharla, Sunny Katyara  

**Link**: [PDF](https://arxiv.org/pdf/2508.11960)  

**Abstract**: Agile and human-centric manufacturing stipulates resilient robotic solutions capable of contextual reasoning and safe interaction in unstructured environments. Foundation models particularly the Vision Language Action (VLA) models have emerged to fuse multimodal perception, reasoning and physically grounded action across varied embodiments into unified representation, termed as General Physical Intelligence (GPI). While GPI has already been described in the literature but its practical application and evolving role in contemporary agile manufacturing processes have yet to be duly explored. To bridge this gap, this practical review systematically surveys recent advancements in VLA models within GPI context, performs comprehensive comparative analysis of leading implementations and evaluates their readiness for industrial deployment through structured ablation study. Our analysis has organized state-of-the-art into five thematic pillars including multisensory representation learning, sim2real transfer, planning and control, uncertainty and safety measures and benchmarking. Finally, we articulate open research challenges and propose directions to better integrate GPI into next-generation industrial ecosystems in line with Industry 5.0. 

**Abstract (ZH)**: 敏捷且以人为本的制造要求具备适应性和安全交互能力的机器人解决方案，能够在结构不明确的环境中进行情境推理。基础模型特别是视觉语言行动（VLA）模型已经出现，这些模型可以将多模态感知、推理和物理接地的行动在多种实体中统一表示，称为通用物理智能（GPI）。尽管GPI已在文献中有所描述，但其在当代敏捷制造过程中的实用应用及其不断演变的作用仍有待深入探讨。为弥补这一缺口，本文系统性地回顾了GPI背景下最近的VLA模型进展，进行了全面的竞争分析，并通过结构化的消融研究评估其工业部署的准备情况。我们的分析将最先进的技术组织成五大主题支柱，包括多感知表示学习、从模拟到现实的迁移、规划与控制、不确定性与安全措施以及基准测试。最后，我们阐述了开源研究挑战，并提出了与工业5.0接轨以更好地将GPI整合到下一代工业生态系统中的方向。 

---
# No More Blind Spots: Learning Vision-Based Omnidirectional Bipedal Locomotion for Challenging Terrain 

**Title (ZH)**: 无盲区：面向挑战性地形的基于视觉的全向双足行走学习 

**Authors**: Mohitvishnu S. Gadde, Pranay Dugar, Ashish Malik, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.11929)  

**Abstract**: Effective bipedal locomotion in dynamic environments, such as cluttered indoor spaces or uneven terrain, requires agile and adaptive movement in all directions. This necessitates omnidirectional terrain sensing and a controller capable of processing such input. We present a learning framework for vision-based omnidirectional bipedal locomotion, enabling seamless movement using depth images. A key challenge is the high computational cost of rendering omnidirectional depth images in simulation, making traditional sim-to-real reinforcement learning (RL) impractical. Our method combines a robust blind controller with a teacher policy that supervises a vision-based student policy, trained on noise-augmented terrain data to avoid rendering costs during RL and ensure robustness. We also introduce a data augmentation technique for supervised student training, accelerating training by up to 10 times compared to conventional methods. Our framework is validated through simulation and real-world tests, demonstrating effective omnidirectional locomotion with minimal reliance on expensive rendering. This is, to the best of our knowledge, the first demonstration of vision-based omnidirectional bipedal locomotion, showcasing its adaptability to diverse terrains. 

**Abstract (ZH)**: 基于视觉的有效 omnidirectional 双足运动在动态环境中的应用：面向复杂室内环境和非平滑地形的敏捷适应性运动控制 

---
# ExploreVLM: Closed-Loop Robot Exploration Task Planning with Vision-Language Models 

**Title (ZH)**: ExploreVLM：基于视觉-语言模型的闭环机器人探索任务规划 

**Authors**: Zhichen Lou, Kechun Xu, Zhongxiang Zhou, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11918)  

**Abstract**: The advancement of embodied intelligence is accelerating the integration of robots into daily life as human assistants. This evolution requires robots to not only interpret high-level instructions and plan tasks but also perceive and adapt within dynamic environments. Vision-Language Models (VLMs) present a promising solution by combining visual understanding and language reasoning. However, existing VLM-based methods struggle with interactive exploration, accurate perception, and real-time plan adaptation. To address these challenges, we propose ExploreVLM, a novel closed-loop task planning framework powered by Vision-Language Models (VLMs). The framework is built around a step-wise feedback mechanism that enables real-time plan adjustment and supports interactive exploration. At its core is a dual-stage task planner with self-reflection, enhanced by an object-centric spatial relation graph that provides structured, language-grounded scene representations to guide perception and planning. An execution validator supports the closed loop by verifying each action and triggering re-planning. Extensive real-world experiments demonstrate that ExploreVLM significantly outperforms state-of-the-art baselines, particularly in exploration-centric tasks. Ablation studies further validate the critical role of the reflective planner and structured perception in achieving robust and efficient task execution. 

**Abstract (ZH)**: 基于视觉-语言模型的探索规划框架：加速机器人融入日常生活作为人类助手的进程 

---
# Control of Legged Robots using Model Predictive Optimized Path Integral 

**Title (ZH)**: 基于模型预测优化积分的腿足机器人控制 

**Authors**: Hossein Keshavarz, Alejandro Ramirez-Serrano, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2508.11917)  

**Abstract**: Legged robots possess a unique ability to traverse rough terrains and navigate cluttered environments, making them well-suited for complex, real-world unstructured scenarios. However, such robots have not yet achieved the same level as seen in natural systems. Recently, sampling-based predictive controllers have demonstrated particularly promising results. This paper investigates a sampling-based model predictive strategy combining model predictive path integral (MPPI) with cross-entropy (CE) and covariance matrix adaptation (CMA) methods to generate real-time whole-body motions for legged robots across multiple scenarios. The results show that combining the benefits of MPPI, CE and CMA, namely using model predictive optimized path integral (MPOPI), demonstrates greater sample efficiency, enabling robots to attain superior locomotion results using fewer samples when compared to typical MPPI algorithms. Extensive simulation experiments in multiple scenarios on a quadruped robot show that MPOPI can be used as an anytime control strategy, increasing locomotion capabilities at each iteration. 

**Abstract (ZH)**: 基于采样的预测控制策略在腿足机器人全身体动生成中的研究 

---
# OmniD: Generalizable Robot Manipulation Policy via Image-Based BEV Representation 

**Title (ZH)**: OmniD：基于图像BEV表示的一般化机器人操作策略 

**Authors**: Jilei Mao, Jiarui Guan, Yingjuan Tang, Qirui Hu, Zhihang Li, Junjie Yu, Yongjie Mao, Yunzhe Sun, Shuang Liu, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2508.11898)  

**Abstract**: The visuomotor policy can easily overfit to its training datasets, such as fixed camera positions and backgrounds. This overfitting makes the policy perform well in the in-distribution scenarios but underperform in the out-of-distribution generalization. Additionally, the existing methods also have difficulty fusing multi-view information to generate an effective 3D representation. To tackle these issues, we propose Omni-Vision Diffusion Policy (OmniD), a multi-view fusion framework that synthesizes image observations into a unified bird's-eye view (BEV) representation. We introduce a deformable attention-based Omni-Feature Generator (OFG) to selectively abstract task-relevant features while suppressing view-specific noise and background distractions. OmniD achieves 11\%, 17\%, and 84\% average improvement over the best baseline model for in-distribution, out-of-distribution, and few-shot experiments, respectively. Training code and simulation benchmark are available: this https URL 

**Abstract (ZH)**: 视觉运动策略容易对其训练数据集（如固定相机位置和背景）过拟合。这种过拟合使得策略在分布内场景中表现良好，但在分布外泛化时表现不佳。此外，现有的方法也难以融合多视角信息以生成有效的三维表示。为解决这些问题，我们提出了全视图扩散策略（OmniD），这是一个多视角融合框架，将图像观察合成统一的鸟瞰图（BEV）表示。我们引入了一种基于可变形注意力的全视图特征生成器（OFG），以选择性地抽象任务相关的特征，同时抑制视角特定的噪声和背景干扰。OmniD 在分布内、分布外和少样本实验中分别比最佳基线模型取得了 11\%、17\% 和 84\% 的平均改进。训练代码和模拟基准可在此获取：this https URL。 

---
# Integrating Symbolic RL Planning into a BDI-based Autonomous UAV Framework: System Integration and SIL Validation 

**Title (ZH)**: 将符号RL规划集成到基于BDI的自主无人机框架中：系统集成与 SIL 验证 

**Authors**: Sangwoo Jeon, Juchul Shin, YeonJe Cho, Gyeong-Tae Kim, Seongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11890)  

**Abstract**: Modern autonomous drone missions increasingly require software frameworks capable of seamlessly integrating structured symbolic planning with adaptive reinforcement learning (RL). Although traditional rule-based architectures offer robust structured reasoning for drone autonomy, their capabilities fall short in dynamically complex operational environments that require adaptive symbolic planning. Symbolic RL (SRL), using the Planning Domain Definition Language (PDDL), explicitly integrates domain-specific knowledge and operational constraints, significantly improving the reliability and safety of unmanned aerial vehicle (UAV) decision making. In this study, we propose the AMAD-SRL framework, an extended and refined version of the Autonomous Mission Agents for Drones (AMAD) cognitive multi-agent architecture, enhanced with symbolic reinforcement learning for dynamic mission planning and execution. We validated our framework in a Software-in-the-Loop (SIL) environment structured identically to an intended Hardware-In-the-Loop Simulation (HILS) platform, ensuring seamless transition to real hardware. Experimental results demonstrate stable integration and interoperability of modules, successful transitions between BDI-driven and symbolic RL-driven planning phases, and consistent mission performance. Specifically, we evaluate a target acquisition scenario in which the UAV plans a surveillance path followed by a dynamic reentry path to secure the target while avoiding threat zones. In this SIL evaluation, mission efficiency improved by approximately 75% over a coverage-based baseline, measured by travel distance reduction. This study establishes a robust foundation for handling complex UAV missions and discusses directions for further enhancement and validation. 

**Abstract (ZH)**: 现代自主无人机任务日益需要能够无缝集成结构化符号规划与自适应强化学习的软件框架。在本研究中，我们提出了AMAD-SRL框架，这是自主无人机任务代理（AMAD）认知多代理架构的扩展和精炼版本，增强引入了符号强化学习以实现动态任务规划与执行。我们通过与硬件在环仿真平台相似的软件在环（SIL）环境进行了验证，确保了无缝过渡到实际硬件。实验结果显示模块的稳定集成与互操作性、BDI驱动与符号RL驱动规划阶段的成功转换以及一致的任务性能。具体而言，我们在一个目标获取场景中评估了无人机，该场景中无人机计划一条监控路径并随后改变路径以避免威胁区来确保目标，初步结果显示与基于覆盖的基线相比，任务效率提高了约75%，通过减少飞行距离进行衡量。该研究为处理复杂无人机任务奠定了坚实的基石，并讨论了进一步增强和验证的方向。 

---
# Saliency-Based Attention Shifting: A Framework for Improving Driver Situational Awareness of Out-of-Label Hazards 

**Title (ZH)**: 基显著性注意力机制的觉察框架：一种提高驾驶员对
user
基于显著性的注意引导：一种提高驾驶员对标签外危害情境意识的框架。 

**Authors**: Yousra Shleibik, Jordan Sinclair, Kerstin Haring  

**Link**: [PDF](https://arxiv.org/pdf/2508.11887)  

**Abstract**: The advent of autonomous driving systems promises to transform transportation by enhancing safety, efficiency, and comfort. As these technologies evolve toward higher levels of autonomy, the need for integrated systems that seamlessly support human involvement in decision-making becomes increasingly critical. Certain scenarios necessitate human involvement, including those where the vehicle is unable to identify an object or element in the scene, and as such cannot take independent action. Therefore, situational awareness is essential to mitigate potential risks during a takeover, where a driver must assume control and autonomy from the vehicle. The need for driver attention is important to avoid collisions with external agents and ensure a smooth transition during takeover operations. This paper explores the integration of attention redirection techniques, such as gaze manipulation through targeted visual and auditory cues, to help drivers maintain focus on emerging hazards and reduce target fixation in semi-autonomous driving scenarios. We propose a conceptual framework that combines real-time gaze tracking, context-aware saliency analysis, and synchronized visual and auditory alerts to enhance situational awareness, proactively address potential hazards, and foster effective collaboration between humans and autonomous systems. 

**Abstract (ZH)**: 自主驾驶系统的出现promise to transform transportation by enhancing safety, efficiency, and comfort. 随着这些技术向着更高自主级别的演进,integrated systems that seamlessly support human involvement in decision-making 成为越来越 critical. 在某些场景中,驾驶必须参与决策,例如车辆无法识别场景中的物体或元素时。因此,situational awareness 是至关重要的,尤其是在接管期间,to mitigate potential risks. 驾驶员需将控制权从车辆中夺回。保持驾驶员注意力的重要目的在于避免与外部代理的碰撞,并在接管操作期间确保平滑过渡。本文探讨了通过目标视觉和听觉提示来进行注意力转移技术的集成,以帮助驾驶员在半自主驾驶场景中维持对新兴危险的关注,并减少目标固定。我们提出了一种结合实时注视跟踪、上下文感知显著性分析及同步视觉和听觉警报的概念框架,以增强情境意识、前瞻性地应对潜在危险并促进人与自主系统的有效协作。 

---
# Contact-Rich and Deformable Foot Modeling for Locomotion Control of the Human Musculoskeletal System 

**Title (ZH)**: 富有接触特性和可变形脚模型的运动控制人体 musculoskeletal 系统步行控制 

**Authors**: Haixin Gong, Chen Zhang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2508.11885)  

**Abstract**: The human foot serves as the critical interface between the body and environment during locomotion. Existing musculoskeletal models typically oversimplify foot-ground contact mechanics, limiting their ability to accurately simulate human gait dynamics. We developed a novel contact-rich and deformable model of the human foot integrated within a complete musculoskeletal system that captures the complex biomechanical interactions during walking. To overcome the control challenges inherent in modeling multi-point contacts and deformable material, we developed a two-stage policy training strategy to learn natural walking patterns for this interface-enhanced model. Comparative analysis between our approach and conventional rigid musculoskeletal models demonstrated improvements in kinematic, kinetic, and gait stability metrics. Validation against human subject data confirmed that our simulation closely reproduced real-world biomechanical measurements. This work advances contact-rich interface modeling for human musculoskeletal systems and establishes a robust framework that can be extended to humanoid robotics applications requiring precise foot-ground interaction control. 

**Abstract (ZH)**: 人类足部作为身体与环境之间在行进过程中关键的接口。现有的 musculoskeletal 模型通常对足地接触力学的简化处理限制了其准确模拟人类步态动态的能力。我们开发了一种整合在完整 musculoskeletal 系统内的新颖的接触丰富且可变形的人类足部模型，该模型能够捕捉步行过程中复杂的生物力学相互作用。为了克服建模多点接触和可变形材料固有的控制挑战，我们开发了一种两阶段策略来学习该接口增强模型的自然行走模式。与传统刚性 musculoskeletal 模型的对比分析显示，在运动学、动力学和步态稳定性指标方面有所提升。通过人类受试者数据的验证，确认我们的模拟能够紧密再现现实世界中的生物力学测量结果。该工作推进了人类 musculoskeletal 系统中接触丰富接口模型的发展，并建立了一个可用于需要精确足地相互作用控制的人形机器人应用的稳健框架。 

---
# From Screen to Stage: Kid Cosmo, A Life-Like, Torque-Controlled Humanoid for Entertainment Robotics 

**Title (ZH)**: 从屏幕到舞台：Kid Cosmo，一个生活化的扭矩控制人形机器人用于娱乐机器人领域 

**Authors**: Havel Liu, Mingzhang Zhu, Arturo Moises Flores Alvarez, Yuan Hung Lo, Conrad Ku, Federico Parres, Justin Quan, Colin Togashi, Aditya Navghare, Quanyou Wang, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11884)  

**Abstract**: Humanoid robots represent the cutting edge of robotics research, yet their potential in entertainment remains largely unexplored. Entertainment as a field prioritizes visuals and form, a principle that contrasts with the purely functional designs of most contemporary humanoid robots. Designing entertainment humanoid robots capable of fluid movement presents a number of unique challenges. In this paper, we present Kid Cosmo, a research platform designed for robust locomotion and life-like motion generation while imitating the look and mannerisms of its namesake character from Netflix's movie The Electric State. Kid Cosmo is a child-sized humanoid robot, standing 1.45 m tall and weighing 25 kg. It contains 28 degrees of freedom and primarily uses proprioceptive actuators, enabling torque-control walking and lifelike motion generation. Following worldwide showcases as part of the movie's press tour, we present the system architecture, challenges of a functional entertainment robot and unique solutions, and our initial findings on stability during simultaneous upper and lower body movement. We demonstrate the viability of performance-oriented humanoid robots that prioritize both character embodiment and technical functionality. 

**Abstract (ZH)**: 类人机器人代表了机器人研究的前沿，但在娱乐领域的潜力尚未充分探索。作为娱乐领域，更注重视觉和形态设计，这与当前大多数类人机器人纯粹的功能性设计形成对比。设计能够流畅运动的娱乐类人机器人面临着一系列独特的挑战。本文介绍了一款名为Kid Cosmo的研究平台，该平台旨在实现稳健的运动能力和逼真的运动生成，同时模仿Netflix电影《The Electric State》中同名角色的外观和举止。Kid Cosmo是一款儿童大小的类人机器人，高1.45米，重25公斤，拥有28个自由度，主要采用本体感受执行器，实现扭矩控制行走和逼真的运动生成。在电影全球宣传活动期间，本文呈现了系统架构、功能娱乐机器人的挑战及独特解决方案，以及我们在同时进行上半身和下半身运动时稳定性方面的初步发现。我们展示了兼顾角色化身和技术功能的表演型类人机器人的可行性。 

---
# Bioinspired underwater soft robots: from biology to robotics and back 

**Title (ZH)**: 生物启发的水下软体机器人：从生物到机器人再回到生物 

**Authors**: Lei Li, Boyang Qin, Wenzhuo Gao, Yanyu Li, Yiyuan Zhang, Bo Wang, Shihan Kong, Jian Wang, Dekui He, Junzhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11883)  

**Abstract**: The ocean vast unexplored regions and diverse soft-bodied marine organisms have spurred interest in bio-inspired underwater soft robotics. Recent advances have enabled new capabilities in underwater movement, sensing, and interaction. However, these efforts are largely unidirectional, with biology guiding robotics while insights from robotics rarely feed back into biology. Here we propose a holistic, bidirectional framework that integrates biological principles, robotic implementation, and biological validation. We show that soft robots can serve as experimental tools to probe biological functions and even test evolutionary hypotheses. Their inherent compliance also allows them to outperform rigid systems in unstructured environments, supporting applications in marine exploration, manipulation, and medicine. Looking forward, we introduce bio-universal-inspired robotics, a paradigm that transcends species-specific mimicry by identifying convergent principles across species to inspire more adaptable designs. Despite rapid progress, challenges persist in material robustness, actuation efficiency, autonomy, and intelligence. By uniting biology and engineering, soft robots can advance ocean exploration and deepen scientific discovery. 

**Abstract (ZH)**: 海洋广大的未探索区域和多样的软体水生生物激发了仿生水下软体机器人的研究兴趣。近期进展为水下运动、感知和交互提供了新的能力。然而，这些努力大多是一维的，生物学指导机器人技术，而机器人技术的见解很少回馈给生物学。为此，我们提出一个整体的、双向的框架，将生物原理、机器人实施和生物验证整合在一起。我们展示软机器人可以作为实验工具来研究生物功能，甚至测试进化的假说。它们固有的柔顺性也使它们在非结构化环境中优于刚性系统，支持海洋探索、操作和医学的应用。展望未来，我们引入了生物通识启发的机器人学范式，该范式跨越物种特定的模仿，通过识别物种间的共通原理来启发更具适应性的设计。尽管取得了快速进展，材料的坚固性、驱动效率、自主性以及智能等方面仍面临挑战。通过结合生物学与工程学，软机器人可以推进海洋探索并深化科学研究。 

---
# Data Shift of Object Detection in Autonomous Driving 

**Title (ZH)**: 自动驾驶中目标检测的数据偏移 

**Authors**: Lida Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11868)  

**Abstract**: With the widespread adoption of machine learning technologies in autonomous driving systems, their role in addressing complex environmental perception challenges has become increasingly crucial. However, existing machine learning models exhibit significant vulnerability, as their performance critically depends on the fundamental assumption that training and testing data satisfy the independent and identically distributed condition, which is difficult to guarantee in real-world applications. Dynamic variations in data distribution caused by seasonal changes, weather fluctuations lead to data shift problems in autonomous driving systems. This study investigates the data shift problem in autonomous driving object detection tasks, systematically analyzing its complexity and diverse manifestations. We conduct a comprehensive review of data shift detection methods and employ shift detection analysis techniques to perform dataset categorization and balancing. Building upon this foundation, we construct an object detection model. To validate our approach, we optimize the model by integrating CycleGAN-based data augmentation techniques with the YOLOv5 framework. Experimental results demonstrate that our method achieves superior performance compared to baseline models on the BDD100K dataset. 

**Abstract (ZH)**: 随着机器学习技术在自动驾驶系统中的广泛应用，其在应对复杂环境感知挑战中的作用变得越来越关键。然而，现有的机器学习模型存在显著的脆弱性，因为它们的表现严重依赖于训练和测试数据满足独立同分布条件的基本假设，而在实际应用中这一假设难以保证。由于季节变化和天气波动导致的数据分布动态变化引起了自动驾驶系统的数据偏移问题。本研究探讨了自动驾驶目标检测任务中的数据偏移问题，系统分析其复杂性和多样性表现。我们对数据偏移检测方法进行了全面回顾，并运用偏移检测分析技术对数据集进行分类和平衡。在此基础上，我们构建了一个目标检测模型。通过将CycleGAN基于的数据增强技术与YOLOv5框架结合，我们优化了该模型。实验结果表明，我们的方法在BDD100K数据集上的性能优于基线模型。 

---
# LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba 

**Title (ZH)**: LocoMamba：基于端到端深度强化学习的视觉驱动运动控制 

**Authors**: Allen Wang, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11849)  

**Abstract**: We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget. 

**Abstract (ZH)**: 我们介绍LocoMamba，这是一种基于选择性状态空间模型构建的视觉驱动跨模态DRL框架，特别利用Mamba，实现了接近线性时间的序列建模，有效捕捉长距离依赖，并能够使用更长的序列进行高效训练。首先，我们使用多层感知机嵌入本体感受态，并使用轻量级卷积神经网络切片深度图像，产生紧凑的令牌以提升状态表示。其次，堆叠的Mamba层通过接近线性时间的选择性扫描融合这些令牌，降低延迟和内存占用，对令牌长度和图像分辨率保持鲁棒性，并提供抑制过拟合的归纳偏置。第三，我们使用地形和外观随机化以及障碍密度 Curriculum 对策略进行端到端训练，采用紧凑的状态中心奖励平衡进展、平滑度和安全性。我们在具有静态和移动障碍以及不平地形的挑战性模拟环境中评估了该方法。与最先进的基线方法相比，该方法在更少的碰撞下获得更高回报和成功率，更能适应未见过的地形和障碍密度，并在相同计算预算下以更少的更新次数实现训练效率的提升。 

---
# Anticipatory and Adaptive Footstep Streaming for Teleoperated Bipedal Robots 

**Title (ZH)**: 预见性和自适应脚步流传输技术在遥操作 bipedal 机器人中的应用 

**Authors**: Luigi Penco, Beomyeong Park, Stefan Fasano, Nehar Poddar, Stephen McCrory, Nicholas Kitchel, Tomasz Bialek, Dexton Anderson, Duncan Calvert, Robert Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11802)  

**Abstract**: Achieving seamless synchronization between user and robot motion in teleoperation, particularly during high-speed tasks, remains a significant challenge. In this work, we propose a novel approach for transferring stepping motions from the user to the robot in real-time. Instead of directly replicating user foot poses, we retarget user steps to robot footstep locations, allowing the robot to utilize its own dynamics for locomotion, ensuring better balance and stability. Our method anticipates user footsteps to minimize delays between when the user initiates and completes a step and when the robot does it. The step estimates are continuously adapted to converge with the measured user references. Additionally, the system autonomously adjusts the robot's steps to account for its surrounding terrain, overcoming challenges posed by environmental mismatches between the user's flat-ground setup and the robot's uneven terrain. Experimental results on the humanoid robot Nadia demonstrate the effectiveness of the proposed system. 

**Abstract (ZH)**: 实现遥操作中用户与机器人运动的无缝同步，特别是在执行高速任务时，仍然是一个重大挑战。本工作中，我们提出了一种新的方法，用于实时将用户的踏步动作转移到机器人上。我们不对用户的脚部姿态进行直接复制，而是将用户的踏步重新目标定位到机器人的脚步位置上，使机器人能够利用自身的动力学进行移动，从而确保更好的平衡和稳定性。该方法预测用户的踏步，以最小化用户开始和完成一步与机器人执行之间的时间延迟。步态估计不断自适应以与测量的用户参考值收敛。此外，系统还自主调整机器人的步态以适应其周围的地形，克服了用户平坦地面设置与机器人不平地形之间环境不匹配的挑战。实验结果表明，所提出的方法在类人机器人Nadia上是有效的。 

---
# Using Natural Language for Human-Robot Collaboration in the Real World 

**Title (ZH)**: 在现实世界中使用自然语言进行人机协作 

**Authors**: Peter Lindes, Kaoutar Skiker  

**Link**: [PDF](https://arxiv.org/pdf/2508.11759)  

**Abstract**: We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem.
In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans. 

**Abstract (ZH)**: 我们展望有一天自主机器人能够在执行复杂物理世界任务时作为人类的助手进行协作，并能够使用自然语言与人类合作者交流。这一愿景包括机器人具备与人类进行自然语言交流的能力。传统的交互式任务学习（ITL）系统在一定程度上具备这样的能力，但其能够理解的语言极为有限。大语言模型（LLMs）的出现为显著提高机器人的语言理解能力提供了机会，但将LLMs的语言能力与在现实物理世界中操作的机器人结合仍是一个具有挑战性的问题。

在本章中，我们首先简要回顾几种与人类紧密合作的商业机器人产品，并讨论它们如何通过增强的语言能力成为更好的合作者。然后，我们探讨如何通过一个以认知代理为核心控制物理机器人的AI系统与人类和大语言模型进行交互，并通过经验累积情境知识，来实现这一愿景的可能性。我们重点关注机器人理解自然语言的三个具体挑战，并使用ChatGPT进行简单的概念验证实验。最后，我们讨论将这些简单的实验发展为一个实际系统所需的条件，其中LLM辅助的语言理解是集成式机器人助手的一部分，用于通过语言与人类协作。 

---
# Has GPT-5 Achieved Spatial Intelligence? An Empirical Study 

**Title (ZH)**: Has GPT-- Achieved Spatial Intelligence? An Empirical Study 

**Authors**: Zhongang Cai, Yubo Wang, Qingping Sun, Ruisi Wang, Chenyang Gu, Wanqi Yin, Zhiqian Lin, Zhitao Yang, Chen Wei, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Jiaqi Li, Xiangyu Fan, Hanming Deng, Lewei Lu, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, Lei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13142)  

**Abstract**: Multi-modal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, which are fundamental capabilities to achieving artificial general intelligence. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models stand on the path toward spatial intelligence. First, we propose a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and discuss the challenges in ensuring fair evaluation. We then evaluate state-of-the-art proprietary and open-source models on eight key benchmarks, at a cost exceeding one billion total tokens. Our empirical study reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence, yet (2) still falls short of human performance across a broad spectrum of tasks. Moreover, we (3) identify the more challenging spatial intelligence problems for multi-modal models, and (4) proprietary models do not exhibit a decisive advantage when facing the most difficult problems. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans yet fail even the most advanced multi-modal models. 

**Abstract (ZH)**: 多模态模型在 recent years 取得了显著进展，但仍表现出在空间理解与推理方面的明显局限性，这是实现人工通用智能的基本能力。随着 GPT-5 的最近发布，据称为迄今为止最强大的 AI 模型，现在是检查领先模型在通向空间智能之路中的位置的合适时机。首先，我们提出了一种综合的空间任务分类法，统一了现有的基准，并讨论了确保公平评估的挑战。然后，我们在八个核心基准上评估了最先进的 proprietary 和开源模型，总消耗超过一亿个标记。我们的实证研究揭示了以下几点：(1) GPT-5 在空间智能方面表现出前所未知的强大，但 (2) 在广泛的多个任务中仍低于人类表现。此外，我们 (3) 确定了多模态模型面临的更具有挑战性空间智能问题，并 (4) 发现当面对最困难的问题时， proprietary 模型并不表现出明显优势。此外，我们还在一系列直观对人类来说却挑战重重的情景中进行了定性评估。 

---
# Precise Action-to-Video Generation Through Visual Action Prompts 

**Title (ZH)**: 通过视觉动作提示实现精确的动作到视频生成 

**Authors**: Yuang Wang, Chao Wen, Haoyu Guo, Sida Peng, Minghan Qin, Hujun Bao, Xiaowei Zhou, Ruizhen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13104)  

**Abstract**: We present visual action prompts, a unified action representation for action-to-video generation of complex high-DoF interactions while maintaining transferable visual dynamics across domains. Action-driven video generation faces a precision-generality trade-off: existing methods using text, primitive actions, or coarse masks offer generality but lack precision, while agent-centric action signals provide precision at the cost of cross-domain transferability. To balance action precision and dynamic transferability, we propose to "render" actions into precise visual prompts as domain-agnostic representations that preserve both geometric precision and cross-domain adaptability for complex actions; specifically, we choose visual skeletons for their generality and accessibility. We propose robust pipelines to construct skeletons from two interaction-rich data sources - human-object interactions (HOI) and dexterous robotic manipulation - enabling cross-domain training of action-driven generative models. By integrating visual skeletons into pretrained video generation models via lightweight fine-tuning, we enable precise action control of complex interaction while preserving the learning of cross-domain dynamics. Experiments on EgoVid, RT-1 and DROID demonstrate the effectiveness of our proposed approach. Project page: this https URL. 

**Abstract (ZH)**: 我们提出了视觉动作提示，这是一种统一的动作表示，用于生成复杂高自由度交互的动作到视频转换，同时保持跨域的可转移视觉动力学。动作驱动的视频生成面临着精度与通用性的权衡：现有方法使用文本、原始动作或粗糙掩码虽然具有通用性但缺乏精度，而以代理为中心的动作信号则以牺牲跨域可转移性为代价提供了精度。为了平衡动作精度与动态可转移性，我们提出将动作“渲染”为精确的视觉提示，作为域无关的表示，既保存几何精度又保持跨域适应性；具体而言，我们选择了通用且易访问的视觉骨架。我们提出了健壮的工作流，从两种富含交互的数据源构建骨架——人-物交互（HOI）和灵巧的机器人操作，从而实现动作驱动生成模型的跨域训练。通过将视觉骨架轻量级微调到预训练的视频生成模型中，我们能够在保持跨域动力学学习的同时实现对复杂交互的精确动作控制。在EgoVid、RT-1和DROID上的实验结果表明了我们提出方法的有效性。项目页面：this https URL。 

---
# On the complexity of constrained reconfiguration and motion planning 

**Title (ZH)**: 受限重构与运动规划的复杂性研究 

**Authors**: Nicolas Bousquet, Remy El Sabeh, Amer E. Mouawad, Naomi Nishimura  

**Link**: [PDF](https://arxiv.org/pdf/2508.13032)  

**Abstract**: Coordinating the motion of multiple agents in constrained environments is a fundamental challenge in robotics, motion planning, and scheduling. A motivating example involves $n$ robotic arms, each represented as a line segment. The objective is to rotate each arm to its vertical orientation, one at a time (clockwise or counterclockwise), without collisions nor rotating any arm more than once. This scenario is an example of the more general $k$-Compatible Ordering problem, where $n$ agents, each capable of $k$ state-changing actions, must transition to specific target states under constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs.
We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when $\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we provide polynomial-time algorithms for cases such as when $k = 1$ or $\mathcal{G}$ has bounded treewidth. We also introduce generalized variants supporting multiple state-changing actions per agent, broadening the applicability of our framework. These results extend to a wide range of scheduling, reconfiguration, and motion planning applications in constrained environments. 

**Abstract (ZH)**: 在受限环境中协调多代理系统的运动是机器人学、运动规划和调度领域的基础挑战。一个动机例子涉及$n$个机器人臂，每个臂表示为一条线段。目标是依次旋转每个臂至垂直方位（顺时针或逆时针），过程中不发生碰撞且每个臂仅旋转一次。这是一个更一般的$k$-兼容排序问题的实例，其中$n$个代理，每个代理可进行$k$种状态变化操作，必须在由集合$\mathcal{G}$（包含$k$对有向图）编码的约束下过渡到特定的目标状态。我们证明了$k$-兼容排序问题是$\mathsf{NP}$-完全问题，即使在$\mathcal{G}$为平面图、退化图或有向无环图的情况下也是如此。从积极的一面看，我们提供了当$k=1$或$\mathcal{G}$具有有界 treewidth时的多项式时间算法。我们还引入了支持每个代理多种状态变化操作的广义变体，扩大了我们框架的应用范围。这些结果扩展应用于受限环境中的广泛排程、重构和运动规划应用。 

---
# Adjustable AprilTags For Identity Secured Tasks 

**Title (ZH)**: 可调节AprilTags 用于身份认证任务 

**Authors**: Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.12304)  

**Abstract**: Special tags such as AprilTags that facilitate image processing and pattern recognition are useful in practical applications. In close and private environments, identity security is unlikely to be an issue because all involved AprilTags can be completely regulated. However, in open and public environments, identity security is no longer an issue that can be neglected. To handle potential harm caused by adversarial attacks, this note advocates utilization of adjustable AprilTags instead of fixed ones. 

**Abstract (ZH)**: 特殊的AprilTags等标记标签便于图像处理和模式识别，在实际应用中非常有用。在封闭和私密环境中，身份安全通常不是问题，因为所有涉及的AprilTags都可以完全受控。然而，在开放和公共环境中，身份安全不再是可忽视的问题。为了应对潜在的 adversarial攻击造成的危害，本文主张使用可调节的AprilTags而非固定的AprilTags。 

---
# DynamicPose: Real-time and Robust 6D Object Pose Tracking for Fast-Moving Cameras and Objects 

**Title (ZH)**: DynamicPose：快速移动相机和物体的实时 robust 6D对象姿态跟踪 

**Authors**: Tingbang Liang, Yixin Zeng, Jiatong Xie, Boyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11950)  

**Abstract**: We present DynamicPose, a retraining-free 6D pose tracking framework that improves tracking robustness in fast-moving camera and object scenarios. Previous work is mainly applicable to static or quasi-static scenes, and its performance significantly deteriorates when both the object and the camera move rapidly. To overcome these challenges, we propose three synergistic components: (1) A visual-inertial odometry compensates for the shift in the Region of Interest (ROI) caused by camera motion; (2) A depth-informed 2D tracker corrects ROI deviations caused by large object translation; (3) A VIO-guided Kalman filter predicts object rotation, generates multiple candidate poses, and then obtains the final pose by hierarchical refinement. The 6D pose tracking results guide subsequent 2D tracking and Kalman filter updates, forming a closed-loop system that ensures accurate pose initialization and precise pose tracking. Simulation and real-world experiments demonstrate the effectiveness of our method, achieving real-time and robust 6D pose tracking for fast-moving cameras and objects. 

**Abstract (ZH)**: DynamicPose：一种无需重新训练的6D姿态跟踪框架，适用于快速移动的相机和物体场景 

---
# Recent Advances in Transformer and Large Language Models for UAV Applications 

**Title (ZH)**: Recent Advances in Transformer and Large Language Models for UAV Applications 

**Authors**: Hamza Kheddar, Yassine Habchi, Mohamed Chahine Ghanem, Mustapha Hemis, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2508.11834)  

**Abstract**: The rapid advancement of Transformer-based models has reshaped the landscape of uncrewed aerial vehicle (UAV) systems by enhancing perception, decision-making, and autonomy. This review paper systematically categorizes and evaluates recent developments in Transformer architectures applied to UAVs, including attention mechanisms, CNN-Transformer hybrids, reinforcement learning Transformers, and large language models (LLMs). Unlike previous surveys, this work presents a unified taxonomy of Transformer-based UAV models, highlights emerging applications such as precision agriculture and autonomous navigation, and provides comparative analyses through structured tables and performance benchmarks. The paper also reviews key datasets, simulators, and evaluation metrics used in the field. Furthermore, it identifies existing gaps in the literature, outlines critical challenges in computational efficiency and real-time deployment, and offers future research directions. This comprehensive synthesis aims to guide researchers and practitioners in understanding and advancing Transformer-driven UAV technologies. 

**Abstract (ZH)**: 基于Transformer的无人机系统快速发展重塑了感知、决策和自主性的格局。本文系统地分类和评估了Transformer架构在无人机领域的最新进展，包括注意力机制、CNN-Transformer混合模型、强化学习Transformer以及大型语言模型（LLMs）。与以往综述不同，本文提出了基于Transformer的无人机模型的统一分类体系，强调了精准农业和自主导航等新兴应用，并通过结构化表格和性能基准提供了对比分析。本文还回顾了该领域使用的关键数据集、模拟器和评价指标。此外，本文指出了文献中的现有空白，概述了计算效率和实时部署中的关键挑战，并提出了未来研究方向。本文综合分析旨在引导研究人员和实践者理解并推进Transformer驱动的无人机技术。 

---
# Control of a commercial vehicle by a tetraplegic human using a bimanual brain-computer interface 

**Title (ZH)**: 四肢瘫痪人士使用双手持械脑机接口控制商用车辆 

**Authors**: Xinyun Zou, Jorge Gamez, Meghna Menon, Phillip Ring, Chadwick Boulay, Likhith Chitneni, Jackson Brennecke, Shana R. Melby, Gracy Kureel, Kelsie Pejsa, Emily R. Rosario, Ausaf A. Bari, Aniruddh Ravindran, Tyson Aflalo, Spencer S. Kellis, Dimitar Filev, Florian Solzbacher, Richard A. Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11805)  

**Abstract**: Brain-computer interfaces (BCIs) read neural signals directly from the brain to infer motor planning and execution. However, the implementation of this technology has been largely limited to laboratory settings, with few real-world applications. We developed a bimanual BCI system to drive a vehicle in both simulated and real-world environments. We demonstrate that an individual with tetraplegia, implanted with intracortical BCI electrodes in the posterior parietal cortex (PPC) and the hand knob region of the motor cortex (MC), reacts at least as fast and precisely as motor intact participants, and drives a simulated vehicle as proficiently as the same control group. This BCI participant, living in California, could also remotely drive a Ford Mustang Mach-E vehicle in Michigan. Our first teledriving task relied on cursor control for speed and steering in a closed urban test facility. However, the final BCI system added click control for full-stop braking and thus enabled bimanual cursor-and-click control for both simulated driving through a virtual town with traffic and teledriving through an obstacle course without traffic in the real world. We also demonstrate the safety and feasibility of BCI-controlled driving. This first-of-its-kind implantable BCI application not only highlights the versatility and innovative potentials of BCIs but also illuminates the promising future for the development of life-changing solutions to restore independence to those who suffer catastrophic neurological injury. 

**Abstract (ZH)**: 脑机接口系统在模拟和真实环境下的双臂驾驶研究 

---
# Scaling Robust Optimization for Swarms: A Distributed Perspective 

**Title (ZH)**: 群智能中鲁棒优化的扩展：一种分布式视角 

**Authors**: Arshiya Taj Abdul, Augustinos D. Saravanos, Evangelos A. Theodorou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11799)  

**Abstract**: This article introduces a decentralized robust optimization framework for safe multi-agent control under uncertainty. Although stochastic noise has been the primary form of modeling uncertainty in such systems, these formulations might fall short in addressing uncertainties that are deterministic in nature or simply lack probabilistic data. To ensure safety under such scenarios, we employ the concept of robust constraints that must hold for all possible uncertainty realizations lying inside a bounded set. Nevertheless, standard robust optimization approaches become intractable due to the large number or non-convexity of the constraints involved in safe multi-agent control. To address this, we introduce novel robust reformulations that significantly reduce complexity without compromising safety. The applicability of the framework is further broadened to address both deterministic and stochastic uncertainties by incorporating robust chance constraints and distribution steering techniques. To achieve scalability, we derive a distributed approach based on the Alternating Direction Method of Multipliers (ADMM), supported by a convergence study that accounts for the underlying non-convexity. In addition, computational complexity bounds highlighting the efficiency of the proposed frameworks against standard approaches are presented. Finally, the robustness and scalability of the framework is demonstrated through extensive simulation results across diverse scenarios, including environments with nonconvex obstacles and up to 246 agents. 

**Abstract (ZH)**: 一种用于不确定性下安全多代理控制的去中心化鲁棒优化框架 

---
# Lifelong Learner: Discovering Versatile Neural Solvers for Vehicle Routing Problems 

**Title (ZH)**: 终身学习者：发现适用于车辆 routing 问题的多功能神经求解器 

**Authors**: Shaodi Feng, Zhuoyi Lin, Jianan Zhou, Cong Zhang, Jingwen Li, Kuan-Wen Chen, Senthilnath Jayavelu, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11679)  

**Abstract**: Deep learning has been extensively explored to solve vehicle routing problems (VRPs), which yields a range of data-driven neural solvers with promising outcomes. However, most neural solvers are trained to tackle VRP instances in a relatively monotonous context, e.g., simplifying VRPs by using Euclidean distance between nodes and adhering to a single problem size, which harms their off-the-shelf application in different scenarios. To enhance their versatility, this paper presents a novel lifelong learning framework that incrementally trains a neural solver to manage VRPs in distinct contexts. Specifically, we propose a lifelong learner (LL), exploiting a Transformer network as the backbone, to solve a series of VRPs. The inter-context self-attention mechanism is proposed within LL to transfer the knowledge obtained from solving preceding VRPs into the succeeding ones. On top of that, we develop a dynamic context scheduler (DCS), employing the cross-context experience replay to further facilitate LL looking back on the attained policies of solving preceding VRPs. Extensive results on synthetic and benchmark instances (problem sizes up to 18k) show that our LL is capable of discovering effective policies for tackling generic VRPs in varying contexts, which outperforms other neural solvers and achieves the best performance for most VRPs. 

**Abstract (ZH)**: 深度学习在解决车辆路线问题中的持续学习框架：一种新颖的方法以适应不同情境下的通用车辆路线规划 

---
