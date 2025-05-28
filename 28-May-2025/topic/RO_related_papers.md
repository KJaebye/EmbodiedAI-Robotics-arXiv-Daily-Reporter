# Collision Probability Estimation for Optimization-based Vehicular Motion Planning 

**Title (ZH)**: 基于优化的车辆运动规划中的碰撞概率估计 

**Authors**: Leon Tolksdorf, Arturo Tejada, Christian Birkner, Nathan van de Wouw  

**Link**: [PDF](https://arxiv.org/pdf/2505.21161)  

**Abstract**: Many motion planning algorithms for automated driving require estimating the probability of collision (POC) to account for uncertainties in the measurement and estimation of the motion of road users. Common POC estimation techniques often utilize sampling-based methods that suffer from computational inefficiency and a non-deterministic estimation, i.e., each estimation result for the same inputs is slightly different. In contrast, optimization-based motion planning algorithms require computationally efficient POC estimation, ideally using deterministic estimation, such that typical optimization algorithms for motion planning retain feasibility. Estimating the POC analytically, however, is challenging because it depends on understanding the collision conditions (e.g., vehicle's shape) and characterizing the uncertainty in motion prediction. In this paper, we propose an approach in which we estimate the POC between two vehicles by over-approximating their shapes by a multi-circular shape approximation. The position and heading of the predicted vehicle are modelled as random variables, contrasting with the literature, where the heading angle is often neglected. We guarantee that the provided POC is an over-approximation, which is essential in providing safety guarantees, and present a computationally efficient algorithm for computing the POC estimate for Gaussian uncertainty in the position and heading. This algorithm is then used in a path-following stochastic model predictive controller (SMPC) for motion planning. With the proposed algorithm, the SMPC generates reproducible trajectories while the controller retains its feasibility in the presented test cases and demonstrates the ability to handle varying levels of uncertainty. 

**Abstract (ZH)**: 基于多圆弧逼近的碰撞概率估算方法在自动驾驶路径规划中的应用 

---
# SCALOFT: An Initial Approach for Situation Coverage-Based Safety Analysis of an Autonomous Aerial Drone in a Mine Environment 

**Title (ZH)**: SCALOFT: 一种基于情境覆盖的自主无人机矿井环境安全分析初步方法 

**Authors**: Nawshin Mannan Proma, Victoria J Hodge, Rob Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2505.20969)  

**Abstract**: The safety of autonomous systems in dynamic and hazardous environments poses significant challenges. This paper presents a testing approach named SCALOFT for systematically assessing the safety of an autonomous aerial drone in a mine. SCALOFT provides a framework for developing diverse test cases, real-time monitoring of system behaviour, and detection of safety violations. Detected violations are then logged with unique identifiers for detailed analysis and future improvement. SCALOFT helps build a safety argument by monitoring situation coverage and calculating a final coverage measure. We have evaluated the performance of this approach by deliberately introducing seeded faults into the system and assessing whether SCALOFT is able to detect those faults. For a small set of plausible faults, we show that SCALOFT is successful in this. 

**Abstract (ZH)**: 自动驾驶系统在动态和危险环境中的安全性测试方法：SCALOFT在矿用自主无人机安全评估中的应用 

---
# COM Adjustment Mechanism Control for Multi-Configuration Motion Stability of Unmanned Deformable Vehicle 

**Title (ZH)**: 无人变形车多配置运动稳定性COM调整机制控制 

**Authors**: Jun Liu, Hongxun Liu, Cheng Zhang, Jiandang Xing, Shang Jiang, Ping Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20926)  

**Abstract**: An unmanned deformable vehicle is a wheel-legged robot transforming between two configurations: vehicular and humanoid states, with different motion modes and stability characteristics. To address motion stability in multiple configurations, a center-of-mass adjustment mechanism was designed. Further, a motion stability hierarchical control algorithm was proposed, and an electromechanical model based on a two-degree-of-freedom center-of-mass adjustment mechanism was established. An unmanned-deformable-vehicle vehicular-state steady-state steering dynamics model and a gait planning kinematic model of humanoid state walking were established. A stability hierarchical control strategy was designed to realize the stability control. The results showed that the steady-state steering stability in vehicular state and the walking stability in humanoid state could be significantly improved by controlling the slider motion. 

**Abstract (ZH)**: 一种变形无人车通过车辆态和人形态两种配置转换，具有不同的运动模式和稳定性特征。为了在多种配置下实现运动稳定性，设计了一种质心调节机制，并提出了一种运动稳定性分层控制算法，建立了基于两自由度质心调节机制的机电模型。建立了无人车车辆态稳态转向动力学模型和人形态步行姿态规划运动模型。设计了一种稳定性分层控制策略以实现稳定性控制。结果表明，通过控制滑块运动可以显著提高车辆态稳态转向稳定性和人形态步行稳定性。 

---
# HS-SLAM: A Fast and Hybrid Strategy-Based SLAM Approach for Low-Speed Autonomous Driving 

**Title (ZH)**: HS-SLAM：一种适用于低速自主驾驶的快速混合策略SLAM方法 

**Authors**: Bingxiang Kang, Jie Zou, Guofa Li, Pengwei Zhang, Jie Zeng, Kan Wang, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20906)  

**Abstract**: Visual-inertial simultaneous localization and mapping (SLAM) is a key module of robotics and low-speed autonomous vehicles, which is usually limited by the high computation burden for practical applications. To this end, an innovative strategy-based hybrid framework HS-SLAM is proposed to integrate the advantages of direct and feature-based methods for fast computation without decreasing the performance. It first estimates the relative positions of consecutive frames using IMU pose estimation within the tracking thread. Then, it refines these estimates through a multi-layer direct method, which progressively corrects the relative pose from coarse to fine, ultimately achieving accurate corner-based feature matching. This approach serves as an alternative to the conventional constant-velocity tracking model. By selectively bypassing descriptor extraction for non-critical frames, HS-SLAM significantly improves the tracking speed. Experimental evaluations on the EuRoC MAV dataset demonstrate that HS-SLAM achieves higher localization accuracies than ORB-SLAM3 while improving the average tracking efficiency by 15%. 

**Abstract (ZH)**: 视觉惯性同时定位与建图（SLAM）是机器人和低速自动驾驶车辆中的一个关键模块，通常受限于实际应用中的高计算负担。为此，提出了一种创新策略导向的混合框架HS-SLAM，以集成直接法和特征法的优势，在不降低性能的情况下实现快速计算。该框架首先在跟踪线程中使用IMU位姿估计来估算连续帧的相对位置。然后，通过多层直接方法逐步修正相对姿态，从粗到细最终实现精确的角点特征匹配。该方法作为一种常速度跟踪模型的替代方案。通过有选择地跳过非关键帧的描述子提取，HS-SLAM显著提高了跟踪速度。实验评估表明，HS-SLAM在Euroc MAV数据集上的定位精度高于ORB-SLAM3，同时平均跟踪效率提高了15%。 

---
# Developing a Robotic Surgery Training System for Wide Accessibility and Research 

**Title (ZH)**: 开发一种广泛 accessible 的机器人手术培训系统及研究 

**Authors**: Walid Shaker, Mustafa Suphi Erden  

**Link**: [PDF](https://arxiv.org/pdf/2505.20562)  

**Abstract**: Robotic surgery represents a major breakthrough in medical interventions, which has revolutionized surgical procedures. However, the high cost and limited accessibility of robotic surgery systems pose significant challenges for training purposes. This study addresses these issues by developing a cost-effective robotic laparoscopy training system that closely replicates advanced robotic surgery setups to ensure broad access for both on-site and remote users. Key innovations include the design of a low-cost robotic end-effector that effectively mimics high-end laparoscopic instruments. Additionally, a digital twin platform was established, facilitating detailed simulation, testing, and real-time monitoring, which enhances both system development and deployment. Furthermore, teleoperation control was optimized, leading to improved trajectory tracking while maintaining remote center of motion (RCM) constraint, with a RMSE of 5 {\mu}m and reduced system latency to 0.01 seconds. As a result, the system provides smooth, continuous motion and incorporates essential safety features, making it a highly effective tool for laparoscopic training. 

**Abstract (ZH)**: 低成本 Surgical Robotic Laparoscopy 培训系统的发展及其应用研究 

---
# HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval 

**Title (ZH)**: HAND Me the Data: 快速机器人适应性提升via 手部路径检索 

**Authors**: Matthew Hong, Anthony Liang, Kevin Kim, Harshitha Rajaprakash, Jesse Thomason, Erdem Bıyık, Jesse Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20455)  

**Abstract**: We hand the community HAND, a simple and time-efficient method for teaching robots new manipulation tasks through human hand demonstrations. Instead of relying on task-specific robot demonstrations collected via teleoperation, HAND uses easy-to-provide hand demonstrations to retrieve relevant behaviors from task-agnostic robot play data. Using a visual tracking pipeline, HAND extracts the motion of the human hand from the hand demonstration and retrieves robot sub-trajectories in two stages: first filtering by visual similarity, then retrieving trajectories with similar behaviors to the hand. Fine-tuning a policy on the retrieved data enables real-time learning of tasks in under four minutes, without requiring calibrated cameras or detailed hand pose estimation. Experiments also show that HAND outperforms retrieval baselines by over 2x in average task success rates on real robots. Videos can be found at our project website: this https URL. 

**Abstract (ZH)**: 我们提供了社区HAND，这是一种简单且高效的方法，通过人类手部演示来教机器人执行新的操作任务。HAND不依赖于通过远程操作收集的任务特定机器人演示，而是使用易于提供的手部演示从任务无关的机器人玩耍数据中检索相关行为。通过视觉跟踪流水线，HAND从手部演示中提取人类手部的运动，在两个阶段检索机器人子轨迹：首先通过视觉相似性进行过滤，然后检索具有相似行为的轨迹。在检索的数据上微调策略可以在四分钟内实现实时任务学习，无需校准摄像头或详细的手部姿态估计。实验结果还显示，与检索基线相比，HAND在实际机器人上的平均任务成功率提高了超过2倍。更多视频详情请访问我们的项目网站：this https URL。 

---
# Co-Design of Soft Gripper with Neural Physics 

**Title (ZH)**: 软 gripper 与神经物理的联合设计 

**Authors**: Sha Yi, Xueqian Bai, Adabhav Singh, Jianglong Ye, Michael T Tolley, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20404)  

**Abstract**: For robot manipulation, both the controller and end-effector design are crucial. Soft grippers are generalizable by deforming to different geometries, but designing such a gripper and finding its grasp pose remains challenging. In this paper, we propose a co-design framework that generates an optimized soft gripper's block-wise stiffness distribution and its grasping pose, using a neural physics model trained in simulation. We derived a uniform-pressure tendon model for a flexure-based soft finger, then generated a diverse dataset by randomizing both gripper pose and design parameters. A neural network is trained to approximate this forward simulation, yielding a fast, differentiable surrogate. We embed that surrogate in an end-to-end optimization loop to optimize the ideal stiffness configuration and best grasp pose. Finally, we 3D-print the optimized grippers of various stiffness by changing the structural parameters. We demonstrate that our co-designed grippers significantly outperform baseline designs in both simulation and hardware experiments. 

**Abstract (ZH)**: 基于co-design框架的优化软 gripper 及其抓取姿态设计 

---
