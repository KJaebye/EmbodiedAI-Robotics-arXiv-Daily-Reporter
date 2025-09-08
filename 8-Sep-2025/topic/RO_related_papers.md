# Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers 

**Title (ZH)**: 基于感知观测器的自主车辆鲁棒模型预测控制设计 

**Authors**: Nariman Niknejad, Gokul S. Sankar, Bahare Kiumarsi, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2509.05201)  

**Abstract**: This paper presents a robust model predictive control (MPC) framework that explicitly addresses the non-Gaussian noise inherent in deep learning-based perception modules used for state estimation. Recognizing that accurate uncertainty quantification of the perception module is essential for safe feedback control, our approach departs from the conventional assumption of zero-mean noise quantification of the perception error. Instead, it employs set-based state estimation with constrained zonotopes to capture biased, heavy-tailed uncertainties while maintaining bounded estimation errors. To improve computational efficiency, the robust MPC is reformulated as a linear program (LP), using a Minkowski-Lyapunov-based cost function with an added slack variable to prevent degenerate solutions. Closed-loop stability is ensured through Minkowski-Lyapunov inequalities and contractive zonotopic invariant sets. The largest stabilizing terminal set and its corresponding feedback gain are then derived via an ellipsoidal approximation of the zonotopes. The proposed framework is validated through both simulations and hardware experiments on an omnidirectional mobile robot along with a camera and a convolutional neural network-based perception module implemented within a ROS2 framework. The results demonstrate that the perception-aware MPC provides stable and accurate control performance under heavy-tailed noise conditions, significantly outperforming traditional Gaussian-noise-based designs in terms of both state estimation error bounding and overall control performance. 

**Abstract (ZH)**: 一种考虑深度学习感知模块非高斯噪声的鲁棒模型预测控制框架 

---
# Lyapunov-Based Deep Learning Control for Robots with Unknown Jacobian 

**Title (ZH)**: 基于Lyapunov的深度学习控制方法用于未知雅各比矩阵的机器人 

**Authors**: Koji Matsuno, Chien Chern Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2509.04984)  

**Abstract**: Deep learning, with its exceptional learning capabilities and flexibility, has been widely applied in various applications. However, its black-box nature poses a significant challenge in real-time robotic applications, particularly in robot control, where trustworthiness and robustness are critical in ensuring safety. In robot motion control, it is essential to analyze and ensure system stability, necessitating the establishment of methodologies that address this need. This paper aims to develop a theoretical framework for end-to-end deep learning control that can be integrated into existing robot control theories. The proposed control algorithm leverages a modular learning approach to update the weights of all layers in real time, ensuring system stability based on Lyapunov-like analysis. Experimental results on industrial robots are presented to illustrate the performance of the proposed deep learning controller. The proposed method offers an effective solution to the black-box problem in deep learning, demonstrating the possibility of deploying real-time deep learning strategies for robot kinematic control in a stable manner. This achievement provides a critical foundation for future advancements in deep learning based real-time robotic applications. 

**Abstract (ZH)**: 基于深度学习的端到端控制理论框架及其在机器人运动控制中的应用 

---
# Ground-Aware Octree-A* Hybrid Path Planning for Memory-Efficient 3D Navigation of Ground Vehicles 

**Title (ZH)**: 基于地面感知的八叉树-A*混合路径规划方法及其在地面车辆记忆高效3D导航中的应用 

**Authors**: Byeong-Il Ham, Hyun-Bin Kim, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.04950)  

**Abstract**: In this paper, we propose a 3D path planning method that integrates the A* algorithm with the octree structure. Unmanned Ground Vehicles (UGVs) and legged robots have been extensively studied, enabling locomotion across a variety of terrains. Advances in mobility have enabled obstacles to be regarded not only as hindrances to be avoided, but also as navigational aids when beneficial. A modified 3D A* algorithm generates an optimal path by leveraging obstacles during the planning process. By incorporating a height-based penalty into the cost function, the algorithm enables the use of traversable obstacles to aid locomotion while avoiding those that are impassable, resulting in more efficient and realistic path generation. The octree-based 3D grid map achieves compression by merging high-resolution nodes into larger blocks, especially in obstacle-free or sparsely populated areas. This reduces the number of nodes explored by the A* algorithm, thereby improving computational efficiency and memory usage, and supporting real-time path planning in practical environments. Benchmark results demonstrate that the use of octree structure ensures an optimal path while significantly reducing memory usage and computation time. 

**Abstract (ZH)**: 本文提出了一种结合A*算法和八叉树结构的3D路径规划方法。无人驾驶地面车辆（UGVs）和腿式机器人已被广泛研究，使它们能够在多种地形上移动。移动性的进步使障碍物不仅可以被视为需要避免的阻碍，还可以在有益时作为导航辅助。修改后的3D A*算法在规划过程中利用障碍物生成最优路径。通过将高度相关的惩罚纳入成本函数中，算法能够在利用可通行障碍物辅助移动的同时避开不可通行的障碍物，从而生成更高效、更真实的路径。基于八叉树的3D网格地图通过在无障碍或稀疏区域将高分辨率节点合并为较大块体实现压缩，从而减少A*算法探索的节点数量，提高计算效率和内存使用率，并支持实际环境中的实时路径规划。基准测试结果表明，使用八叉树结构可确保最优路径并大幅减少内存使用和计算时间。 

---
# UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis 

**Title (ZH)**: 基于无人机的智能交通 surveillance 系统：实时车辆检测、分类、跟踪及行为分析 

**Authors**: Ali Khanpour, Tianyi Wang, Afra Vahidi-Shams, Wim Ectors, Farzam Nakhaie, Amirhossein Taheri, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2509.04624)  

**Abstract**: Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities. 

**Abstract (ZH)**: 基于无人机的交通监控系统：实时城市环境中车辆检测、分类、跟踪和行为分析 

---
# PRREACH: Probabilistic Risk Assessment Using Reachability for UAV Control 

**Title (ZH)**: PRREACH：基于可达性方法的UAV控制概率风险评估 

**Authors**: Nicole Fronda, Hariharan Narayanan, Sadia Afrin Ananna, Steven Weber, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2509.04451)  

**Abstract**: We present a new approach for designing risk-bounded controllers for Uncrewed Aerial Vehicles (UAVs). Existing frameworks for assessing risk of UAV operations rely on knowing the conditional probability of an incident occurring given different causes. Limited data for computing these probabilities makes real-world implementation of these frameworks difficult. Furthermore, existing frameworks do not include control methods for risk mitigation. Our approach relies on UAV dynamics, and employs reachability analysis for a probabilistic risk assessment over all feasible UAV trajectories. We use this holistic risk assessment to formulate a control optimization problem that minimally changes a UAV's existing control law to be bounded by an accepted risk threshold. We call our approach PRReach. Public and readily available UAV dynamics models and open source spatial data for mapping hazard outcomes enables practical implementation of PRReach for both offline pre-flight and online in-flight risk assessment and mitigation. We evaluate PRReach through simulation experiments on real-world data. Results show that PRReach controllers reduce risk by up to 24% offline, and up to 53% online from classical controllers. 

**Abstract (ZH)**: 一种用于无人驾驶航空车辆（UAVs）的风险界判定控制器设计新方法 

---
