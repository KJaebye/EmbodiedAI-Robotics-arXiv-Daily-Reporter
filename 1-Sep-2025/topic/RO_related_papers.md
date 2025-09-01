# Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators 

**Title (ZH)**: 具有碰撞避免保证的鲁棒凸模型预测控制机器人 manipulator 

**Authors**: Bernhard Wullt, Johannes Köhler, Per Mattsson, Mikeal Norrlöf, Thomas B. Schön  

**Link**: [PDF](https://arxiv.org/pdf/2508.21677)  

**Abstract**: Industrial manipulators are normally operated in cluttered environments, making safe motion planning important. Furthermore, the presence of model-uncertainties make safe motion planning more difficult. Therefore, in practice the speed is limited in order to reduce the effect of disturbances. There is a need for control methods that can guarantee safe motions that can be executed fast. We address this need by suggesting a novel model predictive control (MPC) solution for manipulators, where our two main components are a robust tube MPC and a corridor planning algorithm to obtain collision-free motion. Our solution results in a convex MPC, which we can solve fast, making our method practically useful. We demonstrate the efficacy of our method in a simulated environment with a 6 DOF industrial robot operating in cluttered environments with uncertainties in model parameters. We outperform benchmark methods, both in terms of being able to work under higher levels of model uncertainties, while also yielding faster motion. 

**Abstract (ZH)**: 工业 manipulator 在复杂环境中的安全运动规划至关重要，模型不确定性进一步增加了这一挑战。为保证快速执行的安全运动，我们提出了一种新型的模型预测控制（MPC）方法，该方法结合了鲁棒管型 MPC 和走廊规划算法以获得无碰撞路径。我们的解决方案形成了一个凸型 MPC，可以通过快速求解，使该方法具有实际应用价值。我们在具有模型参数不确定性且环境复杂的 6 自由度工业机器人仿真实验中展示了该方法的有效性，并在应对更高水平的模型不确定性方面优于基准方法，同时实现了更快的运动。 

---
# Estimated Informed Anytime Search for Sampling-Based Planning via Adaptive Sampler 

**Title (ZH)**: 基于自适应采样的估计知情任意时间搜索采样基于规划算法 

**Authors**: Liding Zhang, Kuanqi Cai, Yu Zhang, Zhenshan Bing, Chaoqun Wang, Fan Wu, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.21549)  

**Abstract**: Path planning in robotics often involves solving continuously valued, high-dimensional problems. Popular informed approaches include graph-based searches, such as A*, and sampling-based methods, such as Informed RRT*, which utilize informed set and anytime strategies to expedite path optimization incrementally. Informed sampling-based planners define informed sets as subsets of the problem domain based on the current best solution cost. However, when no solution is found, these planners re-sample and explore the entire configuration space, which is time-consuming and computationally expensive. This article introduces Multi-Informed Trees (MIT*), a novel planner that constructs estimated informed sets based on prior admissible solution costs before finding the initial solution, thereby accelerating the initial convergence rate. Moreover, MIT* employs an adaptive sampler that dynamically adjusts the sampling strategy based on the exploration process. Furthermore, MIT* utilizes length-related adaptive sparse collision checks to guide lazy reverse search. These features enhance path cost efficiency and computation times while ensuring high success rates in confined scenarios. Through a series of simulations and real-world experiments, it is confirmed that MIT* outperforms existing single-query, sampling-based planners for problems in R^4 to R^16 and has been successfully applied to real-world robot manipulation tasks. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 机器人路径规划经常涉及到连续值的高维问题。流行的启发式方法包括基于图的搜索，如A*，以及基于采样的方法，如Informed RRT*，这些方法利用启发式集合和即时策略来逐步加速路径优化。启发式基于采样规划器根据当前最佳解的成本定义启发式集合。然而，当未找到解时，这些规划器需要重新采样并探索整个配置空间，这既耗时又计算成本高。本文介绍了一种新的规划器——多启发式树（MIT*），它在找到初始解之前基于先前可接纳解的成本估计启发式集合，从而加速初始收敛率。此外，MIT*采用了一种自适应采样器，根据探索过程动态调整采样策略。此外，MIT*利用与长度相关的自适应稀疏碰撞检查来引导懒惰的反向搜索。这些功能在确保在受限场景中的高成功率的同时，提高了路径成本效率和计算时间。通过一系列仿真和现实世界的实验，验证了MIT*在从R^4到R^16的问题中优于现有的单查询采样基于规划器，并成功应用于实际的机器人操作任务。我们的实验结果演示视频可在以下链接查看：this https URL。 

---
# Dynamics-Compliant Trajectory Diffusion for Super-Nominal Payload Manipulation 

**Title (ZH)**: 符合动力学要求的超额定载荷轨迹扩散 

**Authors**: Anuj Pasricha, Joewie Koh, Jay Vakil, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2508.21375)  

**Abstract**: Nominal payload ratings for articulated robots are typically derived from worst-case configurations, resulting in uniform payload constraints across the entire workspace. This conservative approach severely underutilizes the robot's inherent capabilities -- our analysis demonstrates that manipulators can safely handle payloads well above nominal capacity across broad regions of their workspace while staying within joint angle, velocity, acceleration, and torque limits. To address this gap between assumed and actual capability, we propose a novel trajectory generation approach using denoising diffusion models that explicitly incorporates payload constraints into the planning process. Unlike traditional sampling-based methods that rely on inefficient trial-and-error, optimization-based methods that are prohibitively slow, or kinodynamic planners that struggle with problem dimensionality, our approach generates dynamically feasible joint-space trajectories in constant time that can be directly executed on physical hardware without post-processing. Experimental validation on a 7 DoF Franka Emika Panda robot demonstrates that up to 67.6% of the workspace remains accessible even with payloads exceeding 3 times the nominal capacity. This expanded operational envelope highlights the importance of a more nuanced consideration of payload dynamics in motion planning algorithms. 

**Abstract (ZH)**: articulated 机器人的名义载荷评级通常基于最坏情况配置得出，导致整个工作空间内的载荷限制均匀分布。这种保守的方法严重低估了机器人的固有能力——我们的分析表明，当处于关节角度、速度、加速度和扭矩限制范围内时，操作器可以在其工作空间的广大区域中安全地处理远超名义容量的载荷。为了弥补假设能力和实际能力之间的差距，我们提出了一种使用去噪扩散模型的新型轨迹生成方法，该方法在规划过程中明确包含了载荷限制。不同于依赖低效试错的基于采样的方法、难以实现优化的优化方法，或在处理问题维度时挣扎的运动动力学规划器，我们的方法可以在常数时间内生成动态可行的关节空间轨迹，可以直接在物理硬件上执行而不需要后处理。在7自由度的Franka Emika Panda机器人上的实验验证表明，即使载荷超过名义容量的3倍，仍有高达67.6%的工作空间保持可访问。这种扩展的工作空间范围突显了在运动规划算法中更细致地考虑载荷动力学的重要性。 

---
# Cooperative Sensing Enhanced UAV Path-Following and Obstacle Avoidance with Variable Formation 

**Title (ZH)**: 基于可变编队合作感知的无人机路径跟踪与避障技术 

**Authors**: Changheng Wang, Zhiqing Wei, Wangjun Jiang, Haoyue Jiang, Zhiyong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21316)  

**Abstract**: The high mobility of unmanned aerial vehicles (UAVs) enables them to be used in various civilian fields, such as rescue and cargo transport. Path-following is a crucial way to perform these tasks while sensing and collision avoidance are essential for safe flight. In this paper, we investigate how to efficiently and accurately achieve path-following, obstacle sensing and avoidance subtasks, as well as their conflict-free fusion scheduling. Firstly, a high precision deep reinforcement learning (DRL)-based UAV formation path-following model is developed, and the reward function with adaptive weights is designed from the perspective of distance and velocity errors. Then, we use integrated sensing and communication (ISAC) signals to detect the obstacle and derive the Cramer-Rao lower bound (CRLB) for obstacle sensing by information-level fusion, based on which we propose the variable formation enhanced obstacle position estimation (VFEO) algorithm. In addition, an online obstacle avoidance scheme without pretraining is designed to solve the sparse reward. Finally, with the aid of null space based (NSB) behavioral method, we present a hierarchical subtasks fusion strategy. Simulation results demonstrate the effectiveness and superiority of the subtask algorithms and the hierarchical fusion strategy. 

**Abstract (ZH)**: 无人驾驶航空车辆（UAVs）的高机动性使其能够在救援和货物运输等众多民用领域应用，路径跟踪是执行这些任务的关键方式，而感测和避障是确保安全飞行的必要条件。本文探讨了如何高效准确地实现路径跟踪、障碍感测与避障子任务及其冲突-free融合调度。首先，提出了一种基于深度强化学习（DRL）的高精度UAV编队路径跟踪模型，并从距离和速度误差的角度设计了自适应权重的奖励函数。然后，利用综合感知与通信（ISAC）信号进行障碍检测，并基于信息级融合推导出障碍感测的克拉默- Rao下界（CRLB），在此基础上提出了变形成编队增强障碍位置估计（VFEO）算法。此外，设计了一种在线避障方案以解决稀疏奖励问题。最后，借助基于 null 空间方法（NSB）的行为方法，提出了分层次子任务融合策略。仿真实验结果表明了子任务算法和分层次融合策略的有效性和优越性。 

---
