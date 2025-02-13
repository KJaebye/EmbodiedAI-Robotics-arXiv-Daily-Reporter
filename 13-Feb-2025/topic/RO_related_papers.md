# Re$^3$Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation 

**Title (ZH)**: Re$^3$Sim: 通过3D逼真实时转换生成机器人操作的高保真模拟数据 

**Authors**: Xiaoshen Han, Minghuan Liu, Yilun Chen, Junqiu Yu, Xiaoyang Lyu, Yang Tian, Bolun Wang, Weinan Zhang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08645)  

**Abstract**: Real-world data collection for robotics is costly and resource-intensive, requiring skilled operators and expensive hardware. Simulations offer a scalable alternative but often fail to achieve sim-to-real generalization due to geometric and visual gaps. To address these challenges, we propose a 3D-photorealistic real-to-sim system, namely, RE$^3$SIM, addressing geometric and visual sim-to-real gaps. RE$^3$SIM employs advanced 3D reconstruction and neural rendering techniques to faithfully recreate real-world scenarios, enabling real-time rendering of simulated cross-view cameras within a physics-based simulator. By utilizing privileged information to collect expert demonstrations efficiently in simulation, and train robot policies with imitation learning, we validate the effectiveness of the real-to-sim-to-real pipeline across various manipulation task scenarios. Notably, with only simulated data, we can achieve zero-shot sim-to-real transfer with an average success rate exceeding 58%. To push the limit of real-to-sim, we further generate a large-scale simulation dataset, demonstrating how a robust policy can be built from simulation data that generalizes across various objects. Codes and demos are available at: this http URL. 

**Abstract (ZH)**: 真实世界数据收集对于机器人技术来说成本高且资源密集，需要 skilled 操作员和昂贵的硬件。模拟提供了可扩展的替代方案，但由于几何和视觉缺口，往往无法实现从模拟到现实的有效迁移。为了解决这些挑战，我们提出了一个3D-写实的真实世界到模拟系统，即 RE$^3$SIM，以解决几何和视觉上的模拟到现实的缺口。RE$^3$SIM 使用先进的3D重建和神经渲染技术忠实再现真实世界场景，在基于物理的模拟器中实时渲染模拟交叉视角相机。通过利用特权信息高效地在模拟中收集专家演示，并使用模仿学习训练机器人策略，我们验证了真实世界到模拟再到现实流水线在各种操作任务场景中的有效性。值得注意的是，仅使用模拟数据，我们可以实现零样本的模拟到现实迁移，成功率平均超过58%。为进一步推动真实世界到模拟的极限，我们还生成了一个大规模的模拟数据集，展示了如何从模拟数据中构建一个鲁棒策略，并在各种物体上有效推广。代码和演示可在：this http URL 获取。 

---
# Ground-Optimized 4D Radar-Inertial Odometry via Continuous Velocity Integration using Gaussian Process 

**Title (ZH)**: 基于地面优化的4D雷达-惯性里程计通过高斯过程进行连续速度积分 

**Authors**: Wooseong Yang, Hyesu Jang, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.08093)  

**Abstract**: Radar ensures robust sensing capabilities in adverse weather conditions, yet challenges remain due to its high inherent noise level. Existing radar odometry has overcome these challenges with strategies such as filtering spurious points, exploiting Doppler velocity, or integrating with inertial measurements. This paper presents two novel improvements beyond the existing radar-inertial odometry: ground-optimized noise filtering and continuous velocity preintegration. Despite the widespread use of ground planes in LiDAR odometry, imprecise ground point distributions of radar measurements cause naive plane fitting to fail. Unlike plane fitting in LiDAR, we introduce a zone-based uncertainty-aware ground modeling specifically designed for radar. Secondly, we note that radar velocity measurements can be better combined with IMU for a more accurate preintegration in radar-inertial odometry. Existing methods often ignore temporal discrepancies between radar and IMU by simplifying the complexities of asynchronous data streams with discretized propagation models. Tackling this issue, we leverage GP and formulate a continuous preintegration method for tightly integrating 3-DOF linear velocity with IMU, facilitating full 6-DOF motion directly from the raw measurements. Our approach demonstrates remarkable performance (less than 1% vertical drift) in public datasets with meticulous conditions, illustrating substantial improvement in elevation accuracy. The code will be released as open source for the community: this https URL. 

**Abstract (ZH)**: 雷达在恶劣天气条件下确保了 robust 的感知能力，但其固有的高噪声水平仍带来挑战。现有雷达里程计通过过滤虚假点、利用多普勒速度或与惯性测量融合等策略克服了这些挑战。本文提出两种超越现有雷达惯性里程计的新改进：地面优化噪声过滤和连续速度预积分。尽管在 LiDAR 里程计中广泛使用地面平面，但雷达测量的不精确地面点分布导致简单的平面拟合失效。不同于 LiDAR，我们引入基于区域的不确定性感知地面建模，专门设计用于雷达。其次，我们注意到雷达速度测量可以更好地与 IMU 结合，以提高雷达惯性里程计中的预积分精度。现有方法通常通过简化异步数据流的复杂性来忽略雷达和 IMU 之间的时序差异，仅用离散传播模型。为解决这一问题，我们利用高斯过程（GP）并提出一种连续预积分方法，使线性速度的 3-DOF 与 IMU 紧密集成，直接从原始测量中实现完整的 6-DOF 运动。我们的方法在公开数据集中的表现卓越（垂直漂移低于 1%），展示了显著的高程精度改进。代码将作为开源发布给社区：this https URL。 

---
# Fast and Safe Scheduling of Robots 

**Title (ZH)**: 快速且安全的机器人调度 

**Authors**: Duncan Adamson, Nathan Flaherty, Igor Potapov, Paul G. Spirakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.07851)  

**Abstract**: In this paper, we present an experimental analysis of a fast heuristic algorithm that was designed to generate a fast, collision-free schedule for a set of robots on a path graph. The experiments confirm the algorithm's effectiveness in producing collision-free schedules as well as achieving the optimal solution when all tasks assigned to the robots are of equal duration. Additionally, we provide an integer linear programming formulation that guarantees an optimal solution for this scheduling problem on any input graph, at the expense of significantly greater computational resources. We prove the correctness of our integer linear program. By comparing the solutions of these two algorithms, including the time required by the schedule itself, and the run time of each algorithm, we show that the heuristic algorithm is optimal or near optimal in nearly all cases, with a far faster run time than the integer linear program. 

**Abstract (ZH)**: 本文提出了一个快速启发式算法的实验分析，该算法用于生成路径图上一组机器人无碰撞调度方案。实验结果验证了该算法在所有任务持续时间相等时能够产生无碰撞调度方案并找到最优解。同时，我们提供了一个整数线性规划模型，该模型能在任何输入图上保证找到最优解，但需要消耗显著更多的计算资源。我们证明了该整数线性规划模型的正确性。通过对这两种算法的解及执行时间进行比较，我们证明了启发式算法在几乎所有情况下都是最优或近似最优的，并且执行速度远快于整数线性规划模型。 

---
