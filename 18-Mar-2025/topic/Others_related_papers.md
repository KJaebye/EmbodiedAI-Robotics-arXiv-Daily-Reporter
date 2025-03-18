# HybridGen: VLM-Guided Hybrid Planning for Scalable Data Generation of Imitation Learning 

**Title (ZH)**: HybridGen: 基于VLM的混合规划方法实现可扩展的 imitation 学习数据生成 

**Authors**: Wensheng Wang, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13171)  

**Abstract**: The acquisition of large-scale and diverse demonstration data are essential for improving robotic imitation learning generalization. However, generating such data for complex manipulations is challenging in real-world settings. We introduce HybridGen, an automated framework that integrates Vision-Language Model (VLM) and hybrid planning. HybridGen uses a two-stage pipeline: first, VLM to parse expert demonstrations, decomposing tasks into expert-dependent (object-centric pose transformations for precise control) and plannable segments (synthesizing diverse trajectories via path planning); second, pose transformations substantially expand the first-stage data. Crucially, HybridGen generates a large volume of training data without requiring specific data formats, making it broadly applicable to a wide range of imitation learning algorithms, a characteristic which we also demonstrate empirically across multiple algorithms. Evaluations across seven tasks and their variants demonstrate that agents trained with HybridGen achieve substantial performance and generalization gains, averaging a 5% improvement over state-of-the-art methods. Notably, in the most challenging task variants, HybridGen achieves significant improvement, reaching a 59.7% average success rate, significantly outperforming Mimicgen's 49.5%. These results demonstrating its effectiveness and practicality. 

**Abstract (ZH)**: 大规模多样示例数据的获取对于提高机器人模仿学习泛化能力至关重要。然而，在现实环境中生成复杂操作的此类数据具有挑战性。我们引入了HybridGen，这是一种结合了视觉语言模型(VLM)和混合规划的自动化框架。HybridGen采用两阶段流水线：首先使用VLM解析专家演示，将任务分解为专家依赖部分（对象为中心的姿态变换以实现精确控制）和可规划部分（通过路径规划合成多样化轨迹）；其次，姿态变换极大地扩展了第一阶段的数据。至关重要的是，HybridGen能够在不需要特定数据格式的情况下生成大量训练数据，使其广泛适用于多种模仿学习算法，我们也在多个算法上通过实验验证了这一点。跨七个任务及其变体的评估结果表明，使用HybridGen训练的智能体实现了显著的性能和泛化提升，平均提高了5%以上，超过了最新方法。特别地，在最具有挑战性的任务变体中，HybridGen实现了显著的改进，平均成功率达到了59.7%，远超Mimicgen的49.5%。这些结果证明了其有效性和实用性。 

---
# Rapid and Inexpensive Inertia Tensor Estimation from a Single Object Throw 

**Title (ZH)**: 单次物体投掷的快速和经济惯性张量估计 

**Authors**: Till M. Blaha, Mike M. Kuijper, Radu Pop, Ewoud J.J. Smeur  

**Link**: [PDF](https://arxiv.org/pdf/2503.13137)  

**Abstract**: The inertia tensor is an important parameter in many engineering fields, but measuring it can be cumbersome and involve multiple experiments or accurate and expensive equipment. We propose a method to measure the moment of inertia tensor of a rigid body from a single spinning throw, by attaching a small and inexpensive stand-alone measurement device consisting of a gyroscope, accelerometer and a reaction wheel. The method includes a compensation for the increase of moment of inertia due to adding the measurement device to the body, and additionally obtains the location of the centre of gravity of the body as an intermediate result. Experiments performed with known rigid bodies show that the mean accuracy is around 2\%. 

**Abstract (ZH)**: 一种基于单次旋转测量刚体惯性张量的方法 

---
# MIXPINN: Mixed-Material Simulations by Physics-Informed Neural Network 

**Title (ZH)**: MIXPINN: 由物理知情神经网络实现的混合材料模拟 

**Authors**: Xintian Yuan, Yunke Ao, Boqi Chen, Philipp Fuernstahl  

**Link**: [PDF](https://arxiv.org/pdf/2503.13123)  

**Abstract**: Simulating the complex interactions between soft tissues and rigid anatomy is critical for applications in surgical training, planning, and robotic-assisted interventions. Traditional Finite Element Method (FEM)-based simulations, while accurate, are computationally expensive and impractical for real-time scenarios. Learning-based approaches have shown promise in accelerating predictions but have fallen short in modeling soft-rigid interactions effectively. We introduce MIXPINN, a physics-informed Graph Neural Network (GNN) framework for mixed-material simulations, explicitly capturing soft-rigid interactions using graph-based augmentations. Our approach integrates Virtual Nodes (VNs) and Virtual Edges (VEs) to enhance rigid body constraint satisfaction while preserving computational efficiency. By leveraging a graph-based representation of biomechanical structures, MIXPINN learns high-fidelity deformations from FEM-generated data and achieves real-time inference with sub-millimeter accuracy. We validate our method in a realistic clinical scenario, demonstrating superior performance compared to baseline GNN models and traditional FEM methods. Our results show that MIXPINN reduces computational cost by an order of magnitude while maintaining high physical accuracy, making it a viable solution for real-time surgical simulation and robotic-assisted procedures. 

**Abstract (ZH)**: 混合材料间软硬交互的物理指导图神经网络模拟：一种提高实时外科模拟和机器人辅助手术效率的方法 

---
# MT-PCR: Leveraging Modality Transformation for Large-Scale Point Cloud Registration with Limited Overlap 

**Title (ZH)**: MT-PCR：利用模态转换进行大规模点云配准以利用有限重叠区域 

**Authors**: Yilong Wu, Yifan Duan, Yuxi Chen, Xinran Zhang, Yedong Shen, Jianmin Ji, Yanyong Zhang, Lu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12833)  

**Abstract**: Large-scale scene point cloud registration with limited overlap is a challenging task due to computational load and constrained data acquisition. To tackle these issues, we propose a point cloud registration method, MT-PCR, based on Modality Transformation. MT-PCR leverages a BEV capturing the maximal overlap information to improve the accuracy and utilizes images to provide complementary spatial features. Specifically, MT-PCR converts 3D point clouds to BEV images and eastimates correspondence by 2D image keypoints extraction and matching. Subsequently, the 2D correspondence estimates are then transformed back to 3D point clouds using inverse mapping. We have applied MT-PCR to Terrestrial Laser Scanning and Aerial Laser Scanning point cloud registration on the GrAco dataset, involving 8 low-overlap, square-kilometer scale registration scenarios. Experiments and comparisons with commonly used methods demonstrate that MT-PCR can achieve superior accuracy and robustness in large-scale scenes with limited overlap. 

**Abstract (ZH)**: 基于模态变换的大规模场景点云注册方法：MT-PCR 

---
# CDKFormer: Contextual Deviation Knowledge-Based Transformer for Long-Tail Trajectory Prediction 

**Title (ZH)**: CDKFormer：基于上下文偏差知识的 transformers 在长尾轨迹预测中的应用 

**Authors**: Yuansheng Lian, Ke Zhang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12695)  

**Abstract**: Predicting the future movements of surrounding vehicles is essential for ensuring the safe operation and efficient navigation of autonomous vehicles (AVs) in urban traffic environments. Existing vehicle trajectory prediction methods primarily focus on improving overall performance, yet they struggle to address long-tail scenarios effectively. This limitation often leads to poor predictions in rare cases, significantly increasing the risk of safety incidents. Taking Argoverse 2 motion forecasting dataset as an example, we first investigate the long-tail characteristics in trajectory samples from two perspectives, individual motion and group interaction, and deriving deviation features to distinguish abnormal from regular scenarios. On this basis, we propose CDKFormer, a Contextual Deviation Knowledge-based Transformer model for long-tail trajectory prediction. CDKFormer integrates an attention-based scene context fusion module to encode spatiotemporal interaction and road topology. An additional deviation feature fusion module is proposed to capture the dynamic deviations in the target vehicle status. We further introduce a dual query-based decoder, supported by a multi-stream decoder block, to sequentially decode heterogeneous scene deviation features and generate multimodal trajectory predictions. Extensive experiments demonstrate that CDKFormer achieves state-of-the-art performance, significantly enhancing prediction accuracy and robustness for long-tailed trajectories compared to existing methods, thus advancing the reliability of AVs in complex real-world environments. 

**Abstract (ZH)**: 基于上下文偏差知识的Transformer模型：面向长尾轨迹预测 

---
# Non-Normalized Solutions of Generalized Nash Equilibrium in Autonomous Racing 

**Title (ZH)**: 广义纳什均衡在自动驾驶赛车中的非正规解 

**Authors**: Mark Pustilnik, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.12002)  

**Abstract**: In dynamic games with shared constraints, Generalized Nash Equilibria (GNE) are often computed using the normalized solution concept, which assumes identical Lagrange multipliers for shared constraints across all players. While widely used, this approach excludes other potentially valuable GNE. This paper addresses the limitations of normalized solutions in racing scenarios through three key contributions. First, we highlight the shortcomings of normalized solutions with a simple racing example. Second, we propose a novel method based on the Mixed Complementarity Problem (MCP) formulation to compute non-normalized Generalized Nash Equilibria (GNE). Third, we demonstrate that our proposed method overcomes the limitations of normalized GNE solutions and enables richer multi-modal interactions in realistic racing scenarios. 

**Abstract (ZH)**: 在共享约束的动态博弈中，通用纳什均衡（GNE）通常使用正则化解的概念进行计算，该概念假定所有玩家对共享约束的拉格朗日乘子相同。虽然被广泛使用，但这种方法排除了其他潜在有价值的GNE。本文通过三个方面解决了正则化解在竞速场景中的局限性。首先，我们通过一个简单的竞速示例突显了正则化解的不足。其次，我们提出了一种基于混合互补问题（MCP）形式化的新方法来计算非正则化通用纳什均衡（GNE）。最后，我们证明了我们提出的方法克服了正则化GNE解的局限性，并能在现实竞速场景中实现更丰富的多模态互动。 

---
# Controllable Latent Diffusion for Traffic Simulation 

**Title (ZH)**: 可控潜在扩散交通模拟 

**Authors**: Yizhuo Xiao, Mustafa Suphi Erden, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11771)  

**Abstract**: The validation of autonomous driving systems benefits greatly from the ability to generate scenarios that are both realistic and precisely controllable. Conventional approaches, such as real-world test drives, are not only expensive but also lack the flexibility to capture targeted edge cases for thorough evaluation. To address these challenges, we propose a controllable latent diffusion that guides the training of diffusion models via reinforcement learning to automatically generate a diverse and controllable set of driving scenarios for virtual testing. Our approach removes the reliance on large-scale real-world data by generating complex scenarios whose properties can be finely tuned to challenge and assess autonomous vehicle systems. Experimental results show that our approach has the lowest collision rate of $0.098$ and lowest off-road rate of $0.096$, demonstrating superiority over existing baselines. The proposed approach significantly improves the realism, stability and controllability of the generated scenarios, enabling more nuanced safety evaluation of autonomous vehicles. 

**Abstract (ZH)**: 自主驾驶系统的验证极大地受益于能够生成既逼真又可精确控制的场景的能力。传统的方法，如实地测试驾驶，不仅成本高，而且缺乏灵活性，无法捕获有针对性的边界案例以进行彻底的评估。为了解决这些问题，我们提出了一种可控的潜在扩散方法，通过强化学习引导扩散模型的训练，以自动生成多样且可控的驾驶场景进行虚拟测试。该方法通过生成复杂场景，使其性质可以精细调整，从而挑战和评估自主车辆系统。实验结果表明，我们的方法具有最低的碰撞率（0.098）和最低的离路率（0.096），证明了其优于现有基线的方法。所提出的方法显著提高了生成场景的逼真性、稳定性和可控性，从而能够进行更为细致的自主车辆安全性评估。 

---
# NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models 

**Title (ZH)**: NuPlanQA：多视图驾驶场景理解的大规模数据集及基准测试 

**Authors**: Sung-Yeon Park, Can Cui, Yunsheng Ma, Ahmadreza Moradipari, Rohit Gupta, Kyungtae Han, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12772)  

**Abstract**: Recent advances in multi-modal large language models (MLLMs) have demonstrated strong performance across various domains; however, their ability to comprehend driving scenes remains less proven. The complexity of driving scenarios, which includes multi-view information, poses significant challenges for existing MLLMs. In this paper, we introduce NuPlanQA-Eval, a multi-view, multi-modal evaluation benchmark for driving scene understanding. To further support generalization to multi-view driving scenarios, we also propose NuPlanQA-1M, a large-scale dataset comprising 1M real-world visual question-answering (VQA) pairs. For context-aware analysis of traffic scenes, we categorize our dataset into nine subtasks across three core skills: Road Environment Perception, Spatial Relations Recognition, and Ego-Centric Reasoning. Furthermore, we present BEV-LLM, integrating Bird's-Eye-View (BEV) features from multi-view images into MLLMs. Our evaluation results reveal key challenges that existing MLLMs face in driving scene-specific perception and spatial reasoning from ego-centric perspectives. In contrast, BEV-LLM demonstrates remarkable adaptability to this domain, outperforming other models in six of the nine subtasks. These findings highlight how BEV integration enhances multi-view MLLMs while also identifying key areas that require further refinement for effective adaptation to driving scenes. To facilitate further research, we publicly release NuPlanQA at this https URL. 

**Abstract (ZH)**: Recent advances in 多模态大型语言模型（MLLMs）在各种领域中展示了强大的性能；然而，它们理解驾驶场景的能力仍然不够证明。驾驶场景的复杂性，包括多视角信息，对现有MLLMs构成了重大挑战。本文介绍了NuPlanQA-Eval，一个用于驾驶场景理解的多视角多模态评估基准。为进一步支持多视角驾驶场景的泛化，我们还提出了包含100万真实世界视觉问答（VQA）对的大规模数据集NuPlanQA-1M。为了进行交通场景的上下文感知分析，我们将数据集按三项核心技能分为九个子任务：道路环境感知、空间关系识别和以自身为中心的推理。此外，我们提出了BEV-LLM，将多视角图像中的鸟瞰图（BEV）特征集成到MLLMs中。我们的评估结果显示，现有MLLMs在驾驶场景特定感知和以自身为中心的空间推理方面面临关键挑战。相比之下，BEV-LLM在九个子任务中的六个子任务中表现出色，展现了其在这一领域的显著适应性。这些发现突显了BEV集成如何增强多视角MLLMs，并指出了需要进一步完善的关键领域，以便更有效地适应驾驶场景。为促进进一步研究，我们在此公开发布了NuPlanQA。 

---
# Polytope Volume Monitoring Problem: Formulation and Solution via Parametric Linear Program Based Control Barrier Function 

**Title (ZH)**: 多面体体积监控问题：基于参数线性规划屏障函数的建模与求解 

**Authors**: Shizhen Wu, Jinyang Dong, Xu Fang, Ning Sun, Yongchun Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12546)  

**Abstract**: Motivated by the latest research on feasible space monitoring of multiple control barrier functions (CBFs) as well as polytopic collision avoidance, this paper studies the Polytope Volume Monitoring (PVM) problem, whose goal is to design a control law for inputs of nonlinear systems to prevent the volume of some state-dependent polytope from decreasing to zero. Recent studies have explored the idea of applying Chebyshev ball method in optimization theory to solve the case study of PVM; however, the underlying difficulties caused by nonsmoothness have not been addressed. This paper continues the study on this topic, where our main contribution is to establish the relationship between nonsmooth CBF and parametric optimization theory through directional derivatives for the first time, so as to solve PVM problems more conveniently. In detail, inspired by Chebyshev ball approach, a parametric linear program (PLP) based nonsmooth barrier function candidate is established for PVM, and then, sufficient conditions for it to be a nonsmooth CBF are proposed, based on which a quadratic program (QP) based safety filter with guaranteed feasibility is proposed to address PVM problems. Finally, a numerical simulation example is given to show the efficiency of the proposed safety filter. 

**Abstract (ZH)**: 基于最新关于多个控制 barrier 函数（CBFs）可行空间监控及多面体避碰的最新研究，本文研究了多面体容积监控（PVM）问题，其目标是设计非线性系统输入的控制律以防止某些状态相关的多面体体积减小至零。近期研究探索了将切比雪夫球方法应用于优化理论以解决PVM案例，但底层的非光滑性导致的困难尚未解决。本文继续探讨这一课题，我们的主要贡献是通过方向导数首次建立了非光滑CBF与参数优化理论之间的关系，以便更方便地解决PVM问题。具体而言，受到切比雪夫球方法的启发，我们为PVM建立了一个基于参数线性规划（PLP）的非光滑 barrier 函数候选，并提出了使其成为非光滑CBF的充分条件，基于此提出了一种基于二次规划（QP）的安全滤波器以确保可行性来解决PVM问题。最后，给出了一个数值仿真例子以展示所提出的安全滤波器的效率。 

---
# Value Gradients with Action Adaptive Search Trees in Continuous (PO)MDPs 

**Title (ZH)**: 在连续(PO)MDPs中基于动作自适应搜索树的价值梯度方法 

**Authors**: Idan Lev-Yehudi, Michael Novitsky, Moran Barenboim, Ron Benchetrit, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2503.12181)  

**Abstract**: Solving Partially Observable Markov Decision Processes (POMDPs) in continuous state, action and observation spaces is key for autonomous planning in many real-world mobility and robotics applications. Current approaches are mostly sample based, and cannot hope to reach near-optimal solutions in reasonable time. We propose two complementary theoretical contributions. First, we formulate a novel Multiple Importance Sampling (MIS) tree for value estimation, that allows to share value information between sibling action branches. The novel MIS tree supports action updates during search time, such as gradient-based updates. Second, we propose a novel methodology to compute value gradients with online sampling based on transition likelihoods. It is applicable to MDPs, and we extend it to POMDPs via particle beliefs with the application of the propagated belief trick. The gradient estimator is computed in practice using the MIS tree with efficient Monte Carlo sampling. These two parts are combined into a new planning algorithm Action Gradient Monte Carlo Tree Search (AGMCTS). We demonstrate in a simulated environment its applicability, advantages over continuous online POMDP solvers that rely solely on sampling, and we discuss further implications. 

**Abstract (ZH)**: 在连续状态、动作和观测空间中解决部分可观测量决策过程（POMDPs）是许多实际移动性和机器人应用自主规划的关键。当前的方法主要是基于采样的，无法在合理的时间内达到接近最优的解决方案。我们提出了两个互补的理论贡献。首先，我们提出了一个新的多重重要性采样（MIS）树来估计价值，该树允许在行动分支之间共享价值信息。新的MIS树在搜索过程中支持基于梯度的行动更新。其次，我们提出了一种新的方法，利用转移概率进行在线采样来计算价值梯度。该方法适用于MDP，并通过粒子信念将其扩展到POMDP。梯度估计器在实践中使用带有效率的蒙特卡洛采样的MIS树进行计算。这两部分结合成一个新的规划算法：行动梯度蒙特卡洛树搜索（AGMCTS）。我们在模拟环境中展示了其适用性，与仅依赖采样的连续在线POMDP求解器相比的优势，并讨论了进一步的含义。 

---
# Learning Closed-Loop Parametric Nash Equilibria of Multi-Agent Collaborative Field Coverage 

**Title (ZH)**: 多代理协作场域覆盖的闭环参数纳什均衡学习 

**Authors**: Jushan Chen, Santiago Paternain  

**Link**: [PDF](https://arxiv.org/pdf/2503.11829)  

**Abstract**: Multi-agent reinforcement learning is a challenging and active field of research due to the inherent nonstationary property and coupling between agents. A popular approach to modeling the multi-agent interactions underlying the multi-agent RL problem is the Markov Game. There is a special type of Markov Game, termed Markov Potential Game, which allows us to reduce the Markov Game to a single-objective optimal control problem where the objective function is a potential function. In this work, we prove that a multi-agent collaborative field coverage problem, which is found in many engineering applications, can be formulated as a Markov Potential Game, and we can learn a parameterized closed-loop Nash Equilibrium by solving an equivalent single-objective optimal control problem. As a result, our algorithm is 10x faster during training compared to a game-theoretic baseline and converges faster during policy execution. 

**Abstract (ZH)**: 多智能体强化学习是一种由于固有的非平稳性和智能体间的耦合而具有挑战性和活跃的研究领域。一种用于建模多智能体强化学习问题下智能体间交互的方法是马尔科夫游戏。其中，一种特殊的马尔科夫游戏称为马尔科夫势能游戏，允许我们将马尔科夫游戏转换为单一目标的最优控制问题，其中目标函数为势能函数。在本文中，我们证明了一种存在于许多工程应用中的多智能体合作区域覆盖问题可以形式化为马尔科夫势能游戏，并通过求解等价的单一目标最优控制问题学习了一个参数化的闭环纳什均衡。因此，与博弈论基线相比，我们的算法在训练过程中快10倍，并且在策略执行过程中收敛更快。 

---
# Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation 

**Title (ZH)**: 工业级传感器模拟 via 高斯点绘：一种模块化框架，用于可扩展编辑和全流程验证 

**Authors**: Xianming Zeng, Sicong Du, Qifeng Chen, Lizhe Liu, Haoyu Shu, Jiaxuan Gao, Jiarun Liu, Jiulong Xu, Jianyun Xu, Mingxia Chen, Yiru Zhao, Peng Chen, Yapeng Xue, Chunming Zhao, Sheng Yang, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11731)  

**Abstract**: Sensor simulation is pivotal for scalable validation of autonomous driving systems, yet existing Neural Radiance Fields (NeRF) based methods face applicability and efficiency challenges in industrial workflows. This paper introduces a Gaussian Splatting (GS) based system to address these challenges: We first break down sensor simulator components and analyze the possible advantages of GS over NeRF. Then in practice, we refactor three crucial components through GS, to leverage its explicit scene representation and real-time rendering: (1) choosing the 2D neural Gaussian representation for physics-compliant scene and sensor modeling, (2) proposing a scene editing pipeline to leverage Gaussian primitives library for data augmentation, and (3) coupling a controllable diffusion model for scene expansion and harmonization. We implement this framework on a proprietary autonomous driving dataset supporting cameras and LiDAR sensors. We demonstrate through ablation studies that our approach reduces frame-wise simulation latency, achieves better geometric and photometric consistency, and enables interpretable explicit scene editing and expansion. Furthermore, we showcase how integrating such a GS-based sensor simulator with traffic and dynamic simulators enables full-stack testing of end-to-end autonomy algorithms. Our work provides both algorithmic insights and practical validation, establishing GS as a cornerstone for industrial-grade sensor simulation. 

**Abstract (ZH)**: 基于高斯点云的传感器模拟在自主驾驶系统工业级验证中的应用与挑战克服：构建高效可解析场景编辑与扩展的传感器模拟系统 

---
# Fed-Joint: Joint Modeling of Nonlinear Degradation Signals and Failure Events for Remaining Useful Life Prediction using Federated Learning 

**Title (ZH)**: Fed-Joint：联邦学习下的非线性退化信号与故障事件联合建模以预测剩余使用寿命 

**Authors**: Cheoljoon Jeong, Xubo Yue, Seokhyun Chung  

**Link**: [PDF](https://arxiv.org/pdf/2503.13404)  

**Abstract**: Many failure mechanisms of machinery are closely related to the behavior of condition monitoring (CM) signals. To achieve a cost-effective preventive maintenance strategy, accurate remaining useful life (RUL) prediction based on the signals is of paramount importance. However, the CM signals are often recorded at different factories and production lines, with limited amounts of data. Unfortunately, these datasets have rarely been shared between the sites due to data confidentiality and ownership issues, a lack of computing and storage power, and high communication costs associated with data transfer between sites and a data center. Another challenge in real applications is that the CM signals are often not explicitly specified \textit{a priori}, meaning that existing methods, which often usually a parametric form, may not be applicable. To address these challenges, we propose a new prognostic framework for RUL prediction using the joint modeling of nonlinear degradation signals and time-to-failure data within a federated learning scheme. The proposed method constructs a nonparametric degradation model using a federated multi-output Gaussian process and then employs a federated survival model to predict failure times and probabilities for in-service machinery. The superiority of the proposed method over other alternatives is demonstrated through comprehensive simulation studies and a case study using turbofan engine degradation signal data that include run-to-failure events. 

**Abstract (ZH)**: 机械设备的许多失效机理与条件监控信号的行为密切相关。为了实现有效的预防性维护策略，基于信号的准确剩余使用寿命（RUL）预测至关重要。然而，这些信号往往在不同工厂和生产线上以有限的数据量记录。这些数据集在不同站点之间由于数据保密性和所有权问题、计算和存储能力不足以及数据传输和数据中心之间的高昂通信成本，很少被共享。在实际应用中，另一个挑战是条件监控信号经常不是先验明确指定的，这意味着现有的方法，通常是以参数形式表示的，可能不适用。为应对这些挑战，我们提出了一种新的基于联邦学习的剩余使用寿命预测框架，该框架结合了非线性退化信号和失效时间数据的联合模型构建方法。该方法利用联邦多输出高斯过程构建非参数退化模型，然后使用联邦生存模型预测在用设备的失效时间和概率。通过全面的模拟研究和基于涡扇发动机退化信号数据的案例研究（包括运行至失效事件），展示了所提出方法的优势。 

---
# Rapfi: Distilling Efficient Neural Network for the Game of Gomoku 

**Title (ZH)**: Rapfi: 一种用于五子棋游戏的高效神经网络蒸馏方法 

**Authors**: Zhanggen Jin, Haobin Duan, Zhiyang Hang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13178)  

**Abstract**: Games have played a pivotal role in advancing artificial intelligence, with AI agents using sophisticated techniques to compete. Despite the success of neural network based game AIs, their performance often requires significant computational resources. In this paper, we present Rapfi, an efficient Gomoku agent that outperforms CNN-based agents in limited computation environments. Rapfi leverages a compact neural network with a pattern-based codebook distilled from CNNs, and an incremental update scheme that minimizes computation when input changes are minor. This new network uses computation that is orders of magnitude less to reach a similar accuracy of much larger neural networks such as Resnet. Thanks to our incremental update scheme, depth-first search methods such as the alpha-beta search can be significantly accelerated. With a carefully tuned evaluation and search, Rapfi reached strength surpassing Katagomo, the strongest open-source Gomoku AI based on AlphaZero's algorithm, under limited computational resources where accelerators like GPUs are absent. Rapfi ranked first among 520 Gomoku agents on Botzone and won the championship in GomoCup 2024. 

**Abstract (ZH)**: Rapfi：在有限计算资源环境中超越CNN基游戏AI的高效五子棋代理 

---
# Intra-neuronal attention within language models Relationships between activation and semantics 

**Title (ZH)**: 语言模型内的神经元注意力激活与语义之间的关系 

**Authors**: Michael Pichat, William Pogrund, Paloma Pichat, Armanouche Gasparian, Samuel Demarchi, Corbet Alois Georgeon, Michael Veillet-Guillem  

**Link**: [PDF](https://arxiv.org/pdf/2503.12992)  

**Abstract**: This study investigates the ability of perceptron-type neurons in language models to perform intra-neuronal attention; that is, to identify different homogeneous categorical segments within the synthetic thought category they encode, based on a segmentation of specific activation zones for the tokens to which they are particularly responsive. The objective of this work is therefore to determine to what extent formal neurons can establish a homomorphic relationship between activation-based and categorical segmentations. The results suggest the existence of such a relationship, albeit tenuous, only at the level of tokens with very high activation levels. This intra-neuronal attention subsequently enables categorical restructuring processes at the level of neurons in the following layer, thereby contributing to the progressive formation of high-level categorical abstractions. 

**Abstract (ZH)**: 本研究探讨了语言模型中感知器型神经元执行内神经注意的能力；即，根据它们特别响应的标记的特定激活区域对划分，识别编码的合成思想类别中不同的同质类别片段。本工作的目标因此是确定形式神经元在基于激活和类别划分之间建立同构关系方面的程度。结果表明，只有在激活水平非常高的标记级别上，才可能存在这种关系，尽管这种关系是脆弱的。这种内神经注意随后使下一层神经元中的类别重构过程成为可能，从而有助于高级类别抽象的逐步形成。 

---
# Verification Learning: Make Unsupervised Neuro-Symbolic System Feasible 

**Title (ZH)**: 验证学习：使无监督神经符号系统可行 

**Authors**: Lin-Han Jia, Wen-Chao Hu, Jie-Jing Shao, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12917)  

**Abstract**: The current Neuro-Symbolic (NeSy) Learning paradigm suffers from an over-reliance on labeled data. If we completely disregard labels, it leads to less symbol information, a larger solution space, and more shortcuts-issues that current Nesy systems cannot resolve. This paper introduces a novel learning paradigm, Verification Learning (VL), which addresses this challenge by transforming the label-based reasoning process in Nesy into a label-free verification process. VL achieves excellent learning results solely by relying on unlabeled data and a function that verifies whether the current predictions conform to the rules. We formalize this problem as a Constraint Optimization Problem (COP) and propose a Dynamic combinatorial Sorting (DCS) algorithm that accelerates the solution by reducing verification attempts, effectively lowering computational costs to the level of a Constraint Satisfaction Problem (CSP). To further enhance performance, we introduce a prior alignment method to address potential shortcuts. Our theoretical analysis points out which tasks in Nesy systems can be completed without labels and explains why rules can replace infinite labels, such as in addition, for some tasks, while for others, like Sudoku, the rules have no effect. We validate the proposed framework through several fully unsupervised tasks including addition, sort, match, and chess, each showing significant performance and efficiency improvements. 

**Abstract (ZH)**: 基于验证的学习（VL）：无标示数据下的神经符号学习范式 

---
# Analyzing sequential activity and travel decisions with interpretable deep inverse reinforcement learning 

**Title (ZH)**: 基于可解释的深度逆强化学习分析序贯活动和出行决策 

**Authors**: Yuebing Liang, Shenhao Wang, Jiangbo Yu, Zhan Zhao, Jinhua Zhao, Sandy Pentland  

**Link**: [PDF](https://arxiv.org/pdf/2503.12761)  

**Abstract**: Travel demand modeling has shifted from aggregated trip-based models to behavior-oriented activity-based models because daily trips are essentially driven by human activities. To analyze the sequential activity-travel decisions, deep inverse reinforcement learning (DIRL) has proven effective in learning the decision mechanisms by approximating a reward function to represent preferences and a policy function to replicate observed behavior using deep neural networks (DNNs). However, most existing research has focused on using DIRL to enhance only prediction accuracy, with limited exploration into interpreting the underlying decision mechanisms guiding sequential decision-making. To address this gap, we introduce an interpretable DIRL framework for analyzing activity-travel decision processes, bridging the gap between data-driven machine learning and theory-driven behavioral models. Our proposed framework adapts an adversarial IRL approach to infer the reward and policy functions of activity-travel behavior. The policy function is interpreted through a surrogate interpretable model based on choice probabilities from the policy function, while the reward function is interpreted by deriving both short-term rewards and long-term returns for various activity-travel patterns. Our analysis of real-world travel survey data reveals promising results in two key areas: (i) behavioral pattern insights from the policy function, highlighting critical factors in decision-making and variations among socio-demographic groups, and (ii) behavioral preference insights from the reward function, indicating the utility individuals gain from specific activity sequences. 

**Abstract (ZH)**: 基于行为的活动-行程决策过程的可解释深度逆强化学习框架 

---
# Understanding Driver Cognition and Decision-Making Behaviors in High-Risk Scenarios: A Drift Diffusion Perspective 

**Title (ZH)**: 高风险场景中驾驶员认知与决策行为的理解：从漂移扩散视角探析 

**Authors**: Heye Huang, Zheng Li, Hao Cheng, Haoran Wang, Junkai Jiang, Xiaopeng Li, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2503.12637)  

**Abstract**: Ensuring safe interactions between autonomous vehicles (AVs) and human drivers in mixed traffic systems remains a major challenge, particularly in complex, high-risk scenarios. This paper presents a cognition-decision framework that integrates individual variability and commonalities in driver behavior to quantify risk cognition and model dynamic decision-making. First, a risk sensitivity model based on a multivariate Gaussian distribution is developed to characterize individual differences in risk cognition. Then, a cognitive decision-making model based on the drift diffusion model (DDM) is introduced to capture common decision-making mechanisms in high-risk environments. The DDM dynamically adjusts decision thresholds by integrating initial bias, drift rate, and boundary parameters, adapting to variations in speed, relative distance, and risk sensitivity to reflect diverse driving styles and risk preferences. By simulating high-risk scenarios with lateral, longitudinal, and multidimensional risk sources in a driving simulator, the proposed model accurately predicts cognitive responses and decision behaviors during emergency maneuvers. Specifically, by incorporating driver-specific risk sensitivity, the model enables dynamic adjustments of key DDM parameters, allowing for personalized decision-making representations in diverse scenarios. Comparative analysis with IDM, Gipps, and MOBIL demonstrates that DDM more precisely captures human cognitive processes and adaptive decision-making in high-risk scenarios. These findings provide a theoretical basis for modeling human driving behavior and offer critical insights for enhancing AV-human interaction in real-world traffic environments. 

**Abstract (ZH)**: 确保自动驾驶车辆（AVs）与人类驾驶员在混合交通系统中的安全互动仍然是一个主要挑战，特别是在复杂高风险场景中。本文提出了一种认知-决策框架，整合了驾驶员行为中的个体差异与共同特征以量化风险认知并建模动态决策。首先，基于多元高斯分布的风险敏感性模型被开发以描述个体间的风险认知差异。然后，引入基于漂移扩散模型（DDM）的认知决策模型以捕捉高风险环境中的共同决策机制。DDM动态调整决策阈值，通过整合初始偏差、漂移速率和边界参数，适应速度、相对距离和风险敏感性的变化，以反映不同的驾驶风格和风险偏好。通过在驾驶模拟器中模拟具有横向、纵向和多维度风险源的高风险场景，所提出的模型能够准确预测紧急操作中的认知反应和决策行为。具体而言，通过纳入驾驶员特定的风险敏感性，模型能够实现关键DDM参数的动态调整，在不同场景中提供个性化的决策表示。与IDM、Gipps和MOBIL模型的比较分析表明，DDM更精准地捕捉了高风险场景中的人类认知过程和适应性决策。这些发现为建模人类驾驶行为提供了理论基础，并为进一步优化AV-人类交互在实际交通环境中的表现提供了关键见解。 

---
# Automated Planning for Optimal Data Pipeline Instantiation 

**Title (ZH)**: 自动规划以获得最优数据管道实例化 

**Authors**: Leonardo Rosa Amado, Adriano Vogel, Dalvan Griebler, Gabriel Paludo Licks, Eric Simon, Felipe Meneguzzi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12626)  

**Abstract**: Data pipeline frameworks provide abstractions for implementing sequences of data-intensive transformation operators, automating the deployment and execution of such transformations in a cluster. Deploying a data pipeline, however, requires computing resources to be allocated in a data center, ideally minimizing the overhead for communicating data and executing operators in the pipeline while considering each operator's execution requirements. In this paper, we model the problem of optimal data pipeline deployment as planning with action costs, where we propose heuristics aiming to minimize total execution time. Experimental results indicate that the heuristics can outperform the baseline deployment and that a heuristic based on connections outperforms other strategies. 

**Abstract (ZH)**: 数据管道部署优化建模为带有动作成本的规划问题：基于连接的启发式优于其他策略 

---
# FedGAI: Federated Style Learning with Cloud-Edge Collaboration for Generative AI in Fashion Design 

**Title (ZH)**: FedGAI：基于云边协作的服装设计生成式人工智能风格学习 federated learning 

**Authors**: Mingzhu Wu, Jianan Jiang, Xinglin Li, Hanhui Deng, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12389)  

**Abstract**: Collaboration can amalgamate diverse ideas, styles, and visual elements, fostering creativity and innovation among different designers. In collaborative design, sketches play a pivotal role as a means of expressing design creativity. However, designers often tend to not openly share these meticulously crafted sketches. This phenomenon of data island in the design area hinders its digital transformation under the third wave of AI. In this paper, we introduce a Federated Generative Artificial Intelligence Clothing system, namely FedGAI, employing federated learning to aid in sketch design. FedGAI is committed to establishing an ecosystem wherein designers can exchange sketch styles among themselves. Through FedGAI, designers can generate sketches that incorporate various designers' styles from their peers, drawing inspiration from collaboration without the need for data disclosure or upload. Extensive performance evaluations indicate that our FedGAI system can produce multi-styled sketches of comparable quality to human-designed ones while significantly enhancing efficiency compared to hand-drawn sketches. 

**Abstract (ZH)**: 协作可以融合多样化的理念、风格和视觉元素，促进不同设计师之间的创新。在协作设计中，草图是表达设计理念的关键手段。然而，设计师往往不愿意公开分享这些精心制作的草图。设计领域的数据孤岛阻碍了在第三次人工智能浪潮下的数字化转型。本文介绍了一种基于联邦学习的生成型人工智能服装系统，即FedGAI，旨在帮助设计师进行草图设计。FedGAI致力于建立一个设计师之间可以交流草图风格的生态系统。通过FedGAI，设计师可以从 peers 的多种设计风格中汲取灵感生成草图，而无需披露或上传数据。大量的性能评估表明，我们的FedGAI系统可以生成与人工设计质量相当、在效率上显著优于手绘草图的多风格草图。 

---
# IPCGRL: Language-Instructed Reinforcement Learning for Procedural Level Generation 

**Title (ZH)**: IPCGRL：基于语言的强化学习程序化水平生成 

**Authors**: In-Chang Baek, Sung-Hyun Kim, Seo-yung Lee, Dong-Hyun Lee, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12358)  

**Abstract**: Recent research has highlighted the significance of natural language in enhancing the controllability of generative models. While various efforts have been made to leverage natural language for content generation, research on deep reinforcement learning (DRL) agents utilizing text-based instructions for procedural content generation remains limited. In this paper, we propose IPCGRL, an instruction-based procedural content generation method via reinforcement learning, which incorporates a sentence embedding model. IPCGRL fine-tunes task-specific embedding representations to effectively compress game-level conditions. We evaluate IPCGRL in a two-dimensional level generation task and compare its performance with a general-purpose embedding method. The results indicate that IPCGRL achieves up to a 21.4% improvement in controllability and a 17.2% improvement in generalizability for unseen instructions. Furthermore, the proposed method extends the modality of conditional input, enabling a more flexible and expressive interaction framework for procedural content generation. 

**Abstract (ZH)**: 基于指令的强化学习程序化内容生成方法（IPCGRL） 

---
# A Transformer-based survival model for prediction of all-cause mortality in heart failure patients: a multi-cohort study 

**Title (ZH)**: 基于变压器的生存模型在心力衰竭患者全因死亡率预测中的应用：一项多队列研究 

**Authors**: Shishir Rao, Nouman Ahmed, Gholamreza Salimi-Khorshidi, Christopher Yau, Huimin Su, Nathalie Conrad, Folkert W Asselbergs, Mark Woodward, Rod Jackson, John GF Cleland, Kazem Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12317)  

**Abstract**: We developed and validated TRisk, a Transformer-based AI model predicting 36-month mortality in heart failure patients by analysing temporal patient journeys from UK electronic health records (EHR). Our study included 403,534 heart failure patients (ages 40-90) from 1,418 English general practices, with 1,063 practices for model derivation and 355 for external validation. TRisk was compared against the MAGGIC-EHR model across various patient subgroups. With median follow-up of 9 months, TRisk achieved a concordance index of 0.845 (95% confidence interval: [0.841, 0.849]), significantly outperforming MAGGIC-EHR's 0.728 (0.723, 0.733) for predicting 36-month all-cause mortality. TRisk showed more consistent performance across sex, age, and baseline characteristics, suggesting less bias. We successfully adapted TRisk to US hospital data through transfer learning, achieving a C-index of 0.802 (0.789, 0.816) with 21,767 patients. Explainability analyses revealed TRisk captured established risk factors while identifying underappreciated predictors like cancers and hepatic failure that were important across both cohorts. Notably, cancers maintained strong prognostic value even a decade after diagnosis. TRisk demonstrated well-calibrated mortality prediction across both healthcare systems. Our findings highlight the value of tracking longitudinal health profiles and revealed risk factors not included in previous expert-driven models. 

**Abstract (ZH)**: 基于Transformer的TRisk模型：预测心力衰竭患者36个月全因 Mortality 的开发与验证 

---
# Aristotle's Original Idea: For and Against Logic in the era of AI 

**Title (ZH)**: 亚里士多德的原创理念：在人工智能时代为逻辑与反对逻辑之争 

**Authors**: Antonis C. Kakas  

**Link**: [PDF](https://arxiv.org/pdf/2503.12161)  

**Abstract**: Aristotle is generally accepted as the father of logic. The ideas that he raised in his study of logical reasoning carried the development of science over the centuries. Today, in the era of AI, this title of the fatherhood of logic has a renewed significance. Behind it lies his original idea that human reasoning could be studied as a process and that perhaps there exist universal systems of reasoning that underly all human reasoning irrespective of the content of what we are reasoning about. In this article, we look into Aristotle's work on human thought, his work on reasoning itself but also on how it relates to science and human endeavor more generally, from a modern perspective of Artificial Intelligence and ask if this can help enlighten our understanding of AI and Science more generally. 

**Abstract (ZH)**: 亚里士多德一般被视为逻辑之父。他在逻辑推理研究中提出的观点推动了科学数个世纪的发展。在人工智能时代，这一“逻辑之父”的称号获得了新的意义。背后蕴含着他关于人类推理可以被视为一个过程的原创思想，以及可能存在着适用于所有人类推理的普遍推理系统的观点。在本文中，我们从人工智能的现代视角审视亚里士多德关于人类思维、推理本身及其与更广泛的科学和人类努力的关系的工作，探讨这是否能帮助我们更深入地理解人工智能和科学。 

---
# Counterfactual Realizability 

**Title (ZH)**: 反事实可实现性 

**Authors**: Arvind Raghavan, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2503.11870)  

**Abstract**: It is commonly believed that, in a real-world environment, samples can only be drawn from observational and interventional distributions, corresponding to Layers 1 and 2 of the Pearl Causal Hierarchy. Layer 3, representing counterfactual distributions, is believed to be inaccessible by definition. However, Bareinboim, Forney, and Pearl (2015) introduced a procedure that allows an agent to sample directly from a counterfactual distribution, leaving open the question of what other counterfactual quantities can be estimated directly via physical experimentation. We resolve this by introducing a formal definition of realizability, the ability to draw samples from a distribution, and then developing a complete algorithm to determine whether an arbitrary counterfactual distribution is realizable given fundamental physical constraints, such as the inability to go back in time and subject the same unit to a different experimental condition. We illustrate the implications of this new framework for counterfactual data collection using motivating examples from causal fairness and causal reinforcement learning. While the baseline approach in these motivating settings typically follows an interventional or observational strategy, we show that a counterfactual strategy provably dominates both. 

**Abstract (ZH)**: 普遍认为，在现实环境中，样本只能来自于观察性和干预性分布，对应佩尔因果层次结构中的第1层和第2层。第3层的反事实分布被认为由于定义原因而不可访问。然而，Bareinboim、Forney和Pearl（2015）提出了一种方法，使代理可以直接从反事实分布中抽样，从而提出了通过物理实验直接估计其他反事实量的问题。我们通过引入可实现性的正式定义——从分布中抽样的能力——并开发了一种完全算法来确定在基本物理约束（如不能时间倒流和对同一单位施加不同实验条件）下，任意反事实分布是否可实现。我们利用因果公平性和因果强化学习中的激励性示例，阐述了这一新框架对反事实数据收集的影响。虽然这些激励性场景下基线方法通常遵循干预性或观察性策略，我们展示了反事实策略在理论上优于两者。 

---
# Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning 

**Title (ZH)**: 幻象的安全性：虚假相关性如何损害VLM安全性微调 

**Authors**: Yiwei Chen, Yuguang Yao, Yihua Zhang, Bingquan Shen, Gaowen Liu, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11832)  

**Abstract**: Recent vision-language models (VLMs) have made remarkable strides in generative modeling with multimodal inputs, particularly text and images. However, their susceptibility to generating harmful content when exposed to unsafe queries raises critical safety concerns. While current alignment strategies primarily rely on supervised safety fine-tuning with curated datasets, we identify a fundamental limitation we call the "safety mirage" where supervised fine-tuning inadvertently reinforces spurious correlations between superficial textual patterns and safety responses, rather than fostering deep, intrinsic mitigation of harm. We show that these spurious correlations leave fine-tuned VLMs vulnerable even to a simple one-word modification-based attack, where substituting a single word in text queries with a spurious correlation-inducing alternative can effectively bypass safeguards. Additionally, these correlations contribute to the over prudence, causing fine-tuned VLMs to refuse benign queries unnecessarily. To address this issue, we show machine unlearning (MU) as a powerful alternative to supervised safety fine-tuning as it avoids biased feature-label mappings and directly removes harmful knowledge from VLMs while preserving their general capabilities. Extensive evaluations across safety benchmarks show that under one-word attacks, MU-based alignment reduces the attack success rate by up to 60.17% and cuts unnecessary rejections by over 84.20%. Codes are available at this https URL. WARNING: There exist AI generations that may be offensive in nature. 

**Abstract (ZH)**: 近期的多模态视觉-语言模型（VLMs）在基于文本和图像的生成建模方面取得了显著进展。然而，它们在接触到不安全查询时生成有害内容的倾向引发了重要的安全问题。当前的对齐策略主要依赖于监督安全微调和精心策划的数据集，但我们发现了一个根本性的局限性，我们称之为“安全幻象”，即监督微调无意中加强了表面上的文字模式与安全响应之间的虚假相关性，而非培养深层次的本质性的有害内容缓解机制。我们展示了这些虚假相关性使得经过微调的VLMs即使在简单的单词修改攻击下也容易受损，其中通过用含有虚假相关性诱导的替代词替换文本查询中的单个词可以有效绕过安全防护。此外，这些相关性导致过度谨慎，使经过微调的VLMs对本不应拒绝的查询进行不必要的拒绝。为解决这一问题，我们展示了机器遗忘（MU）作为一种强大的替代方法，因为它避免了有偏的特征-标签映射，并直接从VLMs中移除有害知识，同时保留其一般能力。在多个安全基准上的广泛评估显示，在单词攻击下，基于MU的对齐将攻击成功率降低了高达60.17%，并减少了超过84.20%的不必要的拒绝。相关的代码可在以下链接获取：这个 https URL。警告：可能存在具有不礼貌性质的AI生成内容。 

---
# An Algebraic Approach to Moralisation and Triangulation of Probabilistic Graphical Models 

**Title (ZH)**: 概率图形模型的道德化与三角化代数方法 

**Authors**: Antonio Lorenzin, Fabio Zanasi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11820)  

**Abstract**: Moralisation and Triangulation are transformations allowing to switch between different ways of factoring a probability distribution into a graphical model. Moralisation allows to view a Bayesian network (a directed model) as a Markov network (an undirected model), whereas triangulation works in the opposite direction. We present a categorical framework where these transformations are modelled as functors between a category of Bayesian networks and one of Markov networks. The two kinds of network (the objects of these categories) are themselves represented as functors, from a `syntax' domain to a `semantics' codomain. Notably, moralisation and triangulation are definable inductively on such syntax, and operate as a form of functor pre-composition. This approach introduces a modular, algebraic perspective in the theory of probabilistic graphical models. 

**Abstract (ZH)**: 道德化和三角化是允许在不同方式下分解概率分布到图形模型之间进行转换的变换。道德化允许将有向模型（贝叶斯网络）视作无向模型（马尔科夫网络），而三角化则反之。我们提出一种范畴论框架，其中这些变换被表示为贝叶斯网络范畴与马尔科夫网络范畴之间的函子。这两种类型的网络（这些范畴的对象）本身被表示为从“语法”领域到“语义”陪域的函子。值得注意的是，道德化和三角化可以在这种语法上归纳定义，并且作为函子预合成的形式进行操作。这种方法为概率图形模型理论引入了一种模块化和代数化的视角。 

---
# PUBLICSPEAK: Hearing the Public with a Probabilistic Framework in Local Government 

**Title (ZH)**: PUBLICSPEAK：基于概率框架倾听公众在地方政府中的声音 

**Authors**: Tianliang Xu, Eva Maxfield Brown, Dustin Dwyer, Sabina Tomkins  

**Link**: [PDF](https://arxiv.org/pdf/2503.11743)  

**Abstract**: Local governments around the world are making consequential decisions on behalf of their constituents, and these constituents are responding with requests, advice, and assessments of their officials at public meetings. So many small meetings cannot be covered by traditional newsrooms at scale. We propose PUBLICSPEAK, a probabilistic framework which can utilize meeting structure, domain knowledge, and linguistic information to discover public remarks in local government meetings. We then use our approach to inspect the issues raised by constituents in 7 cities across the United States. We evaluate our approach on a novel dataset of local government meetings and find that PUBLICSPEAK improves over state-of-the-art by 10% on average, and by up to 40%. 

**Abstract (ZH)**: 全球各地的地方政府正代表其选民做出重要决策，而这些选民则通过在公共会议上提出请求、提供建议和评估官员表现作出回应。由于众多小型会议无法被传统新闻机构大规模覆盖，我们提出一种概率性框架——PUBLICSPEAK，该框架能够利用会议结构、领域知识和语言信息来发现地方政府部门会议中的公众言论。我们随后利用该方法检查了美国7个城市中民众提出的议题。我们在一个新型的地方政府会议数据集上评估了该方法，并发现与现有最佳方法相比，PUBLICSPEAK在平均性能上提高了10%，最高提升了40%。 

---
# Physics-based simulation ontology: an ontology to support modelling and reuse of data for physics-based simulation 

**Title (ZH)**: 基于物理的仿真本体：一种支持物理基于仿真数据建模和重用的本体 

**Authors**: Hyunmin Cheong, Adrian Butscher  

**Link**: [PDF](https://arxiv.org/pdf/2503.11723)  

**Abstract**: The current work presents an ontology developed for physics-based simulation in engineering design, called Physics-based Simulation Ontology (PSO). The purpose of the ontology is to assist in modelling the physical phenomenon of interest in a veridical manner, while capturing the necessary and reusable information for physics-based simulation solvers. The development involved extending an existing upper ontology, Basic Formal Ontology (BFO), to define lower-level terms of PSO. PSO has two parts: PSO-Physics, which consists of terms and relations used to model physical phenomena based on the perspective of classical mechanics involving partial differential equations, and PSO-Sim, which consists of terms used to represent the information artefacts that are about the physical phenomena modelled with PSO-Physics. The former terms are used to model the physical phenomenon of interest independent of solver-specific interpretations, which can be reused across different solvers, while the latter terms are used to instantiate solver-specific input data. A case study involving two simulation solvers was conducted to demonstrate this capability of PSO. Discussion around the benefits and limitations of using BFO for the current work is also provided, which should be valuable for any future work that extends an existing upper ontology to develop ontologies for engineering applications. 

**Abstract (ZH)**: 基于物理的工程设计仿真本体（PSO）发展研究 

---
# The Relativity of Causal Knowledge 

**Title (ZH)**: 因果知识的相对性 

**Authors**: Gabriele D'Acunto, Claudio Battiloro  

**Link**: [PDF](https://arxiv.org/pdf/2503.11718)  

**Abstract**: Recent advances in artificial intelligence reveal the limits of purely predictive systems and call for a shift toward causal and collaborative reasoning. Drawing inspiration from the revolution of Grothendieck in mathematics, we introduce the relativity of causal knowledge, which posits structural causal models (SCMs) are inherently imperfect, subjective representations embedded within networks of relationships. By leveraging category theory, we arrange SCMs into a functor category and show that their observational and interventional probability measures naturally form convex structures. This result allows us to encode non-intervened SCMs with convex spaces of probability measures. Next, using sheaf theory, we construct the network sheaf and cosheaf of causal knowledge. These structures enable the transfer of causal knowledge across the network while incorporating interventional consistency and the perspective of the subjects, ultimately leading to the formal, mathematical definition of relative causal knowledge. 

**Abstract (ZH)**: Recent advances in artificial intelligence揭示纯预测系统之局限，呼吁转向因果和协作推理。受到Grothendieck在数学革命的启发，我们引入因果知识的相对性，认为结构因果模型（SCMs）本质上是嵌入在关系网络中的不完美、主观的表示。通过运用范畴论，我们将SCMs组织成一个函子范畴，并展示它们的观察概率度量和干预概率度量自然形成了凸结构。这一结果使我们能够使用概率度量的凸空间来编码非干预的SCMs。接着，利用层论，我们构建因果知识的网络层和余层。这些结构使因果知识在网络中传输的同时，结合了干预一致性以及主体的视角，最终形成了相对因果知识的形式化数学定义。 

---
# Deep Belief Markov Models for POMDP Inference 

**Title (ZH)**: 深度信念马尔可夫模型在部分可观察马尔可夫决策过程中的推断 

**Authors**: Giacomo Arcieri, Konstantinos G. Papakonstantinou, Daniel Straub, Eleni Chatzi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13438)  

**Abstract**: This work introduces a novel deep learning-based architecture, termed the Deep Belief Markov Model (DBMM), which provides efficient, model-formulation agnostic inference in Partially Observable Markov Decision Process (POMDP) problems. The POMDP framework allows for modeling and solving sequential decision-making problems under observation uncertainty. In complex, high-dimensional, partially observable environments, existing methods for inference based on exact computations (e.g., via Bayes' theorem) or sampling algorithms do not scale well. Furthermore, ground truth states may not be available for learning the exact transition dynamics. DBMMs extend deep Markov models into the partially observable decision-making framework and allow efficient belief inference entirely based on available observation data via variational inference methods. By leveraging the potency of neural networks, DBMMs can infer and simulate non-linear relationships in the system dynamics and naturally scale to problems with high dimensionality and discrete or continuous variables. In addition, neural network parameters can be dynamically updated efficiently based on data availability. DBMMs can thus be used to infer a belief variable, thus enabling the derivation of POMDP solutions over the belief space. We evaluate the efficacy of the proposed methodology by evaluating the capability of model-formulation agnostic inference of DBMMs in benchmark problems that include discrete and continuous variables. 

**Abstract (ZH)**: 基于深度学习的Deep Belief Markov模型（DBMM）在部分可观测量马尔可夫决策过程（POMDP）问题中高效无模型描述符推理的研究 

---
# Securing Virtual Reality Experiences: Unveiling and Tackling Cybersickness Attacks with Explainable AI 

**Title (ZH)**: 保障虚拟现实体验：可解释人工智能揭示并应对网络眩晕攻击 

**Authors**: Ripan Kumar Kundu, Matthew Denton, Genova Mongalo, Prasad Calyam, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2503.13419)  

**Abstract**: The synergy between virtual reality (VR) and artificial intelligence (AI), specifically deep learning (DL)-based cybersickness detection models, has ushered in unprecedented advancements in immersive experiences by automatically detecting cybersickness severity and adaptively various mitigation techniques, offering a smooth and comfortable VR experience. While this DL-enabled cybersickness detection method provides promising solutions for enhancing user experiences, it also introduces new risks since these models are vulnerable to adversarial attacks; a small perturbation of the input data that is visually undetectable to human observers can fool the cybersickness detection model and trigger unexpected mitigation, thus disrupting user immersive experiences (UIX) and even posing safety risks. In this paper, we present a new type of VR attack, i.e., a cybersickness attack, which successfully stops the triggering of cybersickness mitigation by fooling DL-based cybersickness detection models and dramatically hinders the UIX. Next, we propose a novel explainable artificial intelligence (XAI)-guided cybersickness attack detection framework to detect such attacks in VR to ensure UIX and a comfortable VR experience. We evaluate the proposed attack and the detection framework using two state-of-the-art open-source VR cybersickness datasets: Simulation 2021 and Gameplay dataset. Finally, to verify the effectiveness of our proposed method, we implement the attack and the XAI-based detection using a testbed with a custom-built VR roller coaster simulation with an HTC Vive Pro Eye headset and perform a user study. Our study shows that such an attack can dramatically hinder the UIX. However, our proposed XAI-guided cybersickness attack detection can successfully detect cybersickness attacks and trigger the proper mitigation, effectively reducing VR cybersickness. 

**Abstract (ZH)**: 虚拟现实与人工 Intelligence在基于深度学习的晕动症检测模型中的协同效应及其新风险：一种新型的晕动症攻击及其检测框架 

---
# A Comprehensive Survey on Multi-Agent Cooperative Decision-Making: Scenarios, Approaches, Challenges and Perspectives 

**Title (ZH)**: 全面综述多agent协同决策：场景、方法、挑战与视角 

**Authors**: Weiqiang Jin, Hongyang Du, Biao Zhao, Xingwu Tian, Bohang Shi, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13415)  

**Abstract**: With the rapid development of artificial intelligence, intelligent decision-making techniques have gradually surpassed human levels in various human-machine competitions, especially in complex multi-agent cooperative task scenarios. Multi-agent cooperative decision-making involves multiple agents working together to complete established tasks and achieve specific objectives. These techniques are widely applicable in real-world scenarios such as autonomous driving, drone navigation, disaster rescue, and simulated military confrontations. This paper begins with a comprehensive survey of the leading simulation environments and platforms used for multi-agent cooperative decision-making. Specifically, we provide an in-depth analysis for these simulation environments from various perspectives, including task formats, reward allocation, and the underlying technologies employed. Subsequently, we provide a comprehensive overview of the mainstream intelligent decision-making approaches, algorithms and models for multi-agent systems (MAS). Theseapproaches can be broadly categorized into five types: rule-based (primarily fuzzy logic), game theory-based, evolutionary algorithms-based, deep multi-agent reinforcement learning (MARL)-based, and large language models(LLMs)reasoning-based. Given the significant advantages of MARL andLLMs-baseddecision-making methods over the traditional rule, game theory, and evolutionary algorithms, this paper focuses on these multi-agent methods utilizing MARL and LLMs-based techniques. We provide an in-depth discussion of these approaches, highlighting their methodology taxonomies, advantages, and drawbacks. Further, several prominent research directions in the future and potential challenges of multi-agent cooperative decision-making are also detailed. 

**Abstract (ZH)**: 随着人工智能的快速发展，智能决策技术在各种人机竞赛中已经超过了人类的水平，特别是在复杂的多agent协同任务场景中。多agent协同决策涉及多个agent共同协作以完成既定任务并实现特定目标。这些技术在自动驾驶、无人机导航、灾难救援和模拟军事对抗等实际场景中有着广泛的应用。本文首先对多agent协同决策的主要仿真环境和平台进行了全面综述，并从任务格式、奖励分配以及所采用的底层技术等多个视角进行了深入分析。随后，本文对主流的多agent系统智能决策方法、算法和模型进行了综合概述。这些方法可以大致归为五类：基于规则的方法（主要为模糊逻辑）、基于博弈理论的方法、基于进化算法的方法、基于深度多agent强化学习（MARL）的方法以及基于大规模语言模型（LLMs）推理的方法。鉴于MARL和LLMs基方法相对于传统规则、博弈理论和进化算法的巨大优势，本文重点关注利用MARL和LLMs技术的多agent方法，并对其进行了深入探讨，突出了这些方法的分类、优点和局限性。此外，本文还详细讨论了未来多agent协同决策研究的几个主要方向以及可能面临的挑战。 

---
# Scale Efficient Training for Large Datasets 

**Title (ZH)**: 大规模数据集的高效训练 

**Authors**: Qing Zhou, Junyu Gao, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13385)  

**Abstract**: The rapid growth of dataset scales has been a key driver in advancing deep learning research. However, as dataset scale increases, the training process becomes increasingly inefficient due to the presence of low-value samples, including excessive redundant samples, overly challenging samples, and inefficient easy samples that contribute little to model this http URL address this challenge, we propose Scale Efficient Training (SeTa) for large datasets, a dynamic sample pruning approach that losslessly reduces training time. To remove low-value samples, SeTa first performs random pruning to eliminate redundant samples, then clusters the remaining samples according to their learning difficulty measured by loss. Building upon this clustering, a sliding window strategy is employed to progressively remove both overly challenging and inefficient easy clusters following an easy-to-hard this http URL conduct extensive experiments on large-scale synthetic datasets, including ToCa, SS1M, and ST+MJ, each containing over 3 million this http URL reduces training costs by up to 50\% while maintaining or improving performance, with minimal degradation even at 70\% cost reduction. Furthermore, experiments on various scale real datasets across various backbones (CNNs, Transformers, and Mambas) and diverse tasks (instruction tuning, multi-view stereo, geo-localization, composed image retrieval, referring image segmentation) demonstrate the powerful effectiveness and universality of our approach. Code is available at this https URL. 

**Abstract (ZH)**: 大数据集高效训练方法：面向低价值样本的动态采样修剪（SeTa） 

---
# Scalable Runtime Architecture for Data-driven, Hybrid HPC and ML Workflow Applications 

**Title (ZH)**: 面向数据驱动、混合HPC和ML工作流应用的大规模运行时架构 

**Authors**: Andre Merzky, Mikhail Titov, Matteo Turilli, Ozgur Kilic, Tianle Wang, Shantenu Jha  

**Link**: [PDF](https://arxiv.org/pdf/2503.13343)  

**Abstract**: Hybrid workflows combining traditional HPC and novel ML methodologies are transforming scientific computing. This paper presents the architecture and implementation of a scalable runtime system that extends RADICAL-Pilot with service-based execution to support AI-out-HPC workflows. Our runtime system enables distributed ML capabilities, efficient resource management, and seamless HPC/ML coupling across local and remote platforms. Preliminary experimental results show that our approach manages concurrent execution of ML models across local and remote HPC/cloud resources with minimal architectural overheads. This lays the foundation for prototyping three representative data-driven workflow applications and executing them at scale on leadership-class HPC platforms. 

**Abstract (ZH)**: 混合 workflows 结合传统HPC和新型ML方法正在变革科学计算。本文介绍了将 RADICAL-Pilot 扩展为基于服务的执行以支持AI-out-HPC workflows 的可扩展运行时系统的架构和实现。该运行时系统实现了分布式ML能力、高效的资源管理，并在本地和远程平台间无缝耦合HPC/ML。初步实验结果表明，该方法在本地和远程HPC/云资源上实现了ML模型的并发执行，且具有最小的架构开销。这为在旗舰级HPC平台大规模原型设计三种代表性数据驱动workflow应用程序奠定了基础。 

---
# Valid Text-to-SQL Generation with Unification-based DeepStochLog 

**Title (ZH)**: 基于统一的深度马尔可夫文本到SQL生成 

**Authors**: Ying Jiao, Luc De Raedt, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2503.13342)  

**Abstract**: Large language models have been used to translate natural language questions to SQL queries. Without hard constraints on syntax and database schema, they occasionally produce invalid queries that are not executable. These failures limit the usage of these systems in real-life scenarios. We propose a neurosymbolic framework that imposes SQL syntax and schema constraints with unification-based definite clause grammars and thus guarantees the generation of valid queries. Our framework also builds a bi-directional interface to language models to leverage their natural language understanding abilities. The evaluation results on a subset of SQL grammars show that all our output queries are valid. This work is the first step towards extending language models with unification-based grammars. We demonstrate this extension enhances the validity, execution accuracy, and ground truth alignment of the underlying language model by a large margin. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型已被用于将自然语言问题转换为SQL查询。通过使用基于统一的确定性子句语法，我们提出了一种神经符号框架，以施加SQL语法和模式约束，从而保证生成有效的查询。我们的框架还构建了一个双向接口，利用语言模型的自然语言理解能力。在SQL语法子集上的评估结果表明，所有输出查询都是有效的。这是使用基于统一的语法扩展语言模型的第一步。我们的扩展显著提高了底层语言模型的有效性、执行准确性和地面 truth 对齐。相关代码可在以下网址获取。 

---
# RainScaleGAN: a Conditional Generative Adversarial Network for Rainfall Downscaling 

**Title (ZH)**: RainScaleGAN：一种条件生成对抗网络用于降雨下标缩放 

**Authors**: Marcello Iotti, Paolo Davini, Jost von Hardenberg, Giuseppe Zappa  

**Link**: [PDF](https://arxiv.org/pdf/2503.13316)  

**Abstract**: To this day, accurately simulating local-scale precipitation and reliably reproducing its distribution remains a challenging task. The limited horizontal resolution of Global Climate Models is among the primary factors undermining their skill in this context. The physical mechanisms driving the onset and development of precipitation, especially in extreme events, operate at spatio-temporal scales smaller than those numerically resolved, thus struggling to be captured accurately. In order to circumvent this limitation, several downscaling approaches have been developed over the last decades to address the discrepancy between the spatial resolution of models output and the resolution required by local-scale applications. In this paper, we introduce RainScaleGAN, a conditional deep convolutional Generative Adversarial Network (GAN) for precipitation downscaling. GANs have been effectively used in image super-resolution, an approach highly relevant for downscaling tasks. RainScaleGAN's capabilities are tested in a perfect-model setup, where the spatial resolution of a precipitation dataset is artificially degraded from 0.25$^{\circ}\times$0.25$^{\circ}$ to 2$^{\circ}\times$2$^\circ$, and RainScaleGAN is used to restore it. The developed model outperforms one of the leading precipitation downscaling method found in the literature. RainScaleGAN not only generates a synthetic dataset featuring plausible high-resolution spatial patterns and intensities, but also produces a precipitation distribution with statistics closely mirroring those of the ground-truth dataset. Given that RainScaleGAN's approach is agnostic with respect to the underlying physics, the method has the potential to be applied to other physical variables such as surface winds or temperature. 

**Abstract (ZH)**: 到目前为止，准确模拟局地尺度降水并可靠再现其分布仍是一项具有挑战性的任务。全球气候模型的有限水平分辨率是其在这一方面的能力受到限制的主要因素之一。驱动降水发生和发展，尤其是极端事件的物理机制，其时空尺度小于数值分辨率，难以被准确捕捉。为了克服这一限制，近几十年来发展出了多种降尺度方法，以解决模型输出的空间分辨率与局地尺度应用所需的分辨率之间的矛盾。本文介绍了一种基于条件深度卷积生成对抗网络(GAN)的降水降尺度方法——RainScaleGAN。GAN已在图像超分辨率领域取得了成功应用，这与降尺度任务密切相关。通过在完美模型设置中测试RainScaleGAN的功能，即将降水数据集的空间分辨率从0.25°×0.25°人为降级为2°×2°，并使用RainScaleGAN重建原始分辨率。所开发的模型在文献中发现的领先降水降尺度方法中表现更优。RainScaleGAN不仅能生成具有合理高分辨率空间模式和强度的合成数据集，还能产生统计特征与真实数据集非常接近的降水分布。由于RainScaleGAN方法对底层物理过程具有无关性，该方法有潜力应用于其他物理变量，如地表风速或温度。 

---
# Generative AI for Software Architecture. Applications, Trends, Challenges, and Future Directions 

**Title (ZH)**: 生成式AI在软件架构中的应用、趋势、挑战及未来方向 

**Authors**: Matteo Esposito, Xiaozhou Li, Sergio Moreschini, Noman Ahmad, Tomas Cerny, Karthik Vaidhyanathan, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13310)  

**Abstract**: Context: Generative Artificial Intelligence (GenAI) is transforming much of software development, yet its application in software architecture is still in its infancy, and no prior study has systematically addressed the topic. Aim: We aim to systematically synthesize the use, rationale, contexts, usability, and future challenges of GenAI in software architecture. Method: We performed a multivocal literature review (MLR), analyzing peer-reviewed and gray literature, identifying current practices, models, adoption contexts, and reported challenges, extracting themes via open coding. Results: Our review identified significant adoption of GenAI for architectural decision support and architectural reconstruction. OpenAI GPT models are predominantly applied, and there is consistent use of techniques such as few-shot prompting and retrieved-augmented generation (RAG). GenAI has been applied mostly to initial stages of the Software Development Life Cycle (SDLC), such as Requirements-to-Architecture and Architecture-to-Code. Monolithic and microservice architectures were the dominant targets. However, rigorous testing of GenAI outputs was typically missing from the studies. Among the most frequent challenges are model precision, hallucinations, ethical aspects, privacy issues, lack of architecture-specific datasets, and the absence of sound evaluation frameworks. Conclusions: GenAI shows significant potential in software design, but several challenges remain on its path to greater adoption. Research efforts should target designing general evaluation methodologies, handling ethics and precision, increasing transparency and explainability, and promoting architecture-specific datasets and benchmarks to bridge the gap between theoretical possibilities and practical use. 

**Abstract (ZH)**: 生成式人工智能在软件架构中的应用、理据、情境、可用性及未来挑战系统综述 

---
# $ϕ$-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation 

**Title (ZH)**: $\phi$-解码：自适应前瞻性采样以实现平衡的推理时探索与利用 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Jun Liu, Qika Lin, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13288)  

**Abstract**: Inference-time optimization scales computation to derive deliberate reasoning steps for effective performance. While previous search-based strategies address the short-sightedness of auto-regressive generation, the vast search space leads to excessive exploration and insufficient exploitation. To strike an efficient balance to derive the optimal step, we frame the decoding strategy as foresight sampling, leveraging simulated future steps to obtain globally optimal step estimation. Built on it, we propose a novel decoding strategy, named $\phi$-Decoding. To provide a precise and expressive estimation of step value, $\phi$-Decoding approximates two distributions via foresight and clustering. Sampling from the joint distribution, the optimal steps can be selected for exploitation. To support adaptive computation allocation, we propose in-width and in-depth pruning strategies, featuring a light-weight solution to achieve inference efficiency. Extensive experiments across seven benchmarks show $\phi$-Decoding outperforms strong baselines in both performance and efficiency. Additional analysis demonstrates its generalization across various LLMs and scalability across a wide range of computing budgets. The code will be released at this https URL, and the open-source PyPI package is coming soon. 

**Abstract (ZH)**: 推理时的优化扩展计算以获取有目的地推理步骤，从而提高性能。虽然基于搜索的策略解决了自回归生成的短视问题，但庞大的搜索空间导致过度探索和不足的开发。为了在开发最优步骤时取得高效的平衡，我们将解码策略框定为前瞻采样，利用模拟的未来步骤来获取全局最优步骤估计。在此基础上，我们提出了一种新的解码策略，名为$\phi$-解码。为提供精确且表达性强的步骤值估计，$\phi$-解码通过前瞻和聚类近似两种分布。从联合分布采样，可以选择最优步骤进行开发。为了支持适应性计算分配，我们提出了在宽和深方向上的剪枝策略，这是一种轻量级解决方案以实现推理效率。在七个基准上的广泛实验显示，$\phi$-解码在性能和效率上均优于强基线。额外的分析表明，其在各种大型语言模型中具有泛化能力，并且在广泛计算预算范围内具有可扩展性。代码将在此链接中发布，开源的PyPI包即将上线。 

---
# A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening 

**Title (ZH)**: 适用于遥感融合的通用自适应双层次加权机制 

**Authors**: Jie Huang, Haorui Chen, Jiaxuan Ren, Siran Peng, Liangjian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2503.13214)  

**Abstract**: Currently, deep learning-based methods for remote sensing pansharpening have advanced rapidly. However, many existing methods struggle to fully leverage feature heterogeneity and redundancy, thereby limiting their effectiveness. We use the covariance matrix to model the feature heterogeneity and redundancy and propose Correlation-Aware Covariance Weighting (CACW) to adjust them. CACW captures these correlations through the covariance matrix, which is then processed by a nonlinear function to generate weights for adjustment. Building upon CACW, we introduce a general adaptive dual-level weighting mechanism (ADWM) to address these challenges from two key perspectives, enhancing a wide range of existing deep-learning methods. First, Intra-Feature Weighting (IFW) evaluates correlations among channels within each feature to reduce redundancy and enhance unique information. Second, Cross-Feature Weighting (CFW) adjusts contributions across layers based on inter-layer correlations, refining the final output. Extensive experiments demonstrate the superior performance of ADWM compared to recent state-of-the-art (SOTA) methods. Furthermore, we validate the effectiveness of our approach through generality experiments, redundancy visualization, comparison experiments, key variables and complexity analysis, and ablation studies. Our code is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的遥感 pansharpening 方法已取得 rapid 进展，但许多现有方法难以充分利用特征异质性和冗余性，从而限制了其有效性。我们利用协方差矩阵建模特征异质性和冗余性，并提出相关感知协方差加权（CACW）来调整它们。CACW 通过协方差矩阵捕获这些相关性，然后通过非线性函数生成调整权重。在此基础上，我们引入了一种通用的自适应双层加权机制（ADWM），从两个关键角度解决这些挑战，增强了一系列现有的深度学习方法。首先，Intra-Feature 加权（IFW）评估每个特征内部通道之间的相关性以减少冗余性和增强独特信息。其次，Cross-Feature 加权（CFW）基于层间相关性调整各层的贡献，精炼最终输出。大量实验表明，ADWM 在性能上优于最近的先进方法（SOTA）。此外，我们通过通用实验、冗余可视化、比较实验、关键变量和复杂性分析以及消融研究验证了我们方法的有效性。我们的代码可在以下网址访问：this https URL。 

---
# 3DAxisPrompt: Promoting the 3D Grounding and Reasoning in GPT-4o 

**Title (ZH)**: 3DAxisPrompt: 促进GPT-4o的3D定位与推理 

**Authors**: Dingning Liu, Cheng Wang, Peng Gao, Renrui Zhang, Xinzhu Ma, Yuan Meng, Zhihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13185)  

**Abstract**: Multimodal Large Language Models (MLLMs) exhibit impressive capabilities across a variety of tasks, especially when equipped with carefully designed visual prompts. However, existing studies primarily focus on logical reasoning and visual understanding, while the capability of MLLMs to operate effectively in 3D vision remains an ongoing area of exploration. In this paper, we introduce a novel visual prompting method, called 3DAxisPrompt, to elicit the 3D understanding capabilities of MLLMs in real-world scenes. More specifically, our method leverages the 3D coordinate axis and masks generated from the Segment Anything Model (SAM) to provide explicit geometric priors to MLLMs and then extend their impressive 2D grounding and reasoning ability to real-world 3D scenarios. Besides, we first provide a thorough investigation of the potential visual prompting formats and conclude our findings to reveal the potential and limits of 3D understanding capabilities in GPT-4o, as a representative of MLLMs. Finally, we build evaluation environments with four datasets, i.e., ScanRefer, ScanNet, FMB, and nuScene datasets, covering various 3D tasks. Based on this, we conduct extensive quantitative and qualitative experiments, which demonstrate the effectiveness of the proposed method. Overall, our study reveals that MLLMs, with the help of 3DAxisPrompt, can effectively perceive an object's 3D position in real-world scenarios. Nevertheless, a single prompt engineering approach does not consistently achieve the best outcomes for all 3D tasks. This study highlights the feasibility of leveraging MLLMs for 3D vision grounding/reasoning with prompt engineering techniques. 

**Abstract (ZH)**: 多模态大型语言模型在三维视觉理解中的三维轴提示方法 

---
# GC-Fed: Gradient Centralized Federated Learning with Partial Client Participation 

**Title (ZH)**: GC-Fed: 带有部分客户端参与的梯度集中联邦学习 

**Authors**: Jungwon Seo, Ferhat Ozgur Catak, Chunming Rong, Kibeom Hong, Minhoe Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.13180)  

**Abstract**: Multi-source information fusion (MSIF) leverages diverse data streams to enhance decision-making, situational awareness, and system resilience. Federated Learning (FL) enables MSIF while preserving privacy but suffers from client drift under high data heterogeneity, leading to performance degradation. Traditional mitigation strategies rely on reference-based gradient adjustments, which can be unstable in partial participation settings. To address this, we propose Gradient Centralized Federated Learning (GC-Fed), a reference-free gradient correction method inspired by Gradient Centralization (GC). We introduce Local GC and Global GC, applying GC during local training and global aggregation, respectively. Our hybrid GC-Fed approach selectively applies GC at the feature extraction layer locally and at the classifier layer globally, improving training stability and model performance. Theoretical analysis and empirical results demonstrate that GC-Fed mitigates client drift and achieves state-of-the-art accuracy gains of up to 20% in heterogeneous settings. 

**Abstract (ZH)**: 多源信息融合的梯度集中联邦学习（GC-Fed）：应对数据异构性引起的客户端漂移 

---
# Efficient Imitation Under Misspecification 

**Title (ZH)**: 在模型错误情况下的高效模仿 

**Authors**: Nicolas Espinosa-Dice, Sanjiban Choudhury, Wen Sun, Gokul Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.13162)  

**Abstract**: Interactive imitation learning (IL) is a powerful paradigm for learning to make sequences of decisions from an expert demonstrating how to perform a task. Prior work in efficient imitation learning has focused on the realizable setting, where the expert's policy lies within the learner's policy class (i.e. the learner can perfectly imitate the expert in all states). However, in practice, perfect imitation of the expert is often impossible due to differences in state information and action space expressiveness (e.g. morphological differences between robots and humans.) In this paper, we consider the more general misspecified setting, where no assumptions are made about the expert policy's realizability. We introduce a novel structural condition, reward-agnostic policy completeness, and prove that it is sufficient for interactive IL algorithms to efficiently avoid the quadratically compounding errors that stymie offline approaches like behavioral cloning. We address an additional practical constraint-the case of limited expert data-and propose a principled method for using additional offline data to further improve the sample-efficiency of interactive IL algorithms. Finally, we empirically investigate the optimal reset distribution in efficient IL under misspecification with a suite of continuous control tasks. 

**Abstract (ZH)**: 交互式imitation learning（IL）是一种从专家展示任务执行过程学会决策序列的强大范式。先前在高效imitation learning方面的研究主要集中在专家策略可以被学习者策略类完美拟合的实现设置上。然而，在实践中，由于状态信息和动作空间表达能力的差异（例如，机器人和人类之间的形态差异），完全模仿专家往往是不可能的。在本文中，我们考虑了更一般的未指定设置，即不假设专家策略的实现性。我们提出了一个新颖的结构条件——奖励无关心的策略完备性，并证明了它对于交互式IL算法来说是充分的，能够有效避免离线方法（如行为克隆）中的二次递归错误。我们还处理了另一个实际约束——专家数据有限的情况，并提出了一种原则性的方法，利用额外的离线数据进一步提高交互式IL算法的学习效率。最后，我们在未指定情况下，通过一系列连续控制任务，实证研究了高效IL的理想重置分布。 

---
# Beyond Propagation of Chaos: A Stochastic Algorithm for Mean Field Optimization 

**Title (ZH)**: 超越混沌传播：一种用于均 field 优化的随机算法 

**Authors**: Chandan Tankala, Dheeraj M. Nagaraj, Anant Raj  

**Link**: [PDF](https://arxiv.org/pdf/2503.13115)  

**Abstract**: Gradient flow in the 2-Wasserstein space is widely used to optimize functionals over probability distributions and is typically implemented using an interacting particle system with $n$ particles. Analyzing these algorithms requires showing (a) that the finite-particle system converges and/or (b) that the resultant empirical distribution of the particles closely approximates the optimal distribution (i.e., propagation of chaos). However, establishing efficient sufficient conditions can be challenging, as the finite particle system may produce heavily dependent random variables.
In this work, we study the virtual particle stochastic approximation, originally introduced for Stein Variational Gradient Descent. This method can be viewed as a form of stochastic gradient descent in the Wasserstein space and can be implemented efficiently. In popular settings, we demonstrate that our algorithm's output converges to the optimal distribution under conditions similar to those for the infinite particle limit, and it produces i.i.d. samples without the need to explicitly establish propagation of chaos bounds. 

**Abstract (ZH)**: 梯度流在2-Wasserstein空间中的应用及其虚拟粒子随机近似研究 

---
# Test-Time Domain Generalization via Universe Learning: A Multi-Graph Matching Approach for Medical Image Segmentation 

**Title (ZH)**: 基于宇宙学习的测试时域泛化：医疗图像分割的多图匹配方法 

**Authors**: Xingguo Lv, Xingbo Dong, Liwen Wang, Jiewen Yang, Lei Zhao, Bin Pu, Zhe Jin, Xuejun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.13012)  

**Abstract**: Despite domain generalization (DG) has significantly addressed the performance degradation of pre-trained models caused by domain shifts, it often falls short in real-world deployment. Test-time adaptation (TTA), which adjusts a learned model using unlabeled test data, presents a promising solution. However, most existing TTA methods struggle to deliver strong performance in medical image segmentation, primarily because they overlook the crucial prior knowledge inherent to medical images. To address this challenge, we incorporate morphological information and propose a framework based on multi-graph matching. Specifically, we introduce learnable universe embeddings that integrate morphological priors during multi-source training, along with novel unsupervised test-time paradigms for domain adaptation. This approach guarantees cycle-consistency in multi-matching while enabling the model to more effectively capture the invariant priors of unseen data, significantly mitigating the effects of domain shifts. Extensive experiments demonstrate that our method outperforms other state-of-the-art approaches on two medical image segmentation benchmarks for both multi-source and single-source domain generalization tasks. The source code is available at this https URL. 

**Abstract (ZH)**: 尽管泛化域（Domain Generalization, DG）在缓解由于域偏移引起的预训练模型性能下降方面取得了显著进展，但在实际部署中依然存在局限性。测试时自适应（Test-time Adaptation, TTA），即通过未标记的测试数据调整已学习模型，提供了一种有潜力的解决方案。然而，现有的大多数TTA方法在医学图像分割任务中未能表现出色，主要原因是它们忽略了医学图像固有的关键先验知识。为应对这一挑战，我们引入了形态学信息，并提出了一种基于多图匹配的框架。具体来说，我们引入可学习的宇宙嵌入，将形态学先验知识集成到多源训练中，并提出了一种新颖的无监督测试时领域自适应方案。该方法保证了多匹配的一致性，并使模型能够更有效地捕捉未见数据的不变先验，显著减少了域偏移的影响。广泛实验证明，我们的方法在两个医学图像分割基准数据集的多源和单源泛化任务中均优于其他最先进的方法。源代码可在以下链接获取。 

---
# Open3DBench: Open-Source Benchmark for 3D-IC Backend Implementation and PPA Evaluation 

**Title (ZH)**: Open3DBench：用于3D-IC后端实现和PPA评估的开源基准测试 

**Authors**: Yunqi Shi, Chengrui Gao, Wanqi Ren, Siyuan Xu, Ke Xue, Mingxuan Yuan, Chao Qian, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.12946)  

**Abstract**: This work introduces Open3DBench, an open-source 3D-IC backend implementation benchmark built upon the OpenROAD-flow-scripts framework, enabling comprehensive evaluation of power, performance, area, and thermal metrics. Our proposed flow supports modular integration of 3D partitioning, placement, 3D routing, RC extraction, and thermal simulation, aligning with advanced 3D flows that rely on commercial tools and in-house scripts. We present two foundational 3D placement algorithms: Open3D-Tiling, which emphasizes regular macro placement, and Open3D-DMP, which enhances wirelength optimization through cross-die co-placement with analytical placer DREAMPlace. Experimental results show significant improvements in area (51.19%), wirelength (24.06%), timing (30.84%), and power (5.72%) compared to 2D flows. The results also highlight that better wirelength does not necessarily lead to PPA gain, emphasizing the need of developing PPA-driven methods. Open3DBench offers a standardized, reproducible platform for evaluating 3D EDA methods, effectively bridging the gap between open-source tools and commercial solutions in 3D-IC design. 

**Abstract (ZH)**: This work introduces Open3DBench，一个基于OpenROAD-flow-scripts框架的开源3D-IC后端实现基准，用于全面评估功耗、性能、面积和热特性指标。我们提出的流程支持3D切分、布局、3D布线、RC提取和热仿真等模块化集成，与依赖商用工具和内部脚本的先进3D流程相契合。我们提出了两种基础的3D布局算法：Open3D-Tiling，侧重于规则宏布局；Open3D-DMP，通过芯片间协同布局结合分析型布局器DREAMPlace优化布线长度。实验结果表明，与2D流程相比，在面积（51.19%）、布线长度（24.06%）、时序（30.84%）和功耗（5.72%）方面均取得了显著改善。结果还表明，更好的布线长度并不一定能带来PPA收益，突显了开发PPA驱动方法的必要性。Open3DBench提供了一个标准化、可重现的平台，用于评估3D EDA方法，有效填补了开源工具与商业解决方案在3D-IC设计中的差距。 

---
# Federated Continual Instruction Tuning 

**Title (ZH)**: 联邦连续指令调优 

**Authors**: Haiyang Guo, Fanhu Zeng, Fei Zhu, Wenzhuo Liu, Da-Han Wang, Jian Xu, Xu-Yao Zhang, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12897)  

**Abstract**: A vast amount of instruction tuning data is crucial for the impressive performance of Large Multimodal Models (LMMs), but the associated computational costs and data collection demands during supervised fine-tuning make it impractical for most researchers. Federated learning (FL) has the potential to leverage all distributed data and training resources to reduce the overhead of joint training. However, most existing methods assume a fixed number of tasks, while in real-world scenarios, clients continuously encounter new knowledge and often struggle to retain old tasks due to memory constraints. In this work, we introduce the Federated Continual Instruction Tuning (FCIT) benchmark to model this real-world challenge. Our benchmark includes two realistic scenarios, encompassing four different settings and twelve carefully curated instruction tuning datasets. To address the challenges posed by FCIT, we propose dynamic knowledge organization to effectively integrate updates from different tasks during training and subspace selective activation to allocate task-specific output during inference. Extensive experimental results demonstrate that our proposed method significantly enhances model performance across varying levels of data heterogeneity and catastrophic forgetting. Our source code and dataset will be made publicly available. 

**Abstract (ZH)**: 联邦持续指令调优（FCIT）基准 

---
# CompMarkGS: Robust Watermarking for Compression 3D Gaussian Splatting 

**Title (ZH)**: CompMarkGS：鲁棒的压缩3D高斯散列水印技术 

**Authors**: Sumin In, Youngdong Jang, Utae Jeong, MinHyuk Jang, Hyeongcheol Park, Eunbyung Park, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12836)  

**Abstract**: 3D Gaussian Splatting (3DGS) enables rapid differentiable rendering for 3D reconstruction and novel view synthesis, leading to its widespread commercial use. Consequently, copyright protection via watermarking has become critical. However, because 3DGS relies on millions of Gaussians, which require gigabytes of storage, efficient transfer and storage require compression. Existing 3DGS watermarking methods are vulnerable to quantization-based compression, often resulting in the loss of the embedded watermark. To address this challenge, we propose a novel watermarking method that ensures watermark robustness after model compression while maintaining high rendering quality. In detail, we incorporate a quantization distortion layer that simulates compression during training, preserving the watermark under quantization-based compression. Also, we propose a learnable watermark embedding feature that embeds the watermark into the anchor feature, ensuring structural consistency and seamless integration into the 3D scene. Furthermore, we present a frequency-aware anchor growing mechanism to enhance image quality in high-frequency regions by effectively identifying Guassians within these regions. Experimental results confirm that our method preserves the watermark and maintains superior image quality under high compression, validating it as a promising approach for a secure 3DGS model. 

**Abstract (ZH)**: 基于3D高斯点的快速可微渲染及其水印保护方法 

---
# SparseLUT: Sparse Connectivity Optimization for Lookup Table-based Deep Neural Networks 

**Title (ZH)**: SparseLUT: 基于查找表的深度神经网络的稀疏连接优化 

**Authors**: Binglei Lou, Ruilin Wu, Philip Leong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12829)  

**Abstract**: The deployment of deep neural networks (DNNs) on resource-constrained edge devices such as field-programmable gate arrays (FPGAs) requires a careful balance of latency, power, and resource usage while maintaining high accuracy. Existing Lookup Table (LUT)-based DNNs, including LogicNets, PolyLUT, PolyLUT-Add, and NeuraLUT, exploit native FPGA resources with random sparse connectivity. This paper introduces SparseLUT, a connectivity-centric training technique tailored for LUT-based DNNs. SparseLUT leverages a non-greedy training strategy that prioritizes the pruning of less significant connections and strategically regrows alternative ones, resulting in efficient convergence to the target sparsity. Experimental results show consistent accuracy improvements across benchmarks, including up to a 2.13\% increase on MNIST and a 0.94\% improvement for Jet Substructure Classification compared to random sparsity. This is done without any hardware overhead and achieves state-of-the-art results for LUT-based DNNs. 

**Abstract (ZH)**: 基于LUT的深度神经网络在受限资源边缘设备上的部署需要在延迟、功耗和资源使用之间寻求仔细平衡，同时保持高精度。现有的基于LUT的DNNs，包括LogicNets、PolyLUT、PolyLUT-Add和NeuraLUT，利用随机稀疏连接来利用FPGA的固有资源。本文介绍了SparseLUT，这是一种针对基于LUT的DNNs的以连接为中心的训练技术。SparseLUT利用一种非贪婪的训练策略，优先剪枝不重要的连接并战略性地重新生长替代连接，以实现高效的目标稀疏化收敛。实验结果表明，SparseLUT在多个基准测试中实现了一致的准确率提升，包括在MNIST上的2.13%提升和在Jet子结构分类上的0.94%提升，这都超过了随机稀疏化的效果，且不增加任何硬件开销，实现了基于LUT的DNNs的最佳结果。 

---
# Adaptive Deep Learning for Multiclass Breast Cancer Classification via Misprediction Risk Analysis 

**Title (ZH)**: 基于误预测风险分析的多类乳腺癌适.adjusted 深度学习分类 

**Authors**: Gul Sheeraz, Qun Chen, Liu Feiyu, Zhou Fengjin MD  

**Link**: [PDF](https://arxiv.org/pdf/2503.12778)  

**Abstract**: Breast cancer remains one of the leading causes of cancer-related deaths worldwide. Early detection is crucial for improving patient outcomes, yet the diagnostic process is often complex and prone to inconsistencies among pathologists. Computer-aided diagnostic approaches have significantly enhanced breast cancer detection, particularly in binary classification (benign vs. malignant). However, these methods face challenges in multiclass classification, leading to frequent mispredictions. In this work, we propose a novel adaptive learning approach for multiclass breast cancer classification using H&E-stained histopathology images. First, we introduce a misprediction risk analysis framework that quantifies and ranks the likelihood of an image being mislabeled by a classifier. This framework leverages an interpretable risk model that requires only a small number of labeled samples for training. Next, we present an adaptive learning strategy that fine-tunes classifiers based on the specific characteristics of a given dataset. This approach minimizes misprediction risk, allowing the classifier to adapt effectively to the target workload. We evaluate our proposed solutions on real benchmark datasets, demonstrating that our risk analysis framework more accurately identifies mispredictions compared to existing methods. Furthermore, our adaptive learning approach significantly improves the performance of state-of-the-art deep neural network classifiers. 

**Abstract (ZH)**: 乳腺癌仍然是全球癌症相关死亡的主要原因之一。早期检测对于改善患者预后至关重要，但诊断过程往往复杂且病理学家之间容易出现不一致。计算机辅助诊断方法在提高乳腺癌检测方面取得了显著进展，尤其是在二分类（良性 vs. 恶性）中。然而，这些方法在多分类中面临挑战，导致频繁的误预测。本工作中，我们提出了一种用于HE染色组织病理学图像的多分类乳腺癌分类新型自适应学习方法。首先，我们引入了一种误预测风险分析框架，该框架量化并排名图像被分类器误标的可能性。该框架利用一种可解释的风险模型，只需少量标注样本即可训练。接着，我们展示了一种自适应学习策略，根据给定数据集的特定特征微调分类器。该方法最小化误预测风险，使分类器能够有效适应目标负载。我们在实际基准数据集上评估了我们提出的解决方案，结果显示，我们的风险分析框架比现有方法更准确地识别误预测。此外，我们的自适应学习方法显著提高了最先进的深度神经网络分类器的性能。 

---
# SafeSlice: Enabling SLA-Compliant O-RAN Slicing via Safe Deep Reinforcement Learning 

**Title (ZH)**: SafeSlice: 通过安全深度强化学习实现合规SLA切片的O-RAN 

**Authors**: Ahmad M. Nagib, Hatem Abou-Zeid, Hossam S. Hassanein  

**Link**: [PDF](https://arxiv.org/pdf/2503.12753)  

**Abstract**: Deep reinforcement learning (DRL)-based slicing policies have shown significant success in simulated environments but face challenges in physical systems such as open radio access networks (O-RANs) due to simulation-to-reality gaps. These policies often lack safety guarantees to ensure compliance with service level agreements (SLAs), such as the strict latency requirements of immersive applications. As a result, a deployed DRL slicing agent may make resource allocation (RA) decisions that degrade system performance, particularly in previously unseen scenarios. Real-world immersive applications require maintaining SLA constraints throughout deployment to prevent risky DRL exploration. In this paper, we propose SafeSlice to address both the cumulative (trajectory-wise) and instantaneous (state-wise) latency constraints of O-RAN slices. We incorporate the cumulative constraints by designing a sigmoid-based risk-sensitive reward function that reflects the slices' latency requirements. Moreover, we build a supervised learning cost model as part of a safety layer that projects the slicing agent's RA actions to the nearest safe actions, fulfilling instantaneous constraints. We conduct an exhaustive experiment that supports multiple services, including real virtual reality (VR) gaming traffic, to investigate the performance of SafeSlice under extreme and changing deployment conditions. SafeSlice achieves reductions of up to 83.23% in average cumulative latency, 93.24% in instantaneous latency violations, and 22.13% in resource consumption compared to the baselines. The results also indicate SafeSlice's robustness to changing the threshold configurations of latency constraints, a vital deployment scenario that will be realized by the O-RAN paradigm to empower mobile network operators (MNOs). 

**Abstract (ZH)**: 基于深度 reinforcement 学习的SafeSlice策略：同时满足开放无线接入网络切片的累积和即时延迟约束 

---
# TNCSE: Tensor's Norm Constraints for Unsupervised Contrastive Learning of Sentence Embeddings 

**Title (ZH)**: TNCSE: 张量的范数约束无监督句子嵌入对比学习 

**Authors**: Tianyu Zong, Bingkang Shi, Hongzhu Yi, Jungang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12739)  

**Abstract**: Unsupervised sentence embedding representation has become a hot research topic in natural language processing. As a tensor, sentence embedding has two critical properties: direction and norm. Existing works have been limited to constraining only the orientation of the samples' representations while ignoring the features of their module lengths. To address this issue, we propose a new training objective that optimizes the training of unsupervised contrastive learning by constraining the module length features between positive samples. We combine the training objective of Tensor's Norm Constraints with ensemble learning to propose a new Sentence Embedding representation framework, TNCSE. We evaluate seven semantic text similarity tasks, and the results show that TNCSE and derived models are the current state-of-the-art approach; in addition, we conduct extensive zero-shot evaluations, and the results show that TNCSE outperforms other baselines. 

**Abstract (ZH)**: 无监督句子嵌入表示已成为自然语言处理领域的研究热点。作为张量，句子嵌入具有两个关键属性：方向和范数。现有工作仅限于约束样本表示的方向特征，而忽略了其模长特征。为解决这一问题，我们提出了一种新的训练目标，通过约束正样本之间的模长特征来优化无监督对比学习的训练。我们结合张量范数约束的训练目标与集成学习，提出了一种新的句子嵌入表示框架TNCSE。我们在七个语义文本相似度任务上进行了评估，结果显示TNCSE及其衍生模型是当前最先进的方法；此外，我们还进行了广泛的零样本评估，结果显示TNCSE优于其他基线方法。 

---
# TinySQL: A Progressive Text-to-SQL Dataset for Mechanistic Interpretability Research 

**Title (ZH)**: TinySQL: 一种用于机理可解释性研究的渐进式文本到SQL数据集 

**Authors**: Philip Quirke, Clement Neo, Abir Harrasse, Dhruv Nathawani, Amir Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12730)  

**Abstract**: Mechanistic interpretability research faces a gap between analyzing simple circuits in toy tasks and discovering features in large models. To bridge this gap, we propose text-to-SQL generation as an ideal task to study, as it combines the formal structure of toy tasks with real-world complexity. We introduce TinySQL, a synthetic dataset progressing from basic to advanced SQL operations, and train models ranging from 33M to 1B parameters to establish a comprehensive testbed for interpretability. We apply multiple complementary interpretability techniques, including edge attribution patching and sparse autoencoders, to identify minimal circuits and components supporting SQL generation. Our analysis reveals both the potential and limitations of current interpretability methods, showing how circuits can vary even across similar queries. Lastly, we demonstrate how mechanistic interpretability can identify flawed heuristics in models and improve synthetic dataset design. Our work provides a comprehensive framework for evaluating and advancing interpretability techniques while establishing clear boundaries for their reliable application. 

**Abstract (ZH)**: 机制可解释性研究面临在玩具任务中分析简单电路与在大型模型中发现特征之间的差距。为了弥合这一差距，我们提出将文本生成SQL查询任务作为理想的测试任务，因为它结合了玩具任务的形式结构与现实世界的复杂性。我们引入了TinySQL，这是一个从基本到高级SQL操作的合成数据集，并训练了从33M到1B参数的模型，以建立一个全面的可解释性测试平台。我们应用多种互补的可解释性技术，包括边缘归因修补和稀疏自编码器，来识别支持SQL生成的最小电路和组件。我们的分析揭示了当前可解释性方法的潜力和局限性，展示了即使在相似查询之间电路也可能发生变化。最后，我们展示了机制可解释性如何识别模型中的错误启发式并改进合成数据集设计。我们的研究提供了一个全面的框架来评估和推进可解释性技术，并明确了其可靠应用的边界。 

---
# Dynamic Angle Selection in X-Ray CT: A Reinforcement Learning Approach to Optimal Stopping 

**Title (ZH)**: X射线CT中动态角度选择：基于强化学习的最优停止方法 

**Authors**: Tianyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12688)  

**Abstract**: In industrial X-ray Computed Tomography (CT), the need for rapid in-line inspection is critical. Sparse-angle tomography plays a significant role in this by reducing the required number of projections, thereby accelerating processing and conserving resources. Most existing methods aim to balance reconstruction quality and scanning time, typically relying on fixed scan durations. Adaptive adjustment of the number of angles is essential; for instance, more angles may be required for objects with complex geometries or noisier projections. The concept of optimal stopping, which dynamically adjusts this balance according to varying industrial needs, remains underutilized. Building on our previous work, we integrate optimal stopping into sequential Optimal Experimental Design (OED). We propose a novel method for computing the policy gradient within the Actor-Critic framework, enabling the development of adaptive policies for informative angle selection and scan termination. Additionally, we investigated the gap between simulation and real-world applications in the context of the developed learning-based method. Our trained model, developed using synthetic data, demonstrates reliable performance when applied to real-world data. This approach enhances the flexibility of CT operations and expands the applicability of sparse-angle tomography in industrial settings. 

**Abstract (ZH)**: 工业X射线计算机断层成像（CT）中的快速在线检测需求至关重要。稀疏角度断层成像通过对所需投影数量的减少，加快处理速度并节省资源，发挥了重要作用。现有的大多数方法旨在平衡重建质量和扫描时间，通常依赖固定的扫描时间。适应性调整角度的数量至关重要；例如，复杂几何结构的对象或噪声较大的投影需要更多角度。最优停止的概念，根据变化的工业需求动态调整这种平衡，尚未得到充分利用。基于我们之前的工作，我们将最优停止整合到顺序最优实验设计（OED）中。我们提出了一种在Actor-Critic框架下计算策略梯度的新方法，以开发适应性策略，实现信息性角度选择和扫描终止。此外，我们还探讨了所开发的基于学习的方法与仿真之间的差距。使用合成数据训练的模型在应用于实际数据时表现出可靠的性能。该方法增强了CT操作的灵活性，并扩大了稀疏角度断层成像在工业环境中的应用范围。 

---
# FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization 

**Title (ZH)**: FW-融合：基于Frank-Wolfe优化的模型融合 

**Authors**: Hao Mark Chen, Shell Xu Hu, Wayne Luk, Timothy Hospedales, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.12649)  

**Abstract**: Model merging has emerged as a promising approach for multi-task learning (MTL), offering a data-efficient alternative to conventional fine-tuning. However, with the rapid development of the open-source AI ecosystem and the increasing availability of fine-tuned foundation models, existing model merging methods face two key limitations: (i) They are primarily designed for in-house fine-tuned models, making them less adaptable to diverse model sources with partially unknown model and task information, (ii) They struggle to scale effectively when merging numerous model checkpoints. To address these challenges, we formulate model merging as a constrained optimization problem and introduce a novel approach: Frank-Wolfe Merging (FW-Merging). Inspired by Frank-Wolfe optimization, our approach iteratively selects the most relevant model in the pool to minimize a linear approximation of the objective function and then executes a local merging similar to the Frank-Wolfe update. The objective function is designed to capture the desired behavior of the target-merged model, while the fine-tuned candidate models define the constraint set. More importantly, FW-Merging serves as an orthogonal technique for existing merging methods, seamlessly integrating with them to further enhance accuracy performance. Our experiments show that FW-Merging scales across diverse model sources, remaining stable with 16 irrelevant models and improving by 15.3% with 16 relevant models on 20 CV tasks, while maintaining constant memory overhead, unlike the linear overhead of data-informed merging methods. Compared with the state-of-the-art approaches, FW-Merging surpasses the data-free merging method by 32.8% and outperforms the data-informed Adamerging by 8.39% when merging 20 ViT models. 

**Abstract (ZH)**: 模型融合已成为多任务学习（MTL）的一种有前景的方法，提供了一种与常规微调相比更为数据高效的替代方案。然而，随着开源AI生态系统的快速发展和微调基础模型的日益可用，现有的模型融合方法面临两个关键局限：（i）它们主要针对内部微调模型设计，使得它们对来源多样且部分模型和任务信息未知的模型不太适应；（ii）它们在融合大量模型检查点时难以有效扩展。为了解决这些挑战，我们将模型融合形式化为一个受约束的优化问题，并引入一种新颖的方法：Frank-Wolfe融合（FW-Merging）。受Frank-Wolfe优化启发，我们的方法迭代选择池中最相关的模型以最小化目标函数的线性逼近，并执行类似于Frank-Wolfe更新的局部融合。目标函数设计用于捕获目标融合模型所需的特性，而微调候选模型定义约束集。更重要的是，FW-Merging 是现有融合方法的一种补充技术，可以无缝集成到它们之中以进一步提升准确度性能。实验结果表明，FW-Merging 在多种模型来源上可扩展，即使在有 16 个无关模型的情况下保持稳定，并且在 16 个相关模型下在 20 个CV任务上性能提升了15.3%，同时保持恒定的内存开销，不同于数据导向融合方法的线性内存开销。与最先进的方法相比，当融合 20 个 ViT 模型时，FW-Merging 超过了无数据融合方法 32.8%，并在数据导向的Adamerging 上表现优于后者 8.39%。 

---
# COVID 19 Diagnosis Analysis using Transfer Learning 

**Title (ZH)**: COVID-19 诊断分析使用迁移学习 

**Authors**: Anjali Dharmik  

**Link**: [PDF](https://arxiv.org/pdf/2503.12642)  

**Abstract**: Coronaviruses transmit COVID-19, a rapidly spreading disease. A Coronavirus infection (COVID-19) was first discovered in December 2019 in Wuhan, China, and spread rapidly throughout the planet in exactly some months. because of this, the virus can cause severe symptoms and even death, especially within the elderly and in people with medical conditions. The virus causes acute respiratory infections in humans. the primary case was diagnosed in China in 2019 and the pandemic started in 2020. Since the quantity of cases of COVID-19 is increasing daily, there are only a limited number of test kits available in hospitals. So, to stop COVID-19 from spreading among people, an automatic diagnosis system must be implemented. during this study, three pre-trained neural networks supported convolutional neural networks (VGG16, VGG19, ResNet50) are proposed for detecting Coronavirus pneumonia infected patients through X-rays and computerized tomography (CT). By using cross-validation, we've got implemented binary classifications with two classes (COVID-19, Normal (healthy)). Taking into consideration the results obtained, the pre-trained ResNet50 model provides the simplest classification performance (97.77% accuracy, 100% sensitivity, 93.33% specificity, 98.00% F1-score) among the opposite three used models over 6259 images. 

**Abstract (ZH)**: 冠状病毒传播COVID-19，一种迅速传播的疾病 

---
# Hybrid Learners Do Not Forget: A Brain-Inspired Neuro-Symbolic Approach to Continual Learning 

**Title (ZH)**: 混合学习者不会遗忘：一种受脑启发的神经符号连续学习方法 

**Authors**: Amin Banayeeanzade, Mohammad Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2503.12635)  

**Abstract**: Continual learning is crucial for creating AI agents that can learn and improve themselves autonomously. A primary challenge in continual learning is to learn new tasks without losing previously learned knowledge. Current continual learning methods primarily focus on enabling a neural network with mechanisms that mitigate forgetting effects. Inspired by the two distinct systems in the human brain, System 1 and System 2, we propose a Neuro-Symbolic Brain-Inspired Continual Learning (NeSyBiCL) framework that incorporates two subsystems to solve continual learning: A neural network model responsible for quickly adapting to the most recent task, together with a symbolic reasoner responsible for retaining previously acquired knowledge from previous tasks. Moreover, we design an integration mechanism between these components to facilitate knowledge transfer from the symbolic reasoner to the neural network. We also introduce two compositional continual learning benchmarks and demonstrate that NeSyBiCL is effective and leads to superior performance compared to continual learning methods that merely rely on neural architectures to address forgetting. 

**Abstract (ZH)**: 持续学习对于创建能够自主学习和改进的AI代理至关重要。持续学习的主要挑战之一是在学习新任务时不丢失先前学习的知识。目前的持续学习方法主要侧重于通过机制减轻遗忘效应来增强神经网络。受人类大脑中两个不同系统——系统1和系统2——的启发，我们提出了一种神经符号脑启发式持续学习（NeSyBiCL）框架，该框架包含两个子系统来解决持续学习问题：一个负责快速适应最近任务的神经网络模型，以及一个负责保留先前任务中获得的知识的符号推理器。此外，我们设计了一种这些组件之间的集成机制，以促进符号推理器向神经网络的知识迁移。我们还引入了两个组合式持续学习基准，并证明NeSyBiCL是有效的，并且在仅仅依赖神经架构解决遗忘问题的持续学习方法中表现更优。 

---
# Negotiative Alignment: Embracing Disagreement to Achieve Fairer Outcomes -- Insights from Urban Studies 

**Title (ZH)**: 协商一致：拥抱分歧以达成更加公平的结果——从城市研究中获得的启示 

**Authors**: Rashid Mushkani, Hugo Berard, Shin Koseki  

**Link**: [PDF](https://arxiv.org/pdf/2503.12613)  

**Abstract**: Cities are not monolithic; they are arenas of negotiation among groups that hold varying needs, values, and experiences. Conventional methods of urban assessment -- from standardized surveys to AI-driven evaluations -- frequently rely on a single consensus metric (e.g., an average measure of inclusivity or safety). Although such aggregations simplify design decisions, they risk obscuring the distinct perspectives of marginalized populations. In this paper, we present findings from a community-centered study in Montreal involving 35 residents with diverse demographic and social identities, particularly wheelchair users, seniors, and LGBTQIA2+ individuals. Using rating and ranking tasks on 20 urban sites, we observe that disagreements are systematic rather than random, reflecting structural inequalities, differing cultural values, and personal experiences of safety and accessibility.
Based on these empirical insights, we propose negotiative alignment, an AI framework that treats disagreement as an essential input to be preserved, analyzed, and addressed. Negotiative alignment builds on pluralistic models by dynamically updating stakeholder preferences through multi-agent negotiation mechanisms, ensuring no single perspective is marginalized. We outline how this framework can be integrated into urban analytics -- and other decision-making contexts -- to retain minority viewpoints, adapt to changing stakeholder concerns, and enhance fairness and accountability. The study demonstrates that preserving and engaging with disagreement, rather than striving for an artificial consensus, can produce more equitable and responsive AI-driven outcomes in urban design. 

**Abstract (ZH)**: 城市并非同质；它们是不同群体协商的舞台，这些群体有不同的需求、价值观和经历。传统的城市评估方法——从标准化调查到基于AI的评估——通常依赖单一的共识指标（如包容性或安全性的平均值）。虽然这些聚合简化了设计决策，但可能会掩盖边缘化群体的独特视角。在本文中，我们呈现了在蒙特利尔进行的一项以社区为中心的研究成果，涉及35名具有多元人口和社会身份的居民，特别是轮椅使用者、老年人和LGBTQIA2+人群。通过针对20个城市场地进行评级和排序任务，我们观察到，分歧是有系统性的而非随机的，反映了结构不平等、不同的文化价值观以及个人的安全感和可访问性经验。

基于这些实证洞见，我们提出了协商对齐这一AI框架，该框架视分歧为必不可少的输入，需被保留、分析和解决。协商对齐建立在多元主义模型的基础上，通过多智能体协商机制动态更新利益相关者偏好，确保不会边缘化任何一种视角。我们概述了该框架如何可以集成到城市分析以及其他决策场景中，以保留少数群体的观点，适应变化的利益相关者关注点，并增强公平性和问责制。研究表明，保留并参与分歧，而不是追求人为的一致性，可以在城市设计中产生更加公平和具有响应性的AI驱动成果。 

---
# STEVE: AStep Verification Pipeline for Computer-use Agent Training 

**Title (ZH)**: STEVE: 计算机使用代理训练的步骤验证流水线 

**Authors**: Fanbin Lu, Zhisheng Zhong, Ziqin Wei, Shu Liu, Chi-Wing Fu, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.12532)  

**Abstract**: Developing AI agents to autonomously manipulate graphical user interfaces is a long challenging task. Recent advances in data scaling law inspire us to train computer-use agents with a scaled instruction set, yet using behavior cloning to train agents still requires immense high-quality trajectories. To meet the scalability need, we designed STEVE, a step verification pipeline for computer-use agent training. First, we establish a large instruction set for computer-use agents and collect trajectory data with some suboptimal agents. GPT-4o is used to verify the correctness of each step in the trajectories based on the screens before and after the action execution, assigning each step with a binary label. Last, we adopt the Kahneman and Tversky Optimization to optimize the agent from the binary stepwise labels. Extensive experiments manifest that our agent outperforms supervised finetuning by leveraging both positive and negative actions within a trajectory. Also, STEVE enables us to train a 7B vision-language model as a computer-use agent, achieving leading performance in the challenging live desktop environment WinAgentArena with great efficiency at a reduced cost. Code and data: this https URL. 

**Abstract (ZH)**: 开发自主操作图形用户界面的AI代理是一个长期具有挑战性的任务。 recent advances in data scaling law启发我们使用缩放后的指令集训练计算机使用代理，然而，使用行为克隆训练代理仍然需要大量的高质量轨迹数据。为了满足扩展性需求，我们设计了STEVE，一个计算机使用代理训练的步骤验证流程。首先，我们为计算机使用代理建立了一个大型指令集，并使用一些亚最优代理收集轨迹数据。GPT-4o基于动作执行前后屏幕的情况验证轨迹中每个步骤的正确性，并为每个步骤分配一个二元标签。最后，我们采用了Kahneman和Tversky优化方法从二元步骤标签优化代理。大量实验表明，通过利用轨迹中正负动作，我们的代理在监督微调中表现出色。此外，STEVE使我们能够训练一个7B的视觉-语言模型作为计算机使用代理，在具有挑战性的实时桌面环境WinAgentArena中取得了领先性能，并且在效率和成本方面表现出色。代码和数据：this https URL。 

---
# HyConEx: Hypernetwork classifier with counterfactual explanations 

**Title (ZH)**: HyConEx: 基于反事实解释的超网络分类器 

**Authors**: Patryk Marszałek, Ulvi Movsum-zada, Oleksii Furman, Kamil Książek, Przemysław Spurek, Marek Śmieja  

**Link**: [PDF](https://arxiv.org/pdf/2503.12525)  

**Abstract**: In recent years, there has been a growing interest in explainable AI methods. We want not only to make accurate predictions using sophisticated neural networks but also to understand what the model's decision is based on. One of the fundamental levels of interpretability is to provide counterfactual examples explaining the rationale behind the decision and identifying which features, and to what extent, must be modified to alter the model's outcome. To address these requirements, we introduce HyConEx, a classification model based on deep hypernetworks specifically designed for tabular data. Owing to its unique architecture, HyConEx not only provides class predictions but also delivers local interpretations for individual data samples in the form of counterfactual examples that steer a given sample toward an alternative class. While many explainable methods generated counterfactuals for external models, there have been no interpretable classifiers simultaneously producing counterfactual samples so far. HyConEx achieves competitive performance on several metrics assessing classification accuracy and fulfilling the criteria of a proper counterfactual attack. This makes HyConEx a distinctive deep learning model, which combines predictions and explainers as an all-in-one neural network. The code is available at this https URL. 

**Abstract (ZH)**: 近年来，解释性人工智能方法越来越受到关注。我们不仅希望通过复杂的神经网络进行准确的预测，还想理解模型决策背后的依据。解释性的根本层面之一是提供反事实示例，解释决策的理由，并确定哪些特征需要到何种程度的修改以改变模型的结果。为了满足这些要求，我们引入了基于深度超网络的HyConEx分类模型，该模型专门设计用于表格数据。得益于其独特的架构，HyConEx不仅可以提供类别预测，还能以反事实示例的形式为单个数据样本提供局部解释，引导一个样本向另一个类别变化。虽然许多可解释的方法为外部模型生成了反事实，但目前还没有同时产生可解释分类器和反事实样本的方法。HyConEx在几项评估分类准确性的指标上表现竞争力，并满足适当的反事实攻击标准。这使HyConEx成为一种独特的深度学习模型，将预测和解释器结合在一个神经网络中。代码可在以下链接获取。 

---
# A General Close-loop Predictive Coding Framework for Auditory Working Memory 

**Title (ZH)**: 一种闭-loop预测编码框架：听觉工作记忆 

**Authors**: Zhongju Yuan, Geraint Wiggins, Dick Botteldooren  

**Link**: [PDF](https://arxiv.org/pdf/2503.12506)  

**Abstract**: Auditory working memory is essential for various daily activities, such as language acquisition, conversation. It involves the temporary storage and manipulation of information that is no longer present in the environment. While extensively studied in neuroscience and cognitive science, research on its modeling within neural networks remains limited. To address this gap, we propose a general framework based on a close-loop predictive coding paradigm to perform short auditory signal memory tasks. The framework is evaluated on two widely used benchmark datasets for environmental sound and speech, demonstrating high semantic similarity across both datasets. 

**Abstract (ZH)**: 听觉工作记忆对于语言获取和对话等日常活动至关重要，它涉及对不再存在于环境中的信息进行临时存储和操作。尽管在神经科学和认知科学中已经对其进行了广泛研究，但其在神经网络中的建模研究仍然有限。为填补这一空白，我们提出了一种基于闭环预测编码范式的通用框架，用于执行短时听觉信号记忆任务。该框架在两个广泛使用的环境声音和语音基准数据集上进行评估，显示出高语义相似性。 

---
# Defense Against Model Stealing Based on Account-Aware Distribution Discrepancy 

**Title (ZH)**: 基于账户意识的分布差异防御模型窃取 

**Authors**: Jian-Ping Mei, Weibin Zhang, Jie Chen, Xuyun Zhang, Tiantian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12497)  

**Abstract**: Malicious users attempt to replicate commercial models functionally at low cost by training a clone model with query responses. It is challenging to timely prevent such model-stealing attacks to achieve strong protection and maintain utility. In this paper, we propose a novel non-parametric detector called Account-aware Distribution Discrepancy (ADD) to recognize queries from malicious users by leveraging account-wise local dependency. We formulate each class as a Multivariate Normal distribution (MVN) in the feature space and measure the malicious score as the sum of weighted class-wise distribution discrepancy. The ADD detector is combined with random-based prediction poisoning to yield a plug-and-play defense module named D-ADD for image classification models. Results of extensive experimental studies show that D-ADD achieves strong defense against different types of attacks with little interference in serving benign users for both soft and hard-label settings. 

**Abstract (ZH)**: 恶意用户试图通过训练克隆模型来低成本复制商业模型的功能，利用查询响应进行模型盗取。及时防止此类模型盗窃攻击以实现强有力保护并保持实用性是具有挑战性的。本文提出了一种名为Account-aware Distribution Discrepancy (ADD)的新型非参数检测器，通过利用账户级别的局部依赖性来识别恶意用户的查询。我们将每个类别在特征空间中形式化为多元正态分布（MVN），并通过加权类别分布差异来衡量恶意评分。将ADD检测器与基于随机的预测污染结合，形成一个插即用的防御模块D-ADD，用于图像分类模型。广泛的实验研究结果表明，D-ADD在软标签和硬标签设置下都能有效地防御不同类型攻击，同时对良性用户的使用干扰很小。 

---
# KDSelector: A Knowledge-Enhanced and Data-Efficient Model Selector Learning Framework for Time Series Anomaly Detection 

**Title (ZH)**: KDSelector：一种增强知识和数据高效的时间序列异常检测模型选择学习框架 

**Authors**: Zhiyu Liang, Dongrui Cai, Chenyuan Zhang, Zheng Liang, Chen Liang, Bo Zheng, Shi Qiu, Jin Wang, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12478)  

**Abstract**: Model selection has been raised as an essential problem in the area of time series anomaly detection (TSAD), because there is no single best TSAD model for the highly heterogeneous time series in real-world applications. However, despite the success of existing model selection solutions that train a classification model (especially neural network, NN) using historical data as a selector to predict the correct TSAD model for each series, the NN-based selector learning methods used by existing solutions do not make full use of the knowledge in the historical data and require iterating over all training samples, which limits the accuracy and training speed of the selector. To address these limitations, we propose KDSelector, a novel knowledge-enhanced and data-efficient framework for learning the NN-based TSAD model selector, of which three key components are specifically designed to integrate available knowledge into the selector and dynamically prune less important and redundant samples during the learning. We develop a TSAD model selection system with KDSelector as the internal, to demonstrate how users improve the accuracy and training speed of their selectors by using KDSelector as a plug-and-play module. Our demonstration video is hosted at this https URL. 

**Abstract (ZH)**: 基于知识增强和数据高效方法的NN时间序列异常检测模型选择器框架：KDSelector 

---
# ISLR101: an Iranian Word-Level Sign Language Recognition Dataset 

**Title (ZH)**: ISLR101: 伊朗单词级手语识别数据集 

**Authors**: Hossein Ranjbar, Alireza Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2503.12451)  

**Abstract**: Sign language recognition involves modeling complex multichannel information, such as hand shapes and movements while relying on sufficient sign language-specific data. However, sign languages are often under-resourced, posing a significant challenge for research and development in this field. To address this gap, we introduce ISLR101, the first publicly available Iranian Sign Language dataset for isolated sign language recognition. This comprehensive dataset includes 4,614 videos covering 101 distinct signs, recorded by 10 different signers (3 deaf individuals, 2 sign language interpreters, and 5 L2 learners) against varied backgrounds, with a resolution of 800x600 pixels and a frame rate of 25 frames per second. It also includes skeleton pose information extracted using OpenPose. We establish both a visual appearance-based and a skeleton-based framework as baseline models, thoroughly training and evaluating them on ISLR101. These models achieve 97.01% and 94.02% accuracy on the test set, respectively. Additionally, we publish the train, validation, and test splits to facilitate fair comparisons. 

**Abstract (ZH)**: 伊朗手语识别涉及建模复杂的多通道信息，如手形和手势动作，但依赖于足够的特定于手语的数据。由于手语资源普遍不足，这对该领域的研究和开发构成了重大挑战。为解决这一问题，我们引入ISLR101，这是首个公开的孤立伊朗手语数据集。该综合数据集包含4614个视频，涵盖101个不同手势，由10位不同手语使用者（3名聋人、2名手语翻译员和5名第二语言学习者）在多种背景中记录，分辨率为800x600像素，帧率为每秒25帧，并且包括使用OpenPose提取的骨架姿态信息。我们建立基于视觉外观和基于骨架的框架作为基准模型，并在ISLR101上充分训练和评估这些模型。这些模型在测试集上的准确率分别为97.01%和94.02%。此外，我们还发布了训练、验证和测试分割，以促进公平比较。 

---
# Towards Learnable Anchor for Deep Multi-View Clustering 

**Title (ZH)**: 面向学习的锚点深层多视图聚类 

**Authors**: Bocheng Wang, Chusheng Zeng, Mulin Chen, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12427)  

**Abstract**: Deep multi-view clustering incorporating graph learning has presented tremendous potential. Most methods encounter costly square time consumption w.r.t. data size. Theoretically, anchor-based graph learning can alleviate this limitation, but related deep models mainly rely on manual discretization approaches to select anchors, which indicates that 1) the anchors are fixed during model training and 2) they may deviate from the true cluster distribution. Consequently, the unreliable anchors may corrupt clustering results. In this paper, we propose the Deep Multi-view Anchor Clustering (DMAC) model that performs clustering in linear time. Concretely, the initial anchors are intervened by the positive-incentive noise sampled from Gaussian distribution, such that they can be optimized with a newly designed anchor learning loss, which promotes a clear relationship between samples and anchors. Afterwards, anchor graph convolution is devised to model the cluster structure formed by the anchors, and the mutual information maximization loss is built to provide cross-view clustering guidance. In this way, the learned anchors can better represent clusters. With the optimal anchors, the full sample graph is calculated to derive a discriminative embedding for clustering. Extensive experiments on several datasets demonstrate the superior performance and efficiency of DMAC compared to state-of-the-art competitors. 

**Abstract (ZH)**: Deep多视图锚点聚类结合图学习在线性时间内实现聚类 

---
# Unveiling Pitfalls: Understanding Why AI-driven Code Agents Fail at GitHub Issue Resolution 

**Title (ZH)**: 揭开陷阱：理解为何AI驱动的代码代理在GitHub问题解决中失败 

**Authors**: Zhi Chen, Wei Ma, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12374)  

**Abstract**: AI-driven software development has rapidly advanced with the emergence of software development agents that leverage large language models (LLMs) to tackle complex, repository-level software engineering tasks. These agents go beyond just generation of final code; they engage in multi-step reasoning, utilize various tools for code modification and debugging, and interact with execution environments to diagnose and iteratively resolve issues. However, most existing evaluations focus primarily on static analyses of final code outputs, yielding limited insights into the agents' dynamic problem-solving processes. To fill this gap, we conduct an in-depth empirical study on 3,977 solving-phase trajectories and 3,931 testing-phase logs from 8 top-ranked agents evaluated on 500 GitHub issues in the SWE-Bench benchmark. Our exploratory analysis shows that Python execution errors during the issue resolution phase correlate with lower resolution rates and increased reasoning overheads. We have identified the most prevalent errors -- such as ModuleNotFoundError and TypeError -- and highlighted particularly challenging errors like OSError and database-related issues (e.g., IntegrityError) that demand significantly more debugging effort. Furthermore, we have discovered 3 bugs in the SWE-Bench platform that affect benchmark fairness and accuracy; these issues have been reported to and confirmed by the maintainers. To promote transparency and foster future research, we publicly share our datasets and analysis scripts. 

**Abstract (ZH)**: 基于AI的软件开发随着利用大规模语言模型（LLMs）处理复杂仓库级软件工程任务的软件开发代理的出现而迅速发展。这些代理不仅生成最终代码，还进行多步推理，利用各种工具进行代码修改和调试，并与执行环境交互以诊断和迭代解决问题。然而，现有的大多数评估主要集中在最终代码输出的静态分析上，这限制了对代理动态问题解决过程的深入了解。为填平这一差距，我们对SWE-Bench基准测试中的8个顶级代理在解决500个GitHub问题过程中3,977个解题阶段轨迹和3,931个测试阶段日志进行了深入的实证研究。我们的探索性分析表明，问题解决阶段的Python执行错误与较低的问题解决率和增加的推理开销有关。我们确定了最常见的错误，如ModuleNotFoundError和TypeError，并强调了特别具有挑战性的错误，如OSError以及数据库相关问题（如IntegrityError），这些错误需要更多的调试努力。此外，我们还发现了SWE-Bench平台上的3个影响基准公平性和准确性的错误，这些问题已报告给并得到了维护者的确认。为促进透明度并促进未来的研究，我们公开分享了我们的数据集和分析脚本。 

---
# Synthetic Data for Robust AI Model Development in Regulated Enterprises 

**Title (ZH)**: 受监管企业中 robust AI 模型开发的合成数据应用 

**Authors**: Aditi Godbole  

**Link**: [PDF](https://arxiv.org/pdf/2503.12353)  

**Abstract**: In today's business landscape, organizations need to find the right balance between using their customers' data ethically to power AI solutions and being compliant regarding data privacy and data usage regulations. In this paper, we discuss synthetic data as a possible solution to this dilemma. Synthetic data is simulated data that mimics the real data. We explore how organizations in heavily regulated industries, such as financial institutions or healthcare organizations, can leverage synthetic data to build robust AI solutions while staying compliant. We demonstrate that synthetic data offers two significant advantages by allowing AI models to learn from more diverse data and by helping organizations stay compliant against data privacy laws with the use of synthetic data instead of customer information. We discuss case studies to show how synthetic data can be effectively used in the finance and healthcare sector while discussing the challenges of using synthetic data and some ethical questions it raises. Our research finds that synthetic data could be a game-changer for AI in regulated industries. The potential can be realized when industry, academia, and regulators collaborate to build solutions. We aim to initiate discussions on the use of synthetic data to build ethical, responsible, and effective AI systems in regulated enterprise industries. 

**Abstract (ZH)**: 在当今的商业环境中，组织需要在利用客户数据来驱动AI解决方案与遵守数据隐私和数据使用规定之间找到合适的平衡。本文探讨合成数据作为解决这一困境的可能解决方案。合成数据是模拟的、模拟真实数据的数据。我们探讨了在金融机构或医疗组织等高度监管行业中，组织如何利用合成数据构建稳健的AI解决方案，同时保持合规。我们证明合成数据提供了两大优势：使AI模型能够从更多样化的数据中学习，并通过使用合成数据而非客户信息帮助组织遵守数据隐私法规。我们讨论了金融和医疗保健行业的案例研究，展示了合成数据如何有效使用，同时讨论了使用合成数据面临的挑战及其引发的一些伦理问题。我们的研究发现，合成数据可能成为监管行业中AI游戏规则的改变者。实现其潜力需要行业、学术界和监管者合作构建解决方案。我们旨在发起关于在受监管的企业行业中利用合成数据构建伦理、负责且有效的AI系统的讨论。 

---
# General Table Question Answering via Answer-Formula Joint Generation 

**Title (ZH)**: 基于答案-公式联合生成的一般表格问答 

**Authors**: Zhongyuan Wang, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.12345)  

**Abstract**: Advanced table question answering (TableQA) methods prompt large language models (LLMs) to generate answer text, SQL query, Python code, or custom operations, which impressively improve the complex reasoning problems in the TableQA task. However, these methods lack the versatility to cope with specific question types or table structures. In contrast, the Spreadsheet Formula, the widely-used and well-defined operation language for tabular data, has not been thoroughly explored to solve TableQA. In this paper, we first attempt to use Formula as the logical form for solving complex reasoning on the tables with different structures. Specifically, we construct a large Formula-annotated TableQA dataset \texttt{FromulaQA} from existing datasets. In addition, we propose \texttt{TabAF}, a general table answering framework to solve multiple types of tasks over multiple types of tables simultaneously. Unlike existing methods, \texttt{TabAF} decodes answers and Formulas with a single LLM backbone, demonstrating great versatility and generalization. \texttt{TabAF} based on Llama3.1-70B achieves new state-of-the-art performance on the WikiTableQuestion, HiTab and TabFact. 

**Abstract (ZH)**: 使用Spreadsheet Formula解决具有不同结构的表格复杂推理问题：TabAF框架取得最新性能 

---
# Bi-Criteria Optimization for Combinatorial Bandits: Sublinear Regret and Constraint Violation under Bandit Feedback 

**Title (ZH)**: 组合臂拉动的双标准优化：基于臂反馈下的亚线性遗憾和约束违反而上的优化 

**Authors**: Vaneet Aggarwal, Shweta Jain, Subham Pokhriyal, Christopher John Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2503.12285)  

**Abstract**: In this paper, we study bi-criteria optimization for combinatorial multi-armed bandits (CMAB) with bandit feedback. We propose a general framework that transforms discrete bi-criteria offline approximation algorithms into online algorithms with sublinear regret and cumulative constraint violation (CCV) guarantees. Our framework requires the offline algorithm to provide an $(\alpha, \beta)$-bi-criteria approximation ratio with $\delta$-resilience and utilize $\texttt{N}$ oracle calls to evaluate the objective and constraint functions. We prove that the proposed framework achieves sub-linear regret and CCV, with both bounds scaling as ${O}\left(\delta^{2/3} \texttt{N}^{1/3}T^{2/3}\log^{1/3}(T)\right)$. Crucially, the framework treats the offline algorithm with $\delta$-resilience as a black box, enabling flexible integration of existing approximation algorithms into the CMAB setting. To demonstrate its versatility, we apply our framework to several combinatorial problems, including submodular cover, submodular cost covering, and fair submodular maximization. These applications highlight the framework's broad utility in adapting offline guarantees to online bi-criteria optimization under bandit feedback. 

**Abstract (ZH)**: 本文研究了具有带反馈的组合多臂 bandits (CMAB) 双准则优化问题。我们提出了一种通用框架，将离线双准则近似算法转化为具有亚线性后悔和累积约束违例（CCV）保证的在线算法。该框架要求离线算法提供 $(\alpha, \beta)$-双准则近似比，并具有 $\delta$-稳健性，同时利用 $\texttt{N}$ 次 oracle 调用评估目标函数和约束函数。我们证明了所提出框架实现了亚线性后悔和 CCV，两者界均为 ${O}\left(\delta^{2/3} \texttt{N}^{1/3}T^{2/3}\log^{1/3}(T)\right)$。关键地，该框架将具有 $\delta$-稳健性的离线算法视为黑盒，使得现有的近似算法可以灵活地整合到 CMAB 设置中。为了展示其灵活性，我们将该框架应用于多个组合问题，包括子模覆盖、子模成本覆盖和公平子模最大化。这些应用突显了该框架在带反馈的在线双准则优化中将离线保证适应的广泛用途。 

---
# A Novel Double Pruning method for Imbalanced Data using Information Entropy and Roulette Wheel Selection for Breast Cancer Diagnosis 

**Title (ZH)**: 基于信息熵和轮盘赌选择的乳腺癌症诊断新型双重剪枝方法 

**Authors**: Soufiane Bacha, Huansheng Ning, Belarbi Mostefa, Doreen Sebastian Sarwatt, Sahraoui Dhelim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12239)  

**Abstract**: Accurate illness diagnosis is vital for effective treatment and patient safety. Machine learning models are widely used for cancer diagnosis based on historical medical data. However, data imbalance remains a major challenge, leading to hindering classifier performance and reliability. The SMOTEBoost method addresses this issue by generating synthetic data to balance the dataset, but it may overlook crucial overlapping regions near the decision boundary and can produce noisy samples. This paper proposes RE-SMOTEBoost, an enhanced version of SMOTEBoost, designed to overcome these limitations. Firstly, RE-SMOTEBoost focuses on generating synthetic samples in overlapping regions to better capture the decision boundary using roulette wheel selection. Secondly, it incorporates a filtering mechanism based on information entropy to reduce noise, and borderline cases and improve the quality of generated data. Thirdly, we introduce a double regularization penalty to control the synthetic samples proximity to the decision boundary and avoid class overlap. These enhancements enable higher-quality oversampling of the minority class, resulting in a more balanced and effective training dataset. The proposed method outperforms existing state-of-the-art techniques when evaluated on imbalanced datasets. Compared to the top-performing sampling algorithms, RE-SMOTEBoost demonstrates a notable improvement of 3.22\% in accuracy and a variance reduction of 88.8\%. These results indicate that the proposed model offers a solid solution for medical settings, effectively overcoming data scarcity and severe imbalance caused by limited samples, data collection difficulties, and privacy constraints. 

**Abstract (ZH)**: 精确的疾病诊断对于有效的治疗和患者安全至关重要。基于历史医疗数据的机器学习模型广泛用于癌症诊断。然而，数据不平衡仍然是一个主要挑战，阻碍了分类器性能和可靠性。RE-SMOTEBoost方法通过生成合成数据来平衡数据集，解决了这一问题，但它可能会忽略决策边界附近的重叠区域，并可能生成噪声样本。本文提出了一种增强的RE-SMOTEBoost方法，旨在克服这些限制。首先，RE-SMOTEBoost重点关注在重叠区域生成合成样本，以更好地捕捉决策边界，并使用轮盘赌选择。其次，它结合了基于信息熵的过滤机制来减少噪声和边界案例，提高生成数据的质量。第三，引入双正则化惩罚以控制合成样本与决策边界的 proximity 并避免类别重叠。这些增强使少数类的过采样更具质量，从而生成更平衡和有效的训练数据集。该提出的模型在不平衡数据集上的评估中优于现有最先进的技术。与表现最佳的采样算法相比，RE-SMOTEBoost在准确率上提高了3.22%，方差减少了88.8%。这些结果表明，提出的模型为医疗环境提供了一个有效的解决方案，能够有效克服由样本稀缺性和严重不平衡引起的数据短缺和隐私限制问题。 

---
# Changing Base Without Losing Pace: A GPU-Efficient Alternative to MatMul in DNNs 

**Title (ZH)**: 保持节奏不变而换基底：DNN中MatMul的GPU高效替代方案 

**Authors**: Nir Ailon, Akhiad Bercovich, Omri Weinstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.12211)  

**Abstract**: We propose a cheaper alternative bilinear operator to matrix-multiplication in deep neural networks (DNNs). Unlike many stubborn attempts to accelerate MatMuls in DNN inference, this operator is supported by capabilities of existing GPU hardware, most notably NVIDIA TensorCores. To our knowledge, this is the first GPU-native acceleration technique which \emph{does not decrease} (in fact, increases) the number of trainable parameters of the network, mitigating the accuracy-loss of compression-based techniques. Hence, this operator is at the same time more expressive than MatMul, yet requires substantially \emph{fewer} FLOPs to evaluate. We term this new operator \emph{Strassen-Tile} (STL).
The main idea behind STL$(X,W)$ is a \emph{local} change-of-basis (learnable encoder) on weights and activation \emph{tiles}, after which we perform batched \emph{elementwise} products between tiles, and a final decoding transformation (inspired by algebraic pipelines from fast matrix and polynomial multiplication).
We compare STL against two benchmarks. The first one is SoTA T2T-ViT on Imagenet-1K. Here we show that replacing \emph{all} linear layers with STL and training from scratch, results in factor x2.7 reduction in FLOPs with a 0.5 \emph{accuracy improvement}. Our second speed-accuracy comparison benchmark for pretrained LLMs is the most practical GPU-acceleration technique, \twofour structured Sparsity. Finetuning TinyLlama \cite{tinyllama24} with STL layers on the Slim Pajama dataset, achieves similar accuracy to 2:4, with x2.2 FLOP speedup compared to x1.7 of the latter.
Finally, we discuss a group-theoretic approach for discovering \emph{universal} encoders for STL, which could lead to fast \emph{black-box} acceleration via approximate matrix-multiplication (AMM). 

**Abstract (ZH)**: 我们提出了一种用于深度神经网络中矩阵乘法的更便宜的双线性算子。这一算子通过现有GPU硬件的能力，尤其是NVIDIA TensorCores，支持了这种加速技术。据我们所知，这是首个在不减少（实际上增加了）网络可训练参数数量的情况下，减轻压缩技术精度损失的GPU原生加速技术。因此，这一算子在表达能力上比矩阵乘法更强，同时评估所需的基本运算单元远少于矩阵乘法。我们称这一新算子为Strassen-Tile (STL)。

STL$(X,W)$的主要思想是对权重和激活tile进行局部基底变换（可学习编码器），之后执行tile之间的批量逐元素乘法，并进行最终的解码变换（灵感来源于快速矩阵和多项式乘法的代数管道）。

我们将STL与两个基准进行比较。第一个基准是SoTA T2T-ViT在Imagenet-1K上的表现。实验结果显示，将所有线性层替换为STL并从头训练，FLOPs减少了2.7倍，同时精度提高了0.5倍。我们的第二个速度-精度比较基准是预训练的LLM，即高效的GPU加速技术twofour结构化稀疏性。使用STL层对Slim Pajama数据集进行TinyLlama微调，实现了与2:4类似的效果，并且相比于2:4，性能提升了2.2倍。

最后，我们讨论了一种群论方法，用于发现适用于STL的通用编码器，这一方法可能通过近似矩阵乘法（AMM）实现快速的黑盒加速。 

---
# Probabilistic Graph Circuits: Deep Generative Models for Tractable Probabilistic Inference over Graphs 

**Title (ZH)**: 可计算图概率电路：用于图上可计算概率推理的深度生成模型 

**Authors**: Milan Papež, Martin Rektoris, Václav Šmídl, Tomáš Pevný  

**Link**: [PDF](https://arxiv.org/pdf/2503.12162)  

**Abstract**: Deep generative models (DGMs) have recently demonstrated remarkable success in capturing complex probability distributions over graphs. Although their excellent performance is attributed to powerful and scalable deep neural networks, it is, at the same time, exactly the presence of these highly non-linear transformations that makes DGMs intractable. Indeed, despite representing probability distributions, intractable DGMs deny probabilistic foundations by their inability to answer even the most basic inference queries without approximations or design choices specific to a very narrow range of queries. To address this limitation, we propose probabilistic graph circuits (PGCs), a framework of tractable DGMs that provide exact and efficient probabilistic inference over (arbitrary parts of) graphs. Nonetheless, achieving both exactness and efficiency is challenging in the permutation-invariant setting of graphs. We design PGCs that are inherently invariant and satisfy these two requirements, yet at the cost of low expressive power. Therefore, we investigate two alternative strategies to achieve the invariance: the first sacrifices the efficiency, and the second sacrifices the exactness. We demonstrate that ignoring the permutation invariance can have severe consequences in anomaly detection, and that the latter approach is competitive with, and sometimes better than, existing intractable DGMs in the context of molecular graph generation. 

**Abstract (ZH)**: 可计算的概率图电路：兼具精确性和高效性的图生成模型 

---
# Weighted Graph Structure Learning with Attention Denoising for Node Classification 

**Title (ZH)**: 带有注意力去噪的加权图结构学习节点分类 

**Authors**: Tingting Wang, Jiaxin Su, Haobing Liu, Ruobing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12157)  

**Abstract**: Node classification in graphs aims to predict the categories of unlabeled nodes by utilizing a small set of labeled nodes. However, weighted graphs often contain noisy edges and anomalous edge weights, which can distort fine-grained relationships between nodes and hinder accurate classification. We propose the Edge Weight-aware Graph Structure Learning (EWGSL) method, which combines weight learning and graph structure learning to address these issues. EWGSL improves node classification by redefining attention coefficients in graph attention networks to incorporate node features and edge weights. It also applies graph structure learning to sparsify attention coefficients and uses a modified InfoNCE loss function to enhance performance by adapting to denoised graph weights. Extensive experimental results show that EWGSL has an average Micro-F1 improvement of 17.8% compared with the best baseline. 

**Abstract (ZH)**: 图中节点分类旨在通过利用少量标记节点来预测未标记节点的类别。然而，加权图中经常包含噪声边和异常边权重，这会扭曲节点之间的细粒度关系并妨碍准确分类。我们提出了边权重感知图结构学习（EWGSL）方法，该方法结合了权重学习和图结构学习以解决这些问题。EWGSL 通过在图注意力网络中重新定义注意力系数来纳入节点特征和边权重，从而改善节点分类。它还应用图结构学习来稀疏化注意力系数，并使用修改后的 InfoNCE 损失函数来通过适应去噪后的图权重来增强性能。广泛的经验研究表明，与最佳基线相比，EWGSL 的平均 Micro-F1 改进了17.8%。 

---
# Robust Isolation Forest using Soft Sparse Random Projection and Valley Emphasis Method 

**Title (ZH)**: 鲁棒孤立森林基于软稀疏随机投影和谷值强调方法 

**Authors**: Hun Kang, Kyoungok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12125)  

**Abstract**: Isolation Forest (iForest) is an unsupervised anomaly detection algorithm designed to effectively detect anomalies under the assumption that anomalies are ``few and different." Various studies have aimed to enhance iForest, but the resulting algorithms often exhibited significant performance disparities across datasets. Additionally, the challenge of isolating rare and widely distributed anomalies persisted in research focused on improving splits. To address these challenges, we introduce Robust iForest (RiForest). RiForest leverages both existing features and random hyperplanes obtained through soft sparse random projection to identify superior split features for anomaly detection, independent of datasets. It utilizes the underutilized valley emphasis method for optimal split point determination and incorporates sparsity randomization in soft sparse random projection for enhanced anomaly detection robustness. Across 24 benchmark datasets, experiments demonstrate RiForest's consistent outperformance of existing algorithms in anomaly detection, emphasizing stability and robustness to noise variables. 

**Abstract (ZH)**: Robust iForest (RiForest): An Enhanced Anomaly Detection Algorithm 

---
# ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables 

**Title (ZH)**: ChronosX：适应外生变量的预训练时间序列模型 

**Authors**: Sebastian Pineda Arango, Pedro Mercado, Shubham Kapoor, Abdul Fatir Ansari, Lorenzo Stella, Huibin Shen, Hugo Senetaire, Caner Turkmen, Oleksandr Shchur, Danielle C. Maddix, Michael Bohlke-Schneider, Yuyang Wang, Syama Sundar Rangapuram  

**Link**: [PDF](https://arxiv.org/pdf/2503.12107)  

**Abstract**: Covariates provide valuable information on external factors that influence time series and are critical in many real-world time series forecasting tasks. For example, in retail, covariates may indicate promotions or peak dates such as holiday seasons that heavily influence demand forecasts. Recent advances in pretraining large language model architectures for time series forecasting have led to highly accurate forecasters. However, the majority of these models do not readily use covariates as they are often specific to a certain task or domain. This paper introduces a new method to incorporate covariates into pretrained time series forecasting models. Our proposed approach incorporates covariate information into pretrained forecasting models through modular blocks that inject past and future covariate information, without necessarily modifying the pretrained model in consideration. In order to evaluate our approach, we introduce a benchmark composed of 32 different synthetic datasets with varying dynamics to evaluate the effectivity of forecasting models with covariates. Extensive evaluations on both synthetic and real datasets show that our approach effectively incorporates covariate information into pretrained models, outperforming existing baselines. 

**Abstract (ZH)**: 协变量提供了影响时间序列的外部因素的宝贵信息，在许多实际时间序列预测任务中至关重要。本论文提出了一种新方法，将协变量纳入预训练的时间序列预测模型。我们提出的方法通过模块化块将过去的和未来的协变量信息注入预训练的预测模型中，而无需修改所考虑的预训练模型。为了评估该方法，我们引入了一个由32个不同动态的合成数据集组成的基准，以评估包含协变量的预测模型的效果。在合成数据集和真实数据集上的广泛评估表明，我们的方法有效地将协变量信息纳入预训练模型，并优于现有基线。 

---
# Revisiting Training-Inference Trigger Intensity in Backdoor Attacks 

**Title (ZH)**: 回顾训练-推理触发强度在后门攻击中的作用 

**Authors**: Chenhao Lin, Chenyang Zhao, Shiwei Wang, Longtian Wang, Chao Shen, Zhengyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12058)  

**Abstract**: Backdoor attacks typically place a specific trigger on certain training data, such that the model makes prediction errors on inputs with that trigger during inference. Despite the core role of the trigger, existing studies have commonly believed a perfect match between training-inference triggers is optimal. In this paper, for the first time, we systematically explore the training-inference trigger relation, particularly focusing on their mismatch, based on a Training-Inference Trigger Intensity Manipulation (TITIM) workflow. TITIM specifically investigates the training-inference trigger intensity, such as the size or the opacity of a trigger, and reveals new insights into trigger generalization and overfitting.
These new insights challenge the above common belief by demonstrating that the training-inference trigger mismatch can facilitate attacks in two practical scenarios, posing more significant security threats than previously thought. First, when the inference trigger is fixed, using training triggers with mixed intensities leads to stronger attacks than using any single intensity. For example, on CIFAR-10 with ResNet-18, mixing training triggers with 1.0 and 0.1 opacities improves the worst-case attack success rate (ASR) (over different testing opacities) of the best single-opacity attack from 10.61\% to 92.77\%. Second, intentionally using certain mismatched training-inference triggers can improve the attack stealthiness, i.e., better bypassing defenses. For example, compared to the training/inference intensity of 1.0/1.0, using 1.0/0.7 decreases the area under the curve (AUC) of the Scale-Up defense from 0.96 to 0.62, while maintaining a high attack ASR (99.65\% vs. 91.62\%). The above new insights are validated to be generalizable across different backdoor attacks, models, datasets, tasks, and (digital/physical) domains. 

**Abstract (ZH)**: Backdoor攻击通常在特定训练数据上放置一个特定触发器，使得模型在推理时对于带有该触发器的输入产生预测错误。尽管触发器的核心作用已被公认，但现有研究普遍认为训练-推理触发器的完美匹配是最优的。本文首次系统地探索了训练-推理触发器的关系，特别是关注它们的不匹配情况，基于一次训练-推理触发器强度操控（TITIM）工作流程。TITIM具体研究了训练-推理触发器的强度，如触发器的大小或透明度，并揭示了触发器泛化和过拟合的新见解。这些新的见解通过证明训练-推理触发器不匹配在两种实际场景中能够促进攻击，挑战了上述普遍信念，提高了潜在的安全威胁。首先，当推理触发器固定时，使用混合强度的训练触发器比使用任何单一强度的效果更强。例如，在ResNet-18上的CIFAR-10数据集，混合使用0.1和1.0透明度的训练触发器将最佳单一透明度攻击的最坏情况攻击成功率（ASR）从10.61%提高到92.77%。其次，有意使用特定不匹配的训练-推理触发器可以提高攻击的隐蔽性，即更好地绕过防守。例如，相比训练/推理强度为1.0/1.0，使用1.0/0.7降低了Scale-Up防守的AUC从0.96到0.62，同时保持较高的攻击ASR（99.65% vs. 91.62%）。上述新的见解被验证为在不同的后门攻击、模型、数据集、任务及（数字/物理）领域中具有普适性。 

---
# Ferret: An Efficient Online Continual Learning Framework under Varying Memory Constraints 

**Title (ZH)**: Ferret: 一种在变化的内存约束下高效的在线连续学习框架 

**Authors**: Yuhao Zhou, Yuxin Tian, Jindi Lv, Mingjia Shi, Yuanxi Li, Qing Ye, Shuhao Zhang, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2503.12053)  

**Abstract**: In the realm of high-frequency data streams, achieving real-time learning within varying memory constraints is paramount. This paper presents Ferret, a comprehensive framework designed to enhance online accuracy of Online Continual Learning (OCL) algorithms while dynamically adapting to varying memory budgets. Ferret employs a fine-grained pipeline parallelism strategy combined with an iterative gradient compensation algorithm, ensuring seamless handling of high-frequency data with minimal latency, and effectively counteracting the challenge of stale gradients in parallel training. To adapt to varying memory budgets, its automated model partitioning and pipeline planning optimizes performance regardless of memory limitations. Extensive experiments across 20 benchmarks and 5 integrated OCL algorithms show Ferret's remarkable efficiency, achieving up to 3.7$\times$ lower memory overhead to reach the same online accuracy compared to competing methods. Furthermore, Ferret consistently outperforms these methods across diverse memory budgets, underscoring its superior adaptability. These findings position Ferret as a premier solution for efficient and adaptive OCL framework in real-time environments. 

**Abstract (ZH)**: 在高频率数据流领域，实现不同内存约束下的实时学习至关重要。本文提出Ferret，一种全面框架，旨在提升在线连续学习（OCL）算法的在线准确性，同时动态适应不同的内存预算。Ferret采用细粒度管道并行策略结合迭代梯度补偿算法，确保在最小延迟的情况下无缝处理高频率数据，并有效地克服平行训练中过时梯度的挑战。为了适应不同的内存预算，其自动模型分割和管道规划优化了性能，不受内存限制的影响。在20个基准和5种集成OCL算法上的广泛实验表明，Ferret具有显著的效率，与竞争方法相比，实现相同在线准确性时的内存开销低至3.7倍。此外，Ferret在不同内存预算下也始终表现出优越的性能，突显了其卓越的适应性。这些发现将Ferret定位为实时环境中高效且适应性强的OCL框架的首选解决方案。 

---
# Unsupervised Graph Anomaly Detection via Multi-Hypersphere Heterophilic Graph Learning 

**Title (ZH)**: 无监督图异常检测 via 多超球体异质图学习 

**Authors**: Hang Ni, Jindong Han, Nengjun Zhu, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12037)  

**Abstract**: Graph Anomaly Detection (GAD) plays a vital role in various data mining applications such as e-commerce fraud prevention and malicious user detection. Recently, Graph Neural Network (GNN) based approach has demonstrated great effectiveness in GAD by first encoding graph data into low-dimensional representations and then identifying anomalies under the guidance of supervised or unsupervised signals. However, existing GNN-based approaches implicitly follow the homophily principle (i.e., the "like attracts like" phenomenon) and fail to learn discriminative embedding for anomalies that connect vast normal nodes. Moreover, such approaches identify anomalies in a unified global perspective but overlook diversified abnormal patterns conditioned on local graph context, leading to suboptimal performance. To overcome the aforementioned limitations, in this paper, we propose a Multi-hypersphere Heterophilic Graph Learning (MHetGL) framework for unsupervised GAD. Specifically, we first devise a Heterophilic Graph Encoding (HGE) module to learn distinguishable representations for potential anomalies by purifying and augmenting their neighborhood in a fully unsupervised manner. Then, we propose a Multi-Hypersphere Learning (MHL) module to enhance the detection capability for context-dependent anomalies by jointly incorporating critical patterns from both global and local perspectives. Extensive experiments on ten real-world datasets show that MHetGL outperforms 14 baselines. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 多超球面异ophilous图学习的无监督图异常检测 

---
# Variance-Dependent Regret Lower Bounds for Contextual Bandits 

**Title (ZH)**: 基于上下文的多臂老虎机依赖方差的寄存器下界 

**Authors**: Jiafan He, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12020)  

**Abstract**: Variance-dependent regret bounds for linear contextual bandits, which improve upon the classical $\tilde{O}(d\sqrt{K})$ regret bound to $\tilde{O}(d\sqrt{\sum_{k=1}^K\sigma_k^2})$, where $d$ is the context dimension, $K$ is the number of rounds, and $\sigma^2_k$ is the noise variance in round $k$, has been widely studied in recent years. However, most existing works focus on the regret upper bounds instead of lower bounds. To our knowledge, the only lower bound is from Jia et al. (2024), which proved that for any eluder dimension $d_{\textbf{elu}}$ and total variance budget $\Lambda$, there exists an instance with $\sum_{k=1}^K\sigma_k^2\leq \Lambda$ for which any algorithm incurs a variance-dependent lower bound of $\Omega(\sqrt{d_{\textbf{elu}}\Lambda})$. However, this lower bound has a $\sqrt{d}$ gap with existing upper bounds. Moreover, it only considers a fixed total variance budget $\Lambda$ and does not apply to a general variance sequence $\{\sigma_1^2,\ldots,\sigma_K^2\}$. In this paper, to overcome the limitations of Jia et al. (2024), we consider the general variance sequence under two settings. For a prefixed sequence, where the entire variance sequence is revealed to the learner at the beginning of the learning process, we establish a variance-dependent lower bound of $\Omega(d \sqrt{\sum_{k=1}^K\sigma_k^2 }/\log K)$ for linear contextual bandits. For an adaptive sequence, where an adversary can generate the variance $\sigma_k^2$ in each round $k$ based on historical observations, we show that when the adversary must generate $\sigma_k^2$ before observing the decision set $\mathcal{D}_k$, a similar lower bound of $\Omega(d\sqrt{ \sum_{k=1}^K\sigma_k^2} /\log^6(dK))$ holds. In both settings, our results match the upper bounds of the SAVE algorithm (Zhao et al., 2023) up to logarithmic factors. 

**Abstract (ZH)**: 依赖方差的线性上下文Bandits的懊悔界研究：从$\tilde{O}(d\sqrt{K})$改进到$\tilde{O}(d\sqrt{\sum_{k=1}^K\sigma_k^2})$ 

---
# Winning the MIDST Challenge: New Membership Inference Attacks on Diffusion Models for Tabular Data Synthesis 

**Title (ZH)**: 赢得MIDST挑战：针对表格数据合成的扩散模型会员推理攻击新方法 

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12008)  

**Abstract**: Tabular data synthesis using diffusion models has gained significant attention for its potential to balance data utility and privacy. However, existing privacy evaluations often rely on heuristic metrics or weak membership inference attacks (MIA), leaving privacy risks inadequately assessed. In this work, we conduct a rigorous MIA study on diffusion-based tabular synthesis, revealing that state-of-the-art attacks designed for image models fail in this setting. We identify noise initialization as a key factor influencing attack efficacy and propose a machine-learning-driven approach that leverages loss features across different noises and time steps. Our method, implemented with a lightweight MLP, effectively learns membership signals, eliminating the need for manual optimization. Experimental results from the MIDST Challenge @ SaTML 2025 demonstrate the effectiveness of our approach, securing first place across all tracks. Code is available at this https URL. 

**Abstract (ZH)**: 基于扩散模型的表格数据合成中的会员身份推理研究：揭示先进的图像模型攻击在此场景下的失效，并提出一种机器学习驱动的方法以评估隐私风险。 

---
# Goal-Oriented Source Coding using LDPC Codes for Compressed-Domain Image Classification 

**Title (ZH)**: 基于LDPC码的目标导向源编码在压缩域图像分类中的应用 

**Authors**: Ahcen Aliouat, Elsa Dupraz  

**Link**: [PDF](https://arxiv.org/pdf/2503.11954)  

**Abstract**: In the emerging field of goal-oriented communications, the focus has shifted from reconstructing data to directly performing specific learning tasks, such as classification, segmentation, or pattern recognition, on the received coded data. In the commonly studied scenario of classification from compressed images, a key objective is to enable learning directly on entropy-coded data, thereby bypassing the computationally intensive step of data reconstruction. Conventional entropy-coding methods, such as Huffman and Arithmetic coding, are effective for compression but disrupt the data structure, making them less suitable for direct learning without decoding. This paper investigates the use of low-density parity-check (LDPC) codes -- originally designed for channel coding -- as an alternative entropy-coding approach. It is hypothesized that the structured nature of LDPC codes can be leveraged more effectively by deep learning models for tasks like classification. At the receiver side, gated recurrent unit (GRU) models are trained to perform image classification directly on LDPC-coded data. Experiments on datasets like MNIST, Fashion-MNIST, and CIFAR show that LDPC codes outperform Huffman and Arithmetic coding in classification tasks, while requiring significantly smaller learning models. Furthermore, the paper analyzes why LDPC codes preserve data structure more effectively than traditional entropy-coding techniques and explores the impact of key code parameters on classification performance. These results suggest that LDPC-based entropy coding offers an optimal balance between learning efficiency and model complexity, eliminating the need for prior decoding. 

**Abstract (ZH)**: 目标导向通信中编码数据上的直接学习 

---
# Privacy Ethics Alignment in AI (PEA-AI): A Stakeholder-Centric Based Framework for Ethcial AI 

**Title (ZH)**: 隐私伦理一致性的AI (PEA-AI): 一个基于利益相关者中心的道德AI框架 

**Authors**: Ankur Barthwal, Molly Campbell, Ajay Kumar Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2503.11950)  

**Abstract**: The increasing integration of Artificial Intelligence (AI) in digital ecosystems has reshaped privacy dynamics, particularly for young digital citizens navigating data-driven environments. This study explores evolving privacy concerns across three key stakeholder groups, digital citizens (ages 16-19), parents, educators, and AI professionals, and assesses differences in data ownership, trust, transparency, parental mediation, education, and risk-benefit perceptions. Employing a grounded theory methodology, this research synthesizes insights from 482 participants through structured surveys, qualitative interviews, and focus groups. The findings reveal distinct privacy expectations- Young users emphasize autonomy and digital freedom, while parents and educators advocate for regulatory oversight and AI literacy programs. AI professionals, in contrast, prioritize the balance between ethical system design and technological efficiency. The data further highlights gaps in AI literacy and transparency, emphasizing the need for comprehensive, stakeholder-driven privacy frameworks that accommodate diverse user needs. Using comparative thematic analysis, this study identifies key tensions in privacy governance and develops the novel Privacy-Ethics Alignment in AI (PEA-AI) model, which structures privacy decision-making as a dynamic negotiation between stakeholders. By systematically analyzing themes such as transparency, user control, risk perception, and parental mediation, this research provides a scalable, adaptive foundation for AI governance, ensuring that privacy protections evolve alongside emerging AI technologies and youth-centric digital interactions. 

**Abstract (ZH)**: 人工智能在数字生态系统中的不断整合重塑了隐私动态，特别是在数据驱动环境中 navigating 的年轻数字公民尤为明显。本研究探究了跨三类关键利益相关者——数字公民（16-19岁）、家长、教育者和AI专业人士——的 evolving 隐私关切，并评估了他们在数据所有权、信任、透明度、家长介入、教育和风险效益感知方面的差异。通过扎根理论方法，本研究结合结构化调查、质性访谈和焦点小组，从482名参与者中整合了见解。研究发现表明，年轻用户强调自主和数字自由，而家长和教育者则倡导监管监督和AI素养计划。相比之下，AI专业人士更侧重于伦理系统设计与技术效率之间的平衡。数据进一步突显了AI素养和透明度的缺口，强调了需要全面、以利益相关者为导向的隐私框架，以适应用户多样化的需求。通过比较主题分析，本研究识别了隐私治理中的关键张力，并提出了新的隐私-伦理在AI中的契合模型（PEA-AI），将隐私决策视为利益相关者之间的动态谈判结构。通过对透明度、用户控制、风险感知和家长介入等主题的系统分析，本研究为AI治理提供了可扩展和适应的基础，确保了随着新兴AI技术和以青年为中心的数字互动扩展，隐私保护能同步发展。 

---
# Ethical AI for Young Digital Citizens: A Call to Action on Privacy Governance 

**Title (ZH)**: 面向年轻数字公民的伦理AI：隐私治理行动呼吁 

**Authors**: Austin Shouli, Ankur Barthwal, Molly Campbell, Ajay Kumar Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2503.11947)  

**Abstract**: The rapid expansion of Artificial Intelligence (AI) in digital platforms used by youth has created significant challenges related to privacy, autonomy, and data protection. While AI-driven personalization offers enhanced user experiences, it often operates without clear ethical boundaries, leaving young users vulnerable to data exploitation and algorithmic biases. This paper presents a call to action for ethical AI governance, advocating for a structured framework that ensures youth-centred privacy protections, transparent data practices, and regulatory oversight. We outline key areas requiring urgent intervention, including algorithmic transparency, privacy education, parental data-sharing ethics, and accountability measures. Through this approach, we seek to empower youth with greater control over their digital identities and propose actionable strategies for policymakers, AI developers, and educators to build a fairer and more accountable AI ecosystem. 

**Abstract (ZH)**: 人工智能在青年使用的数字平台上的迅速扩张引发了隐私、自主权和数据保护方面的重大挑战。虽然基于AI的个性化服务提升了用户体验，但往往会缺乏明确的伦理边界，使年轻用户面临数据滥用和算法偏见的风险。本文呼吁建立伦理AI治理框架，确保以青年为中心的数据隐私保护、透明的数据实践和监管监督。我们指出了亟需干预的关键领域，包括算法透明性、隐私教育、家长数据共享伦理和问责措施。通过这种 approach，我们旨在赋予青年对数字身份的更大控制权，并为政策制定者、AI开发者和教育者提出了构建更加公平和负责任的AI生态系统的方法。 

---
# Human Digital Twins in Personalized Healthcare: An Overview and Future Perspectives 

**Title (ZH)**: 个人数字孪生在个性化医疗中的应用：综述与未来展望 

**Authors**: Melvin Mokhtari  

**Link**: [PDF](https://arxiv.org/pdf/2503.11944)  

**Abstract**: Digital twins (DTs) are redefining healthcare by paving the way for more personalized, proactive, and intelligent medical interventions. As the shift toward personalized care intensifies, there is a growing need for an individual's virtual replica that delivers the right treatment at the optimal time and in the most effective manner. The emerging concept of a Human Digital Twin (HDT) holds the potential to revolutionize the traditional healthcare system much like digital twins have transformed manufacturing and aviation. An HDT mirrors the physical entity of a human body through a dynamic virtual model that continuously reflects changes in molecular, physiological, emotional, and lifestyle factors. This digital representation not only supports remote monitoring, diagnosis, and prescription but also facilitates surgery, rehabilitation, and overall personalized care, thereby relieving pressure on conventional healthcare frameworks. Despite its promising advantages, there are considerable research challenges to overcome as HDT technology evolves. In this study, I will initially delineate the distinctions between traditional digital twins and HDTs, followed by an exploration of the networking architecture integral to their operation--from data acquisition and communication to computation, management, and decision-making--thereby offering insights into how these innovations may reshape the modern healthcare industry. 

**Abstract (ZH)**: 数字孪生在重塑个性化、前瞻性和智能医疗服务中的作用：人类数字孪生的前景与挑战 

---
# A Framework for Evaluating Emerging Cyberattack Capabilities of AI 

**Title (ZH)**: 一种评估新兴人工智能攻击能力的框架 

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11917)  

**Abstract**: As frontier models become more capable, the community has attempted to evaluate their ability to enable cyberattacks. Performing a comprehensive evaluation and prioritizing defenses are crucial tasks in preparing for AGI safely. However, current cyber evaluation efforts are ad-hoc, with no systematic reasoning about the various phases of attacks, and do not provide a steer on how to use targeted defenses. In this work, we propose a novel approach to AI cyber capability evaluation that (1) examines the end-to-end attack chain, (2) helps to identify gaps in the evaluation of AI threats, and (3) helps defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation to support red teaming. To achieve these goals, we propose adapting existing cyberattack chain frameworks to AI systems. We analyze over 12,000 instances of real-world attempts to use AI in cyberattacks catalogued by Google's Threat Intelligence Group. Using this analysis, we curate a representative collection of seven cyberattack chain archetypes and conduct a bottleneck analysis to identify areas of potential AI-driven cost disruption. Our evaluation benchmark consists of 50 new challenges spanning different phases of cyberattacks. Based on this, we devise targeted cybersecurity model evaluations, report on the potential for AI to amplify offensive cyber capabilities across specific attack phases, and conclude with recommendations on prioritizing defenses. In all, we consider this to be the most comprehensive AI cyber risk evaluation framework published so far. 

**Abstract (ZH)**: 前沿模型能力不断提升，社区已尝试评估其是否能-enable-黑客攻击。全面评估和优先级排序防御是为安全地准备AGI所需的关键任务。然而，当前的网络评估努力往往是随机进行的，缺乏对攻击各个阶段的系统性分析，也无法为如何使用有针对性的防御提供指导。在本文中，我们提出了一种新的AI网络能力评估方法，该方法（1）考查端到端攻击链，（2）有助于识别AI威胁评估中的缺口，（3）帮助防御者优先考虑有针对性的缓解措施，并进行AI支持的对手仿真以支持红队演练。为了实现这些目标，我们提出将现有的网络攻击链框架适应到AI系统。我们分析了谷歌威胁情报团队记录的超过12,000个实际的AI在网络攻击中的应用实例。基于这些分析，我们精心挑选了七个网络攻击链原型，并进行了瓶颈分析以识别潜在的AI驱动的成本中断领域。我们的评估基准包括50项新的挑战，涵盖了网络攻击的不同阶段。在此基础上，我们设计了有针对性的网络安全模型评估，报告了AI在特定攻击阶段增强 Offensive 网络能力的潜力，并提供了优先防御的建议。总的来说，我们认为这是迄今为止最全面的AI网络风险评估框架。 

---
# How Problematic Writer-AI Interactions (Rather than Problematic AI) Hinder Writers' Idea Generation 

**Title (ZH)**: 如何糟糕的作家-AI交互（而非糟糕的AI）妨碍作家的idea生成 

**Authors**: Khonzoda Umarova, Talia Wise, Zhuoer Lyu, Mina Lee, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11915)  

**Abstract**: Writing about a subject enriches writers' understanding of that subject. This cognitive benefit of writing -- known as constructive learning -- is essential to how students learn in various disciplines. However, does this benefit persist when students write with generative AI writing assistants? Prior research suggests the answer varies based on the type of AI, e.g., auto-complete systems tend to hinder ideation, while assistants that pose Socratic questions facilitate it. This paper adds an additional perspective. Through a case study, we demonstrate that the impact of genAI on students' idea development depends not only on the AI but also on the students and, crucially, their interactions in between. Students who proactively explored ideas gained new ideas from writing, regardless of whether they used auto-complete or Socratic AI assistants. Those who engaged in prolonged, mindless copyediting developed few ideas even with a Socratic AI. These findings suggest opportunities in designing AI writing assistants, not merely by creating more thought-provoking AI, but also by fostering more thought-provoking writer-AI interactions. 

**Abstract (ZH)**: 写作关于某个主题可以丰富作者对该主题的理解。这种写作带来的认知收益——称为建设性学习——是学生在各个学科中学习的重要方式。然而，当学生使用生成性AI写作助手写作时，这种收益是否会持续存在？先前的研究表明，这取决于AI的类型，例如自动补全系统往往会妨碍创新思维，而提出苏格拉底式问题的助手则有助于创新思维。本文通过案例研究增加了新的视角。我们证明生成性AI对学生想法发展的影响不仅取决于AI本身，还取决于学生及其互动。积极探索想法的学生无论使用自动补全还是苏格拉底式AI助手都能获得新想法。而长时间进行机械润色的学生即使使用了苏格拉底式AI也无法产生许多新想法。这些发现表明，在设计AI写作助手时，不仅需要创造更具启发性的AI，还需要促进更具启发性的作家-AI互动。 

---
# RTD-Lite: Scalable Topological Analysis for Comparing Weighted Graphs in Learning Tasks 

**Title (ZH)**: RTD-Lite: 可扩展的拓扑分析方法用于学习任务中加权图的比较 

**Authors**: Eduard Tulchinskii, Daria Voronkova, Ilya Trofimov, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2503.11910)  

**Abstract**: Topological methods for comparing weighted graphs are valuable in various learning tasks but often suffer from computational inefficiency on large datasets. We introduce RTD-Lite, a scalable algorithm that efficiently compares topological features, specifically connectivity or cluster structures at arbitrary scales, of two weighted graphs with one-to-one correspondence between vertices. Using minimal spanning trees in auxiliary graphs, RTD-Lite captures topological discrepancies with $O(n^2)$ time and memory complexity. This efficiency enables its application in tasks like dimensionality reduction and neural network training. Experiments on synthetic and real-world datasets demonstrate that RTD-Lite effectively identifies topological differences while significantly reducing computation time compared to existing methods. Moreover, integrating RTD-Lite into neural network training as a loss function component enhances the preservation of topological structures in learned representations. Our code is publicly available at this https URL 

**Abstract (ZH)**: 基于拓扑的方法在比较加权图方面对各种学习任务很有价值，但在处理大数据集时通常面临计算效率低的问题。我们介绍了RTD-Lite，这是一种可扩展算法，能高效比较具有顶点一对一对应关系的两个加权图的拓扑特征，特别是任意尺度的连通性或聚类结构。利用辅助图的最小生成树，RTD-Lite 在 $O(n^2)$ 的时间复杂性和内存复杂度下捕捉拓扑差异。这种高效性使其能够应用于诸如降维和神经网络训练等任务。在合成和真实数据集上的实验表明，RTD-Lite 有效地识别了拓扑差异，同时显著减少了计算时间，相较于现有方法。此外，将RTD-Lite 集成到神经网络训练中作为损失函数组件，能够增强学习表示中拓扑结构的保留。我们的代码可在以下网址获得。 

---
# Revisiting FastMap: New Applications 

**Title (ZH)**: 重新审视FastMap：新的应用领域 

**Authors**: Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11908)  

**Abstract**: FastMap was first introduced in the Data Mining community for generating Euclidean embeddings of complex objects. In this dissertation, we first present FastMap to generate Euclidean embeddings of graphs in near-linear time: The pairwise Euclidean distances approximate a desired graph-based distance function on the vertices. We then apply the graph version of FastMap to efficiently solve various graph-theoretic problems of significant interest in AI: including facility location, top-K centrality computations, community detection and block modeling, and graph convex hull computations. We also present a novel learning framework, called FastMapSVM, by combining FastMap and Support Vector Machines. We then apply FastMapSVM to predict the satisfiability of Constraint Satisfaction Problems and to classify seismograms in Earthquake Science. 

**Abstract (ZH)**: FastMap在数据挖掘社区首次被引入以生成复杂对象的欧几里得嵌入，在本论文中，我们首先介绍FastMap以在接近线性时间内生成图的欧几里得嵌入：pairwise欧几里得距离近似于顶点上的期望图基距离函数。然后，我们应用图版本的FastMap高效解决人工智能中具有重要意义的多种图论问题，包括设施定位、Top-K中心性计算、社区检测与模块化以及图凸包计算。我们还提出了一种新的学习框架FastMapSVM，将FastMap与支持向量机结合。最后，我们将FastMapSVM应用于预测约束满足问题的可满足性以及在地震科学中对地震检波器的分类。 

---
# Characterizing GPU Resilience and Impact on AI/HPC Systems 

**Title (ZH)**: GPU容错特性及其对AI/HPC系统的影响 

**Authors**: Shengkun Cui, Archit Patke, Ziheng Chen, Aditya Ranjan, Hung Nguyen, Phuong Cao, Saurabh Jha, Brett Bode, Gregory Bauer, Chandra Narayanaswami, Daby Sow, Catello Di Martino, Zbigniew T. Kalbarczyk, Ravishankar K. Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.11901)  

**Abstract**: In this study, we characterize GPU failures in Delta, the current large-scale AI system with over 600 petaflops of peak compute throughput. The system comprises GPU and non-GPU nodes with modern AI accelerators, such as NVIDIA A40, A100, and H100 GPUs. The study uses two and a half years of data on GPU errors. We evaluate the resilience of GPU hardware components to determine the vulnerability of different GPU components to failure and their impact on the GPU and node availability. We measure the key propagation paths in GPU hardware, GPU interconnect (NVLink), and GPU memory. Finally, we evaluate the impact of the observed GPU errors on user jobs. Our key findings are: (i) Contrary to common beliefs, GPU memory is over 30x more reliable than GPU hardware in terms of MTBE (mean time between errors). (ii) The newly introduced GSP (GPU System Processor) is the most vulnerable GPU hardware component. (iii) NVLink errors did not always lead to user job failure, and we attribute it to the underlying error detection and retry mechanisms employed. (iv) We show multiple examples of hardware errors originating from one of the key GPU hardware components, leading to application failure. (v) We project the impact of GPU node availability on larger scales with emulation and find that significant overprovisioning between 5-20% would be necessary to handle GPU failures. If GPU availability were improved to 99.9%, the overprovisioning would be reduced by 4x. 

**Abstract (ZH)**: 本研究characterizes GPU故障情况，涉及当前具有超过600 petaflops峰值计算能力的大规模AI系统Delta。该系统包含GPU节点和非GPU节点，配备了现代AI加速器，如NVIDIA A40、A100和H100 GPU。研究使用了近两年半关于GPU错误的数据。我们评估了GPU硬件组件的韧性，以确定不同GPU组件对其失效的脆弱性及其对GPU和节点可用性的影响。我们测量了GPU硬件的关键传播路径、GPU互连（NVLink）和GPU内存。最后，我们评估了观察到的GPU错误对用户作业的影响。我们的主要发现包括：（i）与普遍认知相反，从平均错误间隔（MTBE）角度来看，GPU内存比GPU硬件可靠30多倍。（ii）新引入的GSP（GPU系统处理器）是最脆弱的GPU硬件组件。（iii）NVLink错误并不总是导致用户作业失败，我们认为这是由于底层的错误检测和重试机制。（iv）我们展示了来自关键GPU硬件组件的多个硬件错误实例，导致应用程序失败。（v）我们通过模拟研究了GPU节点可用性在更大规模的影响，并发现为了处理GPU故障，需要5-20%的超额预订。如果GPU可用性提高到99.9%，超额预订将减少4倍。 

---
# Expressive Music Data Processing and Generation 

**Title (ZH)**: 具有表现力的音乐数据处理与生成 

**Authors**: Jingwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11896)  

**Abstract**: Musical expressivity and coherence are indispensable in music composition and performance, while often neglected in modern AI generative models. In this work, we introduce a listening-based data-processing technique that captures the expressivity in musical performance. This technique derived from Weber's law reflects the human perceptual truth of listening and preserves musical subtlety and expressivity in the training input. To facilitate musical coherence, we model the output interdependencies among multiple arguments in the music data such as pitch, duration, velocity, etc. in the neural networks based on the probabilistic chain rule. In practice, we decompose the multi-output sequential model into single-output submodels and condition previously sampled outputs on the subsequent submodels to induce conditional distributions. Finally, to select eligible sequences from all generations, a tentative measure based on the output entropy was proposed. The entropy sequence is set as a criterion to select predictable and stable generations, which is further studied under the context of informational aesthetic measures to quantify musical pleasure and information gain along the music tendency. 

**Abstract (ZH)**: 音乐的表现力和连贯性在音乐创作与表演中不可或缺，但在现代AI生成模型中往往被忽视。本文引入了一种基于聆听的数据处理技术，以捕捉音乐表演中的表现力。这种技术源自韦伯定律，反映了人类听觉感知的真实性，并在训练输入中保留了音乐的微妙性和表现力。为了促进音乐的连贯性，我们基于概率链规则在神经网络中建模了不同音乐数据参数（如音高、时长、力度等）之间的输出依存关系。实际操作中，我们将多输出序列模型分解为单输出子模型，并以前一步生成的输出条件化后续子模型，从而诱导条件分布。最后，为了从所有生成序列中选择合适的序列，提出了基于输出熵的一个临时度量。熵序列设为标准，以选择可预测和稳定的生成，并进一步在信息美学度量的背景下研究，以量化沿音乐倾向的听觉愉悦和信息增益。 

---
# Transfer Learning for Automated Feedback Generation on Small Datasets 

**Title (ZH)**: 基于少量数据的自动化反馈生成迁移学习 

**Authors**: Oscar Morris  

**Link**: [PDF](https://arxiv.org/pdf/2503.11836)  

**Abstract**: Feedback is a very important part the learning process. However, it is challenging to make this feedback both timely and accurate when relying on human markers. This is the challenge that Automated Feedback Generation attempts to address. In this paper, a technique to train such a system on a very small dataset with very long sequences is presented. Both of these attributes make this a very challenging task, however, by using a three stage transfer learning pipeline state-of-the-art results can be achieved with qualitatively accurate but unhuman sounding results. The use of both Automated Essay Scoring and Automated Feedback Generation systems in the real world is also discussed. 

**Abstract (ZH)**: 自动化反馈生成是学习过程中的一个重要组成部分。然而，依赖人类标记者提供及时且准确的反馈具有挑战性。本文提出了在一个非常小的数据集和非常长的序列上训练此类系统的技术。这两个特性使得这一任务极具挑战性，但通过使用三阶段迁移学习管道，可以实现质量上准确但缺乏人性化声音的结果。本文还讨论了在实际中使用自动化作文评分和自动化反馈生成系统的问题。 

---
# Adaptive Stochastic Gradient Descents on Manifolds with an Application on Weighted Low-Rank Approximation 

**Title (ZH)**: 流形上的自适应随机梯度下降及其在加权低秩逼近中的应用 

**Authors**: Peiqi Yang, Conglong Xu, Hao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11833)  

**Abstract**: We prove a convergence theorem for stochastic gradient descents on manifolds with adaptive learning rate and apply it to the weighted low-rank approximation problem. 

**Abstract (ZH)**: 我们证明了在流形上具有自适应学习率的随机梯度下降的收敛定理，并将其应用于加权低秩逼近问题。 

---
# Semi-Supervised Co-Training of Time and Time-Frequency Models: Application to Bearing Fault Diagnosis 

**Title (ZH)**: 时间模型和时频模型的半监督协同训练：轴承故障诊断应用 

**Authors**: Tuomas Jalonen, Mohammad Al-Sa'd, Serkan Kiranyaz, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2503.11824)  

**Abstract**: Neural networks require massive amounts of annotated data to train intelligent solutions. Acquiring many labeled data in industrial applications is often difficult; therefore, semi-supervised approaches are preferred. We propose a new semi-supervised co-training method, which combines time and time-frequency (TF) machine learning models to improve performance and reliability. The developed framework collaboratively co-trains fast time-domain models by utilizing high-performing TF techniques without increasing the inference complexity. Besides, it operates in cloud-edge networks and offers holistic support for many applications covering edge-real-time monitoring and cloud-based updates and corrections. Experimental results on bearing fault diagnosis verify the superiority of our technique compared to a competing self-training method. The results from two case studies show that our method outperforms self-training for different noise levels and amounts of available data with accuracy gains reaching from 10.6% to 33.9%. They demonstrate that fusing time-domain and TF-based models offers opportunities for developing high-performance industrial solutions. 

**Abstract (ZH)**: 基于时间和时间-频率模型的新型半监督协同训练方法及其应用 

---
# Mitigating Bad Ground Truth in Supervised Machine Learning based Crop Classification: A Multi-Level Framework with Sentinel-2 Images 

**Title (ZH)**: 基于Sentinel-2图像的多级框架在监督机器学习作物分类中缓解错误标注数据的影响 

**Authors**: Sanayya A, Amoolya Shetty, Abhijeet Sharma, Venkatesh Ravichandran, Masthan Wali Gosuvarapalli, Sarthak Jain, Priyamvada Nanjundiah, Ujjal Kr Dutta, Divya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2503.11807)  

**Abstract**: In agricultural management, precise Ground Truth (GT) data is crucial for accurate Machine Learning (ML) based crop classification. Yet, issues like crop mislabeling and incorrect land identification are common. We propose a multi-level GT cleaning framework while utilizing multi-temporal Sentinel-2 data to address these issues. Specifically, this framework utilizes generating embeddings for farmland, clustering similar crop profiles, and identification of outliers indicating GT errors. We validated clusters with False Colour Composite (FCC) checks and used distance-based metrics to scale and automate this verification process. The importance of cleaning the GT data became apparent when the models were trained on the clean and unclean data. For instance, when we trained a Random Forest model with the clean GT data, we achieved upto 70\% absolute percentage points higher for the F1 score metric. This approach advances crop classification methodologies, with potential for applications towards improving loan underwriting and agricultural decision-making. 

**Abstract (ZH)**: 在农业管理中，精确的地面真实数据对于基于机器学习的作物分类至关重要。然而，作物错误标签和土地识别不正确等问题普遍存在。我们提出了一种利用多时相Sentinel-2数据的多层次地面真实数据清理框架来解决这些问题。具体而言，该框架通过生成农田嵌入、聚类相似作物特性以及识别表明地面真实数据错误的异常值来实现。我们通过假colour复合图（FCC）检查验证聚类，并使用基于距离的指标来放大和自动化这一验证过程。当模型在清理和未清理的地面真实数据上训练时，清理地面真实数据的重要性变得尤为明显。例如，当使用清理的地面真实数据训练随机森林模型时，F1得分指标提高了多达70%的绝对百分点。该方法推进了作物分类方法的发展，并有可能应用于贷款审批和农业决策等领域。 

---
# BioMamba: Leveraging Spectro-Temporal Embedding in Bidirectional Mamba for Enhanced Biosignal Classification 

**Title (ZH)**: BioMamba：利用双向Mamba中的频谱-时间嵌入增强生物信号分类 

**Authors**: Jian Qian, Teck Lun Goh, Bingyu Xie, Chengyao Zhu, Biao Wan, Yawen Guan, Patrick Yin Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11741)  

**Abstract**: Biological signals, such as electroencephalograms (EEGs) and electrocardiograms (ECGs), play a pivotal role in numerous clinical practices, such as diagnosing brain and cardiac arrhythmic diseases. Existing methods for biosignal classification rely on Attention-based frameworks with dense Feed Forward layers, which lead to inefficient learning, high computational overhead, and suboptimal performance. In this work, we introduce BioMamba, a Spectro-Temporal Embedding strategy applied to the Bidirectional Mamba framework with Sparse Feed Forward layers to enable effective learning of biosignal sequences. By integrating these three key components, BioMamba effectively addresses the limitations of existing methods. Extensive experiments demonstrate that BioMamba significantly outperforms state-of-the-art methods with marked improvement in classification performance. The advantages of the proposed BioMamba include (1) Reliability: BioMamba consistently delivers robust results, confirmed across six evaluation metrics. (2) Efficiency: We assess both model and training efficiency, the BioMamba demonstrates computational effectiveness by reducing model size and resource consumption compared to existing approaches. (3) Generality: With the capacity to effectively classify a diverse set of tasks, BioMamba demonstrates adaptability and effectiveness across various domains and applications. 

**Abstract (ZH)**: 生物信号，如脑电图（EEGs）和心电图（ECGs），在临床实践中发挥着关键作用，例如诊断脑部和心脏节律性疾病。现有的生物信号分类方法依赖于基于注意力的框架和密集的全连接层，导致学习效率低下、计算开销高和性能不佳。在这项工作中，我们引入了BioMamba，这是一种应用于双向Mamba框架的频谱-时间嵌入策略，结合稀疏全连接层，以实现生物信号序列的有效学习。通过结合这三个关键组件，BioMamba有效地解决了现有方法的局限性。广泛的实验表明，BioMamba在分类性能上显著优于现有最先进的方法，具有显著的改进。所提出BioMamba的优点包括：（1）可靠性：BioMamba在六项评估指标上一致提供稳健的结果。（2）效率：我们在模型和训练效率方面进行评估，BioMamba通过减小模型规模和资源消耗显示出计算上的有效性，优于现有方法。（3）普适性：BioMamba具有有效分类一系列任务的能力，展示了在各种领域和应用中的适应性和有效性。 

---
# Multi-View Node Pruning for Accurate Graph Representation 

**Title (ZH)**: 多视图节点剪枝以实现准确的图表示 

**Authors**: Jiseong Park, Hanjin Kim, Seojin Kim, Jueun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11737)  

**Abstract**: Graph pooling, which compresses a whole graph into a smaller coarsened graph, is an essential component of graph representation learning. To efficiently compress a given graph, graph pooling methods often drop their nodes with attention-based scoring with the task loss. However, this often results in simply removing nodes with lower degrees without consideration of their feature-level relevance to the given task. To fix this problem, we propose a Multi-View Pruning(MVP), a graph pruning method based on a multi-view framework and reconstruction loss. Given a graph, MVP first constructs multiple graphs for different views either by utilizing the predefined modalities or by randomly partitioning the input features, to consider the importance of each node in diverse perspectives. Then, it learns the score for each node by considering both the reconstruction and the task loss. MVP can be incorporated with any hierarchical pooling framework to score the nodes. We validate MVP on multiple benchmark datasets by coupling it with two graph pooling methods, and show that it significantly improves the performance of the base graph pooling method, outperforming all baselines. Further analysis shows that both the encoding of multiple views and the consideration of reconstruction loss are the key to the success of MVP, and that it indeed identifies nodes that are less important according to domain knowledge. 

**Abstract (ZH)**: 基于多视图框架和重构损失的图剪枝方法：MVP 

---
# Class-Level Feature Selection Method Using Feature Weighted Growing Self-Organising Maps 

**Title (ZH)**: 基于特征加权生长自组织映射的类级特征选择方法 

**Authors**: Andrew Starkey, Uduak Idio Akpan, Omaimah AL Hosni, Yaseen Pullissery  

**Link**: [PDF](https://arxiv.org/pdf/2503.11732)  

**Abstract**: There have been several attempts to develop Feature Selection (FS) algorithms capable of identifying features that are relevant in a dataset. Although in certain applications the FS algorithms can be seen to be successful, they have similar basic limitations. In all cases, the global feature selection algorithms seek to select features that are relevant and common to all classes of the dataset. This is a major limitation since there could be features that are specifically useful for a particular class while irrelevant for other classes, and full explanation of the relationship at class level therefore cannot be determined. While the inclusion of such features for all classes could cause improved predictive ability for the relevant class, the same features could be problematic for other classes. In this paper, we examine this issue and also develop a class-level feature selection method called the Feature Weighted Growing Self-Organising Map (FWGSOM). The proposed method carries out feature analysis at class level which enhances its ability to identify relevant features for each class. Results from experiments indicate that our method performs better than other methods, gives explainable results at class level, and has a low computational footprint when compared to other methods. 

**Abstract (ZH)**: 在数据集中识别相关特征的特征选择算法已有若干尝试。尽管在某些应用中特征选择算法可以取得成功，但它们存在类似的基本局限性。在所有情况下，全局特征选择算法都试图选择对所有数据集类别的特征都有意义的特征。这是一个主要的局限性，因为可能存在仅对特定类别有用而不对其他类别有用的特征，从而无法在类别级别上确定这些关系的完整解释。虽然包括这些特征可以提高相关类别预测能力，但这些特征对于其他类别可能存在问题。在本文中，我们探讨了这一问题并开发了一种称为特征加权增长自组织映射（FWGSOM）的类别级别特征选择方法。所提方法在类别级别进行特征分析，增强了其识别每个类别相关特征的能力。实验结果表明，与现有方法相比，我们的方法在类级别提供了可解释的结果，并且与现有方法相比具有较低的计算成本。 

---
# BACE-RUL: A Bi-directional Adversarial Network with Covariate Encoding for Machine Remaining Useful Life Prediction 

**Title (ZH)**: BACE-RUL：一种带有协变量编码的双向对抗网络机器剩余使用寿命预测 

**Authors**: Zekai Zhang, Dan Li, Shunyu Wu, Junya Cai, Bo Zhang, See Kiong Ng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.11730)  

**Abstract**: Prognostic and Health Management (PHM) are crucial ways to avoid unnecessary maintenance for Cyber-Physical Systems (CPS) and improve system reliability. Predicting the Remaining Useful Life (RUL) is one of the most challenging tasks for PHM. Existing methods require prior knowledge about the system, contrived assumptions, or temporal mining to model the life cycles of machine equipment/devices, resulting in diminished accuracy and limited applicability in real-world scenarios. This paper proposes a Bi-directional Adversarial network with Covariate Encoding for machine Remaining Useful Life (BACE-RUL) prediction, which only adopts sensor measurements from the current life cycle to predict RUL rather than relying on previous consecutive cycle recordings. The current sensor measurements of mechanical devices are encoded to a conditional space to better understand the implicit inner mechanical status. The predictor is trained as a conditional generative network with the encoded sensor measurements as its conditions. Various experiments on several real-world datasets, including the turbofan aircraft engine dataset and the dataset collected from degradation experiments of Li-Ion battery cells, show that the proposed model is a general framework and outperforms state-of-the-art methods. 

**Abstract (ZH)**: 基于双向对抗网络和协变量编码的机械剩余寿命预测（BACE-RUL） 

---
# Forecasting Empty Container availability for Vehicle Booking System Application 

**Title (ZH)**: 基于车辆预订系统应用的空箱可用性预测 

**Authors**: Arthur Cartel Foahom Gouabou, Mohammed Al-Kharaz, Faouzi Hakimi, Tarek Khaled, Kenza Amzil  

**Link**: [PDF](https://arxiv.org/pdf/2503.11728)  

**Abstract**: Container terminals, pivotal nodes in the network of empty container movement, hold significant potential for enhancing operational efficiency within terminal depots through effective collaboration between transporters and terminal operators. This collaboration is crucial for achieving optimization, leading to streamlined operations and reduced congestion, thereby benefiting both parties. Consequently, there is a pressing need to develop the most suitable forecasting approaches to address this challenge. This study focuses on developing and evaluating a data-driven approach for forecasting empty container availability at container terminal depots within a Vehicle Booking System (VBS) framework. It addresses the gap in research concerning optimizing empty container dwell time and aims to enhance operational efficiencies in container terminal operations. Four forecasting models-Naive, ARIMA, Prophet, and LSTM-are comprehensively analyzed for their predictive capabilities, with LSTM emerging as the top performer due to its ability to capture complex time series patterns. The research underscores the significance of selecting appropriate forecasting techniques tailored to the specific requirements of container terminal operations, contributing to improved operational planning and management in maritime logistics. 

**Abstract (ZH)**: 集装箱码头：作为空箱流动网络中的关键节点，通过运输商与码头运营商的有效合作，拥有通过优化协作提高码头仓库运营效率的显著潜力。为此，有必要开发合适的预测方法来应对这一挑战。本研究旨在基于车辆预订系统（VBS）的框架，开发并评估一种数据驱动的预测方法，以预测集装箱码头仓库中的空箱可用性，解决优化空箱停留时间的研究缺口，旨在提高集装箱码头运营效率。四种预测模型——朴素法、ARIMA、Prophet和LSTM——被全面分析其预测能力，其中LSTM因其能够捕捉复杂的时间序列模式而表现最佳。研究强调选择合适的预测技术对于满足集装箱码头运营特定需求的重要性，有助于改善海运物流中的运营规划与管理。 

---
# SPECTra: Scalable Multi-Agent Reinforcement Learning with Permutation-Free Networks 

**Title (ZH)**: SPECTra: 可扩展的无排列依赖多智能体 reinforcement 学习 

**Authors**: Hyunwoo Park, Baekryun Seong, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2503.11726)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), the permutation problem where the state space grows exponentially with the number of agents reduces sample efficiency. Additionally, many existing architectures struggle with scalability, relying on a fixed structure tied to a specific number of agents, limiting their applicability to environments with a variable number of entities. While approaches such as graph neural networks (GNNs) and self-attention mechanisms have progressed in addressing these challenges, they have significant limitations as dense GNNs and self-attention mechanisms incur high computational costs. To overcome these limitations, we propose a novel agent network and a non-linear mixing network that ensure permutation-equivariance and scalability, allowing them to generalize to environments with various numbers of agents. Our agent network significantly reduces computational complexity, and our scalable hypernetwork enables efficient weight generation for non-linear mixing. Additionally, we introduce curriculum learning to improve training efficiency. Experiments on SMACv2 and Google Research Football (GRF) demonstrate that our approach achieves superior learning performance compared to existing methods. By addressing both permutation-invariance and scalability in MARL, our work provides a more efficient and adaptable framework for cooperative MARL. Our code is available at this https URL. 

**Abstract (ZH)**: 在合作多智能体强化学习（MARL）中，随着智能体数量增加而指数级增长的状态空间导致采样效率降低。此外，许多现有架构在可扩展性方面存在不足，依赖于固定结构，只能适用于特定数量智能体的环境，限制了它们在智能体数量可变的环境中的应用。尽管图神经网络（GNN）和自注意力机制等方法在解决这些问题方面取得了进步，但密集的GNN和自注意力机制会带来高昂的计算成本。为克服这些限制，我们提出了一种新型智能体网络和非线性混合网络，以确保置换不变性和可扩展性，使其能够适应不同数量智能体的环境。我们的智能体网络显著降低了计算复杂度，而我们的可扩展超网络使非线性混合的权重生成更加高效。此外，我们引入了分级学习以提高训练效率。在SMACv2和Google Research Football（GRF）上的实验表明，我们的方法在学习性能上优于现有方法。通过同时解决MARL中的置换不变性和可扩展性问题，我们的工作为合作MARL提供了更高效和更具适应性的框架。我们的代码可在以下链接获取。 

---
# Privacy-Preserved Automated Scoring using Federated Learning for Educational Research 

**Title (ZH)**: 隐私保护的联邦学习自动评分在教育研究中的应用 

**Authors**: Ehsan Latif, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11711)  

**Abstract**: Data privacy remains a critical concern in educational research, necessitating Institutional Review Board (IRB) certification and stringent data handling protocols to ensure compliance with ethical standards. Traditional approaches rely on anonymization and controlled data-sharing mechanisms to facilitate research while mitigating privacy risks. However, these methods still involve direct access to raw student data, posing potential vulnerabilities and being time-consuming. This study proposes a federated learning (FL) framework for automatic scoring in educational assessments, eliminating the need to share raw data. Our approach leverages client-side model training, where student responses are processed locally on edge devices, and only optimized model parameters are shared with a central aggregation server. To effectively aggregate heterogeneous model updates, we introduce an adaptive weighted averaging strategy, which dynamically adjusts weight contributions based on client-specific learning characteristics. This method ensures robust model convergence while preserving privacy. We evaluate our framework using assessment data from nine middle schools, comparing the accuracy of federated learning-based scoring models with traditionally trained centralized models. A statistical significance test (paired t-test, $t(8) = 2.29, p = 0.051$) confirms that the accuracy difference between the two approaches is not statistically significant, demonstrating that federated learning achieves comparable performance while safeguarding student data. Furthermore, our method significantly reduces data collection, processing, and deployment overhead, accelerating the adoption of AI-driven educational assessments in a privacy-compliant manner. 

**Abstract (ZH)**: 联邦学习在教育评估中的自动评分：保护学生数据隐私的同时实现高性能 

---
# ConjointNet: Enhancing Conjoint Analysis for Preference Prediction with Representation Learning 

**Title (ZH)**: 共轭网络：基于表示学习的联合分析增强偏好预测 

**Authors**: Yanxia Zhang, Francine Chen, Shabnam Hakimi, Totte Harinen, Alex Filipowicz, Yan-Ying Chen, Rumen Iliev, Nikos Arechiga, Kalani Murakami, Kent Lyons, Charlene Wu, Matt Klenk  

**Link**: [PDF](https://arxiv.org/pdf/2503.11710)  

**Abstract**: Understanding consumer preferences is essential to product design and predicting market response to these new products. Choice-based conjoint analysis is widely used to model user preferences using their choices in surveys. However, traditional conjoint estimation techniques assume simple linear models. This assumption may lead to limited predictability and inaccurate estimation of product attribute contributions, especially on data that has underlying non-linear relationships. In this work, we employ representation learning to efficiently alleviate this issue. We propose ConjointNet, which is composed of two novel neural architectures, to predict user preferences. We demonstrate that the proposed ConjointNet models outperform traditional conjoint estimate techniques on two preference datasets by over 5%, and offer insights into non-linear feature interactions. 

**Abstract (ZH)**: 基于选择的联合分析：通过表示学习提高用户偏好预测的非线性特征交互洞察 

---
# Conformal Prediction and Human Decision Making 

**Title (ZH)**: 符合性预测与人类决策making 

**Authors**: Jessica Hullman, Yifan Wu, Dawei Xie, Ziyang Guo, Andrew Gelman  

**Link**: [PDF](https://arxiv.org/pdf/2503.11709)  

**Abstract**: Methods to quantify uncertainty in predictions from arbitrary models are in demand in high-stakes domains like medicine and finance. Conformal prediction has emerged as a popular method for producing a set of predictions with specified average coverage, in place of a single prediction and confidence value. However, the value of conformal prediction sets to assist human decisions remains elusive due to the murky relationship between coverage guarantees and decision makers' goals and strategies. How should we think about conformal prediction sets as a form of decision support? Under what conditions do we expect the support they provide to be superior versus inferior to that of alternative presentations of predictive uncertainty? We outline a decision theoretic framework for evaluating predictive uncertainty as informative signals, then contrast what can be said within this framework about idealized use of calibrated probabilities versus conformal prediction sets. Informed by prior empirical results and theories of human decisions under uncertainty, we formalize a set of possible strategies by which a decision maker might use a prediction set. We identify ways in which conformal prediction sets and posthoc predictive uncertainty quantification more broadly are in tension with common goals and needs in human-AI decision making. We give recommendations for future research in predictive uncertainty quantification to support human decision makers. 

**Abstract (ZH)**: 量化来自任意模型预测不确定性的方法在高风险领域如医学和金融中需求日益增加。条件一致性预测已成为一种流行的生成具有指定平均覆盖范围的预测集的方法，取代单一预测和置信值。然而，条件一致性预测集对辅助人类决策的价值仍然难以明确，因为覆盖保证与决策者的目标和策略之间的关系模糊。我们应如何将条件一致性预测集视为一种决策支持形式？在什么条件下，我们期望它们提供的支持优于或劣于替代预测不确定性表示形式的支持？我们提出了一种决策理论框架，用于评估预测不确定性的信息信号，然后对比了该框架内关于校准概率的理想化使用与条件一致性预测集之间可以阐述的内容。根据先前的实证结果和不确定环境下的人类决策理论，我们正式化了一系列决策者可能采用的策略，使用预测集。我们指出了条件一致性预测集和更广泛的.post hoc不确定性量化与人类-人工智能决策中常见目标和需求之间的矛盾之处。我们为支持人类决策者在未来研究预测不确定性量化中提出建议。 

---
# Refining Filter Global Feature Weighting for Fully-Unsupervised Clustering 

**Title (ZH)**: 改进滤波全局特征权值赋予权对全程无监督聚类的优化 

**Authors**: Fabian Galis, Darian Onchis  

**Link**: [PDF](https://arxiv.org/pdf/2503.11706)  

**Abstract**: In the context of unsupervised learning, effective clustering plays a vital role in revealing patterns and insights from unlabeled data. However, the success of clustering algorithms often depends on the relevance and contribution of features, which can differ between various datasets. This paper explores feature weighting for clustering and presents new weighting strategies, including methods based on SHAP (SHapley Additive exPlanations), a technique commonly used for providing explainability in various supervised machine learning tasks. By taking advantage of SHAP values in a way other than just to gain explainability, we use them to weight features and ultimately improve the clustering process itself in unsupervised scenarios.
Our empirical evaluations across five benchmark datasets and clustering methods demonstrate that feature weighting based on SHAP can enhance unsupervised clustering quality, achieving up to a 22.69\% improvement over other weighting methods (from 0.586 to 0.719 in terms of the Adjusted Rand Index). Additionally, these situations where the weighted data boosts the results are highlighted and thoroughly explored, offering insight for practical applications. 

**Abstract (ZH)**: 在无监督学习.context中，有效的聚类对于揭示未标记数据中的模式和见解起着至关重要的作用。然而，聚类算法的成功往往取决于特征的相关性和贡献，而这些在不同数据集中可能有所不同。本文探讨了聚类中的特征加权，并提出了一种新的加权策略，包括基于SHAP（SHapley Additive exPlanations）的方法，这是一种常用于提供监督机器学习任务可解释性的技术。通过利用SHAP值，我们不仅用来获得可解释性，还用于加权特征，从而在无监督场景中提高聚类过程本身的效果。我们在五个基准数据集和聚类方法上的实证评估表明，基于SHAP的特征加权可以提升无监督聚类的质量，相较于其他加权方法，在调整兰德指数（Adjusted Rand Index）上最高可提高22.69%（从0.586提高到0.719）。同时，加权数据增强效果的情况也被强调并进行了深入探讨，为实际应用提供了见解。 

---
# Timing-Driven Global Placement by Efficient Critical Path Extraction 

**Title (ZH)**: 由高效关键路径提取驱动的时驱动全局布放 

**Authors**: Yunqi Shi, Siyuan Xu, Shixiong Kai, Xi Lin, Ke Xue, Mingxuan Yuan, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11674)  

**Abstract**: Timing optimization during the global placement of integrated circuits has been a significant focus for decades, yet it remains a complex, unresolved issue. Recent analytical methods typically use pin-level timing information to adjust net weights, which is fast and simple but neglects the path-based nature of the timing graph. The existing path-based methods, however, cannot balance the accuracy and efficiency due to the exponential growth of number of critical paths. In this work, we propose a GPU-accelerated timing-driven global placement framework, integrating accurate path-level information into the efficient DREAMPlace infrastructure. It optimizes the fine-grained pin-to-pin attraction objective and is facilitated by efficient critical path extraction. We also design a quadratic distance loss function specifically to align with the RC timing model. Experimental results demonstrate that our method significantly outperforms the current leading timing-driven placers, achieving an average improvement of 40.5% in total negative slack (TNS) and 8.3% in worst negative slack (WNS), as well as an improvement in half-perimeter wirelength (HPWL). 

**Abstract (ZH)**: 全球布线过程中集成电路的定时优化一直是几十年来的研究重点，但仍然是一个复杂且未解决的问题。尽管最近的分析方法通常使用引脚级定时信息来调整网的权重，这种方法快速且简单，但忽略了定时图的路径基础特性。现有的基于路径的方法由于关键路径数量的指数级增长，无法平衡准确性与效率。在这项工作中，我们提出了一种基于GPU加速的驱动定时全局布线框架，将精确的路径级信息整合到高效的DREAMPlace架构中。该框架优化了细粒度的引脚到引脚吸引力目标，并通过高效的关键路径提取来协助。我们还设计了一个二次距离损失函数，专门与RC定时模型对齐。实验结果表明，我们的方法明显优于当前领先的定时驱动布线器，平均在总负时宽(TNS)上提高了40.5%，在最坏负时宽(WNS)上提高了8.3%，并且在半周长布线长度(HPWL)上也有所改进。 

---
# Optimizing Coverage-Driven Verification Using Machine Learning and PyUVM: A Novel Approach 

**Title (ZH)**: 基于机器学习和PyUVM的新型覆盖驱动验证优化方法 

**Authors**: Suruchi Kumari, Deepak Narayan Gadde, Aman Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11666)  

**Abstract**: The escalating complexity of System-on-Chip (SoC) designs has created a bottleneck in verification, with traditional techniques struggling to achieve complete coverage. Existing techniques, such as Constrained Random Verification (CRV) and coverage-driven methodologies, rely on time-consuming and redundant simulation regression, leading to higher verification costs and longer time-to-market due to the manual effort required to adjust constraints and drive the stimuli to achieve coverage objectives. To address this challenge, we propose a novel methodology that leverages supervised Machine Learning (ML) to optimize simulation regressions, resulting in reduced simulation run-time and the number of test simulations required to achieve target coverage goals. We also investigate and compare the effectiveness of various supervised learning algorithms from scikit-learn. Our results demonstrate that these algorithms can achieve at least 99% coverage regain with significantly reduced simulation cycles. We utilize Python Universal Verification Methodology (PyUVM) over SystemVerilog-Universal Verification Methodology (SV-UVM) for testbench creation, enabling simpler constructs using Python and facilitating the reuse of existing ML libraries. Our methodology is applied to three diverse designs, and our results show that it can significantly reduce verification costs, manual efforts, and time-to-market, while enhancing verification productivity and completeness, by automating the testbench update process and achieving target coverage goals. 

**Abstract (ZH)**: SoC设计复杂性的攀升导致验证瓶颈，传统技术难以实现全面覆盖。为应对这一挑战，我们提出了一种新型方法，利用监督机器学习优化仿真回归，从而减少仿真运行时间和达到目标覆盖目标所需的测试仿真次数。我们还研究并比较了scikit-learn中各种监督学习算法的有效性。结果显示，这些算法可以在显著减少仿真周期的前提下实现至少99%的覆盖恢复。我们使用Python Universal Verification Methodology (PyUVM) 而不是SystemVerilog-Universal Verification Methodology (SV-UVM) 创建测试平台，利用Python简化构造并方便重用现有的机器学习库。该方法应用于三个不同的设计，结果显示它可以显著降低验证成本、减少手动努力和缩短时间-to-市场，同时提高验证的生产力和完整性，通过自动化测试平台更新过程并达到目标覆盖目标。 

---
# Lorecast: Layout-Aware Performance and Power Forecasting from Natural Language 

**Title (ZH)**: Lorecast: 基于布局的性能和功率预测方法自然语言处理 

**Authors**: Runzhi Wang, Prianka Sengupta, Yiran Chen, Jiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11662)  

**Abstract**: In chip design planning, obtaining reliable performance and power forecasts for various design options is of critical importance. Traditionally, this involves using system-level models, which often lack accuracy, or trial synthesis, which is both labor-intensive and time-consuming. We introduce a new methodology, called Lorecast, which accepts English prompts as input to rapidly generate layout-aware performance and power estimates. This approach bypasses the need for HDL code development or synthesis, making it both fast and user-friendly. Experimental results demonstrate that Lorecast achieves accuracy within a few percent of error compared to post-layout analysis. 

**Abstract (ZH)**: 在芯片设计规划中，获得各种设计选项的可靠性能和功率预测至关重要。传统方法通常需要使用系统级模型，这些模型往往缺乏准确性，或者进行试合成，这既耗时又耗力。我们提出了一种新的方法，称为Lorecast，该方法接受英文提示作为输入，快速生成布局感知的性能和功率估算。该方法 bypassed 对HDL代码开发或合成的需求，使其既快速又用户友好。实验结果表明，Lorecast 的准确度误差范围在几百分点以内，与后布局分析结果相当。 

---
# A 28 nm AI microcontroller with tightly coupled zero-standby power weight memory featuring standard logic compatible 4 Mb 4-bits/cell embedded flash technology 

**Title (ZH)**: 一种配备紧密耦合零待机功耗权重存储器的28纳米AI微控制器，采用兼容标准逻辑的嵌入式闪存技术4Mb 4-bits/cell 

**Authors**: Daewung Kim, Seong Hwan Jeon, Young Hee Jeon, Kyung-Bae Kwon, Jigon Kim, Yeounghun Choi, Hyunseung Cha, Kitae Kwon, Daesik Park, Jongseuk Lee, Sihwan Kim, Seung-Hwan Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.11660)  

**Abstract**: This study introduces a novel AI microcontroller optimized for cost-effective, battery-powered edge AI applications. Unlike traditional single bit/cell memory configurations, the proposed microcontroller integrates zero-standby power weight memory featuring standard logic compatible 4-bits/cell embedded flash technology tightly coupled to a Near-Memory Computing Unit. This architecture enables efficient and low-power AI acceleration. Advanced state mapping and an overstress-free word line (WL) driver circuit extend verify levels, ensuring robust 16 state cell margin. A ping-pong buffer reduces internal data movement while supporting simultaneous multi-bit processing. The fabricated microcontroller demonstrated high reliability, maintaining accuracy after 160 hours of unpowered baking at 125$^\circ$C. 

**Abstract (ZH)**: 这种研究介绍了一种新型AI微控制器，该微控制器针对成本效益高、电池供电的边缘AI应用进行了优化。该提出的微控制器集成了零待机功率权重内存，该内存采用标准逻辑兼容的4-bit/cell嵌入式闪存技术，并紧密耦合到一个近内存计算单元。该架构使AI加速既高效又低功耗。先进的状态映射和一个无过压力的位线(WL)驱动电路增加了验证级别，确保了16状态单元的稳健余量。乒乓缓冲区减少了内部数据移动，同时支持同时多比特处理。制造出的微控制器表现出高可靠性，在125°C下无电烘烤160小时后仍保持准确度。 

---
# Circuit Diagram Retrieval Based on Hierarchical Circuit Graph Representation 

**Title (ZH)**: 基于分层电路图表示的电路图检索 

**Authors**: Ming Gao, Ruichen Qiu, Zeng Hui Chang, Kanjian Zhang, Haikun Wei, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11658)  

**Abstract**: In the domain of analog circuit design, the retrieval of circuit diagrams has drawn a great interest, primarily due to its vital role in the consultation of legacy designs and the detection of design plagiarism. Existing image retrieval techniques are adept at handling natural images, which converts images into feature vectors and retrieval similar images according to the closeness of these vectors. Nonetheless, these approaches exhibit limitations when applied to the more specialized and intricate domain of circuit diagrams. This paper presents a novel approach to circuit diagram retrieval by employing a graph representation of circuit diagrams, effectively reformulating the retrieval task as a graph retrieval problem. The proposed methodology consists of two principal components: a circuit diagram recognition algorithm designed to extract the circuit components and topological structure of the circuit using proposed GAM-YOLO model and a 2-step connected domain filtering algorithm, and a hierarchical retrieval strategy based on graph similarity and different graph representation methods for analog circuits. Our methodology pioneers the utilization of graph representation in the retrieval of circuit diagrams, incorporating topological features that are commonly overlooked by standard image retrieval methods. The results of our experiments substantiate the efficacy of our approach in retrieving circuit diagrams across of different types. 

**Abstract (ZH)**: 在模拟电路设计领域，电路图的检索受到了极大关注，主要因其在遗产设计咨询和设计抄袭检测中的关键作用。现有的图像检索技术擅长处理自然图像，即将图像转换为特征向量，并根据这些向量的相似性检索相似图像。然而，这些方法在应用于更为专业和复杂的电路图领域时表现出局限性。本文提出了一种新的电路图检索方法，通过采用电路图的图表示法，有效将检索任务重新表述为图检索问题。所提出的方法主要包括两个主要组成部分：一种基于提出的GAM-YOLO模型提取电路元件和拓扑结构的电路图识别算法以及两步连接域过滤算法，以及基于图相似性和不同图表示方法的层次检索策略。我们的方法开创性地将图表示法应用于电路图检索，结合了标准图像检索方法通常忽视的拓扑特征。实验结果证明了该方法在不同类型的电路图检索中的有效性。 

---
