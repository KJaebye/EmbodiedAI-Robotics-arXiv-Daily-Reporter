# TOP: Trajectory Optimization via Parallel Optimization towards Constant Time Complexity 

**Title (ZH)**: TOP: 通过并行优化实现恒定时间复杂度的轨迹优化 

**Authors**: Jiajun Yu, Nanhe Chen, Guodong Liu, Chao Xu, Fei Gao, Yanjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10290)  

**Abstract**: Optimization has been widely used to generate smooth trajectories for motion planning. However, existing trajectory optimization methods show weakness when dealing with large-scale long trajectories. Recent advances in parallel computing have accelerated optimization in some fields, but how to efficiently solve trajectory optimization via parallelism remains an open question. In this paper, we propose a novel trajectory optimization framework based on the Consensus Alternating Direction Method of Multipliers (CADMM) algorithm, which decomposes the trajectory into multiple segments and solves the subproblems in parallel. The proposed framework reduces the time complexity to O(1) per iteration to the number of segments, compared to O(N) of the state-of-the-art (SOTA) approaches. Furthermore, we introduce a closed-form solution that integrates convex linear and quadratic constraints to speed up the optimization, and we also present numerical solutions for general inequality constraints. A series of simulations and experiments demonstrate that our approach outperforms the SOTA approach in terms of efficiency and smoothness. Especially for a large-scale trajectory, with one hundred segments, achieving over a tenfold speedup. To fully explore the potential of our algorithm on modern parallel computing architectures, we deploy our framework on a GPU and show high performance with thousands of segments. 

**Abstract (ZH)**: 基于共识交替方向乘子算法的轨迹优化框架：高效并行求解轨迹规划问题 

---
# Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications 

**Title (ZH)**: 基于非线性传播模型的无迹卡尔曼滤波在导航应用中的研究 

**Authors**: Amit Levy, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2507.10082)  

**Abstract**: The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios. 

**Abstract (ZH)**: 无迹卡尔曼滤波算法在导航应用中的非线性估计算法及其在导航误差状态向量非线性动态模型下的sigma点传播方法：提高滤波精度与导航性能的研究 

---
# TGLD: A Trust-Aware Game-Theoretic Lane-Changing Decision Framework for Automated Vehicles in Heterogeneous Traffic 

**Title (ZH)**: TGLD：一种考虑信任的游戏理论变道决策框架，应用于异构交通中的自动驾驶车辆 

**Authors**: Jie Pan, Tianyi Wang, Yangyang Wang, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10075)  

**Abstract**: Automated vehicles (AVs) face a critical need to adopt socially compatible behaviors and cooperate effectively with human-driven vehicles (HVs) in heterogeneous traffic environment. However, most existing lane-changing frameworks overlook HVs' dynamic trust levels, limiting their ability to accurately predict human driver behaviors. To address this gap, this study proposes a trust-aware game-theoretic lane-changing decision (TGLD) framework. First, we formulate a multi-vehicle coalition game, incorporating fully cooperative interactions among AVs and partially cooperative behaviors from HVs informed by real-time trust evaluations. Second, we develop an online trust evaluation method to dynamically estimate HVs' trust levels during lane-changing interactions, guiding AVs to select context-appropriate cooperative maneuvers. Lastly, social compatibility objectives are considered by minimizing disruption to surrounding vehicles and enhancing the predictability of AV behaviors, thereby ensuring human-friendly and context-adaptive lane-changing strategies. A human-in-the-loop experiment conducted in a highway on-ramp merging scenario validates our TGLD approach. Results show that AVs can effectively adjust strategies according to different HVs' trust levels and driving styles. Moreover, incorporating a trust mechanism significantly improves lane-changing efficiency, maintains safety, and contributes to transparent and adaptive AV-HV interactions. 

**Abstract (ZH)**: 自动车辆（AVs）在异构交通环境中需要采用社会兼容行为，并有效与人类驾驶车辆（HVs）合作。为解决这一问题，本研究提出了一种信任感知博弈论变道决策（TGLD）框架。首先，我们构建了一种多车辆联合游戏，包含AVs之间的完全合作互动和基于实时信任评估的HVs的部分合作行为。其次，我们开发了一种在线信任评价方法，在变道过程中动态估计HVs的信任水平，指导AVs选择合适的合作 maneuvers。最后，通过最小化对周围车辆的干扰和提高AV行为的可预测性，考虑社会兼容性目标，从而确保人类友好和情境适应的变道策略。在高速公路上下道口合并场景中进行的人机环路实验验证了我们的TGLD方法。结果显示，AVs可以根据不同HVs的信任水平和驾驶风格有效调整策略。此外，引入信任机制显著提高了变道效率，维持了安全性，并促进了透明和适应性的AV-HV互动。 

---
# MP-RBFN: Learning-based Vehicle Motion Primitives using Radial Basis Function Networks 

**Title (ZH)**: MP-RBFN：基于径向基函数网络的学习驱动车辆运动 primitives 

**Authors**: Marc Kaufeld, Mattia Piccinini, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10047)  

**Abstract**: This research introduces MP-RBFN, a novel formulation leveraging Radial Basis Function Networks for efficiently learning Motion Primitives derived from optimal control problems for autonomous driving. While traditional motion planning approaches based on optimization are highly accurate, they are often computationally prohibitive. In contrast, sampling-based methods demonstrate high performance but impose constraints on the geometric shape of trajectories. MP-RBFN combines the strengths of both by coupling the high-fidelity trajectory generation of sampling-based methods with an accurate description of vehicle dynamics. Empirical results show compelling performance compared to previous methods, achieving a precise description of motion primitives at low inference times. MP-RBFN yields a seven times higher accuracy in generating optimized motion primitives compared to existing semi-analytic approaches. We demonstrate the practical applicability of MP-RBFN for motion planning by integrating the method into a sampling-based trajectory planner. MP-RBFN is available as open-source software at this https URL. 

**Abstract (ZH)**: MP-RBFN：一种结合径向基函数网络的新型运动基元学习方法及其在自主驾驶中的应用 

---
# Multi-residual Mixture of Experts Learning for Cooperative Control in Multi-vehicle Systems 

**Title (ZH)**: 多残差专家混合学习在多车辆系统协同控制中的应用 

**Authors**: Vindula Jayawardana, Sirui Li, Yashar Farid, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09836)  

**Abstract**: Autonomous vehicles (AVs) are becoming increasingly popular, with their applications now extending beyond just a mode of transportation to serving as mobile actuators of a traffic flow to control flow dynamics. This contrasts with traditional fixed-location actuators, such as traffic signals, and is referred to as Lagrangian traffic control. However, designing effective Lagrangian traffic control policies for AVs that generalize across traffic scenarios introduces a major challenge. Real-world traffic environments are highly diverse, and developing policies that perform robustly across such diverse traffic scenarios is challenging. It is further compounded by the joint complexity of the multi-agent nature of traffic systems, mixed motives among participants, and conflicting optimization objectives subject to strict physical and external constraints. To address these challenges, we introduce Multi-Residual Mixture of Expert Learning (MRMEL), a novel framework for Lagrangian traffic control that augments a given suboptimal nominal policy with a learned residual while explicitly accounting for the structure of the traffic scenario space. In particular, taking inspiration from residual reinforcement learning, MRMEL augments a suboptimal nominal AV control policy by learning a residual correction, but at the same time dynamically selects the most suitable nominal policy from a pool of nominal policies conditioned on the traffic scenarios and modeled as a mixture of experts. We validate MRMEL using a case study in cooperative eco-driving at signalized intersections in Atlanta, Dallas Fort Worth, and Salt Lake City, with real-world data-driven traffic scenarios. The results show that MRMEL consistently yields superior performance-achieving an additional 4%-9% reduction in aggregate vehicle emissions relative to the strongest baseline in each setting. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）在交通控制中的拉格朗日控制研究：一种多残留专家混合学习框架（MRMEL） 

---
# Self-supervised Pretraining for Integrated Prediction and Planning of Automated Vehicles 

**Title (ZH)**: 自动驾驶车辆综合预测与规划的自我监督预训练 

**Authors**: Yangang Ren, Guojian Zhan, Chen Lv, Jun Li, Fenghua Liang, Keqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09537)  

**Abstract**: Predicting the future of surrounding agents and accordingly planning a safe, goal-directed trajectory are crucial for automated vehicles. Current methods typically rely on imitation learning to optimize metrics against the ground truth, often overlooking how scene understanding could enable more holistic trajectories. In this paper, we propose Plan-MAE, a unified pretraining framework for prediction and planning that capitalizes on masked autoencoders. Plan-MAE fuses critical contextual understanding via three dedicated tasks: reconstructing masked road networks to learn spatial correlations, agent trajectories to model social interactions, and navigation routes to capture destination intents. To further align vehicle dynamics and safety constraints, we incorporate a local sub-planning task predicting the ego-vehicle's near-term trajectory segment conditioned on earlier segment. This pretrained model is subsequently fine-tuned on downstream tasks to jointly generate the prediction and planning trajectories. Experiments on large-scale datasets demonstrate that Plan-MAE outperforms current methods on the planning metrics by a large margin and can serve as an important pre-training step for learning-based motion planner. 

**Abstract (ZH)**: 基于掩码自编码器的预测与规划统一预训练框架：Plan-MAE 

---
# TruckV2X: A Truck-Centered Perception Dataset 

**Title (ZH)**: 卡车V2X: 以卡车为中心的感知数据集 

**Authors**: Tenghui Xie, Zhiying Song, Fuxi Wen, Jun Li, Guangzhao Liu, Zijian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09505)  

**Abstract**: Autonomous trucking offers significant benefits, such as improved safety and reduced costs, but faces unique perception challenges due to trucks' large size and dynamic trailer movements. These challenges include extensive blind spots and occlusions that hinder the truck's perception and the capabilities of other road users. To address these limitations, cooperative perception emerges as a promising solution. However, existing datasets predominantly feature light vehicle interactions or lack multi-agent configurations for heavy-duty vehicle scenarios. To bridge this gap, we introduce TruckV2X, the first large-scale truck-centered cooperative perception dataset featuring multi-modal sensing (LiDAR and cameras) and multi-agent cooperation (tractors, trailers, CAVs, and RSUs). We further investigate how trucks influence collaborative perception needs, establishing performance benchmarks while suggesting research priorities for heavy vehicle perception. The dataset provides a foundation for developing cooperative perception systems with enhanced occlusion handling capabilities, and accelerates the deployment of multi-agent autonomous trucking systems. The TruckV2X dataset is available at this https URL. 

**Abstract (ZH)**: 自主 trucks 的自动运输提供了显著的益处，如提高安全性与降低运营成本，但同时面临着独特的感知挑战，由于卡车体积大和动态挂车运动。这些挑战包括广泛的盲区和遮挡，妨碍了卡车的感知能力与其他道路使用者的能力。为应对这些限制，协作感知成为一种有前景的解决方案。然而，现有的数据集主要侧重于轻型车辆的交互或缺乏重载车辆场景中的多智能体配置。为弥合这一缺口，我们引入了 TruckV2X，这是首个以卡车为中心的大型多模态协同感知数据集，集成了多种传感器（激光雷达和摄像头）和多智能体合作（拖拉机、挂车、CAVs 和 RSUs）。我们进一步研究了卡车对协作感知需求的影响，建立了性能基准并提出了重载车辆感知的研究优先级。该数据集为开发具备更强遮挡处理能力的协作感知系统奠定了基础，并加速了多智能体自主卡车系统的部署。TruckV2X 数据集可访问此链接：this https URL。 

---
# Informed Hybrid Zonotope-based Motion Planning Algorithm 

**Title (ZH)**: 基于知情混合zonotope的运动规划算法 

**Authors**: Peng Xie, Johannes Betz, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09309)  

**Abstract**: Optimal path planning in nonconvex free spaces is notoriously challenging, as formulating such problems as mixed-integer linear programs (MILPs) is NP-hard. We propose HZ-MP, an informed Hybrid Zonotope-based Motion Planner, as an alternative approach that decomposes the obstacle-free space and performs low-dimensional face sampling guided by an ellipsotope heuristic, enabling focused exploration along promising transit regions. This structured exploration eliminates the excessive, unreachable sampling that degrades existing informed planners such as AIT* and EIT* in narrow gaps or boxed-goal scenarios. We prove that HZ-MP is probabilistically complete and asymptotically optimal. It converges to near-optimal trajectories in finite time and scales to high-dimensional cluttered scenes. 

**Abstract (ZH)**: 非凸自由空间中的最优路径规划因其将此类问题形式化为混合整数线性规划（MILPs）是NP硬问题而极具挑战性。我们提出了一种名为HZ-MP的启发式混合棱柱体基于 motion planner，该方法通过椭棱柱体启发式指导的低维度面采样分解无障碍空间，实现集中在有前途的过渡区域进行探索。这种结构化的探索消除了AIT*和EIT*等现有启发式规划器在狭窄间隙或盒形目标场景中不必要的、无法到达的采样。我们证明HZ-MP是概率完备的，并且渐近最优。它能够在有限时间内收敛到接近最优的轨迹，并扩展到高维复杂场景。 

---
# DLBAcalib: Robust Extrinsic Calibration for Non-Overlapping LiDARs Based on Dual LBA 

**Title (ZH)**: DLBAcalib：基于双DLBA的非重叠LiDAR鲁棒外参标定 

**Authors**: Han Ye, Yuqiang Jin, Jinyuan Liu, Tao Li, Wen-An Zhang, Minglei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09176)  

**Abstract**: Accurate extrinsic calibration of multiple LiDARs is crucial for improving the foundational performance of three-dimensional (3D) map reconstruction systems. This paper presents a novel targetless extrinsic calibration framework for multi-LiDAR systems that does not rely on overlapping fields of view or precise initial parameter estimates. Unlike conventional calibration methods that require manual annotations or specific reference patterns, our approach introduces a unified optimization framework by integrating LiDAR bundle adjustment (LBA) optimization with robust iterative refinement. The proposed method constructs an accurate reference point cloud map via continuous scanning from the target LiDAR and sliding-window LiDAR bundle adjustment, while formulating extrinsic calibration as a joint LBA optimization problem. This method effectively mitigates cumulative mapping errors and achieves outlier-resistant parameter estimation through an adaptive weighting mechanism. Extensive evaluations in both the CARLA simulation environment and real-world scenarios demonstrate that our method outperforms state-of-the-art calibration techniques in both accuracy and robustness. Experimental results show that for non-overlapping sensor configurations, our framework achieves an average translational error of 5 mm and a rotational error of 0.2°, with an initial error tolerance of up to 0.4 m/30°. Moreover, the calibration process operates without specialized infrastructure or manual parameter tuning. The code is open source and available on GitHub (\underline{this https URL}) 

**Abstract (ZH)**: 多LiDAR系统的无目标精确外参校准对于提高三维地图重建系统的基础性能至关重要。本文提出了一种新的无目标多LiDAR系统外参校准框架，该框架不依赖于重叠的视场或精确的初始参数估计。与需要手动注释或特定参考模式的传统校准方法不同，我们的方法通过将LiDAR束调整（LBA）优化与鲁棒迭代改进相结合，引入了一个统一的优化框架。该方法通过目标LiDAR连续扫描和滑动窗口LiDAR束调整构建了一个准确的参考点云地图，将外参校准表述为联合LBA优化问题。该方法通过自适应加权机制有效地减少了累积建图误差，并实现了抗离群值的参数估计。在CARLA仿真环境和实际场景中的广泛评估表明，我们的方法在准确性和鲁棒性方面优于现有的最先进的校准技术。实验结果表明，在非重叠传感器配置下，我们的框架实现了平均平移误差为5 mm和旋转误差为0.2°，初始误差耐受度可达0.4 m/30°。此外，校准过程无需专用基础设施或手动参数调整。代码开源并可在GitHub上获得（this https URL）。 

---
# MTF-Grasp: A Multi-tier Federated Learning Approach for Robotic Grasping 

**Title (ZH)**: MTF-抓取：一种多层联邦学习方法用于机器人抓取 

**Authors**: Obaidullah Zaland, Erik Elmroth, Monowar Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10158)  

**Abstract**: Federated Learning (FL) is a promising machine learning paradigm that enables participating devices to train privacy-preserved and collaborative models. FL has proven its benefits for robotic manipulation tasks. However, grasping tasks lack exploration in such settings where robots train a global model without moving data and ensuring data privacy. The main challenge is that each robot learns from data that is nonindependent and identically distributed (non-IID) and of low quantity. This exhibits performance degradation, particularly in robotic grasping. Thus, in this work, we propose MTF-Grasp, a multi-tier FL approach for robotic grasping, acknowledging the unique challenges posed by the non-IID data distribution across robots, including quantitative skewness. MTF-Grasp harnesses data quality and quantity across robots to select a set of "top-level" robots with better data distribution and higher sample count. It then utilizes top-level robots to train initial seed models and distribute them to the remaining "low-level" robots, reducing the risk of model performance degradation in low-level robots. Our approach outperforms the conventional FL setup by up to 8% on the quantity-skewed Cornell and Jacquard grasping datasets. 

**Abstract (ZH)**: 联邦学习（FL）是一种有前途的机器学习范式，使参与设备能够训练隐私保护和协作的模型。FL已在机器人操作任务中证明了其优势。然而，抓取任务在无需移动数据和确保数据隐私的情况下训练全球模型的环境中缺乏探索。主要挑战在于每个机器人从非独立同分布（non-IID）且数量较少的数据中学习。这导致了性能下降，尤其是在机器人抓取方面。因此，在本文中，我们提出了一种名为MTF-Grasp的多层联邦学习方法，以应对机器人之间由于非IID数据分布带来的独特挑战，包括数量上的偏差。MTF-Grasp利用跨机器人之间的数据质量和数量，选择具有更好数据分布和更高样本数量的“顶级”机器人。然后利用顶级机器人训练初始种子模型并分发给剩余的“底层”机器人，从而降低底层机器人模型性能下降的风险。我们的方法在数量偏差的Cornell和Jacquard抓取数据集上比传统FL设置提高了最多8%的性能。 

---
# Learning to Control Dynamical Agents via Spiking Neural Networks and Metropolis-Hastings Sampling 

**Title (ZH)**: 通过脉冲神经网络和Metropolis-Hastings采样学习控制动力学智能体 

**Authors**: Ali Safa, Farida Mohsen, Ali Al-Zawqari  

**Link**: [PDF](https://arxiv.org/pdf/2507.09540)  

**Abstract**: Spiking Neural Networks (SNNs) offer biologically inspired, energy-efficient alternatives to traditional Deep Neural Networks (DNNs) for real-time control systems. However, their training presents several challenges, particularly for reinforcement learning (RL) tasks, due to the non-differentiable nature of spike-based communication. In this work, we introduce what is, to our knowledge, the first framework that employs Metropolis-Hastings (MH) sampling, a Bayesian inference technique, to train SNNs for dynamical agent control in RL environments without relying on gradient-based methods. Our approach iteratively proposes and probabilistically accepts network parameter updates based on accumulated reward signals, effectively circumventing the limitations of backpropagation while enabling direct optimization on neuromorphic platforms. We evaluated this framework on two standard control benchmarks: AcroBot and CartPole. The results demonstrate that our MH-based approach outperforms conventional Deep Q-Learning (DQL) baselines and prior SNN-based RL approaches in terms of maximizing the accumulated reward while minimizing network resources and training episodes. 

**Abstract (ZH)**: 基于Metropolis-Hastings采样的生物启发式、能源高效脉冲神经网络在强化学习中的动态代理控制框架 

---
# Consistency Trajectory Planning: High-Quality and Efficient Trajectory Optimization for Offline Model-Based Reinforcement Learning 

**Title (ZH)**: 一致性轨迹规划：离线模型基于强化学习的高质量与高效轨迹优化 

**Authors**: Guanquan Wang, Takuya Hiraoka, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2507.09534)  

**Abstract**: This paper introduces Consistency Trajectory Planning (CTP), a novel offline model-based reinforcement learning method that leverages the recently proposed Consistency Trajectory Model (CTM) for efficient trajectory optimization. While prior work applying diffusion models to planning has demonstrated strong performance, it often suffers from high computational costs due to iterative sampling procedures. CTP supports fast, single-step trajectory generation without significant degradation in policy quality. We evaluate CTP on the D4RL benchmark and show that it consistently outperforms existing diffusion-based planning methods in long-horizon, goal-conditioned tasks. Notably, CTP achieves higher normalized returns while using significantly fewer denoising steps. In particular, CTP achieves comparable performance with over $120\times$ speedup in inference time, demonstrating its practicality and effectiveness for high-performance, low-latency offline planning. 

**Abstract (ZH)**: 基于一致性轨迹模型的一致性轨迹规划方法（CTP）：高效的 Offline 强化学习轨迹优化方法 

---
# Acquiring and Adapting Priors for Novel Tasks via Neural Meta-Architectures 

**Title (ZH)**: 通过神经元元架构获取和适应新任务的先验知识 

**Authors**: Sudarshan Babu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10446)  

**Abstract**: The ability to transfer knowledge from prior experiences to novel tasks stands as a pivotal capability of intelligent agents, including both humans and computational models. This principle forms the basis of transfer learning, where large pre-trained neural networks are fine-tuned to adapt to downstream tasks. Transfer learning has demonstrated tremendous success, both in terms of task adaptation speed and performance. However there are several domains where, due to lack of data, training such large pre-trained models or foundational models is not a possibility - computational chemistry, computational immunology, and medical imaging are examples. To address these challenges, our work focuses on designing architectures to enable efficient acquisition of priors when large amounts of data are unavailable. In particular, we demonstrate that we can use neural memory to enable adaptation on non-stationary distributions with only a few samples. Then we demonstrate that our hypernetwork designs (a network that generates another network) can acquire more generalizable priors than standard networks when trained with Model Agnostic Meta-Learning (MAML). Subsequently, we apply hypernetworks to 3D scene generation, demonstrating that they can acquire priors efficiently on just a handful of training scenes, thereby leading to faster text-to-3D generation. We then extend our hypernetwork framework to perform 3D segmentation on novel scenes with limited data by efficiently transferring priors from earlier viewed scenes. Finally, we repurpose an existing molecular generative method as a pre-training framework that facilitates improved molecular property prediction, addressing critical challenges in computational immunology 

**Abstract (ZH)**: 具备将先前经验的知识转移到新任务的能力是智能代理，包括人类和计算模型的关键能力。这一原则构成了迁移学习的基础，其中大规模预训练神经网络被微调以适应下游任务。迁移学习在任务适应速度和性能方面取得了巨大成功。然而，在缺乏数据的某些领域中，训练如此大规模的预训练模型或基础模型是不可能的——这在计算化学、计算免疫学和医学成像等领域中例证明显。为了解决这些挑战，我们的工作集中在设计架构以在缺乏大量数据时高效获取先验知识。特别地，我们证明可以通过神经记忆在只有少量样本的情况下使模型适应非平稳分布。然后，我们证明我们的超网络设计（生成另一个网络的网络）在使用模型无关元学习（MAML）进行训练时能够获得比标准网络更广泛适用的先验知识。随后，我们将超网络应用于3D场景生成，证明它们可以在少量训练场景下高效地获取先验知识，从而加快文本到3D的生成速度。我们进一步扩展了超网络框架，在有限数据的情况下对新场景进行3D分割，通过高效地从先前观看的场景转移先验知识来实现。最后，我们将现有的分子生成方法重新用于预训练框架，以改善分子性质预测，解决计算免疫学中的关键挑战。 

---
# SentiDrop: A Multi Modal Machine Learning model for Predicting Dropout in Distance Learning 

**Title (ZH)**: SentiDrop: 多模态机器学习模型预测远程学习中的辍学现象 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.10421)  

**Abstract**: School dropout is a serious problem in distance learning, where early detection is crucial for effective intervention and student perseverance. Predicting student dropout using available educational data is a widely researched topic in learning analytics. Our partner's distance learning platform highlights the importance of integrating diverse data sources, including socio-demographic data, behavioral data, and sentiment analysis, to accurately predict dropout risks. In this paper, we introduce a novel model that combines sentiment analysis of student comments using the Bidirectional Encoder Representations from Transformers (BERT) model with socio-demographic and behavioral data analyzed through Extreme Gradient Boosting (XGBoost). We fine-tuned BERT on student comments to capture nuanced sentiments, which were then merged with key features selected using feature importance techniques in XGBoost. Our model was tested on unseen data from the next academic year, achieving an accuracy of 84\%, compared to 82\% for the baseline model. Additionally, the model demonstrated superior performance in other metrics, such as precision and F1-score. The proposed method could be a vital tool in developing personalized strategies to reduce dropout rates and encourage student perseverance 

**Abstract (ZH)**: 远程学习中学生辍学是一个严重的问题，早期检测对于有效干预和学生坚持不懈至关重要。利用可用的教育数据预测学生辍学是学习分析领域的广泛研究课题。我们的合作伙伴的远程学习平台强调集成多样数据源的重要性，包括社会人口统计数据、行为数据和情感分析，以准确预测辍学风险。在本文中，我们提出了一种新颖的模型，该模型结合了使用双向编码器表示形式（BERT）模型进行的学生评论情感分析与通过极端梯度提升（XGBoost）分析的社会人口统计学和行为数据。我们针对学生评论对BERT进行了微调，以捕捉细微的情感，然后将这些情感与XGBoost特征重要性技术选择的关键特征合并。该模型在下一年未见过的数据上进行了测试，准确率为84%，而基线模型为82%。此外，该模型在其他指标（如精确度和F1分数）上也表现出色。所提出的方法可以成为开发个性化策略以降低辍学率和鼓励学生坚持不懈的重要工具。 

---
# Survey for Categorising Explainable AI Studies Using Data Analysis Task Frameworks 

**Title (ZH)**: 基于数据分析任务框架的可解释人工智能研究分类综述 

**Authors**: Hamzah Ziadeh, Hendrik Knoche  

**Link**: [PDF](https://arxiv.org/pdf/2507.10208)  

**Abstract**: Research into explainable artificial intelligence (XAI) for data analysis tasks suffer from a large number of contradictions and lack of concrete design recommendations stemming from gaps in understanding the tasks that require AI assistance. In this paper, we drew on multiple fields such as visual analytics, cognition, and dashboard design to propose a method for categorising and comparing XAI studies under three dimensions: what, why, and who. We identified the main problems as: inadequate descriptions of tasks, context-free studies, and insufficient testing with target users. We propose that studies should specifically report on their users' domain, AI, and data analysis expertise to illustrate the generalisability of their findings. We also propose study guidelines for designing and reporting XAI tasks to improve the XAI community's ability to parse the rapidly growing field. We hope that our contribution can help researchers and designers better identify which studies are most relevant to their work, what gaps exist in the research, and how to handle contradictory results regarding XAI design. 

**Abstract (ZH)**: 关于可解释人工智能（XAI）在数据分析任务中的研究受到大量矛盾和具体设计建议不足的问题，这些问题是由于对需要AI辅助的任务理解不足造成的。在本文中，我们借鉴了视觉分析、认知和仪表板设计等多个领域，提出了根据三个维度（是什么、为什么、为谁）对XAI研究进行分类和比较的方法。我们发现主要问题包括任务描述不充分、脱离上下文的研究以及对目标用户的测试不足。我们建议研究应具体报告其用户在领域、AI以及数据分析方面的专长，以阐明其研究发现的普遍性。我们还提出了设计和报告XAI任务的研究指南，以提高XAI社区处理快速发展领域中结果矛盾的能力。我们希望我们的贡献能够帮助研究人员和设计师更好地识别哪些研究与他们的工作最为相关，研究中存在哪些缺口，以及如何处理关于XAI设计的矛盾结果。 

---
# Adaptability in Multi-Agent Reinforcement Learning: A Framework and Unified Review 

**Title (ZH)**: 多代理强化学习中的适应性：一个框架与统一综述 

**Authors**: Siyi Hu, Mohamad A Hady, Jianglin Qiao, Jimmy Cao, Mahardhika Pratama, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.10142)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown clear effectiveness in coordinating multiple agents across simulated benchmarks and constrained scenarios. However, its deployment in real-world multi-agent systems (MAS) remains limited, primarily due to the complex and dynamic nature of such environments. These challenges arise from multiple interacting sources of variability, including fluctuating agent populations, evolving task goals, and inconsistent execution conditions. Together, these factors demand that MARL algorithms remain effective under continuously changing system configurations and operational demands. To better capture and assess this capacity for adjustment, we introduce the concept of \textit{adaptability} as a unified and practically grounded lens through which to evaluate the reliability of MARL algorithms under shifting conditions, broadly referring to any changes in the environment dynamics that may occur during learning or execution. Centred on the notion of adaptability, we propose a structured framework comprising three key dimensions: learning adaptability, policy adaptability, and scenario-driven adaptability. By adopting this adaptability perspective, we aim to support more principled assessments of MARL performance beyond narrowly defined benchmarks. Ultimately, this survey contributes to the development of algorithms that are better suited for deployment in dynamic, real-world multi-agent systems. 

**Abstract (ZH)**: 多代理强化学习的适应性在动态现实世界多代理系统中的评估与应用 

---
# Analysis of AI Techniques for Orchestrating Edge-Cloud Application Migration 

**Title (ZH)**: 基于AI技术的边缘-云应用迁移编排分析 

**Authors**: Sadig Gojayev, Ahmad Anaqreh, Carolina Fortuna  

**Link**: [PDF](https://arxiv.org/pdf/2507.10119)  

**Abstract**: Application migration in edge-cloud system enables high QoS and cost effective service delivery. However, automatically orchestrating such migration is typically solved with heuristic approaches. Starting from the Markov Decision Process (MDP), in this paper, we identify, analyze and compare selected state-of-the-art Artificial Intelligence (AI) planning and Reinforcement Learning (RL) approaches for solving the class of edge-cloud application migration problems that can be modeled as Towers of Hanoi (ToH) problems. We introduce a new classification based on state space definition and analyze the compared models also through this lense. The aim is to understand available techniques capable of orchestrating such application migration in emerging computing continuum environments. 

**Abstract (ZH)**: 边缘-云系统中的应用迁移能实现高质量服务和成本效益交付。然而，自动 orchestrating 这样的迁移通常通过启发式方法解决。基于马尔可夫决策过程（MDP），本文识别、分析并比较了用于解决可以建模为汉诺塔问题（ToH）的应用迁移问题的最新人工智能（AI）规划和强化学习（RL）方法。我们引入了一种基于状态空间定义的新分类，并通过这种视角分析所比较的模型。目标是理解在新兴计算 continuum 环境中 orchestrating 这种应用迁移可用的技术。 

---
# BlueGlass: A Framework for Composite AI Safety 

**Title (ZH)**: BlueGlass：一种复合AI安全框架 

**Authors**: Harshal Nandigramwar, Syed Qutub, Kay-Ulrich Scholl  

**Link**: [PDF](https://arxiv.org/pdf/2507.10106)  

**Abstract**: As AI systems become increasingly capable and ubiquitous, ensuring the safety of these systems is critical. However, existing safety tools often target different aspects of model safety and cannot provide full assurance in isolation, highlighting a need for integrated and composite methodologies. This paper introduces BlueGlass, a framework designed to facilitate composite AI safety workflows by providing a unified infrastructure enabling the integration and composition of diverse safety tools that operate across model internals and outputs. Furthermore, to demonstrate the utility of this framework, we present three safety-oriented analyses on vision-language models for the task of object detection: (1) distributional evaluation, revealing performance trade-offs and potential failure modes across distributions; (2) probe-based analysis of layer dynamics highlighting shared hierarchical learning via phase transition; and (3) sparse autoencoders identifying interpretable concepts. More broadly, this work contributes foundational infrastructure and findings for building more robust and reliable AI systems. 

**Abstract (ZH)**: 随着AI系统的能力不断增强和普及，确保这些系统的安全至关重要。然而，现有的安全工具通常针对模型安全的不同方面，在孤立情况下无法提供全面保证，突显出集成和综合方法的需求。本文介绍了BlueGlass框架，该框架旨在通过提供统一基础设施来促进多样化的安全工具的集成与组合，这些工具可以跨越模型内部和输出进行操作。此外，为了展示该框架的应用价值，我们对视觉-语言模型进行了三项面向安全性的分析，用于目标检测任务：（1）分布评估，揭示不同数据分布下的性能权衡和潜在故障模式；（2）基于探针的层动态分析，突出通过相变共享的层级学习；（3）稀疏自编码器识别可解释的概念。更广泛地说，这项工作为构建更加稳健和可靠的AI系统提供了基础架构和发现。 

---
# On Gradual Semantics for Assumption-Based Argumentation 

**Title (ZH)**: 基于假设的论证渐进语义学 

**Authors**: Anna Rapberger, Fabrizio Russo, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.10076)  

**Abstract**: In computational argumentation, gradual semantics are fine-grained alternatives to extension-based and labelling-based semantics . They ascribe a dialectical strength to (components of) arguments sanctioning their degree of acceptability. Several gradual semantics have been studied for abstract, bipolar and quantitative bipolar argumentation frameworks (QBAFs), as well as, to a lesser extent, for some forms of structured argumentation. However, this has not been the case for assumption-based argumentation (ABA), despite it being a popular form of structured argumentation with several applications where gradual semantics could be useful. In this paper, we fill this gap and propose a family of novel gradual semantics for equipping assumptions, which are the core components in ABA frameworks, with dialectical strengths. To do so, we use bipolar set-based argumentation frameworks as an abstraction of (potentially non-flat) ABA frameworks and generalise state-of-the-art modular gradual semantics for QBAFs. We show that our gradual ABA semantics satisfy suitable adaptations of desirable properties of gradual QBAF semantics, such as balance and monotonicity. We also explore an argument-based approach that leverages established QBAF modular semantics directly, and use it as baseline. Finally, we conduct experiments with synthetic ABA frameworks to compare our gradual ABA semantics with its argument-based counterpart and assess convergence. 

**Abstract (ZH)**: 在计算论辩中，渐进语义是基于扩展和基于标签语义的细粒度替代方案。它们赋予论据（或论据的组成部分） dialectical 强度，以表明它们的接受程度。已经研究了几种渐进语义，适用于抽象论辩框架、双极性和定量双极性论辩框架（QBAFs），以及在一定程度上适用于某些结构化论辩形式。然而，这些研究尚未应用于假设基础论辩（ABA），尽管ABA是广泛应用于多种应用场景的一种结构化论辩形式，并且渐进语义在其中可能非常有用。在本文中，我们填补了这一空白，并提出了一种为假设（ABA框架的核心组成部分）赋予 dialectical 强度的新型渐进语义家族。为此，我们使用双极性集合论辩框架作为潜在非平滑的ABA框架的抽象，并对QBAFs的最先进模块化渐进语义进行了泛化。我们证明，我们的渐进ABA语义满足适合的渐进QBAF语义的可取性质的适当改编，例如平衡性和单调性。我们还探索了一种基于论辩的方法，直接利用已建立的QBAF模块化语义，并将其用作基线。最后，我们使用合成的ABA框架进行实验，比较我们的渐进ABA语义与其论辩基线，并评估收敛性。 

---
# Improving monotonic optimization in heterogeneous multi-agent reinforcement learning with optimal marginal deterministic policy gradient 

**Title (ZH)**: 在最优边际确定性策略梯度方法下，提高异构多agent reinforcement学习中的单调优化 

**Authors**: Xiaoyang Yu, Youfang Lin, Shuo Wang, Sheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.09989)  

**Abstract**: In heterogeneous multi-agent reinforcement learning (MARL), achieving monotonic improvement plays a pivotal role in enhancing performance. The HAPPO algorithm proposes a feasible solution by introducing a sequential update scheme, which requires independent learning with No Parameter-sharing (NoPS). However, heterogeneous MARL generally requires Partial Parameter-sharing (ParPS) based on agent grouping to achieve high cooperative performance. Our experiments prove that directly combining ParPS with the sequential update scheme leads to the policy updating baseline drift problem, thereby failing to achieve improvement. To solve the conflict between monotonic improvement and ParPS, we propose the Optimal Marginal Deterministic Policy Gradient (OMDPG) algorithm. First, we replace the sequentially computed $Q_{\psi}^s(s,a_{1:i})$ with the Optimal Marginal Q (OMQ) function $\phi_{\psi}^*(s,a_{1:i})$ derived from Q-functions. This maintains MAAD's monotonic improvement while eliminating the conflict through optimal joint action sequences instead of sequential policy ratio calculations. Second, we introduce the Generalized Q Critic (GQC) as the critic function, employing pessimistic uncertainty-constrained loss to optimize different Q-value estimations. This provides the required Q-values for OMQ computation and stable baselines for actor updates. Finally, we implement a Centralized Critic Grouped Actor (CCGA) architecture that simultaneously achieves ParPS in local policy networks and accurate global Q-function computation. Experimental results in SMAC and MAMuJoCo environments demonstrate that OMDPG outperforms various state-of-the-art MARL baselines. 

**Abstract (ZH)**: 在异构多智能体强化学习（MARL）中实现单调改进对于提升性能至关重要。HAPPO算法通过引入顺序更新方案提出了一种可行的解决方案，该方案要求无参数共享（NoPS）。然而，异构MARL通常需要根据智能体分组实现部分参数共享（ParPS）以达到高协同性能。我们的实验表明，直接将ParPS与顺序更新方案结合会导致策略更新基准漂移问题，从而无法实现改进。为了解决单调改进与ParPS之间的冲突，我们提出了最优边际确定性策略梯度（OMDPG）算法。首先，我们用从Q函数推导出的最优边际Q（OMQ）函数 $\phi_{\psi}^*(s,a_{1:i})$ 替换顺序计算的 $Q_{\psi}^s(s,a_{1:i})$，这在保持马尔可夫平均绝对差异（MAAD）单调改进的同时，通过最优联合动作序列而不是顺序策略比率计算消除了冲突。其次，我们引入广义Q评论器（GQC）作为评论器函数，使用悲观不确定性约束损失优化不同的Q值估计，为OMQ计算提供所需的Q值并为演员更新提供稳定的基线。最后，我们实现了集中评论器分组演员（CCGA）架构，该架构同时在局部策略网络中实现部分参数共享并在准确的全局Q函数计算中实现。SMAC和MAMuJoCo环境中的实验结果表明，OMDPG优于各种最新的MARL基准。 

---
# Technical Requirements for Halting Dangerous AI Activities 

**Title (ZH)**: Technical Requirements for Stopping Dangerous AI Activities 

**Authors**: Peter Barnett, Aaron Scher, David Abecassis  

**Link**: [PDF](https://arxiv.org/pdf/2507.09801)  

**Abstract**: The rapid development of AI systems poses unprecedented risks, including loss of control, misuse, geopolitical instability, and concentration of power. To navigate these risks and avoid worst-case outcomes, governments may proactively establish the capability for a coordinated halt on dangerous AI development and deployment. In this paper, we outline key technical interventions that could allow for a coordinated halt on dangerous AI activities. We discuss how these interventions may contribute to restricting various dangerous AI activities, and show how these interventions can form the technical foundation for potential AI governance plans. 

**Abstract (ZH)**: AI系统rapid发展带来的前所未有的风险，包括失控风险、误用风险、地缘政治不稳定性和权力集中。为应对这些风险并避免最糟糕的结果，政府可能需要主动建立协调停止单一危险AI研发和部署的能力。在本文中，我们概述了关键的技术干预措施，这些措施可以实现对危险AI活动的协调停止。我们讨论了这些干预措施如何限制各种危险AI活动，并展示了这些干预措施如何成为潜在AI治理计划的技术基础。 

---
# Causality-informed Anomaly Detection in Partially Observable Sensor Networks: Moving beyond Correlations 

**Title (ZH)**: 基于因果关系的部分可观测传感器网络异常检测：超越相关性 

**Authors**: Xiaofeng Xiao, Bo Shen, Xubo Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.09742)  

**Abstract**: Nowadays, as AI-driven manufacturing becomes increasingly popular, the volume of data streams requiring real-time monitoring continues to grow. However, due to limited resources, it is impractical to place sensors at every location to detect unexpected shifts. Therefore, it is necessary to develop an optimal sensor placement strategy that enables partial observability of the system while detecting anomalies as quickly as possible. Numerous approaches have been proposed to address this challenge; however, most existing methods consider only variable correlations and neglect a crucial factor: Causality. Moreover, although a few techniques incorporate causal analysis, they rely on interventions-artificially creating anomalies-to identify causal effects, which is impractical and might lead to catastrophic losses. In this paper, we introduce a causality-informed deep Q-network (Causal DQ) approach for partially observable sensor placement in anomaly detection. By integrating causal information at each stage of Q-network training, our method achieves faster convergence and tighter theoretical error bounds. Furthermore, the trained causal-informed Q-network significantly reduces the detection time for anomalies under various settings, demonstrating its effectiveness for sensor placement in large-scale, real-world data streams. Beyond the current implementation, our technique's fundamental insights can be applied to various reinforcement learning problems, opening up new possibilities for real-world causality-informed machine learning methods in engineering applications. 

**Abstract (ZH)**: 基于因果信息的深度Q网络在异常检测中的部分可观测传感器布置方法 

---
# The Hidden Costs of AI: A Review of Energy, E-Waste, and Inequality in Model Development 

**Title (ZH)**: AI隐性成本：模型开发中的能源、电子废物与不平等回顾 

**Authors**: Jenis Winsta  

**Link**: [PDF](https://arxiv.org/pdf/2507.09611)  

**Abstract**: Artificial intelligence (AI) has made remarkable progress in recent years, yet its rapid expansion brings overlooked environmental and ethical challenges. This review explores four critical areas where AI's impact extends beyond performance: energy consumption, electronic waste (e-waste), inequality in compute access, and the hidden energy burden of cybersecurity systems. Drawing from recent studies and institutional reports, the paper highlights systemic issues such as high emissions from model training, rising hardware turnover, global infrastructure disparities, and the energy demands of securing AI. By connecting these concerns, the review contributes to Responsible AI discourse by identifying key research gaps and advocating for sustainable, transparent, and equitable development practices. Ultimately, it argues that AI's progress must align with ethical responsibility and environmental stewardship to ensure a more inclusive and sustainable technological future. 

**Abstract (ZH)**: 人工智能（AI）在近年来取得了显著进步，但其快速扩张带来了未被忽视的环境和伦理挑战。本文综述了AI影响超越性能的四个关键领域：能源消耗、电子废物（e-waste）、计算资源获取不平以及网络安全系统的隐含能源负担。通过参考近期的研究和机构报告，文章指出了系统性问题，如模型训练的高排放、硬件更新频率上升、全球基础设施差距以及AI安全的能源需求。通过将这些担忧联系起来，本文综述为负责任的AI讨论做出了贡献，识别了关键研究缺口，并倡导可持续、透明和公平的发展实践。最终，本文认为AI的发展必须与伦理责任和环境 stewardship 相匹配，以确保一个更加包容和可持续的技术未来。 

---
# A Taxonomy of Omnicidal Futures Involving Artificial Intelligence 

**Title (ZH)**: 人工智能涉及的万劫不复的未来分类 

**Authors**: Andrew Critch, Jacob Tsimerman  

**Link**: [PDF](https://arxiv.org/pdf/2507.09369)  

**Abstract**: This report presents a taxonomy and examples of potential omnicidal events resulting from AI: scenarios where all or almost all humans are killed. These events are not presented as inevitable, but as possibilities that we can work to avoid. Insofar as large institutions require a degree of public support in order to take certain actions, we hope that by presenting these possibilities in public, we can help to support preventive measures against catastrophic risks from AI. 

**Abstract (ZH)**: 本报告提出了一种关于由AI引发的潜在全人类灭绝事件的分类及其例子：描述了可能导致所有或几乎所有人被杀的情景。这些事件并不被视为不可避免，而是我们可以通过努力来避免的可能性。鉴于大型机构需要一定程度的公众支持才能采取某些行动，我们希望通过在公共场合呈现这些可能性，来支持防止AI带来的灾难性风险的预防措施。 

---
# Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity 

**Title (ZH)**: 衡量2025年前AI对资深开源开发者 productivity 的影响 

**Authors**: Joel Becker, Nate Rush, Elizabeth Barnes, David Rein  

**Link**: [PDF](https://arxiv.org/pdf/2507.09089)  

**Abstract**: Despite widespread adoption, the impact of AI tools on software development in the wild remains understudied. We conduct a randomized controlled trial (RCT) to understand how AI tools at the February-June 2025 frontier affect the productivity of experienced open-source developers. 16 developers with moderate AI experience complete 246 tasks in mature projects on which they have an average of 5 years of prior experience. Each task is randomly assigned to allow or disallow usage of early 2025 AI tools. When AI tools are allowed, developers primarily use Cursor Pro, a popular code editor, and Claude 3.5/3.7 Sonnet. Before starting tasks, developers forecast that allowing AI will reduce completion time by 24%. After completing the study, developers estimate that allowing AI reduced completion time by 20%. Surprisingly, we find that allowing AI actually increases completion time by 19%--AI tooling slowed developers down. This slowdown also contradicts predictions from experts in economics (39% shorter) and ML (38% shorter). To understand this result, we collect and evaluate evidence for 20 properties of our setting that a priori could contribute to the observed slowdown effect--for example, the size and quality standards of projects, or prior developer experience with AI tooling. Although the influence of experimental artifacts cannot be entirely ruled out, the robustness of the slowdown effect across our analyses suggests it is unlikely to primarily be a function of our experimental design. 

**Abstract (ZH)**: 尽管AI工具已被广泛采用，但它们在实际软件开发中的影响仍研究不足。我们开展一项随机对照试验（RCT），以了解2025年2月至6月前沿的AI工具如何影响有经验的开源开发者的工作效率。16名拥有中等AI经验的开发者在成熟项目中完成了246项任务，这些项目他们平均已有5年的开发经验。每项任务均随机分配，允许或不允许使用早期2025年的AI工具。当允许使用AI工具时，开发者主要使用流行的代码编辑器Cursor Pro，以及Claude 3.5/3.7 Sonnet。在开始任务前，开发者预测允许使用AI可将完成时间减少24%。完成研究后，开发者估计允许使用AI将完成时间减少了20%。令人惊讶的是，我们发现允许使用AI实际上将完成时间延长了19%——AI工具反而减慢了开发者的速度。这种减慢也与经济学专家（缩短39%）和机器学习专家（缩短38%）的预测不符。为了理解这一结果，我们收集并评估了可能对观察到的减慢效果有贡献的20种设置属性的证据，例如项目的规模和质量标准，或开发者先前使用AI工具的经验。虽然无法完全排除实验 artefacts 的影响，但我们在分析中发现的减慢效果的稳健性表明，这不太可能是由于实验设计的功能。 

---
# BioAnalyst: A Foundation Model for Biodiversity 

**Title (ZH)**: BioAnalyst: 生物多样性基础模型 

**Authors**: Athanasios Trantas, Martino Mensio, Stylianos Stasinos, Sebastian Gribincea, Taimur Khan, Damian Podareanu, Aliene van der Veen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09080)  

**Abstract**: The accelerating loss of biodiversity presents critical challenges for ecological research and conservation strategies. The preservation of biodiversity is paramount for maintaining ecological balance and ensuring the sustainability of ecosystems. However, biodiversity faces numerous threats, including habitat loss, climate change, and the proliferation of invasive species. Addressing these and other ecology-related challenges, both at local and global scales, requires comprehensive monitoring, predictive and conservation planning capabilities. Artificial Intelligence (AI) Foundation Models (FMs) have gained significant momentum in numerous scientific domains by leveraging vast datasets to learn general-purpose representations adaptable to various downstream tasks. This paradigm holds immense promise for biodiversity conservation. In response, we introduce BioAnalyst, the first Foundation Model tailored for biodiversity analysis and conservation planning. BioAnalyst employs a transformer-based architecture, pre-trained on extensive multi-modal datasets encompassing species occurrence records, remote sensing indicators, climate and environmental variables. BioAnalyst is designed for adaptability, allowing for fine-tuning of a range of downstream tasks, such as species distribution modelling, habitat suitability assessments, invasive species detection, and population trend forecasting. We evaluate the model's performance on two downstream use cases, demonstrating its generalisability compared to existing methods, particularly in data-scarce scenarios for two distinct use-cases, establishing a new accuracy baseline for ecological forecasting. By openly releasing BioAnalyst and its fine-tuning workflows to the scientific community, we aim to foster collaborative efforts in biodiversity modelling and advance AI-driven solutions to pressing ecological challenges. 

**Abstract (ZH)**: 生物多样性的加速丧失对生态研究和保护策略提出了关键性挑战。生物多样性的维持对于维持生态平衡和确保生态系统的可持续性至关重要。然而，生物多样性面临着诸多威胁，包括栖息地丧失、气候变化和入侵物种的扩散。针对这些及其它生态相关挑战，无论是局部还是全球尺度，都要求具备全面监测、预测和保护规划的能力。通过利用大量数据集来学习适应各种下游任务的泛化表示，人工智能基础模型（AI Foundation Models, FMs）在诸多科学领域获得了显著进展。这一范式为生物多样性保护提供了巨大的潜力。为此，我们引入了BioAnalyst，这是首个专门针对生物多样性分析和保护规划的基础模型。BioAnalyst采用基于变换器的架构，预训练于包含物种分布记录、遥感指标、气候和环境变量的多模态大数据集上。BioAnalyst设计灵活，允许对多种下游任务进行微调，例如物种分布建模、栖息地适宜性评估、入侵物种检测和种群趋势预测。我们对模型进行了两个下游应用场景的评估，展示了其在数据稀缺条件下的一般适用性，特别是在两个不同的应用场景中建立了生态预测的新准确率基线。通过公开发布BioAnalyst及其微调工作流程，我们旨在促进生物多样性建模的协作努力，并推动以人工智能驱动的解决方案应对紧迫的生态挑战。 

---
# Multi-Actor Generative Artificial Intelligence as a Game Engine 

**Title (ZH)**: 多行为体生成人工智能作为游戏引擎 

**Authors**: Alexander Sasha Vezhnevets, Jayd Matyas, Logan Cross, Davide Paglieri, Minsuk Chang, William A. Cunningham, Simon Osindero, William S. Isaac, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08892)  

**Abstract**: Generative AI can be used in multi-actor environments with purposes ranging from social science modeling to interactive narrative and AI evaluation. Supporting this diversity of use cases -- which we classify as Simulationist, Dramatist, and Evaluationist -- demands a flexible scenario definition framework. We argue here that a good approach is to take inspiration from tabletop role-playing games (TTRPGs), where a Game Master (GM) is responsible for the environment and generates all parts of the story not directly determined by the voluntary actions of player characters. We argue that the Entity-Component architectural pattern is useful here. In such a system, the GM is not a hardcoded computer game but is itself a configurable entity, composed of components just like any other actor. By design, the approach allows for a separation between the underlying implementation details handled by an engineer, the creation of reusable components, and their composition and configuration managed by a designer who constructs entities from the components. This separation of concerns is instrumental for achieving rapid iteration, maintaining modularity, and ultimately to ensure scalability. We describe the ongoing evolution of the Concordia library in terms of this philosophy, demonstrating how it allows users to effectively configure scenarios that align with their specific goals. 

**Abstract (ZH)**: 生成式AI可以用于多 actors 环境，用途涵盖社会科学建模、互动叙事和AI评估等。为了支持这一多样性用途——我们将其分类为模拟主义、戏剧主义和评价主义——需要一个灵活的场景定义框架。我们认为从中桌游（TTRPG）抽取灵感的一种方法是有效的。在游戏中，主持人（GM）负责环境并生成所有不由玩家角色自愿行为直接决定的故事部分。我们认为实体-组件架构模式在此很有用。在这种系统中，GM 不是一个硬编码的计算机游戏，而是自己就是一个可配置的实体，与任何其他演员一样由组件构成。这一设计方法允许工程师处理底层实现细节，设计人员通过组件的组合和配置创建实体，从而实现关注点的分离。这一点对于实现快速迭代、保持模块化以及最终确保可扩展性至关重要。我们描述了康考迪亚库的持续演变，展示了它是如何让用户能够有效配置与他们具体目标相一致的场景。 

---
# A New Approach for Multicriteria Assessment in the Ranking of Alternatives Using Cardinal and Ordinal Data 

**Title (ZH)**: 基于卡片和序位数据的多准则评估在替代方案排序中的新方法 

**Authors**: Fuh-Hwa Franklin Liu, Su-Chuan Shih  

**Link**: [PDF](https://arxiv.org/pdf/2507.08875)  

**Abstract**: Modern methods for multi-criteria assessment (MCA), such as Data Envelopment Analysis (DEA), Stochastic Frontier Analysis (SFA), and Multiple Criteria Decision-Making (MCDM), are utilized to appraise a collection of Decision-Making Units (DMUs), also known as alternatives, based on several criteria. These methodologies inherently rely on assumptions and can be influenced by subjective judgment to effectively tackle the complex evaluation challenges in various fields. In real-world scenarios, it is essential to incorporate both quantitative and qualitative criteria as they consist of cardinal and ordinal data. Despite the inherent variability in the criterion values of different alternatives, the homogeneity assumption is often employed, significantly affecting evaluations. To tackle these challenges and determine the most appropriate alternative, we propose a novel MCA approach that combines two Virtual Gap Analysis (VGA) models. The VGA framework, rooted in linear programming, is pivotal in the MCA methodology. This approach improves efficiency and fairness, ensuring that evaluations are both comprehensive and dependable, thus offering a strong and adaptive solution. Two comprehensive numerical examples demonstrate the accuracy and transparency of our proposed method. The goal is to encourage continued advancement and stimulate progress in automated decision systems and decision support systems. 

**Abstract (ZH)**: 现代多准则评估方法（MCA）如数据包络分析（DEA）、随机前沿分析（SFA）和多准则决策方法（MCDM）被用于基于多个标准评估决策单元（DMUs）或替代方案的集合。这些方法固有地依赖于假设，并可能受到主观判断的影响，以有效应对各种领域中的复杂评估挑战。在实际场景中，必须同时纳入定性和定量标准，因为这些标准包括基数和序数数据。尽管不同替代方案的指标值具有固有的变异性，但通常会采用同质性假设，这对评估产生了显著影响。为应对这些挑战并确定最合适的替代方案，我们提出了结合两种虚拟差距分析（VGA）模型的新颖MCA方法。VGA框架根植于线性规划，对于MCA方法至关重要。此方法提高了效率和公平性，确保评估既全面又可靠，从而提供了一个强大且适应性强的解决方案。两个综合的数值示例展示了我们提出方法的准确性和透明度。目标是促进持续的进步，并激励自动化决策系统和决策支持系统的进展。 

---
# Disentangling Neural Disjunctive Normal Form Models 

**Title (ZH)**: 拆解神经析取范型模型 

**Authors**: Kexin Gu Baugh, Vincent Perreault, Matthew Baugh, Luke Dickens, Katsumi Inoue, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2507.10546)  

**Abstract**: Neural Disjunctive Normal Form (DNF) based models are powerful and interpretable approaches to neuro-symbolic learning and have shown promising results in classification and reinforcement learning settings without prior knowledge of the tasks. However, their performance is degraded by the thresholding of the post-training symbolic translation process. We show here that part of the performance degradation during translation is due to its failure to disentangle the learned knowledge represented in the form of the networks' weights. We address this issue by proposing a new disentanglement method; by splitting nodes that encode nested rules into smaller independent nodes, we are able to better preserve the models' performance. Through experiments on binary, multiclass, and multilabel classification tasks (including those requiring predicate invention), we demonstrate that our disentanglement method provides compact and interpretable logical representations for the neural DNF-based models, with performance closer to that of their pre-translation counterparts. Our code is available at this https URL. 

**Abstract (ZH)**: 基于神经析取范式(DNF)的模型是神经符号学习的强大且可解释的方法，在无需任务先验知识的情况下，在分类和强化学习设置中显示出有希望的结果。然而，它们的性能在训练后符号转换过程的阈值化过程中受损。我们在此表明，在转换过程中性能下降的部分原因是其无法解开以网络权重形式表示的学习知识。我们通过提出一种新的解耦方法来解决这一问题；通过将编码嵌套规则的节点分裂为更小的独立节点，我们能够更好地保持模型的性能。通过二分类、多分类和多标签分类任务（包括需要谓词发明的任务）的实验，我们证明了我们的解耦方法为神经析取范式基于的模型提供了紧凑且可解释的逻辑表示，其性能接近于其转换前的版本。我们的代码可在以下链接获取：这个 https URL。 

---
# WildFX: A DAW-Powered Pipeline for In-the-Wild Audio FX Graph Modeling 

**Title (ZH)**: WildFX：由数字音频工作站驱动的野外音频效果图建模管道 

**Authors**: Qihui Yang, Taylor Berg-Kirkpatrick, Julian McAuley, Zachary Novack  

**Link**: [PDF](https://arxiv.org/pdf/2507.10534)  

**Abstract**: Despite rapid progress in end-to-end AI music generation, AI-driven modeling of professional Digital Signal Processing (DSP) workflows remains challenging. In particular, while there is growing interest in neural black-box modeling of audio effect graphs (e.g. reverb, compression, equalization), AI-based approaches struggle to replicate the nuanced signal flow and parameter interactions used in professional workflows. Existing differentiable plugin approaches often diverge from real-world tools, exhibiting inferior performance relative to simplified neural controllers under equivalent computational constraints. We introduce WildFX, a pipeline containerized with Docker for generating multi-track audio mixing datasets with rich effect graphs, powered by a professional Digital Audio Workstation (DAW) backend. WildFX supports seamless integration of cross-platform commercial plugins or any plugins in the wild, in VST/VST3/LV2/CLAP formats, enabling structural complexity (e.g., sidechains, crossovers) and achieving efficient parallelized processing. A minimalist metadata interface simplifies project/plugin configuration. Experiments demonstrate the pipeline's validity through blind estimation of mixing graphs, plugin/gain parameters, and its ability to bridge AI research with practical DSP demands. The code is available on: this https URL. 

**Abstract (ZH)**: 尽管端到端人工智能音乐生成取得了快速进展，但基于人工智能的专业数字信号处理（DSP）工作流建模依然具有挑战性。特别是，虽然对音频效果图（如混响、压缩、均衡）的神经黑箱建模日益受到关注，但基于人工智能的方法在复制专业工作流中的细微信号流动和参数交互方面仍显得力不从心。现有的可微插件方法往往与实际工具有所偏离，在同等计算约束条件下表现出较低的性能。我们提出了WildFX，这是一种基于Docker封装的工作流程管道，通过专业的数字音频工作站（DAW）后端生成包含丰富效果图的多轨音频混音数据集。WildFX 支持跨平台商业插件或任何野生插件的无缝集成（适用于VST/VST3/LV2/CLAP格式），能够实现结构复杂性（如侧链、分频器）并实现高效并行处理。简约的元数据接口简化了项目/插件配置。实验通过盲估计算法人声图、插件/增益参数，并展示了其将人工智能研究与实际DSP需求衔接的能力。代码可在以下链接获取：this https URL。 

---
# Accurate generation of chemical reaction transition states by conditional flow matching 

**Title (ZH)**: 基于条件流匹配的化学反应过渡态准确生成 

**Authors**: Ping Tuo, Jiale Chen, Ju Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.10530)  

**Abstract**: Transition state (TS) structures define the critical geometries and energy barriers underlying chemical reactivity, yet their fleeting nature renders them experimentally elusive and drives the reliance on costly, high-throughput density functional theory (DFT) calculations. Here, we introduce TS-GEN, a conditional flow-matching generative model that maps samples from a simple Gaussian prior directly to transition-state saddle-point geometries in a single, deterministic pass. By embedding both reactant and product conformations as conditioning information, TS-GEN learns to transport latent noise to true TS structures via an optimal-transport path, effectively replacing the iterative optimization common in nudged-elastic band or string-method algorithms. TS-GEN delivers unprecedented accuracy, achieving a root-mean-square deviation of $0.004\ \rm{\mathring{A}}$ (vs. $0.103\ \rm{\mathring{A}}$ for prior state-of-the-art) and a mean barrier-height error of $1.019\ {\rm kcal/mol}$ (vs. $2.864\ {\rm kcal/mol}$), while requiring only $0.06\ {\rm s}$ GPU time per inference. Over 87% of generated TSs meet chemical-accuracy criteria ($<1.58\ {\rm kcal/mol}$ error), substantially outpacing existing methods. TS-GEN also exhibits strong transferability to out-of-distribution reactions from a larger database. By uniting sub-angstrom precision, sub-second speed, and broad applicability, TS-GEN will be highly useful for high-throughput exploration of complex reaction networks, paving the way to the exploration of novel chemical reaction mechanisms. 

**Abstract (ZH)**: TS-GEN: A Conditional Flow-Matching Generative Model for Accurate and Efficient Transition-State Generation 

---
# Benchmarking and Evaluation of AI Models in Biology: Outcomes and Recommendations from the CZI Virtual Cells Workshop 

**Title (ZH)**: 生物领域中人工智能模型的基准测试与评估：CZI 虚拟细胞工作坊的成果与建议 

**Authors**: Elizabeth Fahsbender, Alma Andersson, Jeremy Ash, Polina Binder, Daniel Burkhardt, Benjamin Chang, Georg K. Gerber, Anthony Gitter, Patrick Godau, Ankit Gupta, Genevieve Haliburton, Siyu He, Trey Ideker, Ivana Jelic, Aly Khan, Yang-Joon Kim, Aditi Krishnapriyan, Jon M. Laurent, Tianyu Liu 28, Emma Lundberg, Shalin B. Mehta, Rob Moccia, Angela Oliveira Pisco, Katherine S. Pollard, Suresh Ramani, Julio Saez-Rodriguez, Yasin Senbabaoglu, Elana Simon, Srinivasan Sivanandan, Gustavo Stolovitzky, Marc Valer, Bo Wang, Xikun Zhang, James Zou, Katrina Kalantar  

**Link**: [PDF](https://arxiv.org/pdf/2507.10502)  

**Abstract**: Artificial intelligence holds immense promise for transforming biology, yet a lack of standardized, cross domain, benchmarks undermines our ability to build robust, trustworthy models. Here, we present insights from a recent workshop that convened machine learning and computational biology experts across imaging, transcriptomics, proteomics, and genomics to tackle this gap. We identify major technical and systemic bottlenecks such as data heterogeneity and noise, reproducibility challenges, biases, and the fragmented ecosystem of publicly available resources and propose a set of recommendations for building benchmarking frameworks that can efficiently compare ML models of biological systems across tasks and data modalities. By promoting high quality data curation, standardized tooling, comprehensive evaluation metrics, and open, collaborative platforms, we aim to accelerate the development of robust benchmarks for AI driven Virtual Cells. These benchmarks are crucial for ensuring rigor, reproducibility, and biological relevance, and will ultimately advance the field toward integrated models that drive new discoveries, therapeutic insights, and a deeper understanding of cellular systems. 

**Abstract (ZH)**: 人工智能在生物学领域的应用前景广阔，但标准化跨领域基准的缺乏阻碍了我们构建稳健可信模型的能力。为此，我们介绍了最近一次研讨会的成果，该研讨会汇聚了来自成像、转录组学、蛋白质组学和基因组学领域的机器学习和计算生物学专家，以解决这一问题。我们指出了数据异质性、噪声、可重复性挑战、偏见以及公开可用资源碎片化等主要的技术和系统瓶颈，并提出了构建跨任务和数据模态比较机器学习模型基准框架的建议。通过促进高质量的数据管理、标准化工具、全面的评价指标以及开放协作平台，我们旨在加速稳健基准的开发，以驱动AI驱动的虚拟细胞领域的发展。这些基准对于确保严格性、可重复性和生物学相关性至关重要，并将最终推动该领域向集成模型发展，从而促进新的发现、治疗洞察以及细胞系统更深刻的理解。 

---
# BenchReAD: A systematic benchmark for retinal anomaly detection 

**Title (ZH)**: BenchReAD: 一种系统性的视网膜异常检测基准 

**Authors**: Chenyu Lian, Hong-Yu Zhou, Zhanli Hu, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10492)  

**Abstract**: Retinal anomaly detection plays a pivotal role in screening ocular and systemic diseases. Despite its significance, progress in the field has been hindered by the absence of a comprehensive and publicly available benchmark, which is essential for the fair evaluation and advancement of methodologies. Due to this limitation, previous anomaly detection work related to retinal images has been constrained by (1) a limited and overly simplistic set of anomaly types, (2) test sets that are nearly saturated, and (3) a lack of generalization evaluation, resulting in less convincing experimental setups. Furthermore, existing benchmarks in medical anomaly detection predominantly focus on one-class supervised approaches (training only with negative samples), overlooking the vast amounts of labeled abnormal data and unlabeled data that are commonly available in clinical practice. To bridge these gaps, we introduce a benchmark for retinal anomaly detection, which is comprehensive and systematic in terms of data and algorithm. Through categorizing and benchmarking previous methods, we find that a fully supervised approach leveraging disentangled representations of abnormalities (DRA) achieves the best performance but suffers from significant drops in performance when encountering certain unseen anomalies. Inspired by the memory bank mechanisms in one-class supervised learning, we propose NFM-DRA, which integrates DRA with a Normal Feature Memory to mitigate the performance degradation, establishing a new SOTA. The benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 视网膜异常检测在眼科和全身疾病筛查中扮演着重要角色。尽管如此，由于缺乏全面且公开可用的标准基准，该领域的进展受到了限制，而标准基准对于方法的公平评估和进步至关重要。由于这一限制，之前与视网膜图像相关的异常检测工作受到了以下限制：（1）异常类型过于有限且过于简单，（2）测试集几乎饱和，以及（3）缺乏泛化评估，从而导致不够令人信服的实验设置。此外，现有的医学异常检测基准主要集中在单类监督方法上（仅使用负样本进行训练），而忽视了临床实践中通常可用的大量标记异常数据和未标记数据。为弥补这些不足，我们提出了一个全面且系统化的视网膜异常检测基准。通过对先前方法进行分类和基准测试，我们发现利用分离表示异常（DRA）的完全监督方法表现最佳，但在遇到某些未见过的异常时性能显著下降。受单类监督学习中记忆库机制的启发，我们提出了NFM-DRA，将DRA与Normal Feature Memory相结合，以缓解性能下降，从而建立了新的SOTA。该基准可在此处公开访问：<https://github.com/alibaba/Qwen-Benchmark>。 

---
# Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation 

**Title (ZH)**: 基于半监督联邦学习和机器人视觉确认的隐私保护多阶段跌倒检测框架 

**Authors**: Seyed Alireza Rahimi Azghadi, Truong-Thanh-Hung Nguyen, Helene Fournier, Monica Wachowicz, Rene Richard, Francis Palma, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10474)  

**Abstract**: The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time. However, to provide timely intervention and avoid unnecessary alarms, detection systems must be effective and reliable while addressing privacy concerns regarding the user. In this work, we propose a framework for detecting falls using several complementary systems: a semi-supervised federated learning-based fall detection system (SF2D), an indoor localization and navigation system, and a vision-based human fall recognition system. A wearable device and an edge device identify a fall scenario in the first system. On top of that, the second system uses an indoor localization technique first to localize the fall location and then navigate a robot to inspect the scenario. A vision-based detection system running on an edge device with a mounted camera on a robot is used to recognize fallen people. Each of the systems of this proposed framework achieves different accuracy rates. Specifically, the SF2D has a 0.81% failure rate equivalent to 99.19% accuracy, while the vision-based fallen people detection achieves 96.3% accuracy. However, when we combine the accuracy of these two systems with the accuracy of the navigation system (95% success rate), our proposed framework creates a highly reliable performance for fall detection, with an overall accuracy of 99.99%. Not only is the proposed framework safe for older adults, but it is also a privacy-preserving solution for detecting falls. 

**Abstract (ZH)**: 老龄化人口快速增长，老年人跌倒的风险也在增加。跌倒是造成伤害的主要原因，及时监测和检测可以大大节省医疗费用和恢复时间。然而，为了提供及时的干预并避免不必要的报警，检测系统必须有效可靠且能够解决用户隐私问题。本研究提出了一种使用多种互补系统的框架：基于半监督联邦学习的跌倒检测系统（SF2D）、室内定位与导航系统以及基于视觉的人体跌倒识别系统。第一系统使用可穿戴设备和边缘设备识别跌倒场景。在此基础上，第二系统首先使用室内定位技术定位跌倒地点，然后导航机器人检查场景。机器人上装有摄像头的边缘设备运行的基于视觉的检测系统用于识别跌倒的人。该框架中的每个系统都实现了不同的准确率。具体来说，SF2D的失败率为0.81%，相当于99.19%的准确率，而基于视觉的人体跌倒检测准确率为96.3%。然而，当我们结合这两大系统与导航系统（95%的成功率）的准确率时，本研究提出框架为跌倒检测创造了高度可靠的性能，整体准确率为99.99%。不仅该框架安全适用于老年人，而且还是一个保护隐私的跌倒检测解决方案。 

---
# An Empirical Evaluation of AI-Powered Non-Player Characters' Perceived Realism and Performance in Virtual Reality Environments 

**Title (ZH)**: 基于AI驱动的非玩家角色在虚拟现实环境中感知现实感与性能的实证评价 

**Authors**: Mikko Korkiakoski, Saeid Sheikhi, Jesper Nyman, Jussi Saariniemi, Kalle Tapio, Panos Kostakos  

**Link**: [PDF](https://arxiv.org/pdf/2507.10469)  

**Abstract**: Advancements in artificial intelligence (AI) have significantly enhanced the realism and interactivity of non-player characters (NPCs) in virtual reality (VR), creating more engaging and believable user experiences. This paper evaluates AI-driven NPCs within a VR interrogation simulator, focusing on their perceived realism, usability, and system performance. The simulator features two AI-powered NPCs, a suspect, and a partner, using GPT-4 Turbo to engage participants in a scenario to determine the suspect's guilt or innocence. A user study with 18 participants assessed the system using the System Usability Scale (SUS), Game Experience Questionnaire (GEQ), and a Virtual Agent Believability Questionnaire, alongside latency measurements for speech-to-text (STT), text-to-speech (TTS), OpenAI GPT-4 Turbo, and overall (cycle) latency. Results showed an average cycle latency of 7 seconds, influenced by the increasing conversational context. Believability scored 6.67 out of 10, with high ratings in behavior, social relationships, and intelligence but moderate scores in emotion and personality. The system achieved a SUS score of 79.44, indicating good usability. These findings demonstrate the potential of large language models to improve NPC realism and interaction in VR while highlighting challenges in reducing system latency and enhancing emotional depth. This research contributes to the development of more sophisticated AI-driven NPCs, revealing the need for performance optimization to achieve increasingly immersive virtual experiences. 

**Abstract (ZH)**: 人工智能（AI）的进步显著提高了虚拟现实（VR）中非玩家角色（NPCs）的 realism 和交互性，创造了更具吸引力和可信度的用户体验。本文评估了AI驱动的NPC在VR审讯模拟器中的应用，重点关注它们的可信度、可用性和系统性能。该模拟器使用GPT-4 Turbo打造了两个NPC角色，一名嫌疑人和一名伙伴，与参与者进行交互，以确定嫌疑人的罪行。该研究包含18名参与者，使用系统可用性量表（SUS）、游戏体验问卷（GEQ）和虚拟代理可信度问卷评估系统，并测量了语音转文本（STT）、文本转语音（TTS）、OpenAI GPT-4 Turbo以及整体（周期）延迟。结果显示平均周期延迟为7秒，受对话上下文增加的影响。可信度评分为6.67分，行为、社会关系和智力方面得分较高，但在情感和个性方面得分较低。该系统获得了SUS评分为79.44，表明良好的可用性。这些发现展示了大型语言模型在提高VR中NPC的realism和交互性方面的潜在价值，同时也指出了减少系统延迟和增强情感深度的挑战。这项研究为开发更高级的AI驱动NPC做出了贡献，揭示了实现日益沉浸式虚拟体验时需要进行性能优化的需求。 

---
# Evaluating Fake Music Detection Performance Under Audio Augmentations 

**Title (ZH)**: 评估音频增强条件下假音乐检测性能 

**Authors**: Tomasz Sroka, Tomasz Wężowicz, Dominik Sidorczuk, Mateusz Modrzejewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.10447)  

**Abstract**: With the rapid advancement of generative audio models, distinguishing between human-composed and generated music is becoming increasingly challenging. As a response, models for detecting fake music have been proposed. In this work, we explore the robustness of such systems under audio augmentations. To evaluate model generalization, we constructed a dataset consisting of both real and synthetic music generated using several systems. We then apply a range of audio transformations and analyze how they affect classification accuracy. We test the performance of a recent state-of-the-art musical deepfake detection model in the presence of audio augmentations. The performance of the model decreases significantly even with the introduction of light augmentations. 

**Abstract (ZH)**: 随着生成音频模型的迅速发展，区分由人类创作和生成的音乐变得日益困难。为应对这一挑战，已经提出了检测假音乐的模型。在本文中，我们探讨了这些系统在音频增强下的鲁棒性。为了评估模型的泛化能力，我们构建了一个包含真实音乐和使用多种系统生成的合成音乐的数据集。然后，我们应用多种音频变换，并分析这些变换如何影响分类准确性。在音频增强存在的情况下，我们测试了一种最新的音乐深度假象检测模型的性能。即使引入轻度增强，模型的性能也显著下降。 

---
# Efficient Federated Learning with Heterogeneous Data and Adaptive Dropout 

**Title (ZH)**: 异质数据下的高效联邦学习与自适应失活 

**Authors**: Ji Liu, Beichen Ma, Yang Zhou, Jingbo Zhou, Ruoming Jin, Dejing Dou, Huaiyu Dai, Haixun Wang, Patrick Valduriez  

**Link**: [PDF](https://arxiv.org/pdf/2507.10430)  

**Abstract**: Federated Learning (FL) is a promising distributed machine learning approach that enables collaborative training of a global model using multiple edge devices. The data distributed among the edge devices is highly heterogeneous. Thus, FL faces the challenge of data distribution and heterogeneity, where non-Independent and Identically Distributed (non-IID) data across edge devices may yield in significant accuracy drop. Furthermore, the limited computation and communication capabilities of edge devices increase the likelihood of stragglers, thus leading to slow model convergence. In this paper, we propose the FedDHAD FL framework, which comes with two novel methods: Dynamic Heterogeneous model aggregation (FedDH) and Adaptive Dropout (FedAD). FedDH dynamically adjusts the weights of each local model within the model aggregation process based on the non-IID degree of heterogeneous data to deal with the statistical data heterogeneity. FedAD performs neuron-adaptive operations in response to heterogeneous devices to improve accuracy while achieving superb efficiency. The combination of these two methods makes FedDHAD significantly outperform state-of-the-art solutions in terms of accuracy (up to 6.7% higher), efficiency (up to 2.02 times faster), and computation cost (up to 15.0% smaller). 

**Abstract (ZH)**: Federated Learning框架FedDHAD：动态异质模型聚合与自适应丢弃 

---
# Energy Efficiency in AI for 5G and Beyond: A DeepRx Case Study 

**Title (ZH)**: AI在5G及更 beyond 的能效研究：DeepRx案例分析 

**Authors**: Amine Lbath, Ibtissam Labriji  

**Link**: [PDF](https://arxiv.org/pdf/2507.10409)  

**Abstract**: This study addresses the challenge of balancing energy efficiency with performance in AI/ML models, focusing on DeepRX, a deep learning receiver based on a fully convolutional ResNet architecture. We evaluate the energy consumption of DeepRX, considering factors including FLOPs/Watt and FLOPs/clock, and find consistency between estimated and actual energy usage, influenced by memory access patterns. The research extends to comparing energy dynamics during training and inference phases. A key contribution is the application of knowledge distillation (KD) to train a compact DeepRX \textit{student} model that emulates the performance of the \textit{teacher} model but with reduced energy consumption. We experiment with different student model sizes, optimal teacher sizes, and KD hyperparameters. Performance is measured by comparing the Bit Error Rate (BER) performance versus Signal-to-Interference \& Noise Ratio (SINR) values of the distilled model and a model trained from scratch. The distilled models demonstrate a lower error floor across SINR levels, highlighting the effectiveness of KD in achieving energy-efficient AI solutions. 

**Abstract (ZH)**: 本研究探讨了在AI/ML模型中平衡能效与性能的挑战，重点关注基于全卷积ResNet架构的DeepRX深度学习接收机。我们评估了DeepRX的能效，考虑了每瓦浮点运算次数(FLOPs/Watt)和每时钟周期浮点运算次数(FLOPs/clock)等因素，并发现了估算的能耗与实际能耗之间的一致性，受内存访问模式的影响。研究还扩展到了训练和推理阶段能效动态的比较。一个主要贡献是应用知识蒸馏(KD)训练一个紧凑的DeepRX学生模型，该模型在能耗降低的情况下模拟了教师模型的性能。我们实验了不同的学生模型大小、最优教师模型大小以及KD超参数。性能通过比较蒸馏模型和从头开始训练的模型的比特错误率(BER)性能与信号干扰与噪声比(SINR)值来进行衡量。蒸馏模型在SINR值不同水平上显示出较低的错误底限，突显了KD在实现能效AI解决方案方面的有效性。 

---
# TAT: Temporal-Aligned Transformer for Multi-Horizon Peak Demand Forecasting 

**Title (ZH)**: TAT：时间对齐变换器在多 horizon 尖峰需求预测中的应用 

**Authors**: Zhiyuan Zhao, Sitan Yang, Kin G. Olivares, Boris N. Oreshkin, Stan Vitebsky, Michael W. Mahoney, B. Aditya Prakash, Dmitry Efimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10349)  

**Abstract**: Multi-horizon time series forecasting has many practical applications such as demand forecasting. Accurate demand prediction is critical to help make buying and inventory decisions for supply chain management of e-commerce and physical retailers, and such predictions are typically required for future horizons extending tens of weeks. This is especially challenging during high-stake sales events when demand peaks are particularly difficult to predict accurately. However, these events are important not only for managing supply chain operations but also for ensuring a seamless shopping experience for customers. To address this challenge, we propose Temporal-Aligned Transformer (TAT), a multi-horizon forecaster leveraging apriori-known context variables such as holiday and promotion events information for improving predictive performance. Our model consists of an encoder and decoder, both embedded with a novel Temporal Alignment Attention (TAA), designed to learn context-dependent alignment for peak demand forecasting. We conduct extensive empirical analysis on two large-scale proprietary datasets from a large e-commerce retailer. We demonstrate that TAT brings up to 30% accuracy improvement on peak demand forecasting while maintaining competitive overall performance compared to other state-of-the-art methods. 

**Abstract (ZH)**: 多 horizons 时间序列预测在需求预测等领域有许多实际应用。在电子商务和实体零售商的供应链管理中，准确的需求预测对于购买和库存决策至关重要，通常需要对未来几周进行预测。尤其是对于高风险销售事件，在需求峰值预测方面更具挑战性。然而，这些事件不仅对供应链操作管理至关重要，也对确保顺畅的购物体验至关重要。为此，我们提出了一种时空对齐变换器（TAT），它利用先验已知的上下文变量（如节假日和促销活动信息）来提高预测性能。我们的模型由编码器和解码器组成，两者都嵌入了一种新型的时空对齐注意力机制（TAA），旨在学习上下文相关的对齐以进行峰值需求预测。我们在一家大型电子商务零售商的两个大规模专有数据集上进行了广泛的经验分析。结果显示，TAT 在峰值需求预测上的准确率最多可提高 30%，同时在总体性能上仍然保持与其他最先进的方法相当。 

---
# Feature Distillation is the Better Choice for Model-Heterogeneous Federated Learning 

**Title (ZH)**: 特征精炼是模型异构联邦学习的更好选择 

**Authors**: Yichen Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.10348)  

**Abstract**: Model-Heterogeneous Federated Learning (Hetero-FL) has attracted growing attention for its ability to aggregate knowledge from heterogeneous models while keeping private data locally. To better aggregate knowledge from clients, ensemble distillation, as a widely used and effective technique, is often employed after global aggregation to enhance the performance of the global model. However, simply combining Hetero-FL and ensemble distillation does not always yield promising results and can make the training process unstable. The reason is that existing methods primarily focus on logit distillation, which, while being model-agnostic with softmax predictions, fails to compensate for the knowledge bias arising from heterogeneous models. To tackle this challenge, we propose a stable and efficient Feature Distillation for model-heterogeneous Federated learning, dubbed FedFD, that can incorporate aligned feature information via orthogonal projection to integrate knowledge from heterogeneous models better. Specifically, a new feature-based ensemble federated knowledge distillation paradigm is proposed. The global model on the server needs to maintain a projection layer for each client-side model architecture to align the features separately. Orthogonal techniques are employed to re-parameterize the projection layer to mitigate knowledge bias from heterogeneous models and thus maximize the distilled knowledge. Extensive experiments show that FedFD achieves superior performance compared to state-of-the-art methods. 

**Abstract (ZH)**: 模型异构联邦学习中的稳定高效特征蒸馏（FedFD） 

---
# Toolsuite for Implementing Multiagent Systems Based on Communication Protocols 

**Title (ZH)**: 基于通信协议实现多agent系统的工具套件 

**Authors**: Amit K. Chopra, Samuel H. Christie V, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10324)  

**Abstract**: Interaction-Oriented Programming (IOP) is an approach to building a multiagent system by modeling the interactions between its roles via a flexible interaction protocol and implementing agents to realize the interactions of the roles they play in the protocol.
In recent years, we have developed an extensive suite of software that enables multiagent system developers to apply IOP. These include tools for efficiently verifying protocols for properties such as liveness and safety and middleware that simplifies the implementation of agents. This paper presents some of that software suite. 

**Abstract (ZH)**: 面向交互的编程（IOP）是一种通过使用灵活的交互协议模型化的角色之间交互来构建多智能体系统的方法，并实现执行协议中角色交互的智能体。近年来，我们开发了一整套软件工具，使多智能体系统开发者能够应用IOP。这些工具包括用于高效验证具有活锁和安全性等性质的协议的工具以及简化智能体实现的中间件。本文介绍了其中的一些软件工具。 

---
# Recognizing Dementia from Neuropsychological Tests with State Space Models 

**Title (ZH)**: 基于态空模型的痴呆识别研究——从神经心理学测试着手 

**Authors**: Liming Wang, Saurabhchand Bhati, Cody Karjadi, Rhoda Au, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2507.10311)  

**Abstract**: Early detection of dementia is critical for timely medical intervention and improved patient outcomes. Neuropsychological tests are widely used for cognitive assessment but have traditionally relied on manual scoring. Automatic dementia classification (ADC) systems aim to infer cognitive decline directly from speech recordings of such tests. We propose Demenba, a novel ADC framework based on state space models, which scale linearly in memory and computation with sequence length. Trained on over 1,000 hours of cognitive assessments administered to Framingham Heart Study participants, some of whom were diagnosed with dementia through adjudicated review, our method outperforms prior approaches in fine-grained dementia classification by 21\%, while using fewer parameters. We further analyze its scaling behavior and demonstrate that our model gains additional improvement when fused with large language models, paving the way for more transparent and scalable dementia assessment tools. Code: this https URL 

**Abstract (ZH)**: early detection of dementia is critical for timely medical intervention and improved patient outcomes. 基于状态空间模型的Demenba：一种线性扩展的自动痴呆分类框架，实现了更细粒度的分类性能提升与参数减少。进一步分析其扩展行为，并展示了将该模型与大型语言模型融合后获得了额外的提升，为更透明和可扩展的痴呆评估工具铺平了道路。 

---
# Visual Analytics for Explainable and Trustworthy Artificial Intelligence 

**Title (ZH)**: 可解释和可信赖的人工智能的可视化分析 

**Authors**: Angelos Chatzimparmpas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10240)  

**Abstract**: Our society increasingly depends on intelligent systems to solve complex problems, ranging from recommender systems suggesting the next movie to watch to AI models assisting in medical diagnoses for hospitalized patients. With the iterative improvement of diagnostic accuracy and efficiency, AI holds significant potential to mitigate medical misdiagnoses by preventing numerous deaths and reducing an economic burden of approximately 450 EUR billion annually. However, a key obstacle to AI adoption lies in the lack of transparency: many automated systems function as "black boxes," providing predictions without revealing the underlying processes. This opacity can hinder experts' ability to trust and rely on AI systems. Visual analytics (VA) provides a compelling solution by combining AI models with interactive visualizations. These specialized charts and graphs empower users to incorporate their domain expertise to refine and improve the models, bridging the gap between AI and human understanding. In this work, we define, categorize, and explore how VA solutions can foster trust across the stages of a typical AI pipeline. We propose a design space for innovative visualizations and present an overview of our previously developed VA dashboards, which support critical tasks within the various pipeline stages, including data processing, feature engineering, hyperparameter tuning, understanding, debugging, refining, and comparing models. 

**Abstract (ZH)**: 我们的社会日益依赖智能系统来解决复杂问题，从推荐系统建议观看的下一部电影到辅助住院患者进行医学诊断的人工智能模型。随着诊断准确性和效率的迭代提升，人工智能在预防大量死亡并减轻约4500亿欧元的经济负担方面具有巨大的潜力。然而，人工智能采纳的关键障碍在于透明度的缺乏：许多自动化系统充当“黑箱”，提供预测而不揭示内部过程。这种不透明性可能会阻碍专家对人工智能系统的信任和依赖。视觉分析（VA）通过结合人工智能模型与互动可视化提供了引人注目的解决方案。这些专门的图表和图形使用户能够结合其领域专业知识来完善和改进模型，弥合人工智能与人类理解之间的差距。在本研究中，我们定义、分类并探讨VA解决方案如何在典型人工智能管道的各个阶段促进信任。我们提出了创新可视化的设计空间，并概述了我们之前开发的VA仪表板，这些仪表板支持各种管道阶段内的关键任务，包括数据处理、特征工程、超参数调整、理解、调试、完善和模型比较。 

---
# Learning Private Representations through Entropy-based Adversarial Training 

**Title (ZH)**: 基于熵的对抗训练学习隐私表示 

**Authors**: Tassilo Klein, Moin Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10194)  

**Abstract**: How can we learn a representation with high predictive power while preserving user privacy? We present an adversarial representation learning method for sanitizing sensitive content from the learned representation. Specifically, we introduce a variant of entropy - focal entropy, which mitigates the potential information leakage of the existing entropy-based approaches. We showcase feasibility on multiple benchmarks. The results suggest high target utility at moderate privacy leakage. 

**Abstract (ZH)**: 如何在保留用户隐私的同时学习具有高预测能力的表示？我们提出了一种对抗表示学习方法，用于清理学习到的表示中的敏感内容。具体而言，我们引入了一种熵的变体——焦点熵，这可以减轻现有基于熵方法潜在的信息泄漏问题。我们在多个基准上展示了其实现可行性。结果表明，在适度的隐私泄漏下，可以实现较高的目标利用率。 

---
# The Second Machine Turn: From Checking Proofs to Creating Concepts 

**Title (ZH)**: 第二次机器革命：从验证证明到创建概念 

**Authors**: Asvin G  

**Link**: [PDF](https://arxiv.org/pdf/2507.10179)  

**Abstract**: We identify a second machine turn in the process of mathematical discovery: after automating proof-checking, AI is now poised to automate the *creation* of mathematical concepts themselves. We discuss the current state of the art, obstacles and potential solutions as well as a preliminary attempt at mathematizing the creation of concepts itself. The paper ends with an assessment of how these capabilities could reshape mathematics and human-machine collaboration, and a few different futures we might find ourselves in. 

**Abstract (ZH)**: 我们在数学发现过程中识别出第二个机器阶段：在 Automation of Proof-Checking 之后，AI 现在准备自动创造数学概念本身。我们讨论当前的技术水平、障碍和潜在解决方案，以及对概念创造本身进行数学化的一个初步尝试。文章结尾评估了这些能力如何重塑数学和人机协作，并设想了几种可能的未来。 

---
# Play Style Identification Using Low-Level Representations of Play Traces in MicroRTS 

**Title (ZH)**: 使用微RTS游戏轨迹的低级表示进行游戏风格识别 

**Authors**: Ruizhe Yu Xia, Jeremy Gow, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10172)  

**Abstract**: Play style identification can provide valuable game design insights and enable adaptive experiences, with the potential to improve game playing agents. Previous work relies on domain knowledge to construct play trace representations using handcrafted features. More recent approaches incorporate the sequential structure of play traces but still require some level of domain abstraction. In this study, we explore the use of unsupervised CNN-LSTM autoencoder models to obtain latent representations directly from low-level play trace data in MicroRTS. We demonstrate that this approach yields a meaningful separation of different game playing agents in the latent space, reducing reliance on domain expertise and its associated biases. This latent space is then used to guide the exploration of diverse play styles within studied AI players. 

**Abstract (ZH)**: 基于无监督CNN-LSTM自编码器的MicroRTS游戏玩法风格识别 

---
# A PBN-RL-XAI Framework for Discovering a "Hit-and-Run'' Therapeutic Strategy in Melanoma 

**Title (ZH)**: 基于PBN-RL-XAI的黑色素瘤“访问-撤离”治疗策略发现框架 

**Authors**: Zhonglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10136)  

**Abstract**: Innate resistance to anti-PD-1 immunotherapy remains a major clinical challenge in metastatic melanoma, with the underlying molecular networks being poorly understood. To address this, we constructed a dynamic Probabilistic Boolean Network model using transcriptomic data from patient tumor biopsies to elucidate the regulatory logic governing therapy response. We then employed a reinforcement learning agent to systematically discover optimal, multi-step therapeutic interventions and used explainable artificial intelligence to mechanistically interpret the agent's control policy. The analysis revealed that a precisely timed, 4-step temporary inhibition of the lysyl oxidase like 2 protein (LOXL2) was the most effective strategy. Our explainable analysis showed that this ``hit-and-run" intervention is sufficient to erase the molecular signature driving resistance, allowing the network to self-correct without requiring sustained intervention. This study presents a novel, time-dependent therapeutic hypothesis for overcoming immunotherapy resistance and provides a powerful computational framework for identifying non-obvious intervention protocols in complex biological systems. 

**Abstract (ZH)**: 先天性对anti-PD-1免疫治疗的抵抗仍是转移性黑色素瘤临床治疗的主要挑战，其 underlying 分子网络尚不明确。为了解决这一问题，我们利用患者肿瘤活检的转录组数据构建了一个动态概率布尔网络模型，以阐明调控治疗反应的调节逻辑。然后，我们使用强化学习代理系统地发现最优的多步治疗干预措施，并利用可解释的人工智能来机制性地解释代理的控制策略。分析表明，精确时间安排的4步暂态抑制LOXL2蛋白是最有效的方法。我们的可解释分析显示，这种“击打即走”干预足以消除驱动抗性的分子标记，使网络能够自我校正，无需持续干预。该研究提出了一种新的、时间依赖性的治疗假设，以克服免疫治疗抗性，并提供了一种强大的计算框架，用于在复杂生物系统中识别非显性干预协议。 

---
# Extending Defeasibility for Propositional Standpoint Logics 

**Title (ZH)**: 扩展命题立场逻辑中的攻否性 

**Authors**: Nicholas Leisegang, Thomas Meyer, Ivan Varzinczak  

**Link**: [PDF](https://arxiv.org/pdf/2507.10133)  

**Abstract**: In this paper, we introduce a new defeasible version of propositional standpoint logic by integrating Kraus et al.'s defeasible conditionals, Britz and Varzinczak's notions of defeasible necessity and distinct possibility, along with Leisegang et al.'s approach to defeasibility into the standpoint logics of Gómez Álvarez and Rudolph. The resulting logical framework allows for the expression of defeasibility on the level of implications, standpoint modal operators, and standpoint-sharpening statements. We provide a preferential semantics for this extended language and propose a tableaux calculus, which is shown to be sound and complete with respect to preferential entailment. We also establish the computational complexity of the tableaux procedure to be in PSpace. 

**Abstract (ZH)**: 本文引入了一种新的命题立场逻辑的可败斥版本，通过整合Kraus等人提出的可败斥条件、Britz和Varzinczak提出的可败斥必然性和独立可能性概念以及Leisegang等人对可败斥性的处理方法，结合Gómez Álvarez和Rudolph的立场逻辑。 resulting logical framework 允许在推论、立场模态运算符和立场细化声明的层面表达可败斥性。我们为此扩展语言提供了优选语义，并提出了一种表格式计算法，该计算法相对于优选蕴含是sound和complete的。我们还建立了表格式计算法的计算复杂性为PSpace。 

---
# Wavelet-Enhanced Neural ODE and Graph Attention for Interpretable Energy Forecasting 

**Title (ZH)**: Wavelet-增强神经ODE和图注意力机制的可解释能源预测 

**Authors**: Usman Gani Joy  

**Link**: [PDF](https://arxiv.org/pdf/2507.10132)  

**Abstract**: Accurate forecasting of energy demand and supply is critical for optimizing sustainable energy systems, yet it is challenged by the variability of renewable sources and dynamic consumption patterns. This paper introduces a neural framework that integrates continuous-time Neural Ordinary Differential Equations (Neural ODEs), graph attention, multi-resolution wavelet transformations, and adaptive learning of frequencies to address the issues of time series prediction. The model employs a robust ODE solver, using the Runge-Kutta method, paired with graph-based attention and residual connections to better understand both structural and temporal patterns. Through wavelet-based feature extraction and adaptive frequency modulation, it adeptly captures and models diverse, multi-scale temporal dynamics. When evaluated across seven diverse datasets: ETTh1, ETTh2, ETTm1, ETTm2 (electricity transformer temperature), and Waste, Solar, and Hydro (renewable energy), this architecture consistently outperforms state-of-the-art baselines in various forecasting metrics, proving its robustness in capturing complex temporal dependencies. Furthermore, the model enhances interpretability through SHAP analysis, making it suitable for sustainable energy applications. 

**Abstract (ZH)**: 准确预测能源需求和供应对于优化可持续能源系统至关重要，但受到可再生能源波动性和动态消费模式的挑战。本文介绍了一种神经框架，该框架集成连续时间神经常微分方程（Neural ODEs）、图注意力、多分辨率小波变换和频率自适应学习，以解决时间序列预测问题。该模型采用鲁棒的ODE求解器，结合图注意力和残差连接，更好地理解结构和时间模式。通过基于小波的特征提取和自适应频率调制，它能够灵活捕捉和建模多尺度的时间动态。该架构在ETTh1、ETTh2、ETTm1、ETTm2（电力变压器温度）以及废物、太阳能和水能（可再生能源）等七个不同数据集上的一系列预测指标中，始终优于最先进的基线模型，证明了其在捕捉复杂时间依赖性方面的稳健性。此外，通过SHAP分析增加模型的可解释性，使其适用于可持续能源应用。 

---
# A Variance-Reduced Cubic-Regularized Newton for Policy Optimization 

**Title (ZH)**: 带有方差减小的三次正则化牛顿法的策略优化 

**Authors**: Cheng Sun, Zhen Zhang, Shaofu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10120)  

**Abstract**: In this paper, we study a second-order approach to policy optimization in reinforcement learning. Existing second-order methods often suffer from suboptimal sample complexity or rely on unrealistic assumptions about importance sampling. To overcome these limitations, we propose VR-CR-PN, a variance-reduced cubic-regularized policy Newton algorithm. To the best of our knowledge, this is the first algorithm that integrates Hessian-aided variance reduction with second-order policy optimization, effectively addressing the distribution shift problem and achieving best-known sample complexity under general nonconvex conditions but without the need for importance sampling. We theoretically establish that VR-CR-PN achieves a sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-3})$ to reach an $\epsilon$-second-order stationary point, significantly improving upon the previous best result of $\tilde{\mathcal{O}}(\epsilon^{-3.5})$ under comparable assumptions. As an additional contribution, we introduce a novel Hessian estimator for the expected return function, which admits a uniform upper bound independent of the horizon length $H$, allowing the algorithm to achieve horizon-independent sample complexity. 

**Abstract (ZH)**: 在本论文中，我们研究了强化学习中政策优化的二阶方法。现有的二阶方法往往面临次优样本复杂度或依赖于重要性采样的不切实际假设。为克服这些局限，我们提出了一种减少方差的立方正则化政策牛顿算法VR-CR-PN。据我们所知，这是首个将Hessian辅助减少方差与二阶政策优化相结合的算法，有效地解决了分布偏移问题，并在一般非凸条件下达到了最佳的样本复杂度，且无需重要性采样。我们理论分析表明，VR-CR-PN可实现$\tilde{\mathcal{O}}(\epsilon^{-3})$的样本复杂度以达到$\epsilon$-二阶稳定点，这一结果显著优于之前在相似假设下的$\tilde{\mathcal{O}}(\epsilon^{-3.5})$的最佳结果。此外，我们引入了一种新的预期回报函数Hessian估计器，其上界与时间_horizon无关，从而使算法能够获得时间_horizon无关的样本复杂度。 

---
# Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning 

**Title (ZH)**: 增强链式思考推理能力的关键表示微调 

**Authors**: Chenxi Huang, Shaotian Yan, Liang Xie, Binbin Lin, Sinan Fan, Yue Xin, Deng Cai, Chen Shen, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.10085)  

**Abstract**: Representation Fine-tuning (ReFT), a recently proposed Parameter-Efficient Fine-Tuning (PEFT) method, has attracted widespread attention for significantly improving parameter efficiency by editing representation space alone. In this work, we investigate applying ReFT to complex reasoning tasks. However, directly using the native ReFT method, which modifies fixed representations at the beginning and end of each layer, yields suboptimal performance, as these fixed-position representations have uncertain impact on the outputs. We observe that, in complex reasoning tasks, there often exist certain critical representations. These representations either integrate significant information from preceding layers or regulate subsequent layer representations. Through layer-by-layer propagation, they exert a substantial influence on the final output. Naturally, fine-tuning these critical representations has the potential to greatly enhance reasoning performance. Building upon these insights, we propose Critical Representation Fine-Tuning (CRFT), a novel method that identifies and optimizes these critical representations through information flow analysis. CRFT operates within a supervised learning framework, dynamically optimizing critical representations in a low-rank linear subspace while freezing the base model. The effectiveness and efficiency of our method are validated across eight benchmarks for arithmetic and commonsense reasoning, using LLaMA and Mistral model families. Furthermore, our method also adapts effectively to few-shot settings, boosting one-shot accuracy by 16.4%. Our work highlights the untapped potential of representation-level optimization for CoT reasoning, offering a lightweight yet powerful alternative to traditional PEFT methods. 

**Abstract (ZH)**: Representation细调（ReFT）：一种最近提出的参数高效细调（PEFT）方法，通过单独编辑表示空间显著提高了参数效率，引起了广泛关注。在本文中，我们探讨了将ReFT应用于复杂推理任务。然而，直接使用原生的ReFT方法，该方法在每一层的开始和结束处修改固定表示，会导致性能不佳，因为这些固定位置的表示对输出的影响尚不确定。我们观察到，在复杂推理任务中，通常存在一些关键表示，这些表示要么从前一层整合了重要信息，要么调节后续层的表示。通过逐层传播，它们对最终输出产生了重大影响。因此，细调这些关键表示有可能大幅提高推理性能。基于这些见解，我们提出了一种名为关键表示细调（CRFT）的新方法，该方法通过信息流分析来识别和优化这些关键表示。CRFT在监督学习框架下运作，动态优化关键表示在低秩线性子空间中的表现，同时冻结基础模型。我们通过LLaMA和Mistral模型家族在八个算术和常识推理基准上验证了该方法的有效性和效率。此外，我们的方法还能够很好地适应少样本设置，将单样本准确率提升了16.4%。我们的工作突显了表示级优化在CoT推理中的未开发潜力，提供了一种轻量级但强大的传统PEFT方法的替代方案。 

---
# PRISM: Fine-Grained Paper-to-Paper Retrieval with Multi-Aspect-Aware Query Optimization 

**Title (ZH)**: PRISM: 多方面感知查询优化的细粒度文献检索 

**Authors**: Sangwoo Park, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10057)  

**Abstract**: Scientific paper retrieval, particularly framed as document-to-document retrieval, aims to identify relevant papers in response to a long-form query paper, rather than a short query string. Previous approaches to this task have focused on abstracts, embedding them into dense vectors as surrogates for full documents and calculating similarity across them, although abstracts provide only sparse and high-level summaries. To address this, we propose PRISM, a novel document-to-document retrieval method that introduces multiple, fine-grained representations for both the query and candidate papers. In particular, each query paper is decomposed into multiple aspect-specific views and individually embedded, which are then matched against candidate papers similarity segmented to consider their multifaceted dimensions. Moreover, we present SciFullBench, a novel benchmark in which the complete and segmented context of full papers for both queries and candidates is available. Then, experimental results show that PRISM improves performance by an average of 4.3% over existing retrieval baselines. 

**Abstract (ZH)**: 科学论文检索，特别是框架下的文档到文档检索，旨在针对长篇查询论文识别相关论文，而非短查询字符串。针对这一任务，先前的方法集中在摘要上，将摘要嵌入为密集向量以代替全文，并计算它们之间的相似性，尽管摘要仅提供了稀疏且高层次的总结。为解决这一问题，我们提出PRISM，一种新颖的文档到文档检索方法，引入了查询和候选论文的多种细粒度表示。特别地，每篇查询论文被分解为多个方面特定的视图并单独嵌入，然后与候选论文相似地分割匹配，以考虑其多维度特征。此外，我们提出SciFullBench，一种新颖的基准，在其中查询和候选论文的完整和分割上下文是可用的。实验结果表明，PRISM在现有检索基线上的性能平均提高4.3%。 

---
# Evolution of Fear and Social Rewards in Prey-Predator Relationship 

**Title (ZH)**: 猎食者-被捕食者关系中恐惧与社会奖励的进化 

**Authors**: Yuji Kanagawa, Kenji Doya  

**Link**: [PDF](https://arxiv.org/pdf/2507.09992)  

**Abstract**: Fear is a critical brain function for detecting danger and learning to avoid specific stimuli that can lead to danger. While fear is believed to have evolved under pressure from predators, experimentally reproducing the evolution is challenging. To investigate the relationship between environmental conditions, the evolution of fear, and the evolution of other rewards, such as food reward and social reward, we developed a distributed evolutionary simulation. In our simulation, prey and predator agents co-evolve their innate reward functions, including a possibly fear-like term for observing predators, and learn behaviors via reinforcement learning. Surprisingly, our simulation revealed that social reward for observing the same species is more important for prey to survive, and fear-like negative reward for observing predators evolves only after acquiring social reward. We also found that the predator with increased hunting ability (larger mouth) amplified fear emergence, but also that fear evolution is more stable with non-evolving predators that are bad at chasing prey. Additionally, unlike for predators, we found that positive rewards evolve in opposition to fear for stationary threats, as areas with abundant leftover food develop around them. These findings suggest that fear and social reward have had a complex interplay with each other through evolution, along with the nature of predators and threats. 

**Abstract (ZH)**: 恐惧是检测危险和避免特定危险刺激的关键大脑功能。虽然恐惧被认为是在捕食者压力下进化的，但实验性地重现这一进化过程具有挑战性。为研究环境条件、恐惧的进化与其他奖励如食物奖励和社会奖励的进化之间的关系，我们开发了一种分布式进化仿真。在我们的仿真中，被捕食者和捕食者代理共同进化其固有的奖励功能，包括可能类似于恐惧的项以观察捕食者，并通过强化学习学习行为。令人惊讶的是，我们的仿真揭示出，观察同物种的社会奖励对于被捕食者生存更为重要，而观察捕食者的类似恐惧的负向奖励仅在获得社会奖励后才会进化。我们还发现，具有增强狩猎能力（更大嘴巴）的捕食者加剧了恐惧的出现，但具有较差追赶能力的非进化捕食者会使恐惧的进化更加稳定。此外，与捕食者不同，我们发现，对于静止的威胁，正面奖励会在它们周围大量食物残余的区域与恐惧进化相反而出现。这些发现表明，恐惧和社交奖励在捕食者和威胁的本性影响下，通过进化过程与彼此产生了复杂相互作用。 

---
# MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora 

**Title (ZH)**: MixLoRA-DSI: 动态可扩展的LoRA混合专家用于动态 CORPORA 的生成性检索 

**Authors**: Tuan-Luc Huynh, Thuy-Trang Vu, Weiqing Wang, Trung Le, Dragan Gašević, Yuan-Fang Li, Thanh-Toan Do  

**Link**: [PDF](https://arxiv.org/pdf/2507.09924)  

**Abstract**: Continually updating model-based indexes in generative retrieval with new documents remains challenging, as full retraining is computationally expensive and impractical under resource constraints. We propose MixLoRA-DSI, a novel framework that combines an expandable mixture of Low-Rank Adaptation experts with a layer-wise out-of-distribution (OOD)-driven expansion strategy. Instead of allocating new experts for each new corpus, our proposed expansion strategy enables sublinear parameter growth by selectively introducing new experts only when significant number of OOD documents are detected. Experiments on NQ320k and MS MARCO Passage demonstrate that MixLoRA-DSI outperforms full-model update baselines, with minimal parameter overhead and substantially lower training costs. 

**Abstract (ZH)**: 基于生成检索的模型本征索引在加入新文档时持续更新仍具有挑战性，因为全面重训在资源受限条件下既耗时又不切实际。我们提出了一种新型框架MixLoRA-DSI，该框架结合了可扩展的低秩适应专家混合体和逐层分布外推策略。我们的扩展策略不仅避免为每个新语料库分配新的专家，还通过仅在检测到大量分布外文档时才引入新专家，实现了亚线性参数增长。实验结果表明，在NQ320k和MS MARCO Passage数据集上，MixLoRA-DSI在最少参数开销的情况下，训练成本显著降低，并优于全面模型更新基准。 

---
# Large Population Models 

**Title (ZH)**: 大型人口模型 

**Authors**: Ayush Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2507.09901)  

**Abstract**: Many of society's most pressing challenges, from pandemic response to supply chain disruptions to climate adaptation, emerge from the collective behavior of millions of autonomous agents making decisions over time. Large Population Models (LPMs) offer an approach to understand these complex systems by simulating entire populations with realistic behaviors and interactions at unprecedented scale. LPMs extend traditional modeling approaches through three key innovations: computational methods that efficiently simulate millions of agents simultaneously, mathematical frameworks that learn from diverse real-world data streams, and privacy-preserving communication protocols that bridge virtual and physical environments. This allows researchers to observe how agent behavior aggregates into system-level outcomes and test interventions before real-world implementation. While current AI advances primarily focus on creating "digital humans" with sophisticated individual capabilities, LPMs develop "digital societies" where the richness of interactions reveals emergent phenomena. By bridging individual agent behavior and population-scale dynamics, LPMs offer a complementary path in AI research illuminating collective intelligence and providing testing grounds for policies and social innovations before real-world deployment. We discuss the technical foundations and some open problems here. LPMs are implemented by the AgentTorch framework (this http URL) 

**Abstract (ZH)**: 社会面临的许多紧迫挑战，从大流行应对到供应链中断再到气候适应，都源于数百万个自主代理随时间做出决策所产生的集体行为。大规模人群模型（LPMs）通过以前所未有的规模模拟整个具有现实行为和交互模式的人群，提供了一种理解这些复杂系统的办法。LPMs通过三种关键创新扩展了传统的建模方法：高效的计算方法可以同时模拟数百万个代理，数学框架可以从多种多样的现实世界数据流中学习，并且保护隐私的通信协议可以连接虚拟和物理环境。这使研究人员能够观察代理行为如何汇总为系统级别的结果，并在实际实施之前测试干预措施。虽然当前的人工智能进步主要侧重于创建具有复杂个体能力的“数字人类”，LPMs则致力于构建“数字社会”，其中丰富的人际互动揭示了 emergent 现象。通过连接个体代理行为和群体规模的动力学，LPMs提供了一条补充的人工智能研究路径，揭示集体智能，并为政策和社会创新提供测试平台，然后再进行实际部署。我们在这里讨论其技术基础和一些开放问题。LPMs由AgentTorch框架实现（ this http URL）。 

---
# Sequence-Model-Guided Measurement Selection for Quantum State Learning 

**Title (ZH)**: 基于序列模型的量子态学习测量选择 

**Authors**: Jiaxin Huang, Yan Zhu, Giulio Chiribella, Ya-Dong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09891)  

**Abstract**: Characterization of quantum systems from experimental data is a central problem in quantum science and technology. But which measurements should be used to gather data in the first place? While optimal measurement choices can be worked out for small quantum systems, the optimization becomes intractable as the system size grows large. To address this problem, we introduce a deep neural network with a sequence model architecture that searches for efficient measurement choices in a data-driven, adaptive manner. The model can be applied to a variety of tasks, including the prediction of linear and nonlinear properties of quantum states, as well as state clustering and state tomography tasks. In all these tasks, we find that the measurement choices identified by our neural network consistently outperform the uniformly random choice. Intriguingly, for topological quantum systems, our model tends to recommend measurements at the system's boundaries, even when the task is to predict bulk properties. This behavior suggests that the neural network may have independently discovered a connection between boundaries and bulk, without having been provided any built-in knowledge of quantum physics. 

**Abstract (ZH)**: 基于实验数据表征量子系统是量子科学与技术中的一个核心问题。但在最初应使用哪些测量来收集数据？虽然小型量子系统中最佳测量选择可以计算得出，但随着系统规模增大，优化变得不可行。为解决这一问题，我们引入了一种具有序列模型架构的深度神经网络，在数据驱动和自适应方式下搜索高效的测量选择。该模型可以应用于预测量子态的线性和非线性性质、状态聚类以及状态Tomography等多种任务。在所有这些任务中，我们发现由我们的神经网络识别出的测量选择始终优于均匀随机选择。有趣的是，对于拓扑量子系统，即使任务是预测体相性质，我们的模型也倾向于建议在系统的边界上进行测量。这一行为表明，神经网络可能独立地发现边界与体相之间的联系，而无需任何内置的量子物理知识。 

---
# Soft Graph Clustering for single-cell RNA Sequencing Data 

**Title (ZH)**: 软图聚类方法在单细胞RNA测序数据中的应用 

**Authors**: Ping Xu, Pengfei Wang, Zhiyuan Ning, Meng Xiao, Min Wu, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.09890)  

**Abstract**: Clustering analysis is fundamental in single-cell RNA sequencing (scRNA-seq) data analysis for elucidating cellular heterogeneity and diversity. Recent graph-based scRNA-seq clustering methods, particularly graph neural networks (GNNs), have significantly improved in tackling the challenges of high-dimension, high-sparsity, and frequent dropout events that lead to ambiguous cell population boundaries. However, their reliance on hard graph constructions derived from thresholded similarity matrices presents challenges:(i) The simplification of intercellular relationships into binary edges (0 or 1) by applying thresholds, which restricts the capture of continuous similarity features among cells and leads to significant information loss.(ii) The presence of significant inter-cluster connections within hard graphs, which can confuse GNN methods that rely heavily on graph structures, potentially causing erroneous message propagation and biased clustering outcomes. To tackle these challenges, we introduce scSGC, a Soft Graph Clustering for single-cell RNA sequencing data, which aims to more accurately characterize continuous similarities among cells through non-binary edge weights, thereby mitigating the limitations of rigid data structures. The scSGC framework comprises three core components: (i) a zero-inflated negative binomial (ZINB)-based feature autoencoder; (ii) a dual-channel cut-informed soft graph embedding module; and (iii) an optimal transport-based clustering optimization module. Extensive experiments across ten datasets demonstrate that scSGC outperforms 13 state-of-the-art clustering models in clustering accuracy, cell type annotation, and computational efficiency. These results highlight its substantial potential to advance scRNA-seq data analysis and deepen our understanding of cellular heterogeneity. 

**Abstract (ZH)**: 基于图的单细胞RNA测序聚类分析在揭示细胞异质性和多样性方面是基础的。软图聚类方法scSGC在处理高维度、高稀疏性和频繁缺失事件导致的细胞群体边界不明确等挑战方面取得了显著改进。然而，这些方法依赖于从阈值相似矩阵派生的硬图构建，这带来了挑战：（i）通过应用阈值将细胞间的关系简化为二元边（0或1），这限制了连续相似性特征的捕捉，导致大量信息丢失。（ii）硬图中存在显著的跨簇连接，这可能使严重依赖图结构的GNN方法产生错误的消息传播和偏向性的聚类结果。为解决这些挑战，我们提出了scSGC，一种基于图的单细胞RNA测序软聚类方法，旨在通过非二元边权重更准确地刻画细胞间的连续相似性，从而缓解刚性数据结构的限制。scSGC框架包括三个核心组件：（i）零 inflation 负二项式（ZINB）特征自编码器；（ii）双通道切割信息软图嵌入模块；（iii）基于最优传输的聚类优化模块。在十个多组学数据集上的广泛实验表明，scSGC在聚类准确性、细胞类型注释和计算效率方面均优于13种最先进的聚类模型。这些结果突显了其在推进单细胞RNA测序数据分析和深化对细胞异质性的理解方面的巨大潜力。 

---
# NeuTSFlow: Modeling Continuous Functions Behind Time Series Forecasting 

**Title (ZH)**: NeuTSFlow：建模时间序列预测背后的连续函数 

**Authors**: Huibo Xu, Likang Wu, Xianquan Wang, Haoning Dang, Chun-Wun Cheng, Angelica I Aviles-Rivero, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09888)  

**Abstract**: Time series forecasting is a fundamental task with broad applications, yet conventional methods often treat data as discrete sequences, overlooking their origin as noisy samples of continuous processes. Crucially, discrete noisy observations cannot uniquely determine a continuous function; instead, they correspond to a family of plausible functions. Mathematically, time series can be viewed as noisy observations of a continuous function family governed by a shared probability measure. Thus, the forecasting task can be framed as learning the transition from the historical function family to the future function family. This reframing introduces two key challenges: (1) How can we leverage discrete historical and future observations to learn the relationships between their underlying continuous functions? (2) How can we model the transition path in function space from the historical function family to the future function family? To address these challenges, we propose NeuTSFlow, a novel framework that leverages Neural Operators to facilitate flow matching for learning path of measure between historical and future function families. By parameterizing the velocity field of the flow in infinite-dimensional function spaces, NeuTSFlow moves beyond traditional methods that focus on dependencies at discrete points, directly modeling function-level features instead. Experiments on diverse forecasting tasks demonstrate NeuTSFlow's superior accuracy and robustness, validating the effectiveness of the function-family perspective. 

**Abstract (ZH)**: 时间序列预测是具有广泛应用的基础任务，但传统方法往往将数据视为离散序列，忽视了它们作为连续过程的嘈杂样本的本质。关键在于，离散的嘈杂观测值不能唯一确定一个连续函数，而对应于由共享概率测度支配的一系列可能的函数。从数学角度来看，时间序列可以被视为由共享概率测度支配的一系列连续函数的嘈杂观测值。因此，预测任务可以重新框定为从历史函数家族到未来函数家族的学习连续函数过渡。这一重新框定引入了两个关键挑战：（1）我们如何利用历史和未来的离散观测值来学习它们背后连续函数之间的关系？（2）我们如何在函数空间中建模从历史函数家族到未来函数家族的过渡路径？为了解决这些挑战，我们提出了一种名为NeuTSFlow的新框架，该框架利用神经算子促进流匹配，以学习历史和未来函数家族之间的测度路径。通过在无限维函数空间中参数化流的速度场，NeuTSFlow超越了传统的关注离散点依赖性的方法，直接建模函数级特征。在多种预测任务上的实验结果表明，NeuTSFlow在准确性和鲁棒性方面具有优势，验证了函数家族视角的有效性。 

---
# TolerantECG: A Foundation Model for Imperfect Electrocardiogram 

**Title (ZH)**: 容忍ECG：一种适用于不完美心电图的基础模型 

**Authors**: Huynh Nguyen Dang, Thang Pham, Ngan Le, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09887)  

**Abstract**: The electrocardiogram (ECG) is an essential and effective tool for diagnosing heart diseases. However, its effectiveness can be compromised by noise or unavailability of one or more leads of the standard 12-lead recordings, resulting in diagnostic errors or uncertainty. To address these challenges, we propose TolerantECG, a foundation model for ECG signals that is robust to noise and capable of functioning with arbitrary subsets of the standard 12-lead ECG. TolerantECG training combines contrastive and self-supervised learning frameworks to jointly learn ECG signal representations alongside their corresponding knowledge-retrieval-based text report descriptions and corrupted or lead-missing signals. Comprehensive benchmarking results demonstrate that TolerantECG consistently ranks as the best or second-best performer across various ECG signal conditions and class levels in the PTB-XL dataset, and achieves the highest performance on the MIT-BIH Arrhythmia Database. 

**Abstract (ZH)**: 心电图（ECG）是诊断心脏疾病的重要而有效的工具。然而，其有效性可能因噪声干扰或标准12导联记录中一个或多个导联的不可用而受损，导致诊断错误或不确定性。为应对这些挑战，我们提出了TolerantECG，这是一种针对噪声具有鲁棒性的基础模型，能够在任意子集的标准化12导联ECG导联缺失的情况下正常工作。TolerantECG的训练结合了对比学习和自我监督学习框架，共同学习ECG信号表示及其相应的基于知识检索的文本报告描述和受损害或导联缺失的信号。全面的基准测试结果表明，在PTB-XL数据集中，TolerantECG在各种ECG信号条件和类级别上始终表现为最佳或第二佳性能，在MIT-BIH心律失常数据库中达到最高性能。 

---
# Covering a Few Submodular Constraints and Applications 

**Title (ZH)**: 覆盖少数子模约束及其应用 

**Authors**: Tanvi Bajpai, Chandra Chekuri, Pooja Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09879)  

**Abstract**: We consider the problem of covering multiple submodular constraints. Given a finite ground set $N$, a cost function $c: N \rightarrow \mathbb{R}_+$, $r$ monotone submodular functions $f_1,f_2,\ldots,f_r$ over $N$ and requirements $b_1,b_2,\ldots,b_r$ the goal is to find a minimum cost subset $S \subseteq N$ such that $f_i(S) \ge b_i$ for $1 \le i \le r$. When $r=1$ this is the well-known Submodular Set Cover problem. Previous work \cite{chekuri2022covering} considered the setting when $r$ is large and developed bi-criteria approximation algorithms, and approximation algorithms for the important special case when each $f_i$ is a weighted coverage function. These are fairly general models and capture several concrete and interesting problems as special cases. The approximation ratios for these problem are at least $\Omega(\log r)$ which is unavoidable when $r$ is part of the input. In this paper, motivated by some recent applications, we consider the problem when $r$ is a \emph{fixed constant} and obtain two main results. For covering multiple submodular constraints we obtain a randomized bi-criteria approximation algorithm that for any given integer $\alpha \ge 1$ outputs a set $S$ such that $f_i(S) \ge$ $(1-1/e^\alpha -\epsilon)b_i$ for each $i \in [r]$ and $\mathbb{E}[c(S)] \le (1+\epsilon)\alpha \cdot \sf{OPT}$. Second, when the $f_i$ are weighted coverage functions from a deletion-closed set system we obtain a $(1+\epsilon)$ $(\frac{e}{e-1})$ $(1+\beta)$-approximation where $\beta$ is the approximation ratio for the underlying set cover instances via the natural LP. These results show that one can obtain nearly as good an approximation for any fixed $r$ as what one would achieve for $r=1$. We mention some applications that follow easily from these general results and anticipate more in the future. 

**Abstract (ZH)**: 多子模约束的覆盖问题 

---
# Task Priors: Enhancing Model Evaluation by Considering the Entire Space of Downstream Tasks 

**Title (ZH)**: 下游任务先验：通过考虑整个下游任务空间提升模型评估 

**Authors**: Niket Patel, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2507.09871)  

**Abstract**: The grand goal of AI research, and particularly Self Supervised Learning (SSL), is to produce systems that can successfully solve any possible task. In contrast, current evaluation methods available to AI researchers typically rely on a fixed collection of hand-picked downstream benchmarks. Hence, a large amount of effort is put into designing and searching for large collection of evaluation tasks that can serve as a proxy of our grand goal. We argue that such a rigid evaluation protocol creates a silent bottleneck in AI research. To remedy that, we define a probabilistic space of downstream tasks obtained by adopting a distribution of tasks and by defining Task Priors. Under this view, one can evaluate a model's performance over the set of all possible downstream tasks. Our framework is the first to provide answers to key questions such as (i) what is the average performance of my model over all possible downstream tasks weighted by the probability to encounter each task? or (ii) what is the variance of my model's performance across all downstream tasks under the defined Task Priors? Beyond establishing a new standard for evaluation, we believe that Task Priors will accelerate the pace of research in SSL - where downstream task evaluation is the sole qualitative signal that researchers have access to. 

**Abstract (ZH)**: AI研究的宏大目标，特别是自监督学习（SSL），是产生能够成功解决任何可能任务的系统。与之形成对比的是，当前可用的AI评价方法通常依赖于固定的手选下游基准。因此，AI研究者需要花费大量努力来设计和寻找作为宏大目标代理的评价任务集合。我们认为，这样一种僵化的评价方案在AI研究中形成了一种隐形的瓶颈。为了弥补这一不足，我们通过采用任务分布并定义任务先验来定义一个概率意义上的下游任务空间。从这一视角出发，可以评估模型在所有可能的下游任务集合上的性能。我们的框架首次提供了关键问题的答案，例如（i）在考虑每个任务出现概率加权的情况下，我的模型在所有可能的下游任务上的平均表现如何？或（ii）在定义的任务先验下，我的模型在所有下游任务上的性能差异是多少？超越建立新的评价标准，我们相信任务先验将加速自监督学习中的研究进展——在自监督学习中，下游任务评估是研究者唯一可以获得的定性信号。 

---
# A Pre-training Framework for Relational Data with Information-theoretic Principles 

**Title (ZH)**: 基于信息论原则的关系数据预训练框架 

**Authors**: Quang Truong, Zhikai Chen, Mingxuan Ju, Tong Zhao, Neil Shah, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09837)  

**Abstract**: Relational databases underpin critical infrastructure across a wide range of domains, yet the design of generalizable pre-training strategies for learning from relational databases remains an open challenge due to task heterogeneity. Specifically, there exist infinitely many possible downstream tasks, as tasks are defined based on relational schema graphs, temporal dependencies, and SQL-defined label logics. An effective pre-training framework is desired to take these factors into account in order to obtain task-aware representations. By incorporating knowledge of the underlying distribution that drives label generation, downstream tasks can benefit from relevant side-channel information. To bridge this gap, we introduce Task Vector Estimation (TVE), a novel pre-training framework that constructs predictive supervisory signals via set-based aggregation over schema traversal graphs, explicitly modeling next-window relational dynamics. We formalize our approach through an information-theoretic lens, demonstrating that task-informed representations retain more relevant signals than those obtained without task priors. Extensive experiments on the RelBench benchmark show that TVE consistently outperforms traditional pre-training baselines. Our findings advocate for pre-training objectives that encode task heterogeneity and temporal structure as design principles for predictive modeling on relational databases. 

**Abstract (ZH)**: 面向关系数据库的预训练框架：建模任务异构性和时序结构以获得任务感知表示 

---
# Generative Cognitive Diagnosis 

**Title (ZH)**: 生成认知诊断 

**Authors**: Jiatong Li, Qi Liu, Mengxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09831)  

**Abstract**: Cognitive diagnosis (CD) models latent cognitive states of human learners by analyzing their response patterns on diagnostic tests, serving as a crucial machine learning technique for educational assessment and evaluation. Traditional cognitive diagnosis models typically follow a transductive prediction paradigm that optimizes parameters to fit response scores and extract learner abilities. These approaches face significant limitations as they cannot perform instant diagnosis for new learners without computationally expensive retraining and produce diagnostic outputs with limited reliability. In this study, we introduces a novel generative diagnosis paradigm that fundamentally shifts CD from predictive to generative modeling, enabling inductive inference of cognitive states without parameter re-optimization. We propose two simple yet effective instantiations of this paradigm: Generative Item Response Theory (G-IRT) and Generative Neural Cognitive Diagnosis Model (G-NCDM), which achieve excellent performance improvements over traditional methods. The generative approach disentangles cognitive state inference from response prediction through a well-designed generation process that incorporates identifiability and monotonicity conditions. Extensive experiments on real-world datasets demonstrate the effectiveness of our methodology in addressing scalability and reliability challenges, especially $\times 100$ speedup for the diagnosis of new learners. Our framework opens new avenues for cognitive diagnosis applications in artificial intelligence, particularly for intelligent model evaluation and intelligent education systems. The code is available at this https URL. 

**Abstract (ZH)**: 认知诊断模型通过分析人类学习者在诊断测试中的反应模式来latent认知状态，作为教育评估与评价中重要的机器学习技术。传统的认知诊断模型通常遵循一种归纳预测范式，通过优化参数来拟合反应分数并提取学习者能力。这些方法存在显著的局限性，无法在不进行昂贵的重新训练的情况下对新学习者进行即时诊断，并且生成的诊断输出可靠性较低。本研究引入了一种新的生成诊断范式，从根本上将认知诊断从预测建模转变为生成建模，从而无需重新优化参数即可进行归纳推理以推断认知状态。我们提出了两种简单而有效的该范式的实例：生成项目反应理论（G-IRT）和生成神经认知诊断模型（G-NCDM），这些方法在传统方法上取得了卓越的性能改进。生成方法通过精心设计的生成过程将认知状态推断与反应预测分离，同时满足鉴别性和单调性条件。在实际数据集上的大量实验表明，我们的方法在解决可扩展性和可靠性挑战方面非常有效，特别是对于新学习者的诊断速度提高了100倍。我们的框架为人工智能中的认知诊断应用开辟了新途径，特别是在智能模型评估和智能教育系统中。代码可在以下链接获取：this https URL。 

---
# Bridging Neural Networks and Dynamic Time Warping for Adaptive Time Series Classification 

**Title (ZH)**: 将神经网络与动态时间规整结合用于自适应时间序列分类 

**Authors**: Jintao Qu, Zichong Wang, Chenhao Wu, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09826)  

**Abstract**: Neural networks have achieved remarkable success in time series classification, but their reliance on large amounts of labeled data for training limits their applicability in cold-start scenarios. Moreover, they lack interpretability, reducing transparency in decision-making. In contrast, dynamic time warping (DTW) combined with a nearest neighbor classifier is widely used for its effectiveness in limited-data settings and its inherent interpretability. However, as a non-parametric method, it is not trainable and cannot leverage large amounts of labeled data, making it less effective than neural networks in rich-resource scenarios. In this work, we aim to develop a versatile model that adapts to cold-start conditions and becomes trainable with labeled data, while maintaining interpretability. We propose a dynamic length-shortening algorithm that transforms time series into prototypes while preserving key structural patterns, thereby enabling the reformulation of the DTW recurrence relation into an equivalent recurrent neural network. Based on this, we construct a trainable model that mimics DTW's alignment behavior. As a neural network, it becomes trainable when sufficient labeled data is available, while still retaining DTW's inherent interpretability. We apply the model to several benchmark time series classification tasks and observe that it significantly outperforms previous approaches in low-resource settings and remains competitive in rich-resource settings. 

**Abstract (ZH)**: 基于动态长度缩短的可训练时间序列分类模型 

---
# Compressed Computation: Dense Circuits in a Toy Model of the Universal-AND Problem 

**Title (ZH)**: 压缩计算：通用AND问题玩具模型中的密集电路 

**Authors**: Adam Newgas  

**Link**: [PDF](https://arxiv.org/pdf/2507.09816)  

**Abstract**: Neural networks are capable of superposition -- representing more features than there are dimensions. Recent work considers the analogous concept for computation instead of storage, proposing theoretical constructions. But there has been little investigation into whether these circuits can be learned in practice. In this work, we investigate a toy model for the Universal-AND problem which computes the AND of all $m\choose 2$ pairs of $m$ sparse inputs. The hidden dimension that determines the number of non-linear activations is restricted to pressure the model to find a compute-efficient circuit, called compressed computation. We find that the training process finds a simple solution that does not correspond to theoretical constructions. It is fully dense -- every neuron contributes to every output. The solution circuit naturally scales with dimension, trading off error rates for neuron efficiency. It is similarly robust to changes in sparsity and other key parameters, and extends naturally to other boolean operations and boolean circuits. We explain the found solution in detail and compute why it is more efficient than the theoretical constructions at low sparsity. Our findings shed light on the types of circuits that models like to form and the flexibility of the superposition representation. This contributes to a broader understanding of network circuitry and interpretability. 

**Abstract (ZH)**: 神经网络具备叠加能力——表示的特征维度超过输入维度。最近的研究考虑了类似的概念，即在计算而非存储中实现叠加，提出了理论构想。但实践中这些电路是否可以被学习尚缺乏探讨。本文探讨了一个玩具模型，用于研究通用-AND问题，该模型计算m个稀疏输入的所有组合中两两输入的AND。隐藏维度限制在非线性激活的数量，以迫使模型找到一个计算高效的电路，称为压缩计算。我们发现训练过程找到了一个简单的解决方案，该解决方案不对应于理论构想。解决方案电路在维度上自然扩展，权衡误差率和神经元效率。该解决方案对稀疏性以及其他关键参数的变化具有类似的鲁棒性，并自然扩展到其他布尔操作和布尔电路。我们详细解释了找到的解决方案，并计算其在低稀疏性下比理论构想更高效的理由。我们的发现揭示了模型倾向于形成的电路类型以及叠加表示的灵活性。这有助于更广泛地理解网络电路和可解释性。 

---
# Federated Learning with Graph-Based Aggregation for Traffic Forecasting 

**Title (ZH)**: 基于图聚合的联邦学习在交通预测中的应用 

**Authors**: Audri Banik, Glaucio Haroldo Silva de Carvalho, Renata Dividino  

**Link**: [PDF](https://arxiv.org/pdf/2507.09805)  

**Abstract**: In traffic prediction, the goal is to estimate traffic speed or flow in specific regions or road segments using historical data collected by devices deployed in each area. Each region or road segment can be viewed as an individual client that measures local traffic flow, making Federated Learning (FL) a suitable approach for collaboratively training models without sharing raw data. In centralized FL, a central server collects and aggregates model updates from multiple clients to build a shared model while preserving each client's data privacy. Standard FL methods, such as Federated Averaging (FedAvg), assume that clients are independent, which can limit performance in traffic prediction tasks where spatial relationships between clients are important. Federated Graph Learning methods can capture these dependencies during server-side aggregation, but they often introduce significant computational overhead. In this paper, we propose a lightweight graph-aware FL approach that blends the simplicity of FedAvg with key ideas from graph learning. Rather than training full models, our method applies basic neighbourhood aggregation principles to guide parameter updates, weighting client models based on graph connectivity. This approach captures spatial relationships effectively while remaining computationally efficient. We evaluate our method on two benchmark traffic datasets, METR-LA and PEMS-BAY, and show that it achieves competitive performance compared to standard baselines and recent graph-based federated learning techniques. 

**Abstract (ZH)**: 基于图的联邦学习在交通预测中的轻量级方法 

---
# BitParticle: Partializing Sparse Dual-Factors to Build Quasi-Synchronizing MAC Arrays for Energy-efficient DNNs 

**Title (ZH)**: BitParticle: 部分化稀疏双因子以构建近同步MAC阵列实现能效DNN 

**Authors**: Feilong Qiaoyuan, Jihe Wang, Zhiyu Sun, Linying Wu, Yuanhua Xiao, Danghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09780)  

**Abstract**: Bit-level sparsity in quantized deep neural networks (DNNs) offers significant potential for optimizing Multiply-Accumulate (MAC) operations. However, two key challenges still limit its practical exploitation. First, conventional bit-serial approaches cannot simultaneously leverage the sparsity of both factors, leading to a complete waste of one factor' s sparsity. Methods designed to exploit dual-factor sparsity are still in the early stages of exploration, facing the challenge of partial product explosion. Second, the fluctuation of bit-level sparsity leads to variable cycle counts for MAC operations. Existing synchronous scheduling schemes that are suitable for dual-factor sparsity exhibit poor flexibility and still result in significant underutilization of MAC units. To address the first challenge, this study proposes a MAC unit that leverages dual-factor sparsity through the emerging particlization-based approach. The proposed design addresses the issue of partial product explosion through simple control logic, resulting in a more area- and energy-efficient MAC unit. In addition, by discarding less significant intermediate results, the design allows for further hardware simplification at the cost of minor accuracy loss. To address the second challenge, a quasi-synchronous scheme is introduced that adds cycle-level elasticity to the MAC array, reducing pipeline stalls and thereby improving MAC unit utilization. Evaluation results show that the exact version of the proposed MAC array architecture achieves a 29.2% improvement in area efficiency compared to the state-of-the-art bit-sparsity-driven architecture, while maintaining comparable energy efficiency. The approximate variant further improves energy efficiency by 7.5%, compared to the exact version. Index-Terms: DNN acceleration, Bit-level sparsity, MAC unit 

**Abstract (ZH)**: 比特级稀疏性在量化深度神经网络中的乘累加操作优化潜力巨大，但仍有两个关键挑战限制了其实用性。首先，传统的位串行方法不能同时利用两个因子的稀疏性，导致一个因子的稀疏性完全浪费。设计用于利用双因子稀疏性的方法仍处于早期探索阶段，面临部分乘积爆炸的挑战。其次，比特级稀疏性的波动导致乘累加操作的循环计数变化。现有的适用于双因子稀疏性的同步调度方案表现出较差的灵活性，仍导致乘累加单元的显著未充分利用。为解决第一个挑战，本研究提出了一种通过新兴的粒子化方法利用双因子稀疏性的乘累加单元。所提出的架构通过简单的控制逻辑解决了部分乘积爆炸的问题，从而实现更小面积和能耗的乘累加单元。此外，通过抛弃较不重要的中间结果，设计允许进一步简化硬件，但会轻微损失准确性。为解决第二个挑战，引入了一种准同步方案，为乘累加阵列增加了循环级弹性，减少流水线停滞，从而改善乘累加单元的利用率。评估结果显示，所提出乘累加阵列架构的精确版本相比最先进的比特稀疏性驱动架构在面积效率上提高了29.2%，同时保持了相当的能效。近似版本进一步提高了7.5%的能效，相比精确版本。索引术语：深度神经网络加速，比特级稀疏性，乘累加单元。 

---
# Toward accurate RUL and SOH estimation using reinforced graph-based PINNs enhanced with dynamic weights 

**Title (ZH)**: 基于强化图的PINNs和动态权重增强的准确剩余使用寿命和健康状态估算 

**Authors**: Mohamadreza Akbari Pour, Ali Ghasemzadeh, MohamadAli Bijarchi, Mohammad Behshad Shafii  

**Link**: [PDF](https://arxiv.org/pdf/2507.09766)  

**Abstract**: Accurate estimation of Remaining Useful Life (RUL) and State of Health (SOH) is essential for Prognostics and Health Management (PHM) across a wide range of industrial applications. We propose a novel framework -- Reinforced Graph-Based Physics-Informed Neural Networks Enhanced with Dynamic Weights (RGPD) -- that combines physics-based supervision with advanced spatio-temporal learning. Graph Convolutional Recurrent Networks (GCRNs) embed graph-convolutional filters within recurrent units to capture how node representations evolve over time. Graph Attention Convolution (GATConv) leverages a self-attention mechanism to compute learnable, edge-wise attention coefficients, dynamically weighting neighbor contributions for adaptive spatial aggregation. A Soft Actor-Critic (SAC) module is positioned between the Temporal Attention Unit (TAU) and GCRN to further improve the spatio-temporal learning. This module improves attention and prediction accuracy by dynamically scaling hidden representations to minimize noise and highlight informative features. To identify the most relevant physical constraints in each area, Q-learning agents dynamically assign weights to physics-informed loss terms, improving generalization across real-time industrial systems and reducing the need for manual tuning. In both RUL and SOH estimation tasks, the proposed method consistently outperforms state-of-the-art models, demonstrating strong robustness and predictive accuracy across varied degradation patterns across three diverse industrial benchmark datasets. 

**Abstract (ZH)**: 基于强化图的物理引导神经网络及其动态权重增强的剩余使用寿命和健康状态估计框架 

---
# EventHunter: Dynamic Clustering and Ranking of Security Events from Hacker Forum Discussions 

**Title (ZH)**: EventHunter: 来自黑客论坛讨论的 security 事件的动态聚类和排名 

**Authors**: Yasir Ech-Chammakhy, Anas Motii, Anass Rabii, Jaafar Chbili  

**Link**: [PDF](https://arxiv.org/pdf/2507.09762)  

**Abstract**: Hacker forums provide critical early warning signals for emerging cybersecurity threats, but extracting actionable intelligence from their unstructured and noisy content remains a significant challenge. This paper presents an unsupervised framework that automatically detects, clusters, and prioritizes security events discussed across hacker forum posts. Our approach leverages Transformer-based embeddings fine-tuned with contrastive learning to group related discussions into distinct security event clusters, identifying incidents like zero-day disclosures or malware releases without relying on predefined keywords. The framework incorporates a daily ranking mechanism that prioritizes identified events using quantifiable metrics reflecting timeliness, source credibility, information completeness, and relevance. Experimental evaluation on real-world hacker forum data demonstrates that our method effectively reduces noise and surfaces high-priority threats, enabling security analysts to mount proactive responses. By transforming disparate hacker forum discussions into structured, actionable intelligence, our work addresses fundamental challenges in automated threat detection and analysis. 

**Abstract (ZH)**: 黑客论坛提供早期关键警告信号以应对新兴网络安全威胁，但从其未结构化和嘈杂的内容中提取可操作的情报仍是一项重大挑战。本文提出了一种无监督框架，该框架可自动检测、聚类并优先处理黑客论坛帖子中讨论的安全事件。该方法利用对比学习微调的Transformer嵌入将相关讨论分组为不同的安全事件集群，无需依赖预定义关键词即可识别像零日披露或恶意软件发布等事件。该框架结合了每日排名机制，使用反映及时性、来源可信度、信息完整性和相关性的可量化指标优先处理识别出的事件。实验评价表明，我们的方法有效地减少了噪音并揭示了高优先级威胁，使安全分析师能够采取主动响应措施。通过将分散的黑客论坛讨论转化为结构化的可操作情报，我们的工作解决了自动威胁检测和分析中的根本性挑战。 

---
# EPT-2 Technical Report 

**Title (ZH)**: EPT-2 技术报告 

**Authors**: Roberto Molinaro, Niall Siegenheim, Niels Poulsen, Jordan Dane Daubinet, Henry Martin, Mark Frey, Kevin Thiart, Alexander Jakob Dautel, Andreas Schlueter, Alex Grigoryev, Bogdan Danciu, Nikoo Ekhtiari, Bas Steunebrink, Leonie Wagner, Marvin Vincent Gabler  

**Link**: [PDF](https://arxiv.org/pdf/2507.09703)  

**Abstract**: We present EPT-2, the latest iteration in our Earth Physics Transformer (EPT) family of foundation AI models for Earth system forecasting. EPT-2 delivers substantial improvements over its predecessor, EPT-1.5, and sets a new state of the art in predicting energy-relevant variables-including 10m and 100m wind speed, 2m temperature, and surface solar radiation-across the full 0-240h forecast horizon. It consistently outperforms leading AI weather models such as Microsoft Aurora, as well as the operational numerical forecast system IFS HRES from the European Centre for Medium-Range Weather Forecasts (ECMWF). In parallel, we introduce a perturbation-based ensemble model of EPT-2 for probabilistic forecasting, called EPT-2e. Remarkably, EPT-2e significantly surpasses the ECMWF ENS mean-long considered the gold standard for medium- to longrange forecasting-while operating at a fraction of the computational cost. EPT models, as well as third-party forecasts, are accessible via the this http URL platform. 

**Abstract (ZH)**: EPT-2：地球物理变换器家族的最新一代地球系统预报基础AI模型 

---
# Frequency-aware Surrogate Modeling With SMT Kernels For Advanced Data Forecasting 

**Title (ZH)**: 频率意识的替代建模方法结合SMT内核用于高级数据预测 

**Authors**: Nicolas Gonel, Paul Saves, Joseph Morlier  

**Link**: [PDF](https://arxiv.org/pdf/2507.09694)  

**Abstract**: This paper introduces a comprehensive open-source framework for developing correlation kernels, with a particular focus on user-defined and composition of kernels for surrogate modeling. By advancing kernel-based modeling techniques, we incorporate frequency-aware elements that effectively capture complex mechanical behaviors and timefrequency dynamics intrinsic to aircraft systems. Traditional kernel functions, often limited to exponential-based methods, are extended to include a wider range of kernels such as exponential squared sine and rational quadratic kernels, along with their respective firstand second-order derivatives. The proposed methodologies are first validated on a sinus cardinal test case and then applied to forecasting Mauna-Loa Carbon Dioxide (CO 2 ) concentrations and airline passenger traffic. All these advancements are integrated into the open-source Surrogate Modeling Toolbox (SMT 2.0), providing a versatile platform for both standard and customizable kernel configurations. Furthermore, the framework enables the combination of various kernels to leverage their unique strengths into composite models tailored to specific problems. The resulting framework offers a flexible toolset for engineers and researchers, paving the way for numerous future applications in metamodeling for complex, frequency-sensitive domains. 

**Abstract (ZH)**: 本文介绍了一个全面的开源框架，用于开发相关核函数，特别关注用户定义的核函数及其组合在代理建模中的应用。通过推进基于核的建模技术，我们引入了频率感知元素，有效地捕捉到飞机系统中固有的复杂机械行为和时频动态。传统的核函数通常局限于基于指数的方法，扩展到了包括指数平方正弦和理性二次核及其一阶和二阶导数在内的更广泛范围的核函数。所提出的方法首先在正弦卡丹测试案例上进行验证，然后应用于预测夏威夷冒纳罗亚二氧化碳浓度和航空旅客流量。所有这些进展整合到了开源代理建模工具箱（SMT 2.0）中，提供了一个既适用于标准配置又易于定制的核配置的多功能平台。此外，该框架允许将多种核函数组合起来，利用它们各自的优点，定制针对特定问题的复合模型。最终形成的框架为工程师和研究人员提供了一个灵活的工具集，为复杂、频率敏感领域的大规模元建模开辟了诸多可能应用。 

---
# Post-Training Quantization of Generative and Discriminative LSTM Text Classifiers: A Study of Calibration, Class Balance, and Robustness 

**Title (ZH)**: 生成性和判别性LSTM文本分类器的后训练量化研究：校准、类别平衡与鲁棒性探索 

**Authors**: Md Mushfiqur Rahaman, Elliot Chang, Tasmiah Haque, Srinjoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.09687)  

**Abstract**: Text classification plays a pivotal role in edge computing applications like industrial monitoring, health diagnostics, and smart assistants, where low latency and high accuracy are both key requirements. Generative classifiers, in particular, have been shown to exhibit robustness to out-of-distribution and noisy data, which is an extremely critical consideration for deployment in such real-time edge environments. However, deploying such models on edge devices faces computational and memory constraints. Post Training Quantization (PTQ) reduces model size and compute costs without retraining, making it ideal for edge deployment. In this work, we present a comprehensive comparative study of generative and discriminative Long Short Term Memory (LSTM)-based text classification models with PTQ using the Brevitas quantization library. We evaluate both types of classifier models across multiple bitwidths and assess their robustness under regular and noisy input conditions. We find that while discriminative classifiers remain robust, generative ones are more sensitive to bitwidth, calibration data used during PTQ, and input noise during quantized inference. We study the influence of class imbalance in calibration data for both types of classifiers, comparing scenarios with evenly and unevenly distributed class samples including their effect on weight adjustments and activation profiles during PTQ. Using test statistics derived from nonparametric hypothesis testing, we identify that using class imbalanced data during calibration introduces insufficient weight adaptation at lower bitwidths for generative LSTM classifiers, thereby leading to degraded performance. This study underscores the role of calibration data in PTQ and when generative classifiers succeed or fail under noise, aiding deployment in edge environments. 

**Abstract (ZH)**: 生成式文本分类模型在边缘计算环境中的后训练量化研究 

---
# OrQstrator: An AI-Powered Framework for Advanced Quantum Circuit Optimization 

**Title (ZH)**: OrQstrator: 一种基于人工智能的高级量子电路优化框架 

**Authors**: Laura Baird, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09682)  

**Abstract**: We propose a novel approach, OrQstrator, which is a modular framework for conducting quantum circuit optimization in the Noisy Intermediate-Scale Quantum (NISQ) era. Our framework is powered by Deep Reinforcement Learning (DRL). Our orchestration engine intelligently selects among three complementary circuit optimizers: A DRL-based circuit rewriter trained to reduce depth and gate count via learned rewrite sequences; a domain-specific optimizer that performs efficient local gate resynthesis and numeric optimization; a parameterized circuit instantiator that improves compilation by optimizing template circuits during gate set translation. These modules are coordinated by a central orchestration engine that learns coordination policies based on circuit structure, hardware constraints, and backend-aware performance features such as gate count, depth, and expected fidelity. The system outputs an optimized circuit for hardware-aware transpilation and execution, leveraging techniques from an existing state-of-the-art approach, called the NISQ Analyzer, to adapt to backend constraints. 

**Abstract (ZH)**: 我们提出了一种新颖的方法OrQstrator，这是一种针对Noisy Intermediate-Scale Quantum (NISQ)时代的量子电路优化的模块化框架，该框架由深度强化学习（DRL）驱动。我们的编排引擎智能地选择三种互补的电路优化器：一种基于DRL的电路重写器，通过学习到的重写序列减少电路深度和门数；一种领域特定的优化器，执行高效的局部门重建和数值优化；一种参数化的电路实例化器，在门集转换过程中通过优化模板电路提高编译效率。这些模块由一个中心编排引擎协调，该引擎根据电路结构、硬件约束以及门数、深度和预期保真度的后端感知性能特征学习协调策略。系统生成一个针对硬件感知转化和执行优化的电路，利用一种现有的先进方法——NISQ Analyzer——的技术来适应后端约束。 

---
# Conformal Prediction for Privacy-Preserving Machine Learning 

**Title (ZH)**: 隐私保护机器学习中的齐性预测方法 

**Authors**: Alexander David Balinsky, Dominik Krzeminski, Alexander Balinsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.09678)  

**Abstract**: We investigate the integration of Conformal Prediction (CP) with supervised learning on deterministically encrypted data, aiming to bridge the gap between rigorous uncertainty quantification and privacy-preserving machine learning. Using AES-encrypted variants of the MNIST dataset, we demonstrate that CP methods remain effective even when applied directly in the encrypted domain, owing to the preservation of data exchangeability under fixed-key encryption. We test traditional $p$-value-based against $e$-value-based conformal predictors. Our empirical evaluation reveals that models trained on deterministically encrypted data retain the ability to extract meaningful structure, achieving 36.88\% test accuracy -- significantly above random guessing (9.56\%) observed with per-instance encryption. Moreover, $e$-value-based CP achieves predictive set coverage of over 60\% with 4.3 loss-threshold calibration, correctly capturing the true label in 4888 out of 5000 test cases. In contrast, the $p$-value-based CP yields smaller predictive sets but with reduced coverage accuracy. These findings highlight both the promise and limitations of CP in encrypted data settings and underscore critical trade-offs between prediction set compactness and reliability. %Our work sets a foundation for principled uncertainty quantification in secure, privacy-aware learning systems. 

**Abstract (ZH)**: 我们研究规范预测（CP）与确定性加密数据上监督学习的整合，旨在弥合严格不确定性量化与隐私保护机器学习之间的差距。使用AES加密的MNIST数据变体，我们证明即使直接在加密域中应用CP方法也能保持有效性，这归因于固定密钥加密下数据可交换性的保留。我们测试了基于$p$-值的传统方法和基于$e$-值的方法。实证评估表明，训练于确定性加密数据上的模型仍能提取有意义的结构，测试准确率达到36.88%，远远高于实例加密时随机猜测的9.56%。此外，基于$e$-值的CP在4.3损失阈值校准下实现了超过60%的预测集覆盖率，正确捕捉到真实标签的有4888个测试案例中的4888个。相比之下，基于$p$-值的CP生成的预测集更小，但覆盖率准确性较低。这些发现突显了CP在加密数据环境中既有的潜力和限制，并强调了预测集紧凑性和可靠性之间的关键权衡。 

---
# SimStep: Chain-of-Abstractions for Incremental Specification and Debugging of AI-Generated Interactive Simulations 

**Title (ZH)**: SimStep: 层次抽象方法实现AI生成互动模拟的增量规范与调试 

**Authors**: Zoe Kaputa, Anika Rajaram, Vryan Almanon Feliciano, Zhuoyue Lyu, Maneesh Agrawala, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2507.09664)  

**Abstract**: Programming-by-prompting with generative AI offers a new paradigm for end-user programming, shifting the focus from syntactic fluency to semantic intent. This shift holds particular promise for non-programmers such as educators, who can describe instructional goals in natural language to generate interactive learning content. Yet in bypassing direct code authoring, many of programming's core affordances - such as traceability, stepwise refinement, and behavioral testing - are lost. We propose the Chain-of-Abstractions (CoA) framework as a way to recover these affordances while preserving the expressive flexibility of natural language. CoA decomposes the synthesis process into a sequence of cognitively meaningful, task-aligned representations that function as checkpoints for specification, inspection, and refinement. We instantiate this approach in SimStep, an authoring environment for teachers that scaffolds simulation creation through four intermediate abstractions: Concept Graph, Scenario Graph, Learning Goal Graph, and UI Interaction Graph. To address ambiguities and misalignments, SimStep includes an inverse correction process that surfaces in-filled model assumptions and enables targeted revision without requiring users to manipulate code. Evaluations with educators show that CoA enables greater authoring control and interpretability in programming-by-prompting workflows. 

**Abstract (ZH)**: 基于生成式AI的提示编程为终端用户编程提供了新范式，重点从语法流畅转向语义意图。这种转变特别适合如教育者等非编程人员，他们可以用自然语言描述教学目标以生成互动学习内容。然而，通过 bypass 直接代码编写，编程的核心功能，如可追踪性、逐步细化和行为测试，都会丢失。我们提出了抽象链（CoA）框架，以在保持自然语言表达灵活性的同时恢复这些功能。CoA 将合成过程分解为一系列认知上有意义、任务对齐的表示，作为规范、检查和细化的检查点。我们在 SimStep 中实例化了这一方法，SimStep 是一种为教师搭建仿真创建环境的作者工具，通过四个中间抽象层次的支持：概念图、情景图、学习目标图和 UI 交互图，帮助教师实现仿真内容的逐步构造。为了应对歧义和不一致，SimStep 包括一个逆向修正过程，该过程揭示了填充的模型假设，并允许有针对性的修订而无需用户操作代码。教育者的研究表明，CoA 使提示编程工作流程中的作者控制和可解释性得以提升。 

---
# Brain Stroke Detection and Classification Using CT Imaging with Transformer Models and Explainable AI 

**Title (ZH)**: 基于变压器模型和可解释人工智能的CT影像脑卒中检测与分类 

**Authors**: Shomukh Qari, Maha A. Thafar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09630)  

**Abstract**: Stroke is one of the leading causes of death globally, making early and accurate diagnosis essential for improving patient outcomes, particularly in emergency settings where timely intervention is critical. CT scans are the key imaging modality because of their speed, accessibility, and cost-effectiveness. This study proposed an artificial intelligence framework for multiclass stroke classification (ischemic, hemorrhagic, and no stroke) using CT scan images from a dataset provided by the Republic of Turkey's Ministry of Health. The proposed method adopted MaxViT, a state-of-the-art Vision Transformer, as the primary deep learning model for image-based stroke classification, with additional transformer variants (vision transformer, transformer-in-transformer, and ConvNext). To enhance model generalization and address class imbalance, we applied data augmentation techniques, including synthetic image generation. The MaxViT model trained with augmentation achieved the best performance, reaching an accuracy and F1-score of 98.00%, outperforming all other evaluated models and the baseline methods. The primary goal of this study was to distinguish between stroke types with high accuracy while addressing crucial issues of transparency and trust in artificial intelligence models. To achieve this, Explainable Artificial Intelligence (XAI) was integrated into the framework, particularly Grad-CAM++. It provides visual explanations of the model's decisions by highlighting relevant stroke regions in the CT scans and establishing an accurate, interpretable, and clinically applicable solution for early stroke detection. This research contributed to the development of a trustworthy AI-assisted diagnostic tool for stroke, facilitating its integration into clinical practice and enhancing access to timely and optimal stroke diagnosis in emergency departments, thereby saving more lives. 

**Abstract (ZH)**: 全球范围内中风是导致死亡的主要原因之一，因此早期和准确的诊断对于改善患者结局尤为重要，尤其是在需要及时干预的急诊环境中。由于CT扫描速度快、易于获取且成本效益高，CT扫描是关键的影像学检查方式。本研究提出了一种基于CT扫描图像的多类中风分类人工智能框架（缺血性中风、出血性中风和无中风），该数据集由土耳其共和国卫生部提供。所提方法采用当前最先进的眼动变换器MaxViT作为基于图像的中风分类的主要深度学习模型，并结合了多种变体（包括眼动变换器、双重眼动变换器和ConvNext）。为增强模型泛化能力和解决类别不平衡问题，我们应用了数据增强技术，包括合成图像生成。使用增强技术训练的MaxViT模型取得了最佳性能，准确率为98.00%，F1分数为98.00%，超过了所有其他评估模型和基线方法。本研究的主要目标是通过提高人工智能模型的透明度和信任度来高精度地区分不同类型的中风。为此，我们整合了可解释的人工智能(XAI)，特别是Grad-CAM++，通过在CT扫描上突出显示相关中风区域来提供模型决策的视觉解释，从而为早期中风检测提供了准确、可解释且临床适用的解决方案。该研究促进了可信赖的人工智能辅助诊断工具的发展，有助于其在临床实践中的应用，从而增强急诊科中及时和最优中风诊断的可及性，挽救更多生命。 

---
# DRAGD: A Federated Unlearning Data Reconstruction Attack Based on Gradient Differences 

**Title (ZH)**: DRAGD：基于梯度差的数据重建联邦遗忘攻击 

**Authors**: Bocheng Ju, Junchao Fan, Jiaqi Liu, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09602)  

**Abstract**: Federated learning enables collaborative machine learning while preserving data privacy. However, the rise of federated unlearning, designed to allow clients to erase their data from the global model, introduces new privacy concerns. Specifically, the gradient exchanges during the unlearning process can leak sensitive information about deleted data. In this paper, we introduce DRAGD, a novel attack that exploits gradient discrepancies before and after unlearning to reconstruct forgotten data. We also present DRAGDP, an enhanced version of DRAGD that leverages publicly available prior data to improve reconstruction accuracy, particularly for complex datasets like facial images. Extensive experiments across multiple datasets demonstrate that DRAGD and DRAGDP significantly outperform existing methods in data this http URL work highlights a critical privacy vulnerability in federated unlearning and offers a practical solution, advancing the security of federated unlearning systems in real-world applications. 

**Abstract (ZH)**: 联邦学习使协作机器学习成为可能的同时保护数据隐私。然而，联邦反学习的兴起旨在允许客户端从全局模型中删除其数据，带来了新的隐私问题。具体而言，反学习过程中的梯度交换可能会泄露被删除数据的敏感信息。在本文中，我们提出了一种新颖的攻击方法DRAGD，利用反学习前后梯度的差异来重建被遗忘的数据。我们还介绍了一种增强版的DRAGD，称为DRAGDP，通过利用公开可用的先验数据来提高复杂数据集（如面部图像）的重建准确性。在多个数据集上的 extensive 实验表明，DRAGD 和 DRAGDP 显著优于现有方法。本文强调了联邦反学习中一个关键的隐私漏洞，并提供了一种实用的解决方案，促进了实际应用中联邦反学习系统的安全性。 

---
# NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual eXploration of Finance 

**Title (ZH)**: NMIXX: 领域适应的神经嵌入在跨语言金融探索中的应用 

**Authors**: Hanwool Lee, Sara Yu, Yewon Hwang, Jonghyun Choi, Heejae Ahn, Sungbum Jung, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09601)  

**Abstract**: General-purpose sentence embedding models often struggle to capture specialized financial semantics, especially in low-resource languages like Korean, due to domain-specific jargon, temporal meaning shifts, and misaligned bilingual vocabularies. To address these gaps, we introduce NMIXX (Neural eMbeddings for Cross-lingual eXploration of Finance), a suite of cross-lingual embedding models fine-tuned with 18.8K high-confidence triplets that pair in-domain paraphrases, hard negatives derived from a semantic-shift typology, and exact Korean-English translations. Concurrently, we release KorFinSTS, a 1,921-pair Korean financial STS benchmark spanning news, disclosures, research reports, and regulations, designed to expose nuances that general benchmarks miss.
When evaluated against seven open-license baselines, NMIXX's multilingual bge-m3 variant achieves Spearman's rho gains of +0.10 on English FinSTS and +0.22 on KorFinSTS, outperforming its pre-adaptation checkpoint and surpassing other models by the largest margin, while revealing a modest trade-off in general STS performance. Our analysis further shows that models with richer Korean token coverage adapt more effectively, underscoring the importance of tokenizer design in low-resource, cross-lingual settings. By making both models and the benchmark publicly available, we provide the community with robust tools for domain-adapted, multilingual representation learning in finance. 

**Abstract (ZH)**: 面向金融领域的通用句子嵌入模型在低资源语言如韩语中往往难以捕捉到专门的金融语义，原因包括领域特定的专业术语、时间意义的转变以及双语词汇的不一致。为解决这些问题，我们提出了NMIXX（Neural eMbeddings for Cross-lingual eXploration of Finance），这是一种使用18800个高置信度三元组微调的跨语言嵌入模型套件，该三元组包括领域内的同义句对、从语义转变类型中派生的硬负例以及精确的韩英对照翻译。同时，我们也发布了KorFinSTS，这是一个包含1921对样本的韩语金融STS基准数据集，涵盖了新闻、披露信息、研究报告和监管文件，旨在揭示通用基准数据集未能捕捉到的细微差异。在与七种开放许可基准模型进行评估时，NMIXX的多语言bge-m3变体在英语金融STS和KorFinSTS上的Spearman’s ρ得分分别提高了0.10和0.22，超越了其预适应版本和其他模型，并揭示了在通用STS性能方面的轻微折衷。我们的分析还表明，拥有更丰富韩语标记覆盖的模型在跨语言设置中适应得更好，强调了低资源环境下的分词器设计的重要性。通过公开发布模型和基准数据集，我们为社区提供了用于金融领域适应的多语言表示学习的稳健工具。 

---
# THOR: Transformer Heuristics for On-Demand Retrieval 

**Title (ZH)**: THOR: Transformer启发式方法用于按需检索 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09592)  

**Abstract**: We introduce the THOR (Transformer Heuristics for On-Demand Retrieval) Module, designed and implemented by eSapiens, a secure, scalable engine that transforms natural-language questions into verified, read-only SQL analytics for enterprise databases. The Text-to-SQL module follows a decoupled orchestration/execution architecture: a Supervisor Agent routes queries, Schema Retrieval dynamically injects table and column metadata, and a SQL Generation Agent emits single-statement SELECT queries protected by a read-only guardrail. An integrated Self-Correction & Rating loop captures empty results, execution errors, or low-quality outputs and triggers up to five LLM-driven regeneration attempts. Finally, a Result Interpretation Agent produces concise, human-readable insights and hands raw rows to the Insight & Intelligence engine for visualization or forecasting.
Smoke tests across finance, sales, and operations scenarios demonstrate reliable ad-hoc querying and automated periodic reporting. By embedding schema awareness, fault-tolerant execution, and compliance guardrails, the THOR Module empowers non-technical users to access live data with zero-SQL simplicity and enterprise-grade safety. 

**Abstract (ZH)**: THOR（Transformer Heuristics for On-Demand Retrieval）模块：安全可扩展的自然语言到SQL转换引擎 

---
# Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems 

**Title (ZH)**: 识别 Offline 计量指标以预测 Online 影响：面向实际推荐系统的实用策略 

**Authors**: Timo Wilm, Philipp Normann  

**Link**: [PDF](https://arxiv.org/pdf/2507.09566)  

**Abstract**: A critical challenge in recommender systems is to establish reliable relationships between offline and online metrics that predict real-world performance. Motivated by recent advances in Pareto front approximation, we introduce a pragmatic strategy for identifying offline metrics that align with online impact. A key advantage of this approach is its ability to simultaneously serve multiple test groups, each with distinct offline performance metrics, in an online experiment controlled by a single model. The method is model-agnostic for systems with a neural network backbone, enabling broad applicability across architectures and domains. We validate the strategy through a large-scale online experiment in the field of session-based recommender systems on the OTTO e-commerce platform. The online experiment identifies significant alignments between offline metrics and real-word click-through rate, post-click conversion rate and units sold. Our strategy provides industry practitioners with a valuable tool for understanding offline-to-online metric relationships and making informed, data-driven decisions. 

**Abstract (ZH)**: 推荐系统中的一个关键挑战是建立可靠的离线和在线指标关系以预测实际性能。受Pareto前沿近似最近进展的启发，我们提出了一种实用策略来识别与在线影响相一致的离线指标。该方法的关键优势在于能够同时为由单个模型控制的在线实验中的多个具有不同离线性能指标的测试组提供服务。该方法对具有神经网络骨干的系统具有模型无关性，使其能够在多种架构和领域中广泛应用。我们通过在OTTO电子商务平台基于会话的推荐系统领域进行大规模在线实验验证了该策略。在线实验发现，离线指标与点击率、点击后转化率和销售单位之间存在显著关联。我们的策略为工业从业者提供了一个有价值的工具，以理解离线到在线指标关系并做出基于数据的决策。 

---
# An Analysis of Action-Value Temporal-Difference Methods That Learn State Values 

**Title (ZH)**: 动作值时序差分方法的研究：学习状态值分析 

**Authors**: Brett Daley, Prabhat Nagarajan, Martha White, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2507.09523)  

**Abstract**: The hallmark feature of temporal-difference (TD) learning is bootstrapping: using value predictions to generate new value predictions. The vast majority of TD methods for control learn a policy by bootstrapping from a single action-value function (e.g., Q-learning and Sarsa). Significantly less attention has been given to methods that bootstrap from two asymmetric value functions: i.e., methods that learn state values as an intermediate step in learning action values. Existing algorithms in this vein can be categorized as either QV-learning or AV-learning. Though these algorithms have been investigated to some degree in prior work, it remains unclear if and when it is advantageous to learn two value functions instead of just one -- and whether such approaches are theoretically sound in general. In this paper, we analyze these algorithmic families in terms of convergence and sample efficiency. We find that while both families are more efficient than Expected Sarsa in the prediction setting, only AV-learning methods offer any major benefit over Q-learning in the control setting. Finally, we introduce a new AV-learning algorithm called Regularized Dueling Q-learning (RDQ), which significantly outperforms Dueling DQN in the MinAtar benchmark. 

**Abstract (ZH)**: TD学习的标志性特征是bootstrapping：使用价值预测来生成新的价值预测。大多数用于控制的TD方法通过从单一的动作价值函数（如Q-learning和Sarsa）bootstrap学习策略。相比之下，较少有研究关注从两个不对称价值函数bootstrap的方法：即在学习动作价值之前学习状态价值的方法。现有的这类算法可以归类为QV-learning或AV-learning。尽管这些算法在先前的研究中有所探讨，但仍不清楚何时以及在什么情况下学习两个价值函数而非一个更有优势——并且这种途径在一般情况下是否具有理论上的合理性。在本文中，我们从收敛性和样本效率的角度分析了这些算法家族。我们发现，在预测任务中，两种家族算法都比Expected Sarsa更高效，但在控制任务中，只有AV-learning方法在一定程度上优于Q-learning。最后，我们提出了一种新的AV-learning算法——正则化对 Dueling Q-learning（RDQ），该算法在MinAtar基准测试中显著优于Dueling DQN。 

---
# QuarterMap: Efficient Post-Training Token Pruning for Visual State Space Models 

**Title (ZH)**: QuarterMap: Visual状态空间模型高效后训练 tokens 裁剪方法 

**Authors**: Tien-Yu Chi, Hung-Yueh Chiang, Diana Marculescu, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09514)  

**Abstract**: State space models (SSMs) reduce the quadratic complexity of transformers by leveraging linear recurrence. Recently, VMamba has emerged as a strong SSM-based vision backbone, yet remains bottlenecked by spatial redundancy in its four-directional scan. We propose QuarterMap, a post-training activation pruning method that removes redundant spatial activations before scanning and restores dimensions via nearest-neighbor upsampling. Our method improves throughput without retraining. On ImageNet-1K, QuarterMap achieves up to 11% speedup on VMamba with less than 0.9% accuracy drop, and yields similar gains on ADE20K segmentation. Beyond VMamba, we validate QuarterMap on MedMamba, a domain-specific model that shares the same four-directional scanning structure, where it consistently improves throughput while preserving accuracy across multiple medical imaging tasks. Compared to token merging methods like ToMe, QuarterMap is tailored for SSMs and avoids costly merge-unmerge operations. Our method offers a plug-and-play tool for deployment-time efficiency without compromising transferability. 

**Abstract (ZH)**: 基于状态空间模型的QuarterMap：一种无重构训练的激活剪枝方法 

---
# HMID-Net: An Exploration of Masked Image Modeling and Knowledge Distillation in Hyperbolic Space 

**Title (ZH)**: HMID-Net：超球面上掩码图像建模与知识蒸馏的探索 

**Authors**: Changli Wang, Fang Yin, Jiafeng Liu, Rui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09487)  

**Abstract**: Visual and semantic concepts are often structured in a hierarchical manner. For instance, textual concept `cat' entails all images of cats. A recent study, MERU, successfully adapts multimodal learning techniques from Euclidean space to hyperbolic space, effectively capturing the visual-semantic hierarchy. However, a critical question remains: how can we more efficiently train a model to capture and leverage this hierarchy? In this paper, we propose the \textit{Hyperbolic Masked Image and Distillation Network} (HMID-Net), a novel and efficient method that integrates Masked Image Modeling (MIM) and knowledge distillation techniques within hyperbolic space. To the best of our knowledge, this is the first approach to leverage MIM and knowledge distillation in hyperbolic space to train highly efficient models. In addition, we introduce a distillation loss function specifically designed to facilitate effective knowledge transfer in hyperbolic space. Our experiments demonstrate that MIM and knowledge distillation techniques in hyperbolic space can achieve the same remarkable success as in Euclidean space. Extensive evaluations show that our method excels across a wide range of downstream tasks, significantly outperforming existing models like MERU and CLIP in both image classification and retrieval. 

**Abstract (ZH)**: 基于双曲空间的掩码图像和蒸馏网络（HMID-Net）：高效捕获和利用视觉-语义层次结构的方法 

---
# Enhancing Clinical Text Classification via Fine-Tuned DRAGON Longformer Models 

**Title (ZH)**: 通过细调DRAGON Longformer模型提升临床文本分类 

**Authors**: Mingchuan Yang, Ziyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09470)  

**Abstract**: This study explores the optimization of the DRAGON Longformer base model for clinical text classification, specifically targeting the binary classification of medical case descriptions. A dataset of 500 clinical cases containing structured medical observations was used, with 400 cases for training and 100 for validation. Enhancements to the pre-trained joeranbosma/dragon-longformer-base-mixed-domain model included hyperparameter tuning, domain-specific preprocessing, and architectural adjustments. Key modifications involved increasing sequence length from 512 to 1024 tokens, adjusting learning rates from 1e-05 to 5e-06, extending training epochs from 5 to 8, and incorporating specialized medical terminology. The optimized model achieved notable performance gains: accuracy improved from 72.0% to 85.2%, precision from 68.0% to 84.1%, recall from 75.0% to 86.3%, and F1-score from 71.0% to 85.2%. Statistical analysis confirmed the significance of these improvements (p < .001). The model demonstrated enhanced capability in interpreting medical terminology, anatomical measurements, and clinical observations. These findings contribute to domain-specific language model research and offer practical implications for clinical natural language processing applications. The optimized model's strong performance across diverse medical conditions underscores its potential for broad use in healthcare settings. 

**Abstract (ZH)**: 本研究探讨了DRAGON Longformer基模型在临床文本分类中的优化，特别针对医疗病例描述的二分类任务。使用了一个包含500个临床病例的结构化医疗观察数据集，其中400个用于训练，100个用于验证。对预训练的joeranbosma/dragon-longformer-base-mixed-domain模型的增强包括超参数调整、领域特定预处理和架构调整。关键修改包括将序列长度从512增加到1024个标记，将学习率从1e-05调整到5e-06，将训练周期从5调整到8，并引入了专门的医疗术语。优化后的模型取得了显著的性能提升：准确率从72.0%提高到85.2%，精确率从68.0%提高到84.1%，召回率从75.0%提高到86.3%，F1分数从71.0%提高到85.2%。统计分析证实了这些改进的显著性（p < .001）。该模型展示了在解释医疗术语、解剖测量和临床观察方面的增强能力。本研究结果为领域特定语言模型研究做出了贡献，并为临床自然语言处理应用提供了实际意义。优化后的模型在多种医学条件下表现出色，表明其在医疗保健领域的广泛应用潜力。 

---
# Enhancing ALS Progression Tracking with Semi-Supervised ALSFRS-R Scores Estimated from Ambient Home Health Monitoring 

**Title (ZH)**: 基于环境家庭健康监测的半监督ALSFRS-R评分增强ALS进展跟踪 

**Authors**: Noah Marchal, William E. Janes, Mihail Popescu, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.09460)  

**Abstract**: Clinical monitoring of functional decline in ALS relies on periodic assessments that may miss critical changes occurring between visits. To address this gap, semi-supervised regression models were developed to estimate rates of decline in a case series cohort by targeting ALSFRS- R scale trajectories with continuous in-home sensor monitoring data. Our analysis compared three model paradigms (individual batch learning and cohort-level batch versus incremental fine-tuned transfer learning) across linear slope, cubic polynomial, and ensembled self-attention pseudo-label interpolations. Results revealed cohort homogeneity across functional domains responding to learning methods, with transfer learning improving prediction error for ALSFRS-R subscales in 28 of 32 contrasts (mean RMSE=0.20(0.04)), and individual batch learning for predicting the composite scale (mean RMSE=3.15(1.25)) in 2 of 3. Self-attention interpolation achieved the lowest prediction error for subscale-level models (mean RMSE=0.19(0.06)), capturing complex nonlinear progression patterns, outperforming linear and cubic interpolations in 20 of 32 contrasts, though linear interpolation proved more stable in all ALSFRS-R composite scale models (mean RMSE=0.23(0.10)). We identified distinct homogeneity-heterogeneity profiles across functional domains with respiratory and speech exhibiting patient-specific patterns benefiting from personalized incremental adaptation, while swallowing and dressing functions followed cohort-level trajectories suitable for transfer models. These findings suggest that matching learning and pseudo-labeling techniques to functional domain-specific homogeneity-heterogeneity profiles enhances predictive accuracy in ALS progression tracking. Integrating adaptive model selection within sensor monitoring platforms could enable timely interventions and scalable deployment in future multi-center studies. 

**Abstract (ZH)**: 基于半监督回归模型的ALS功能衰退临床监测：整合连续居家传感器数据以填补评估间隔期间的变化监测不足 

---
# Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting 

**Title (ZH)**: Fourier 基映射：时间序列预测的时间-频率学习框架 

**Authors**: Runze Yang, Longbing Cao, Xin You, Kun Fang, Jianxun Li, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09445)  

**Abstract**: The integration of Fourier transform and deep learning opens new avenues for time series forecasting. We reconsider the Fourier transform from a basis functions perspective. Specifically, the real and imaginary parts of the frequency components can be regarded as the coefficients of cosine and sine basis functions at tiered frequency levels, respectively. We find that existing Fourier-based methods face inconsistent starting cycles and inconsistent series length issues. They fail to interpret frequency components precisely and overlook temporal information. Accordingly, the novel Fourier Basis Mapping (FBM) method addresses these issues by integrating time-frequency features through Fourier basis expansion and mapping in the time-frequency space. Our approach extracts explicit frequency features while preserving temporal characteristics. FBM supports plug-and-play integration with various types of neural networks by only adjusting the first initial projection layer for better performance. First, we propose FBM-L, FBM-NL, and FBM-NP to enhance linear, MLP-based, and Transformer-based models, respectively, demonstrating the effectiveness of time-frequency features. Next, we propose a synergetic model architecture, termed FBM-S, which decomposes the seasonal, trend, and interaction effects into three separate blocks, each designed to model time-frequency features in a specialized manner. Finally, we introduce several techniques tailored for time-frequency features, including interaction masking, centralization, patching, rolling window projection, and multi-scale down-sampling. The results are validated on diverse real-world datasets for both long-term and short-term forecasting tasks with SOTA performance. 

**Abstract (ZH)**: Fourier变换与深度学习的集成为时间序列预测开辟了新途径：基于基函数的新颖Fourier基映射方法 

---
# Transformers Don't In-Context Learn Least Squares Regression 

**Title (ZH)**: Transformer 不进行上下文学习以求解最小二乘回归 

**Authors**: Joshua Hill, Benjamin Eyre, Elliot Creager  

**Link**: [PDF](https://arxiv.org/pdf/2507.09440)  

**Abstract**: In-context learning (ICL) has emerged as a powerful capability of large pretrained transformers, enabling them to solve new tasks implicit in example input-output pairs without any gradient updates. Despite its practical success, the mechanisms underlying ICL remain largely mysterious. In this work we study synthetic linear regression to probe how transformers implement learning at inference time. Previous works have demonstrated that transformers match the performance of learning rules such as Ordinary Least Squares (OLS) regression or gradient descent and have suggested ICL is facilitated in transformers through the learned implementation of one of these techniques. In this work, we demonstrate through a suite of out-of-distribution generalization experiments that transformers trained for ICL fail to generalize after shifts in the prompt distribution, a behaviour that is inconsistent with the notion of transformers implementing algorithms such as OLS. Finally, we highlight the role of the pretraining corpus in shaping ICL behaviour through a spectral analysis of the learned representations in the residual stream. Inputs from the same distribution as the training data produce representations with a unique spectral signature: inputs from this distribution tend to have the same top two singular vectors. This spectral signature is not shared by out-of-distribution inputs, and a metric characterizing the presence of this signature is highly correlated with low loss. 

**Abstract (ZH)**: 上下文学习（ICL）已成为大规模预训练转换器的一个强大能力，使它们能够在无需任何梯度更新的情况下解决由示例输入-输出对隐含的新任务。尽管其在实践中的成功令人印象深刻，但ICL背后的机制仍 largely神秘。在本文中，我们研究合成线性回归，以探查转换器在推理时如何实现学习。之前的研究表明，转换器能够与如普通最小二乘法（OLS）回归或梯度下降等学习规则匹配，且暗示ICL通过学习实现这些技术之一的方式来促进。在本文中，我们通过一系列出分布泛化实验来证明，用于ICL训练的转换器在提示分布变化后无法泛化，这一行为与转换器实施OLS等算法的观点不一致。最后，我们通过残差流中学习表示的谱分析突显了预训练语料库在塑造ICL行为中的作用。来自与训练数据相同分布的输入产生具有独特谱签名的表示：来自该分布的输入往往具有相同的前两个奇异向量。这种谱签名未由出分布输入共享，且衡量此签名存在性的度量与低损失高度相关。 

---
# Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in Multivariate Time Series 

**Title (ZH)**: 多变量时间序列中可解释因果关系发现的动态稀疏因果注意时序网络 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.09439)  

**Abstract**: Understanding causal relationships in multivariate time series (MTS) is essential for effective decision-making in fields such as finance and marketing, where complex dependencies and lagged effects challenge conventional analytical approaches. We introduce Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in MTS (DyCAST-Net), a novel architecture designed to enhance causal discovery by integrating dilated temporal convolutions and dynamic sparse attention mechanisms. DyCAST-Net effectively captures multiscale temporal dependencies through dilated convolutions while leveraging an adaptive thresholding strategy in its attention mechanism to eliminate spurious connections, ensuring both accuracy and interpretability. A statistical shuffle test validation further strengthens robustness by filtering false positives and improving causal inference reliability. Extensive evaluations on financial and marketing datasets demonstrate that DyCAST-Net consistently outperforms existing models such as TCDF, GCFormer, and CausalFormer. The model provides a more precise estimation of causal delays and significantly reduces false discoveries, particularly in noisy environments. Moreover, attention heatmaps offer interpretable insights, uncovering hidden causal patterns such as the mediated effects of advertising on consumer behavior and the influence of macroeconomic indicators on financial markets. Case studies illustrate DyCAST-Net's ability to detect latent mediators and lagged causal factors, making it particularly effective in high-dimensional, dynamic settings. The model's architecture enhanced by RMSNorm stabilization and causal masking ensures scalability and adaptability across diverse application domains 

**Abstract (ZH)**: 理解和发现多变量时间序列（MTS）中的因果关系对于金融和市场营销等领域中的有效决策至关重要，因为复杂的依赖关系和滞后效应挑战了传统的分析方法。我们提出了Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in MTS（DyCAST-Net），这是一种新颖的架构，旨在通过集成扩张时序卷积和动态稀疏注意机制来提高因果关系的发现能力。DyCAST-Net通过扩张卷积有效捕捉多尺度时间依赖关系，并通过注意机制中的自适应阈值策略消除虚假连接，确保了准确性和可解释性。通过统计混洗测试进一步增强了鲁棒性，通过过滤掉假阳性结果来提高因果推理的可靠性。在金融和市场营销数据集上的广泛评估表明，DyCAST-Net在与TCDF、GCFormer和CausalFormer等现有模型的对比中表现更优。该模型提供了更精确的因果延迟估计，特别是在噪音环境中显著减少了假发现。此外，注意力热图提供了可解释的洞察，揭示了隐藏的因果模式，如广告对消费者行为的中介效应以及宏观经济指标对金融市场的影响。案例研究展示了DyCAST-Net在检测潜在中介和滞后因果因素方面的有效性，特别是在高维动态环境中更为有效。通过RMSNorm稳定化和因果掩码增强的模型架构确保了在各种应用领域的可扩展性和适应性。 

---
# Fair CCA for Fair Representation Learning: An ADNI Study 

**Title (ZH)**: 公平CCA在公平表示学习中的研究：基于ADNI的数据分析 

**Authors**: Bojian Hou, Zhanliang Wang, Zhuoping Zhou, Boning Tong, Zexuan Wang, Jingxuan Bao, Duy Duong-Tran, Qi Long, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09382)  

**Abstract**: Canonical correlation analysis (CCA) is a technique for finding correlations between different data modalities and learning low-dimensional representations. As fairness becomes crucial in machine learning, fair CCA has gained attention. However, previous approaches often overlook the impact on downstream classification tasks, limiting applicability. We propose a novel fair CCA method for fair representation learning, ensuring the projected features are independent of sensitive attributes, thus enhancing fairness without compromising accuracy. We validate our method on synthetic data and real-world data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), demonstrating its ability to maintain high correlation analysis performance while improving fairness in classification tasks. Our work enables fair machine learning in neuroimaging studies where unbiased analysis is essential. 

**Abstract (ZH)**: 公平的主成分分析（CCA）方法：确保投影特征独立于敏感属性，从而在不牺牲准确性的前提下提高公平性 

---
# Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis 

**Title (ZH)**: 基于马尔可夫集成的上下文感知正则化核酸注意力分析 

**Authors**: Mohammadsaleh Refahi, Mahdi Abavisani, Bahrad A. Sokhansanj, James R. Brown, Gail Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09378)  

**Abstract**: Transformers have revolutionized nucleotide sequence analysis, yet capturing long-range dependencies remains challenging. Recent studies show that autoregressive transformers often exhibit Markovian behavior by relying on fixed-length context windows for next-token prediction. However, standard self-attention mechanisms are computationally inefficient for long sequences due to their quadratic complexity and do not explicitly enforce global transition consistency.
We introduce CARMANIA (Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis), a self-supervised pretraining framework that augments next-token (NT) prediction with a transition-matrix (TM) loss. The TM loss aligns predicted token transitions with empirically derived n-gram statistics from each input sequence, encouraging the model to capture higher-order dependencies beyond local context. This integration enables CARMANIA to learn organism-specific sequence structures that reflect both evolutionary constraints and functional organization.
We evaluate CARMANIA across diverse genomic tasks, including regulatory element prediction, functional gene classification, taxonomic inference, antimicrobial resistance detection, and biosynthetic gene cluster classification. CARMANIA outperforms the previous best long-context model by at least 7 percent, matches state-of-the-art on shorter sequences (exceeding prior results on 20 out of 40 tasks while running approximately 2.5 times faster), and shows particularly strong improvements on enhancer and housekeeping gene classification tasks, including up to a 34 percent absolute gain in Matthews correlation coefficient (MCC) for enhancer prediction. The TM loss boosts accuracy in 33 of 40 tasks, especially where local motifs or regulatory patterns drive prediction. 

**Abstract (ZH)**: Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis 

---
# Impute With Confidence: A Framework for Uncertainty Aware Multivariate Time Series Imputation 

**Title (ZH)**: 自信插值：一种考虑不确定性的多变量时间序列插值框架 

**Authors**: Addison Weatherhead, Anna Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.09353)  

**Abstract**: Time series data with missing values is common across many domains. Healthcare presents special challenges due to prolonged periods of sensor disconnection. In such cases, having a confidence measure for imputed values is critical. Most existing methods either overlook model uncertainty or lack mechanisms to estimate it. To address this gap, we introduce a general framework that quantifies and leverages uncertainty for selective imputation. By focusing on values the model is most confident in, highly unreliable imputations are avoided. Our experiments on multiple EHR datasets, covering diverse types of missingness, demonstrate that selectively imputing less-uncertain values not only reduces imputation errors but also improves downstream tasks. Specifically, we show performance gains in a 24-hour mortality prediction task, underscoring the practical benefit of incorporating uncertainty into time series imputation. 

**Abstract (ZH)**: 具有缺失值的时间序列数据在许多领域中都很常见。医疗保健领域因长时间传感器断开连接而面临特殊挑战。在这种情况下，对插补值具有置信度衡量至关重要。现有大多数方法要么忽略了模型不确定性，要么缺乏估计不确定性的机制。为解决这一问题，我们引入了一种一般框架，用于量化和利用不确定性进行选择性插补。通过专注于模型最自信的值，可以避免高度不可靠的插补。我们在多个EHR数据集上的实验涵盖了不同类型的缺失性，表明仅填充较低不确定性的值不仅可以减少插补错误，还可以改善下游任务。具体而言，我们在24小时死亡率预测任务中展示了性能提升，强调了将不确定性纳入时间序列插补中的实践益处。 

---
# A Framework for Predictive Directional Trading Based on Volatility and Causal Inference 

**Title (ZH)**: 基于波动率和因果推断的预测性方向交易框架 

**Authors**: Ivan Letteri  

**Link**: [PDF](https://arxiv.org/pdf/2507.09347)  

**Abstract**: Purpose: This study introduces a novel framework for identifying and exploiting predictive lead-lag relationships in financial markets. We propose an integrated approach that combines advanced statistical methodologies with machine learning models to enhance the identification and exploitation of predictive relationships between equities. Methods: We employed a Gaussian Mixture Model (GMM) to cluster nine prominent stocks based on their mid-range historical volatility profiles over a three-year period. From the resulting clusters, we constructed a multi-stage causal inference pipeline, incorporating the Granger Causality Test (GCT), a customised Peter-Clark Momentary Conditional Independence (PCMCI) test, and Effective Transfer Entropy (ETE) to identify robust, predictive linkages. Subsequently, Dynamic Time Warping (DTW) and a K-Nearest Neighbours (KNN) classifier were utilised to determine the optimal time lag for trade execution. The resulting strategy was rigorously backtested. Results: The proposed volatility-based trading strategy, tested from 8 June 2023 to 12 August 2023, demonstrated substantial efficacy. The portfolio yielded a total return of 15.38%, significantly outperforming the 10.39% return of a comparative Buy-and-Hold strategy. Key performance metrics, including a Sharpe Ratio up to 2.17 and a win rate up to 100% for certain pairs, confirmed the strategy's viability. Conclusion: This research contributes a systematic and robust methodology for identifying profitable trading opportunities derived from volatility-based causal relationships. The findings have significant implications for both academic research in financial modelling and the practical application of algorithmic trading, offering a structured approach to developing resilient, data-driven strategies. 

**Abstract (ZH)**: 目的：本文介绍了一种新型框架，用于识别和利用金融市场的预测领先-滞后关系。我们提出了一种综合方法，结合了高级统计方法和机器学习模型，以增强 Equity 之间的预测关系识别和利用。方法：我们使用高斯混合模型（GMM）根据九只表现突出股票在过去三年中期波动率概况进行聚类。从聚类结果中，我们构建了一个多阶段因果推理管道，结合了格兰杰因果检验（GCT）、自定义佩特-克拉克瞬时条件独立性（PCMCI）检验和有效转移熵（ETE），以识别稳健的预测联系。随后，我们使用动态时间规整（DTW）和K-最近邻（KNN）分类器来确定交易执行的最佳时间滞后。随后，该策略进行了严格的回测。结果：该基于波动率的交易策略从2023年6月8日到2023年8月12日测试，显示出显著的效果。该组合的总收益率为15.38%，显著优于比较的买入并持有策略的10.39%收益率。关键性能指标包括高达2.17的夏普比率和某些配对高达100%的胜率，验证了该策略的有效性。结论：本文贡献了一套系统且稳健的方法，用于从基于波动率的因果关系中识别可盈利的交易机会。研究结果对金融建模的学术研究和算法交易的实际应用具有重要的意义，提供了一种开发稳健的数据驱动策略的结构化方法。 

---
# Enhancing Interpretability in Software Change Management with Chain-of-Thought Reasoning 

**Title (ZH)**: 使用链式思考推理增强软件变更管理的可解释性 

**Authors**: Yongqian Sun, Weihua Kuang, Chao Shen, Xidao Wen, Tinghua Zheng, Heng Liu, Shenglin Zhang, Bo Wu, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2507.09315)  

**Abstract**: In modern online services, frequent software changes introduce significant risks. To tackle this challenge, we propose SCELM (Software Change Evaluation and Lifecycle Management), an end-to-end automated framework for software change management. SCELM aims to manage software changes efficiently and precisely, significantly reducing service failures and economic losses. 

**Abstract (ZH)**: 在现代在线服务中，频繁的软件变更引入了重大的风险。为了应对这一挑战，我们提出了SCELM（软件变更评估与生命周期管理）——一个端到端的自动化软件变更管理体系，旨在高效精准地管理软件变更，显著减少服务失败和经济损失。 

---
# Cross Knowledge Distillation between Artificial and Spiking Neural Networks 

**Title (ZH)**: 人工神经网络与脉冲神经网络之间的跨知识蒸馏 

**Authors**: Shuhan Ye, Yuanbin Qian, Chong Wang, Sunqi Lin, Jiazhen Xu, Jiangbo Qian, Yuqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09269)  

**Abstract**: Recently, Spiking Neural Networks (SNNs) have demonstrated rich potential in computer vision domain due to their high biological plausibility, event-driven characteristic and energy-saving efficiency. Still, limited annotated event-based datasets and immature SNN architectures result in their performance inferior to that of Artificial Neural Networks (ANNs). To enhance the performance of SNNs on their optimal data format, DVS data, we explore using RGB data and well-performing ANNs to implement knowledge distillation. In this case, solving cross-modality and cross-architecture challenges is necessary. In this paper, we propose cross knowledge distillation (CKD), which not only leverages semantic similarity and sliding replacement to mitigate the cross-modality challenge, but also uses an indirect phased knowledge distillation to mitigate the cross-architecture challenge. We validated our method on main-stream neuromorphic datasets, including N-Caltech101 and CEP-DVS. The experimental results show that our method outperforms current State-of-the-Art methods. The code will be available at this https URL 

**Abstract (ZH)**: 近年来，由于其高度的生物合理性、事件驱动特性和节能效率，脉冲神经网络（SNNs）在计算机视觉领域展现出丰富的潜力。然而，受限于有限的标注事件数据集和不成熟的SNN架构，其性能仍低于人工神经网络（ANNs）。为了提升SNNs在最佳数据格式DVS数据上的性能，我们探索使用RGB数据和表现良好的ANNs来实现知识蒸馏。在这种情况下，跨模态和跨架构的挑战需要得到解决。本文提出了一种跨模态知识蒸馏（CKD）方法，该方法不仅利用语义相似性和滑动替换来缓解跨模态挑战，还采用间接分阶段知识蒸馏来缓解跨架构挑战。我们在主流的神经形态数据集N-Caltech101和CEP-DVS上验证了该方法。实验结果表明，该方法优于当前的最先进的方法。代码将在此链接处提供。 

---
# Controllable Patching for Compute-Adaptive Surrogate Modeling of Partial Differential Equations 

**Title (ZH)**: 可控 patching 用于部分微分方程计算自适应代理建模 

**Authors**: Payel Mukhopadhyay, Michael McCabe, Ruben Ohana, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.09264)  

**Abstract**: Patch-based transformer surrogates have become increasingly effective for modeling spatiotemporal dynamics, but the fixed patch size is a major limitation for budget-conscience deployment in production. We introduce two lightweight, architecture-agnostic modules-the Convolutional Kernel Modulator (CKM) and Convolutional Stride Modulator (CSM)-that enable dynamic patch size control at inference in patch based models, without retraining or accuracy loss. Combined with a cyclic patch-size rollout, our method mitigates patch artifacts and improves long-term stability for video-like prediction tasks. Applied to a range of challenging 2D and 3D PDE benchmarks, our approach improves rollout fidelity and runtime efficiency. To our knowledge, this is the first framework to enable inference-time patch-size tunability in patch-based PDE surrogates. Its plug-and-play design makes it broadly applicable across architectures-establishing a general foundation for compute-adaptive modeling in PDE surrogate tasks. 

**Abstract (ZH)**: 基于卷积核调制器和卷积步长调制器的动态 patches 大小控制方法在 PDE 代理模型中的应用 

---
# AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition 

**Title (ZH)**: AGCD-网：注意力引导上下文去偏网络的情绪识别 

**Authors**: Varsha Devi, Amine Bohi, Pardeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09248)  

**Abstract**: Context-aware emotion recognition (CAER) enhances affective computing in real-world scenarios, but traditional methods often suffer from context bias-spurious correlation between background context and emotion labels (e.g. associating ``garden'' with ``happy''). In this paper, we propose \textbf{AGCD-Net}, an Attention Guided Context Debiasing model that introduces \textit{Hybrid ConvNeXt}, a novel convolutional encoder that extends the ConvNeXt backbone by integrating Spatial Transformer Network and Squeeze-and-Excitation layers for enhanced feature recalibration. At the core of AGCD-Net is the Attention Guided - Causal Intervention Module (AG-CIM), which applies causal theory, perturbs context features, isolates spurious correlations, and performs an attention-driven correction guided by face features to mitigate context bias. Experimental results on the CAER-S dataset demonstrate the effectiveness of AGCD-Net, achieving state-of-the-art performance and highlighting the importance of causal debiasing for robust emotion recognition in complex settings. 

**Abstract (ZH)**: 基于注意力引导的上下文去偏模型（AGCD-Net）在情境感知情感识别中的应用 

---
# XiChen: An observation-scalable fully AI-driven global weather forecasting system with 4D variational knowledge 

**Title (ZH)**: XiChen：一种基于4D变分知识的可观察性可伸缩全AI驱动全球天气预报系统 

**Authors**: Wuxin Wang, Weicheng Ni, Lilan Huang, Tao Hao, Ben Fei, Shuo Ma, Taikang Yuan, Yanlai Zhao, Kefeng Deng, Xiaoyong Li, Boheng Duan, Lei Bai, Kaijun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.09202)  

**Abstract**: Recent advancements in Artificial Intelligence (AI) demonstrate significant potential to revolutionize weather forecasting. However, most AI-driven models rely on Numerical Weather Prediction (NWP) systems for initial condition preparation, which often consumes hours on supercomputers. Here we introduce XiChen, the first observation-scalable fully AI-driven global weather forecasting system, whose entire pipeline, from Data Assimilation (DA) to medium-range forecasting, can be accomplished within only 17 seconds. XiChen is built upon a foundation model that is pre-trained for weather forecasting. Meanwhile, this model is subsequently fine-tuned to serve as both observation operators and DA models, thereby scalably assimilating conventional and raw satellite observations. Furthermore, the integration of four-dimensional variational knowledge ensures that XiChen's DA and medium-range forecasting accuracy rivals that of operational NWP systems, amazingly achieving a skillful forecasting lead time exceeding 8.25 days. These findings demonstrate that XiChen holds strong potential toward fully AI-driven weather forecasting independent of NWP systems. 

**Abstract (ZH)**: Recent advancements in Artificial Intelligence (AI)显示了在气象预报领域革命性的潜力。然而，大多数基于AI的模型依赖数值天气预报（NWP）系统进行初始条件准备，这通常需要在超级计算机上耗时数小时。我们介绍了Xichen，这是首个可扩展观测的完全基于AI的全球气象预报系统，其从数据同化（DA）到中期预报的整个管道仅需17秒即可完成。Xichen基于一个为气象预报预训练的基准模型，进而微调以作为观测算子和数据同化模型，从而可扩展地同化常规和原始卫星观测。此外，四维变分知识的整合确保了Xichen的数据同化和中期预报准确度与 operational NWP 系统相当，令人惊讶地实现了超前预报时效超过8.25天。这些发现表明，Xichen有可能完全独立于NWP系统实现基于AI的气象预报。 

---
# Towards Interpretable Drug-Drug Interaction Prediction: A Graph-Based Approach with Molecular and Network-Level Explanations 

**Title (ZH)**: 基于图的分子和网络层面解释的可解释药物-药物相互作用预测 

**Authors**: Mengjie Chen, Ming Zhang, Cunquan Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09173)  

**Abstract**: Drug-drug interactions (DDIs) represent a critical challenge in pharmacology, often leading to adverse drug reactions with significant implications for patient safety and healthcare outcomes. While graph-based methods have achieved strong predictive performance, most approaches treat drug pairs independently, overlooking the complex, context-dependent interactions unique to drug pairs. Additionally, these models struggle to integrate biological interaction networks and molecular-level structures to provide meaningful mechanistic insights. In this study, we propose MolecBioNet, a novel graph-based framework that integrates molecular and biomedical knowledge for robust and interpretable DDI prediction. By modeling drug pairs as unified entities, MolecBioNet captures both macro-level biological interactions and micro-level molecular influences, offering a comprehensive perspective on DDIs. The framework extracts local subgraphs from biomedical knowledge graphs and constructs hierarchical interaction graphs from molecular representations, leveraging classical graph neural network methods to learn multi-scale representations of drug pairs. To enhance accuracy and interpretability, MolecBioNet introduces two domain-specific pooling strategies: context-aware subgraph pooling (CASPool), which emphasizes biologically relevant entities, and attention-guided influence pooling (AGIPool), which prioritizes influential molecular substructures. The framework further employs mutual information minimization regularization to enhance information diversity during embedding fusion. Experimental results demonstrate that MolecBioNet outperforms state-of-the-art methods in DDI prediction, while ablation studies and embedding visualizations further validate the advantages of unified drug pair modeling and multi-scale knowledge integration. 

**Abstract (ZH)**: 基于分子和生物医学知识的药物-药物相互作用预测框架：MolecBioNet 

---
# Advanced Health Misinformation Detection Through Hybrid CNN-LSTM Models Informed by the Elaboration Likelihood Model (ELM) 

**Title (ZH)**: 基于 elaboration likelihood model (ELM) 的混合 CNN-LSTM 模型在先进健康误导信息检测中的应用 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09149)  

**Abstract**: Health misinformation during the COVID-19 pandemic has significantly challenged public health efforts globally. This study applies the Elaboration Likelihood Model (ELM) to enhance misinformation detection on social media using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model. The model aims to enhance the detection accuracy and reliability of misinformation classification by integrating ELM-based features such as text readability, sentiment polarity, and heuristic cues (e.g., punctuation frequency). The enhanced model achieved an accuracy of 97.37%, precision of 96.88%, recall of 98.50%, F1-score of 97.41%, and ROC-AUC of 99.50%. A combined model incorporating feature engineering further improved performance, achieving a precision of 98.88%, recall of 99.80%, F1-score of 99.41%, and ROC-AUC of 99.80%. These findings highlight the value of ELM features in improving detection performance, offering valuable contextual information. This study demonstrates the practical application of psychological theories in developing advanced machine learning algorithms to address health misinformation effectively. 

**Abstract (ZH)**: COVID-19疫情期间的健康 misinformation 对全球公共健康努力构成了重大挑战。本研究运用扩展可能性路径模型（ELM）结合卷积神经网络（CNN）和长短期记忆（LSTM）模型，以提高社交媒体上的 misinformation 检测准确性。该模型通过整合基于ELM的特征（如文本可读性、情感极性和启发性线索，例如标点符号频率）来提高 misinformation 分类的准确性和可靠性。增强模型的准确率为97.37%，精确率为96.88%，召回率为98.50%，F1得分97.41%，ROC-AUC为99.50%。结合特征工程的综合模型进一步提高了性能，精确率为98.88%，召回率为99.80%，F1得分99.41%，ROC-AUC为99.80%。这些发现突显了ELM特征在提高检测性能方面的价值，提供了有价值的语言背景信息。本研究展示了通过运用心理学理论开发高级机器学习算法以有效应对健康 misinformation 的实际应用。 

---
# POIFormer: A Transformer-Based Framework for Accurate and Scalable Point-of-Interest Attribution 

**Title (ZH)**: POIFormer: 基于Transformer的准确可扩展的兴趣点归属框架 

**Authors**: Nripsuta Ani Saxena, Shang-Ling Hsu, Mehul Shetty, Omar Alkhadra, Cyrus Shahabi, Abigail L. Horn  

**Link**: [PDF](https://arxiv.org/pdf/2507.09137)  

**Abstract**: Accurately attributing user visits to specific Points of Interest (POIs) is a foundational task for mobility analytics, personalized services, marketing and urban planning. However, POI attribution remains challenging due to GPS inaccuracies, typically ranging from 2 to 20 meters in real-world settings, and the high spatial density of POIs in urban environments, where multiple venues can coexist within a small radius (e.g., over 50 POIs within a 100-meter radius in dense city centers). Relying on proximity is therefore often insufficient for determining which POI was actually visited. We introduce \textsf{POIFormer}, a novel Transformer-based framework for accurate and efficient POI attribution. Unlike prior approaches that rely on limited spatiotemporal, contextual, or behavioral features, \textsf{POIFormer} jointly models a rich set of signals, including spatial proximity, visit timing and duration, contextual features from POI semantics, and behavioral features from user mobility and aggregated crowd behavior patterns--using the Transformer's self-attention mechanism to jointly model complex interactions across these dimensions. By leveraging the Transformer to model a user's past and future visits (with the current visit masked) and incorporating crowd-level behavioral patterns through pre-computed KDEs, \textsf{POIFormer} enables accurate, efficient attribution in large, noisy mobility datasets. Its architecture supports generalization across diverse data sources and geographic contexts while avoiding reliance on hard-to-access or unavailable data layers, making it practical for real-world deployment. Extensive experiments on real-world mobility datasets demonstrate significant improvements over existing baselines, particularly in challenging real-world settings characterized by spatial noise and dense POI clustering. 

**Abstract (ZH)**: 准确归因用户访问到特定兴趣点（POI）是移动分析、个性化服务、营销和城市规划的基础任务。然而，由于GPS精度问题，在实际环境中通常从2到20米不等，以及城市环境中POI的高度空间密度（例如，在密集城市的100米半径内可能有超过50个POI），依赖于接近性往往不足以确定实际访问的POI。我们介绍了\textsf{POIFormer}，这是一种基于Transformer的新型框架，用于准确和高效地进行POI归因。与依赖有限的时空、上下文或行为特征的先前方法不同，\textsf{POIFormer} 联合建模了丰富的信号，包括空间接近性、访问时间和持续时间、从POI语义中获得的上下文特征以及从用户移动性和聚合人群行为模式中获得的行为特征——利用Transformer的自注意力机制联合建模这些维度上的复杂交互。通过利用Transformer建模用户过去和未来的访问（当前访问被遮蔽）并结合预先计算的核密度估计（KDE）来融入人群层次的行为模式，\textsf{POIFormer} 在大型嘈杂的移动数据集中实现了准确且高效的归因。其架构支持跨不同数据源和地理背景的一般化，同时避免了对难以获取或不可用数据层的依赖，使其适用于实际部署。在实际移动数据集上的 extensively 实验表明，\textsf{POIFormer} 在具有空间噪声和密集POI聚类特征的挑战性实际环境中显著优于现有baseline。 

---
# Heterogeneous Graph Prompt Learning via Adaptive Weight Pruning 

**Title (ZH)**: 异构图提示学习通过自适应权重剪枝 

**Authors**: Chu-Yuan Wei, Shun-Yao Liu, Sheng-Da Zhuo, Chang-Dong Wang, Shu-Qiang Huang, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.09132)  

**Abstract**: Graph Neural Networks (GNNs) have achieved remarkable success in various graph-based tasks (e.g., node classification or link prediction). Despite their triumphs, GNNs still face challenges such as long training and inference times, difficulty in capturing complex relationships, and insufficient feature extraction. To tackle these issues, graph pre-training and graph prompt methods have garnered increasing attention for their ability to leverage large-scale datasets for initial learning and task-specific adaptation, offering potential improvements in GNN performance. However, previous research has overlooked the potential of graph prompts in optimizing models, as well as the impact of both positive and negative graph prompts on model stability and efficiency. To bridge this gap, we propose a novel framework combining graph prompts with weight pruning, called GPAWP, which aims to enhance the performance and efficiency of graph prompts by using fewer of them. We evaluate the importance of graph prompts using an importance assessment function to determine positive and negative weights at different granularities. Through hierarchically structured pruning, we eliminate negative prompt labels, resulting in more parameter-efficient and competitively performing prompts. Extensive experiments on three benchmark datasets demonstrate the superiority of GPAWP, leading to a significant reduction in parameters in node classification tasks. 

**Abstract (ZH)**: 基于图提示与权重剪枝的图神经网络优化框架（GPAWP） 

---
# SPICE: An Automated SWE-Bench Labeling Pipeline for Issue Clarity, Test Coverage, and Effort Estimation 

**Title (ZH)**: SPICE: 一种自动化的SWE-Bench标签流水线，用于问题清晰度、测试覆盖率和努力估计。 

**Authors**: Aaditya Bhatia, Gustavo A. Oliva, Gopi Krishnan Rajbahadur, Haoxiang Zhang, Yihao Chen, Zhilong Chen, Arthur Leung, Dayi Lin, Boyuan Chen, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09108)  

**Abstract**: High-quality labeled datasets are crucial for training and evaluating foundation models in software engineering, but creating them is often prohibitively expensive and labor-intensive. We introduce SPICE, a scalable, automated pipeline for labeling SWE-bench-style datasets with annotations for issue clarity, test coverage, and effort estimation. SPICE combines context-aware code navigation, rationale-driven prompting, and multi-pass consensus to produce labels that closely approximate expert annotations. SPICE's design was informed by our own experience and frustration in labeling more than 800 instances from SWE-Gym. SPICE achieves strong agreement with human-labeled SWE-bench Verified data while reducing the cost of labeling 1,000 instances from around $100,000 (manual annotation) to just $5.10. These results demonstrate SPICE's potential to enable cost-effective, large-scale dataset creation for SE-focused FMs. To support the community, we release both SPICE tool and SPICE Bench, a new dataset of 6,802 SPICE-labeled instances curated from 291 open-source projects in SWE-Gym (over 13x larger than SWE-bench Verified). 

**Abstract (ZH)**: 高质量标注数据集是软件工程中训练和评估基础模型的关键，但创建它们往往代价高昂且劳动密集。我们介绍了SPICE，一种可扩展的自动化流水线，用于标注类似于SWE-bench的 datasets，并提供关于问题清晰度、测试覆盖率和努力估计的标注。SPICE结合上下文感知的代码导航、基于理据的提示和多轮共识，生成与专家标注接近的标签。SPICE的设计基于我们自己在为超过800个实例进行SWE-Gym标注时的经验和挫败感。SPICE在与人工标注的SWE-bench Verified数据达成强烈一致的同时，将1,000个实例的标注成本从约100,000美元（手动标注）降低到仅5.10美元。这些结果表明SPICE有可能促进面向SE的基础模型的大规模、成本效益型数据集创建。为支持社区，我们发布了SPICE工具和包含6,802个SPICE标注实例的新数据集SPICE Bench，这些实例来自291个开源项目（比SWE-bench Verified大13倍以上）。 

---
# Queue up for takeoff: a transferable deep learning framework for flight delay prediction 

**Title (ZH)**: 排队起飞：一种可移植的深度学习框架用于航班延误预测 

**Authors**: Nnamdi Daniel Aghanya, Ta Duong Vu, Amaëlle Diop, Charlotte Deville, Nour Imane Kerroumi, Irene Moulitsas, Jun Li, Desmond Bisandu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09084)  

**Abstract**: Flight delays are a significant challenge in the aviation industry, causing major financial and operational disruptions. To improve passenger experience and reduce revenue loss, flight delay prediction models must be both precise and generalizable across different networks. This paper introduces a novel approach that combines Queue-Theory with a simple attention model, referred to as the Queue-Theory SimAM (QT-SimAM). To validate our model, we used data from the US Bureau of Transportation Statistics, where our proposed QT-SimAM (Bidirectional) model outperformed existing methods with an accuracy of 0.927 and an F1 score of 0.932. To assess transferability, we tested the model on the EUROCONTROL dataset. The results demonstrated strong performance, achieving an accuracy of 0.826 and an F1 score of 0.791. Ultimately, this paper outlines an effective, end-to-end methodology for predicting flight delays. The proposed model's ability to forecast delays with high accuracy across different networks can help reduce passenger anxiety and improve operational decision-making 

**Abstract (ZH)**: 航空延误是航空业的一项重大挑战，会导致严重的财务和运营中断。为了提高乘客体验并减少收入损失，飞行延误预测模型必须既精确又能在不同的网络中泛化。本文提出了一种结合排队理论和简单注意机制的新方法，称为排队理论相似注意力模型（QT-SimAM）。为了验证我们的模型，我们使用了美国交通统计局的数据，我们的提出的QT-SimAM（双向）模型在准确率0.927和F1分数0.932的情况下超过了现有方法。为了评估其可迁移性，我们将模型应用于EUROCONTROL数据集。结果表明，该模型表现强劲，准确率为0.826，F1分数为0.791。最终，本文概述了一种有效的端到端飞行延误预测方法。所提出的模型能够在不同的网络中以高精度预测延误，有助于减轻乘客的焦虑并改善运营决策。 

---
# SetupBench: Assessing Software Engineering Agents' Ability to Bootstrap Development Environments 

**Title (ZH)**: SetupBench: 评估软件工程代理构建开发环境的能力 

**Authors**: Avi Arora, Jinu Jang, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2507.09063)  

**Abstract**: Modern Large Language Model (LLM) agents promise end to end assistance with real-world software tasks, yet existing benchmarks evaluate LLM agents almost exclusively in pre-baked environments where every dependency is pre-installed. To fill this gap, we introduce SetupBench, a 93 instance benchmark that isolates the environment-bootstrap skill: starting from a bare Linux sandbox, an agent must install packages, resolve dependency conflicts, initialize databases, and configure background services. Our tasks span seven language ecosystems, five database engines, and multi-service orchestration scenarios, each accompanies by a natural language problem statement and a deterministic success command. Through evaluation of OpenHands, a state-of-the-art coding agent, we find low success rates across task categories, with particular challenges in repository setup (38.9-57.4%) and local database configuration (20.0-53.3%). Our analysis reveals systematic failure modes including incomplete development tooling installation, hallucinated task constraints, and non-persistent environment modifications that break agent-human collaboration workflows. We identify substantial inefficiencies in agent exploration strategies, with 38-89% of actions being unnecessary compared to optimal human behavior. These findings highlight gaps in current agents' practical environment-bootstrap capabilities. By targeting this critical yet under-evaluated capability, SetupBench provides a rigorous yard-stick for the next generation of software developer agents aiming to solve end to end real-wold tasks. 

**Abstract (ZH)**: SetupBench：隔离环境构建技能的93实例基准 

---
# Analysing Health Misinformation with Advanced Centrality Metrics in Online Social Networks 

**Title (ZH)**: 基于先进中心性指标分析在线社交网络中的健康 misinformation 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09055)  

**Abstract**: The rapid spread of health misinformation on online social networks (OSNs) during global crises such as the COVID-19 pandemic poses challenges to public health, social stability, and institutional trust. Centrality metrics have long been pivotal in understanding the dynamics of information flow, particularly in the context of health misinformation. However, the increasing complexity and dynamism of online networks, especially during crises, highlight the limitations of these traditional approaches. This study introduces and compares three novel centrality metrics: dynamic influence centrality (DIC), health misinformation vulnerability centrality (MVC), and propagation centrality (PC). These metrics incorporate temporal dynamics, susceptibility, and multilayered network interactions. Using the FibVID dataset, we compared traditional and novel metrics to identify influential nodes, propagation pathways, and misinformation influencers. Traditional metrics identified 29 influential nodes, while the new metrics uncovered 24 unique nodes, resulting in 42 combined nodes, an increase of 44.83%. Baseline interventions reduced health misinformation by 50%, while incorporating the new metrics increased this to 62.5%, an improvement of 25%. To evaluate the broader applicability of the proposed metrics, we validated our framework on a second dataset, Monant Medical Misinformation, which covers a diverse range of health misinformation discussions beyond COVID-19. The results confirmed that the advanced metrics generalised successfully, identifying distinct influential actors not captured by traditional methods. In general, the findings suggest that a combination of traditional and novel centrality measures offers a more robust and generalisable framework for understanding and mitigating the spread of health misinformation in different online network contexts. 

**Abstract (ZH)**: 在线社交网络（OSNs）上健康 misinformation 的快速传播：全球危机如COVID-19 pandemic期间对公共健康、社会稳定和机构信任的挑战 

---
# BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis 

**Title (ZH)**: 脑部病灶套件：一个模块化脑部病灶图像分析的灵活易用框架 

**Authors**: Florian Kofler, Marcel Rosier, Mehdi Astaraki, Hendrik Möller, Ilhem Isra Mekki, Josef A. Buchner, Anton Schmick, Arianna Pfiffer, Eva Oswald, Lucas Zimmer, Ezequiel de la Rosa, Sarthak Pati, Julian Canisius, Arianna Piffer, Ujjwal Baid, Mahyar Valizadeh, Akis Linardos, Jan C. Peeken, Surprosanna Shit, Felix Steinbauer, Daniel Rueckert, Rolf Heckemann, Spyridon Bakas, Jan Kirschke, Constantin von See, Ivan Ezhov, Marie Piraud, Benedikt Wiestler, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2507.09036)  

**Abstract**: BrainLesion Suite is a versatile toolkit for building modular brain lesion image analysis pipelines in Python. Following Pythonic principles, BrainLesion Suite is designed to provide a 'brainless' development experience, minimizing cognitive effort and streamlining the creation of complex workflows for clinical and scientific practice. At its core is an adaptable preprocessing module that performs co-registration, atlas registration, and optional skull-stripping and defacing on arbitrary multi-modal input images. BrainLesion Suite leverages algorithms from the BraTS challenge to synthesize missing modalities, inpaint lesions, and generate pathology-specific tumor segmentations. BrainLesion Suite also enables quantifying segmentation model performance, with tools such as panoptica to compute lesion-wise metrics. Although BrainLesion Suite was originally developed for image analysis pipelines of brain lesions such as glioma, metastasis, and multiple sclerosis, it can be adapted for other biomedical image analysis applications. The individual BrainLesion Suite packages and tutorials are accessible on GitHub. 

**Abstract (ZH)**: BrainLesion Suite是用于构建Python中可模块化脑病变图像分析流水线的多功能工具包。遵循Pythonic原则，BrainLesion Suite旨在提供“无脑化”的开发体验，减少认知努力并简化复杂工作流的创建，以满足临床和科学实践的需求。其核心是一个可适应的预处理模块，该模块执行配准、解剖图注册，并可选地进行去头骨和去标识化处理任意多模态输入图像。BrainLesion Suite利用BraTS挑战中的算法来合成缺失的模态、填充病变并生成病理特异性肿瘤分割。BrainLesion Suite还允许通过工具如panoptica等量化分割模型性能，以计算病变级别的指标。尽管BrainLesion Suite最初是为如胶质瘤、转移和多发性硬化等脑病变的图像分析流水线开发的，但它可以适应其他生物医学图像分析应用。BrainLesion Suite的各个包和教程可在GitHub上访问。 

---
# Accelerating Drug Discovery Through Agentic AI: A Multi-Agent Approach to Laboratory Automation in the DMTA Cycle 

**Title (ZH)**: 通过自主智能加速药物发现：DMTA循环中多智能体实验室自动化方法 

**Authors**: Yao Fehlis, Charles Crain, Aidan Jensen, Michael Watson, James Juhasz, Paul Mandel, Betty Liu, Shawn Mahon, Daren Wilson, Nick Lynch-Jonely, Ben Leedom, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.09023)  

**Abstract**: The pharmaceutical industry faces unprecedented challenges in drug discovery, with traditional approaches struggling to meet modern therapeutic development demands. This paper introduces a novel AI framework, Tippy, that transforms laboratory automation through specialized AI agents operating within the Design-Make-Test-Analyze (DMTA) cycle. Our multi-agent system employs five specialized agents - Supervisor, Molecule, Lab, Analysis, and Report, with Safety Guardrail oversight - each designed to excel in specific phases of the drug discovery pipeline. Tippy represents the first production-ready implementation of specialized AI agents for automating the DMTA cycle, providing a concrete example of how AI can transform laboratory workflows. By leveraging autonomous AI agents that reason, plan, and collaborate, we demonstrate how Tippy accelerates DMTA cycles while maintaining scientific rigor essential for pharmaceutical research. The system shows significant improvements in workflow efficiency, decision-making speed, and cross-disciplinary coordination, offering a new paradigm for AI-assisted drug discovery. 

**Abstract (ZH)**: 制药行业在药物发现方面面临着前所未有的挑战，传统方法难以满足现代治疗开发的需求。本文介绍了一种新型AI框架Tippy，通过专门的AI代理在设计-制备-测试-分析（DMTA）循环中运行，从而实现实验室自动化。我们的多代理系统包括五个专门的代理——督导员、分子、实验室、分析和报告，并由安全护栏监管，每个代理都旨在在药物发现管道的特定阶段表现出色。Tippy代表了专门为自动化DMTA循环而开发的第一个生产就绪型专门AI代理的实现，提供了AI如何转型实验室工作流程的典型案例。通过利用能够推理、规划和协作的自主AI代理，我们展示了Tippy如何在保持对于制药研究至关重要的科学严谨性的同时加速DMTA循环。该系统在工作流程效率、决策速度和跨学科协调方面显示出显著改进，提供了一种新的AI辅助药物发现范式。 

---
# Multimodal Cardiovascular Risk Profiling Using Self-Supervised Learning of Polysomnography 

**Title (ZH)**: 基于自我监督学习的多模态心血管风险评估 

**Authors**: Zhengxiao He, Huayu Li, Geng Yuan, William D.S. Killgore, Stuart F. Quan, Chen X. Chen, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09009)  

**Abstract**: Methods: We developed a self-supervised deep learning model that extracts meaningful patterns from multi-modal signals (Electroencephalography (EEG), Electrocardiography (ECG), and respiratory signals). The model was trained on data from 4,398 participants. Projection scores were derived by contrasting embeddings from individuals with and without CVD outcomes. External validation was conducted in an independent cohort with 1,093 participants. The source code is available on this https URL. Results: The projection scores revealed distinct and clinically meaningful patterns across modalities. ECG-derived features were predictive of both prevalent and incident cardiac conditions, particularly CVD mortality. EEG-derived features were predictive of incident hypertension and CVD mortality. Respiratory signals added complementary predictive value. Combining these projection scores with the Framingham Risk Score consistently improved predictive performance, achieving area under the curve values ranging from 0.607 to 0.965 across different outcomes. Findings were robustly replicated and validated in the external testing cohort. Conclusion: Our findings demonstrate that the proposed framework can generate individualized CVD risk scores directly from PSG data. The resulting projection scores have the potential to be integrated into clinical practice, enhancing risk assessment and supporting personalized care. 

**Abstract (ZH)**: 方法：我们开发了一种自监督深度学习模型，从多模态信号（脑电图（EEG）、心电图（ECG）和呼吸信号）中提取有意义的模式。该模型在4,398名参与者的数据上进行训练。通过对比患有和未患有心血管疾病（CVD）结局个体的嵌入表示，得到了投影分数。外部验证在独立的1,093名参与者的队列中进行。源代码可在以下网址获取：this https URL。结果：投影分数揭示了各模态中的独特且具有临床意义的模式。心电图衍生的特征预测了心力衰竭和心血管疾病（CVD）死亡等既往和新发心脏状况，尤其是心血管疾病死亡。脑电图衍生的特征预测了新发高血压和心血管疾病死亡。呼吸信号提供了补充的预测价值。将这些投影分数与弗雷明汉风险评分结合使用，在不同结局上的一致预测性能达到了从0.607到0.965的曲线下面积（AUC）值。该发现在外测试队列中表现出高度的稳健性和验证性。结论：我们的发现表明，所提出的方法可以直接从多导睡眠图（PSG）数据中生成个性化的CVD风险评分。产生的投影分数有潜力融入临床实践，提高风险评估并支持个性化治疗。 

---
# Simulation as Supervision: Mechanistic Pretraining for Scientific Discovery 

**Title (ZH)**: 将模拟作为监督：机理预训练促进科学研究 

**Authors**: Carson Dudley, Reiden Magdaleno, Christopher Harding, Marisa Eisenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.08977)  

**Abstract**: Scientific modeling faces a core limitation: mechanistic models offer interpretability but collapse under real-world complexity, while machine learning models are flexible but require large labeled datasets, cannot infer unobservable quantities, and operate as black boxes. We introduce Simulation-Grounded Neural Networks (SGNNs), a general framework that uses mechanistic simulations as training data for neural networks. SGNNs are pretrained on synthetic corpora spanning diverse model structures, parameter regimes, stochasticity, and observational artifacts. We evaluated SGNNs across scientific disciplines and modeling tasks, and found that SGNNs achieved state-of-the-art results across settings: for prediction tasks, they nearly tripled COVID-19 forecasting skill versus CDC baselines, reduced chemical yield prediction error by one third, and maintained accuracy in ecological forecasting where task specific models failed. For inference tasks, SGNNs also accurately classified the source of information spread in simulated social networks and enabled supervised learning for unobservable targets, such as estimating COVID-19 transmissibility more accurately than traditional methods even in early outbreaks. Finally, SGNNs enable back-to-simulation attribution, a new form of mechanistic interpretability. Given real world input, SGNNs retrieve simulations based on what the model has learned to see as most similar, revealing which underlying dynamics the model believes are active. This provides process-level insight -- what the model thinks is happening -- not just which features mattered. SGNNs unify scientific theory with deep learning flexibility and unlock a new modeling paradigm -- transforming simulations from rigid, post hoc tools into flexible sources of supervision, enabling robust, interpretable inference even when ground truth is missing. 

**Abstract (ZH)**: 基于仿真训练的神经网络：一种结合机理模型和机器学习优势的通用框架 

---
# Simulating Three-dimensional Turbulence with Physics-informed Neural Networks 

**Title (ZH)**: 用物理知情神经网络模拟三维湍流 

**Authors**: Sifan Wang, Shyam Sankaran, Panos Stinis, Paris Perdikaris  

**Link**: [PDF](https://arxiv.org/pdf/2507.08972)  

**Abstract**: Turbulent fluid flows are among the most computationally demanding problems in science, requiring enormous computational resources that become prohibitive at high flow speeds. Physics-informed neural networks (PINNs) represent a radically different approach that trains neural networks directly from physical equations rather than data, offering the potential for continuous, mesh-free solutions. Here we show that appropriately designed PINNs can successfully simulate fully turbulent flows in both two and three dimensions, directly learning solutions to the fundamental fluid equations without traditional computational grids or training data. Our approach combines several algorithmic innovations including adaptive network architectures, causal training, and advanced optimization methods to overcome the inherent challenges of learning chaotic dynamics. Through rigorous validation on challenging turbulence problems, we demonstrate that PINNs accurately reproduce key flow statistics including energy spectra, kinetic energy, enstrophy, and Reynolds stresses. Our results demonstrate that neural equation solvers can handle complex chaotic systems, opening new possibilities for continuous turbulence modeling that transcends traditional computational limitations. 

**Abstract (ZH)**: 湍流流动是科学中计算需求最大的问题之一，在高速流中所需的计算资源变得难以承受。物理导向神经网络（PINNs）代表了一种截然不同的方法，直接从物理方程而非数据训练神经网络，提供了连续、无网格式求解的潜力。我们展示了一种适当设计的PINNs能够成功模拟二维和三维完全湍流流动，直接学习基本流体方程的解，无需传统计算网格或训练数据。我们的方法结合了多种算法创新，包括自适应网络架构、因果训练和高级优化方法，以克服学习混沌动力学的固有挑战。通过在挑战性的湍流问题上进行严格的验证，我们证明PINNs能够准确再现关键流场统计，包括能量谱、动能、涡度和雷诺应力。我们的结果表明，神经方程求解器能够处理复杂的混沌系统，开启了超越传统计算限制的连续湍流建模的新可能性。 

---
# ToxBench: A Binding Affinity Prediction Benchmark with AB-FEP-Calculated Labels for Human Estrogen Receptor Alpha 

**Title (ZH)**: ToxBench: 一种基于AB-FEP计算标签的人类雌激素受体α结合亲和力预测基准 

**Authors**: Meng Liu, Karl Leswing, Simon K. S. Chu, Farhad Ramezanghorbani, Griffin Young, Gabriel Marques, Prerna Das, Anjali Panikar, Esther Jamir, Mohammed Sulaiman Shamsudeen, K. Shawn Watts, Ananya Sen, Hari Priya Devannagari, Edward B. Miller, Muyun Lihan, Howook Hwang, Janet Paulsen, Xin Yu, Kyle Gion, Timur Rvachov, Emine Kucukbenli, Saee Gopal Paliwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.08966)  

**Abstract**: Protein-ligand binding affinity prediction is essential for drug discovery and toxicity assessment. While machine learning (ML) promises fast and accurate predictions, its progress is constrained by the availability of reliable data. In contrast, physics-based methods such as absolute binding free energy perturbation (AB-FEP) deliver high accuracy but are computationally prohibitive for high-throughput applications. To bridge this gap, we introduce ToxBench, the first large-scale AB-FEP dataset designed for ML development and focused on a single pharmaceutically critical target, Human Estrogen Receptor Alpha (ER$\alpha$). ToxBench contains 8,770 ER$\alpha$-ligand complex structures with binding free energies computed via AB-FEP with a subset validated against experimental affinities at 1.75 kcal/mol RMSE, along with non-overlapping ligand splits to assess model generalizability. Using ToxBench, we further benchmark state-of-the-art ML methods, and notably, our proposed DualBind model, which employs a dual-loss framework to effectively learn the binding energy function. The benchmark results demonstrate the superior performance of DualBind and the potential of ML to approximate AB-FEP at a fraction of the computational cost. 

**Abstract (ZH)**: 蛋白-配体结合亲和力预测对于药物发现和毒性评估至关重要。虽然机器学习（ML）承诺能够提供快速且准确的预测，但其进展受限于可靠数据的可用性。相比之下，基于物理的方法如绝对结合自由能突变（AB-FEP）能够提供高精度，但在高通量应用中计算成本过高。为弥合这一差距，我们提出了ToxBench，这是一个用于机器学习开发的大型AB-FEP数据集，专注于单一的药理学关键目标——人类雌激素受体α（ERα）。ToxBench包含8,770个ERα-配体复合物结构，并通过AB-FEP计算了结合自由能，其中部分结构与1.75 kcal/mol的均方根误差实验亲和力进行了验证，同时包含非重叠的配体分割以评估模型的泛化能力。利用ToxBench，我们进一步测试了最先进的机器学习方法，并提出了一种新的DualBind模型，该模型采用双损失框架有效学习结合能量函数。基准测试结果表明，DualBind模型的性能优于现有方法，并展示了机器学习在计算成本极低的情况下逼近AB-FEP的潜力。 

---
# GraphRunner: A Multi-Stage Framework for Efficient and Accurate Graph-Based Retrieval 

**Title (ZH)**: GraphRunner：一种高效准确的图基检索多阶段框架 

**Authors**: Savini Kashmira, Jayanaka L. Dantanarayana, Krisztián Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2507.08945)  

**Abstract**: Conventional Retrieval Augmented Generation (RAG) approaches are common in text-based applications. However, they struggle with structured, interconnected datasets like knowledge graphs, where understanding underlying relationships is crucial for accurate retrieval. A common direction in graph-based retrieval employs iterative, rule-based traversal guided by Large Language Models (LLMs). Such existing iterative methods typically combine reasoning with single hop traversal at each step, making them vulnerable to LLM reasoning errors and hallucinations that ultimately hinder the retrieval of relevant information.
To address these limitations, we propose GraphRunner, a novel graph-based retrieval framework that operates in three distinct stages: planning, verification, and execution. This introduces high-level traversal actions that enable multi-hop exploration in a single step. It also generates a holistic traversal plan, which is verified against the graph structure and pre-defined traversal actions, reducing reasoning errors and detecting hallucinations before execution. GraphRunner significantly reduces LLM reasoning errors and detects hallucinations through validation. Our evaluation using the GRBench dataset shows that GraphRunner consistently outperforms existing approaches, achieving 10-50% performance improvements over the strongest baseline while reducing inference cost by 3.0-12.9x and response generation time by 2.5-7.1x, making it significantly more robust and efficient for graph-based retrieval tasks. 

**Abstract (ZH)**: 基于图的检索增强生成（RAG）方法在文本应用中很常见。然而，它们在处理知识图等结构化且相互连接的数据集时遇到困难，因为在这些数据集中理解潜在关系对于准确检索至关重要。基于图的检索中的一种常见方向是通过大型语言模型（LLMs）引导的迭代、规则导向的遍历。现有的迭代方法通常在每一步结合一次跳跃遍历和推理，这使它们容易受到LLM推理错误和幻觉的影响，从而阻碍了相关信息的检索。

为了解决这些问题，我们提出了GraphRunner，这是一种新颖的基于图的检索框架，分为规划、验证和执行三个阶段。这引入了高层次的遍历动作，允许在单步中进行多跳跃探索。它还生成了一个整体的遍历计划，该计划可以根据图结构和预定义的遍历动作进行验证，从而减少推理错误并检测出执行前的幻觉。GraphRunner通过验证显著减少了LLM的推理错误并检测出了幻觉。使用GRBench数据集的评估表明，GraphRunner在性能上始终优于现有方法，对比最强基准方法在性能上提高了10-50%，同时将推理成本降低了3.0-12.9倍，响应生成时间减少了2.5-7.1倍，使它在基于图的检索任务中变得更加稳健和高效。 

---
# AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model 

**Title (ZH)**: AMix-1：面向测试时可扩展的蛋白质基础模型的一种途径 

**Authors**: Changze Lv, Jiang Zhou, Siyu Long, Lihao Wang, Jiangtao Feng, Dongyu Xue, Yu Pei, Hao Wang, Zherui Zhang, Yuchen Cai, Zhiqiang Gao, Ziyuan Ma, Jiakai Hu, Chaochen Gao, Jingjing Gong, Yuxuan Song, Shuyi Zhang, Xiaoqing Zheng, Deyi Xiong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma, Bowen Zhou, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08920)  

**Abstract**: We introduce AMix-1, a powerful protein foundation model built on Bayesian Flow Networks and empowered by a systematic training methodology, encompassing pretraining scaling laws, emergent capability analysis, in-context learning mechanism, and test-time scaling algorithm. To guarantee robust scalability, we establish a predictive scaling law and reveal the progressive emergence of structural understanding via loss perspective, culminating in a strong 1.7-billion model. Building on this foundation, we devise a multiple sequence alignment (MSA)-based in-context learning strategy to unify protein design into a general framework, where AMix-1 recognizes deep evolutionary signals among MSAs and consistently generates structurally and functionally coherent proteins. This framework enables the successful design of a dramatically improved AmeR variant with an up to $50\times$ activity increase over its wild type. Pushing the boundaries of protein engineering, we further empower AMix-1 with an evolutionary test-time scaling algorithm for in silico directed evolution that delivers substantial, scalable performance gains as verification budgets are intensified, laying the groundwork for next-generation lab-in-the-loop protein design. 

**Abstract (ZH)**: AMix-1：基于贝叶斯流网络的强大蛋白质基础模型及其系统训练方法和多重序列比对指导的上下文学习策略 

---
# Fair-FLIP: Fair Deepfake Detection with Fairness-Oriented Final Layer Input Prioritising 

**Title (ZH)**: Fair-FLIP：面向公平性的最终层输入优先级 deepfake 检测方法 

**Authors**: Tomasz Szandala, Fatima Ezzeddine, Natalia Rusin, Silvia Giordano, Omran Ayoub  

**Link**: [PDF](https://arxiv.org/pdf/2507.08912)  

**Abstract**: Artificial Intelligence-generated content has become increasingly popular, yet its malicious use, particularly the deepfakes, poses a serious threat to public trust and discourse. While deepfake detection methods achieve high predictive performance, they often exhibit biases across demographic attributes such as ethnicity and gender. In this work, we tackle the challenge of fair deepfake detection, aiming to mitigate these biases while maintaining robust detection capabilities. To this end, we propose a novel post-processing approach, referred to as Fairness-Oriented Final Layer Input Prioritising (Fair-FLIP), that reweights a trained model's final-layer inputs to reduce subgroup disparities, prioritising those with low variability while demoting highly variable ones. Experimental results comparing Fair-FLIP to both the baseline (without fairness-oriented de-biasing) and state-of-the-art approaches show that Fair-FLIP can enhance fairness metrics by up to 30% while maintaining baseline accuracy, with only a negligible reduction of 0.25%.
Code is available on Github: this https URL 

**Abstract (ZH)**: 人工智能生成的内容越来越受欢迎，但其恶意使用，尤其是深度伪造，对公众信任和讨论构成了严重威胁。尽管深度伪造检测方法具有高度的预测性能，但在种族和性别等人口统计属性方面常常表现出偏差。在本文中，我们致力于公平深度伪造检测的挑战，旨在减轻这些偏差同时保持检测能力的稳健性。为此，我们提出了一种新颖的后处理方法，称为面向公平性的最终层输入优先级调整（Fair-FLIP），该方法重新加权训练模型的最终层输入以减少子群体间的差异，优先处理低变异性部分，同时降低高变异性部分的权重。将Fair-FLIP与基准方法（未进行公平导向的去偏差处理）和最新方法进行比较的实验结果显示，Fair-FLIP可以在保持基准准确性的基础上，通过提高30%的公平性指标，同时减少不到0.25%的准确性。代码可在Github上获取：this https URL。 

---
# Last Layer Hamiltonian Monte Carlo 

**Title (ZH)**: 最后一层哈密尔顿蒙特卡洛 

**Authors**: Koen Vellenga, H. Joe Steinhauer, Göran Falkman, Jonas Andersson, Anders Sjögren  

**Link**: [PDF](https://arxiv.org/pdf/2507.08905)  

**Abstract**: We explore the use of Hamiltonian Monte Carlo (HMC) sampling as a probabilistic last layer approach for deep neural networks (DNNs). While HMC is widely regarded as a gold standard for uncertainty estimation, the computational demands limit its application to large-scale datasets and large DNN architectures. Although the predictions from the sampled DNN parameters can be parallelized, the computational cost still scales linearly with the number of samples (similar to an ensemble). Last layer HMC (LL--HMC) reduces the required computations by restricting the HMC sampling to the final layer of a DNN, making it applicable to more data-intensive scenarios with limited computational resources. In this paper, we compare LL-HMC against five last layer probabilistic deep learning (LL-PDL) methods across three real-world video datasets for driver action and intention. We evaluate the in-distribution classification performance, calibration, and out-of-distribution (OOD) detection. Due to the stochastic nature of the probabilistic evaluations, we performed five grid searches for different random seeds to avoid being reliant on a single initialization for the hyperparameter configurations. The results show that LL--HMC achieves competitive in-distribution classification and OOD detection performance. Additional sampled last layer parameters do not improve the classification performance, but can improve the OOD detection. Multiple chains or starting positions did not yield consistent improvements. 

**Abstract (ZH)**: 我们探索使用哈密顿蒙特卡洛（HMC）采样作为深度神经网络（DNN）的概率性最后一层方法。虽然HMC通常被视为不确定性估计的金标准，但由于计算需求限制了其在大规模数据集和大型DNN架构上的应用，尽管从采样的DNN参数进行预测可以并行化处理，但计算成本仍然会按样本数量线性增加（类似于集成方法）。最后一层HMC（LL--HMC）通过将HMC采样限制在DNN的最后一层，减少了所需的计算量，使其适用于计算资源有限的数据密集型场景。在本文中，我们将LL-HMC与五个其他最后一层概率深度学习（LL-PDL）方法在三个真实世界的视频数据集（涉及驾驶员行为和意图）上进行比较。我们评估了分布内分类性能、校准性能以及分布外（OOD）检测性能。由于概率评估的随机性，我们进行了五次网格搜索，以不同的随机种子避免依赖单一的超参数配置初始化。结果显示，LL--HMC在分布内分类和分布外检测性能上表现出竞争力。额外的采样最后一层参数不会提高分类性能，但可以改善分布外检测。多个链或起始位置没有一致地提升性能。 

---
# Generation of structure-guided pMHC-I libraries using Diffusion Models 

**Title (ZH)**: 基于结构引导的pMHC-I库生成方法 

**Authors**: Sergio Mares, Ariel Espinoza Weinberger, Nilah M. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08902)  

**Abstract**: Personalized vaccines and T-cell immunotherapies depend critically on identifying peptide-MHC class I (pMHC-I) interactions capable of eliciting potent immune responses. However, current benchmarks and models inherit biases present in mass-spectrometry and binding-assay datasets, limiting discovery of novel peptide ligands. To address this issue, we introduce a structure-guided benchmark of pMHC-I peptides designed using diffusion models conditioned on crystal structure interaction distances. Spanning twenty high-priority HLA alleles, this benchmark is independent of previously characterized peptides yet reproduces canonical anchor residue preferences, indicating structural generalization without experimental dataset bias. Using this resource, we demonstrate that state-of-the-art sequence-based predictors perform poorly at recognizing the binding potential of these structurally stable designs, indicating allele-specific limitations invisible in conventional evaluations. Our geometry-aware design pipeline yields peptides with high predicted structural integrity and higher residue diversity than existing datasets, representing a key resource for unbiased model training and evaluation. Our code, and data are available at: this https URL. 

**Abstract (ZH)**: 个性化疫苗和T细胞免疫治疗依赖于识别能够引发强大免疫反应的肽-MHCI（pMHC-I）相互作用，但当前的基准和模型继承了质谱和结合 assay 数据集中的偏差，限制了新型肽配体的发现。为了解决这一问题，我们引入了一种结构导向的基准，该基准使用条件于晶体结构相互作用距离的扩散模型设计pMHC-I多肽，覆盖了二十个高优先级的HLA等位基因，该基准独立于已知的肽序列，但却再现了经典的锚残基偏好，表明结构上的概括而无实验数据集偏差。使用该资源，我们证明了最先进的基于序列的预测器在识别这些结构上稳定的构想的结合潜力方面表现不佳，表明了在常规评估中看不见的等位基因特异性局限性。我们的几何感知设计流水线产生具有高预测结构完整性和更高残基多样性的肽，代表了无偏模型训练和评估的关键资源。代码和数据可在以下链接获取：this https URL。 

---
# A Multi-Level Strategy for Deepfake Content Moderation under EU Regulation 

**Title (ZH)**: 欧盟法规下的多层次策略对抗虚假内容审核 

**Authors**: Max-Paul Förster, Luca Deck, Raimund Weidlich, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2507.08879)  

**Abstract**: The growing availability and use of deepfake technologies increases risks for democratic societies, e.g., for political communication on online platforms. The EU has responded with transparency obligations for providers and deployers of Artificial Intelligence (AI) systems and online platforms. This includes marking deepfakes during generation and labeling deepfakes when they are shared. However, the lack of industry and enforcement standards poses an ongoing challenge. Through a multivocal literature review, we summarize methods for marking, detecting, and labeling deepfakes and assess their effectiveness under EU regulation. Our results indicate that individual methods fail to meet regulatory and practical requirements. Therefore, we propose a multi-level strategy combining the strengths of existing methods. To account for the masses of content on online platforms, our multi-level strategy provides scalability and practicality via a simple scoring mechanism. At the same time, it is agnostic to types of deepfake technology and allows for context-specific risk weighting. 

**Abstract (ZH)**: 不断增强的深fake技术availability与应用增加了对民主社会的风险，例如在线平台上政治沟通的风险。欧盟对此采取了透明度义务措施，要求人工智能系统和在线平台提供商和部署者遵守，并包括在生成时标记深fake，以及在共享时对其进行标注。然而，缺乏行业和执法标准构成了持续挑战。通过多声腔文献综述，我们总结了标记、检测和标注深fake的方法，并评估它们在欧盟法规下的有效性。我们的结果显示，个别方法无法满足监管和实际要求。因此，我们提议一种多级策略，结合现有方法的优势。为了应对在线平台上大量内容，我们的多级策略通过简单的评分机制提供可扩展性和实用性。同时，它对深fake技术类型保持中立，并允许进行具体的上下文风险加权。 

---
# Next-Generation Travel Demand Modeling with a Generative Framework for Household Activity Coordination 

**Title (ZH)**: 基于生成框架的家庭活动协调的下一代旅行需求建模 

**Authors**: Xishun Liao, Haoxuan Ma, Yifan Liu, Yuxiang Wei, Brian Yueshuai He, Chris Stanford, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08871)  

**Abstract**: Travel demand models are critical tools for planning, policy, and mobility system design. Traditional activity-based models (ABMs), although grounded in behavioral theories, often rely on simplified rules and assumptions, and are costly to develop and difficult to adapt across different regions. This paper presents a learning-based travel demand modeling framework that synthesizes household-coordinated daily activity patterns based on a household's socio-demographic profiles. The whole framework integrates population synthesis, coordinated activity generation, location assignment, and large-scale microscopic traffic simulation into a unified system. It is fully generative, data-driven, scalable, and transferable to other regions. A full-pipeline implementation is conducted in Los Angeles with a 10 million population. Comprehensive validation shows that the model closely replicates real-world mobility patterns and matches the performance of legacy ABMs with significantly reduced modeling cost and greater scalability. With respect to the SCAG ABM benchmark, the origin-destination matrix achieves a cosine similarity of 0.97, and the daily vehicle miles traveled (VMT) in the network yields a 0.006 Jensen-Shannon Divergence (JSD) and a 9.8% mean absolute percentage error (MAPE). When compared to real-world observations from Caltrans PeMS, the evaluation on corridor-level traffic speed and volume reaches a 0.001 JSD and a 6.11% MAPE. 

**Abstract (ZH)**: 基于学习的旅行需求建模框架：基于家庭协调的日活动模式合成 

---
# Privacy-Utility-Fairness: A Balanced Approach to Vehicular-Traffic Management System 

**Title (ZH)**: 隐私-效益-公平： vehicular-traffic 管理系统中的一种平衡方法 

**Authors**: Poushali Sengupta, Sabita Maharjan, frank Eliassen, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08864)  

**Abstract**: Location-based vehicular traffic management faces significant challenges in protecting sensitive geographical data while maintaining utility for traffic management and fairness across regions. Existing state-of-the-art solutions often fail to meet the required level of protection against linkage attacks and demographic biases, leading to privacy leakage and inequity in data analysis. In this paper, we propose a novel algorithm designed to address the challenges regarding the balance of privacy, utility, and fairness in location-based vehicular traffic management systems. In this context, utility means providing reliable and meaningful traffic information, while fairness ensures that all regions and individuals are treated equitably in data use and decision-making. Employing differential privacy techniques, we enhance data security by integrating query-based data access with iterative shuffling and calibrated noise injection, ensuring that sensitive geographical data remains protected. We ensure adherence to epsilon-differential privacy standards by implementing the Laplace mechanism. We implemented our algorithm on vehicular location-based data from Norway, demonstrating its ability to maintain data utility for traffic management and urban planning while ensuring fair representation of all geographical areas without being overrepresented or underrepresented. Additionally, we have created a heatmap of Norway based on our model, illustrating the privatized and fair representation of the traffic conditions across various cities. Our algorithm provides privacy in vehicular traffic 

**Abstract (ZH)**: 基于位置的车辆交通管理在保护敏感地理数据隐私、维持交通管理实用性和区域公平性之间面临着重大挑战。现有先进解决方案往往无法满足对链接攻击和人口统计偏差的保护要求，导致隐私泄露和数据分析中的不公平。本文提出了一种新型算法，旨在解决基于位置的车辆交通管理系统中隐私、实用性和公平性之间的平衡问题。在该上下文中，实用性是指提供可靠和有意义的交通信息，而公平性则确保在数据使用和决策过程中所有地区和个体得到公平对待。我们通过运用差分隐私技术，结合查询驱动的数据访问、迭代洗牌和校准噪声注入，增强数据安全性，确保敏感地理数据的安全。通过实现拉普拉斯机制，我们确保算法符合ε-差分隐私标准。我们在挪威的车辆基于位置的数据上实现了该算法，展示了其在维持交通管理和城市规划的数据实用性的同时，能够公平地代表所有地理区域而不会出现过度代表或不足代表的情况。此外，我们还根据我们的模型创建了挪威的热力图，展示了各种城市中交通状况的私有化和公平表示。该算法提供了车辆交通隐私。 

---
# Foundation models for time series forecasting: Application in conformal prediction 

**Title (ZH)**: 时间序列预测中的基础模型：在可信预测中的应用 

**Authors**: Sami Achour, Yassine Bouher, Duong Nguyen, Nicolas Chesneau  

**Link**: [PDF](https://arxiv.org/pdf/2507.08858)  

**Abstract**: The zero-shot capabilities of foundation models (FMs) for time series forecasting offer promising potentials in conformal prediction, as most of the available data can be allocated to calibration. This study compares the performance of Time Series Foundation Models (TSFMs) with traditional methods, including statistical models and gradient boosting, within a conformal prediction setting. Our findings highlight two key advantages of TSFMs. First, when the volume of data is limited, TSFMs provide more reliable conformalized prediction intervals than classic models, thanks to their superior predictive accuracy. Second, the calibration process is more stable because more data are used for calibration. Morever, the fewer data available, the more pronounced these benefits become, as classic models require a substantial amount of data for effective training. These results underscore the potential of foundation models in improving conformal prediction reliability in time series applications, particularly in data-constrained cases. All the code to reproduce the experiments is available. 

**Abstract (ZH)**: 基础模型在时间序列预测中的零-shot能力为双重校验预测提供了令人鼓舞的潜力，因为大多数可用数据可以分配到校准过程中。本研究在双重校验预测框架下比较了时间序列基础模型（TSFMs）与传统方法（包括统计模型和梯度提升）的性能。我们的研究结果突显了TSFMs的两个主要优势。首先，当数据量受限时，TSFMs提供了比经典模型更可靠的双重校验预测区间，得益于其更高的预测准确性。其次，校准过程更加稳定，因为更多的数据用于校准。此外，随着可用数据的减少，这些优势变得更加显著，因为经典模型需要大量数据才能有效训练。这些结果强调了基础模型在提高时间序列应用中双重校验预测可靠性方面的潜力，特别是在数据受限的情况下。所有重现实验的代码均可获取。 

---
# Clio-X: AWeb3 Solution for Privacy-Preserving AI Access to Digital Archives 

**Title (ZH)**: Clio-X：一种保护隐私的Web3 AI访问数字档案解决方案 

**Authors**: Victoria L. Lemieux, Rosa Gil, Faith Molosiwa, Qihong Zhou, Binming Li, Roberto Garcia, Luis De La Torre Cubillo, Zehua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08853)  

**Abstract**: As archives turn to artificial intelligence to manage growing volumes of digital records, privacy risks inherent in current AI data practices raise critical concerns about data sovereignty and ethical accountability. This paper explores how privacy-enhancing technologies (PETs) and Web3 architectures can support archives to preserve control over sensitive content while still being able to make it available for access by researchers. We present Clio-X, a decentralized, privacy-first Web3 digital solution designed to embed PETs into archival workflows and support AI-enabled reference and access. Drawing on a user evaluation of a medium-fidelity prototype, the study reveals both interest in the potential of the solution and significant barriers to adoption related to trust, system opacity, economic concerns, and governance. Using Rogers' Diffusion of Innovation theory, we analyze the sociotechnical dimensions of these barriers and propose a path forward centered on participatory design and decentralized governance through a Clio-X Decentralized Autonomous Organization. By integrating technical safeguards with community-based oversight, Clio-X offers a novel model to ethically deploy AI in cultural heritage contexts. 

**Abstract (ZH)**: 随着档案机构利用人工智能管理日益增长的数字记录数量，现有AI数据实践中的隐私风险引发了关于数据主权和伦理责任的关键关注。本文探讨了如何通过增强隐私的技术（PETs）和Web3架构，支持档案机构在保留对敏感内容控制权的同时，仍能让研究者访问相关内容。我们介绍了Clio-X，这是一种去中心化的、以隐私为中心的Web3数字解决方案，旨在将PETs嵌入到档案工作流程中，并支持基于AI的参考和访问。通过一个中保真度原型的用户评估，研究揭示了对该解决方案潜在兴趣以及信托、系统透明度、经济考量和治理方面的显著采用障碍。我们运用罗杰斯的创新扩散理论分析这些障碍的社技维度，并提出以参与式设计和通过Clio-X去中心化自治组织实现去中心化治理为中心的发展路径。通过将技术保障与社区监督相结合，Clio-X提供了一种新的模型，以伦理方式在文化遗产领域部署人工智能。 

---
# Assuring the Safety of Reinforcement Learning Components: AMLAS-RL 

**Title (ZH)**: 保证强化学习组件的安全性：AMLAS-RL 

**Authors**: Calum Corrie Imrie, Ioannis Stefanakos, Sepeedeh Shahbeigi, Richard Hawkins, Simon Burton  

**Link**: [PDF](https://arxiv.org/pdf/2507.08848)  

**Abstract**: The rapid advancement of machine learning (ML) has led to its increasing integration into cyber-physical systems (CPS) across diverse domains. While CPS offer powerful capabilities, incorporating ML components introduces significant safety and assurance challenges. Among ML techniques, reinforcement learning (RL) is particularly suited for CPS due to its capacity to handle complex, dynamic environments where explicit models of interaction between system and environment are unavailable or difficult to construct. However, in safety-critical applications, this learning process must not only be effective but demonstrably safe. Safe-RL methods aim to address this by incorporating safety constraints during learning, yet they fall short in providing systematic assurance across the RL lifecycle. The AMLAS methodology offers structured guidance for assuring the safety of supervised learning components, but it does not directly apply to the unique challenges posed by RL. In this paper, we adapt AMLAS to provide a framework for generating assurance arguments for an RL-enabled system through an iterative process; AMLAS-RL. We demonstrate AMLAS-RL using a running example of a wheeled vehicle tasked with reaching a target goal without collision. 

**Abstract (ZH)**: 机器学习快速进步使其在跨多个领域的网络物理系统(CPS)中日益集成。尽管CPS提供了强大的功能，但集成机器学习组件引入了重要的安全性和保障挑战。在机器学习技术中，强化学习(RL)特别适用于CPS，因为它们能够处理缺乏或难以构建系统与环境交互模型的复杂动态环境。但在关键安全应用中，这一学习过程不仅需要有效，还必须是可验证的安全的。安全的RL方法旨在通过在学习过程中引入安全约束来解决这一问题，但它们在提供整个RL生命周期中的系统保障方面仍有所欠缺。AMLAS方法提供了监督学习组件安全性的结构化指导，但未能直接适用于RL所提出的所有独特挑战。在本文中，我们通过对AMLAS进行适应，提出了一种通过迭代过程生成RL使能系统保障论据的框架——AMLAS-RL。我们通过一辆需在不发生碰撞的情况下到达目标点的轮式车辆运行示例来展示AMLAS-RL。 

---
# DAFOS: Dynamic Adaptive Fanout Optimization Sampler 

**Title (ZH)**: DAFOS：动态自适应扇出优化采样器 

**Authors**: Irfan Ullah, Young-Koo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.08845)  

**Abstract**: Graph Neural Networks (GNNs) are becoming an essential tool for learning from graph-structured data, however uniform neighbor sampling and static fanout settings frequently limit GNNs' scalability and efficiency. In this paper, we propose the Dynamic Adaptive Fanout Optimization Sampler (DAFOS), a novel approach that dynamically adjusts the fanout based on model performance and prioritizes important nodes during training. Our approach leverages node scoring based on node degree to focus computational resources on structurally important nodes, incrementing the fanout as the model training progresses. DAFOS also integrates an early stopping mechanism to halt training when performance gains diminish. Experiments conducted on three benchmark datasets, ogbnarxiv, Reddit, and ogbn-products, demonstrate that our approach significantly improves training speed and accuracy compared to a state-of-the-art approach. DAFOS achieves a 3.57x speedup on the ogbn-arxiv dataset and a 12.6x speedup on the Reddit dataset while improving the F1 score from 68.5% to 71.21% on ogbn-arxiv and from 73.78% to 76.88% on the ogbn-products dataset, respectively. These results highlight the potential of DAFOS as an efficient and scalable solution for large-scale GNN training. 

**Abstract (ZH)**: 动态自适应扇出优化抽样器（DAFOS）：一种基于模型性能动态调整扇出并优先处理重要节点的方法 

---
# Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing 

**Title (ZH)**: 梯度作为行动：通过适应性行动共享实现高效联邦推荐系统的研究 

**Authors**: Zhufeng Lu, Chentao Jia, Ming Hu, Xiaofei Xie, Mingsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08842)  

**Abstract**: As a promising privacy-aware collaborative model training paradigm, Federated Learning (FL) is becoming popular in the design of distributed recommender systems. However, Federated Recommender Systems (FedRecs) greatly suffer from two major problems: i) extremely high communication overhead due to massive item embeddings involved in recommendation systems, and ii) intolerably low training efficiency caused by the entanglement of both heterogeneous network environments and client devices. Although existing methods attempt to employ various compression techniques to reduce communication overhead, due to the parameter errors introduced by model compression, they inevitably suffer from model performance degradation. To simultaneously address the above problems, this paper presents a communication-efficient FedRec framework named FedRAS, which adopts an action-sharing strategy to cluster the gradients of item embedding into a specific number of model updating actions for communication rather than directly compressing the item embeddings. In this way, the cloud server can use the limited actions from clients to update all the items. Since gradient values are significantly smaller than item embeddings, constraining the directions of gradients (i.e., the action space) introduces smaller errors compared to compressing the entire item embedding matrix into a reduced space. To accommodate heterogeneous devices and network environments, FedRAS incorporates an adaptive clustering mechanism that dynamically adjusts the number of actions. Comprehensive experiments on well-known datasets demonstrate that FedRAS can reduce the size of communication payloads by up to 96.88%, while not sacrificing recommendation performance within various heterogeneous scenarios. We have open-sourced FedRAS at this https URL. 

**Abstract (ZH)**: 作为一种有前景的隐私感知协作模型训练范式，联邦学习（Federated Learning, FL）在分布式推荐系统的设计中正变得流行。然而，联邦推荐系统（FedRecs）受到两大主要问题的严重影响：一是由于推荐系统中涉及大量项嵌入而导致极大的通信开销；二是由于异构网络环境和客户端设备的纠缠而导致不堪忍受的低训练效率。尽管现有方法尝试使用各种压缩技术来减少通信开销，但由于模型压缩引入的参数误差，它们不可避免地会牺牲模型性能。为了同时解决上述问题，本文提出了一种名为FedRAS的通信高效FedRec框架，该框架采用动作共享策略将项嵌入的梯度聚类成特定数量的模型更新动作用于通信，而不是直接压缩项嵌入。这样一来，云服务器可以利用客户端的有限动作来更新所有项。由于梯度值远小于项嵌入，限制梯度的方向（即动作空间）引入的误差比将整个项嵌入矩阵压缩到较小空间要小。为了适应异构设备和网络环境，FedRAS整合了一种自适应聚类机制，能够动态调整动作数量。在多种异构场景下的综合实验表明，FedRAS可以将通信负载大小减少高达96.88%，且不会牺牲推荐性能。我们已将FedRAS开源于此链接。 

---
# Zero-Shot Neural Architecture Search with Weighted Response Correlation 

**Title (ZH)**: 零样本神经架构搜索：带加权响应相关性方法 

**Authors**: Kun Jing, Luoyu Chen, Jungang Xu, Jianwei Tai, Yiyu Wang, Shuaimin Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08841)  

**Abstract**: Neural architecture search (NAS) is a promising approach for automatically designing neural network architectures. However, the architecture estimation of NAS is computationally expensive and time-consuming because of training multiple architectures from scratch. Although existing zero-shot NAS methods use training-free proxies to accelerate the architecture estimation, their effectiveness, stability, and generality are still lacking. We present a novel training-free estimation proxy called weighted response correlation (WRCor). WRCor utilizes correlation coefficient matrices of responses across different input samples to calculate the proxy scores of estimated architectures, which can measure their expressivity and generalizability. Experimental results on proxy evaluation demonstrate that WRCor and its voting proxies are more efficient estimation strategies than existing proxies. We also apply them with different search strategies in architecture search. Experimental results on architecture search show that our zero-shot NAS algorithm outperforms most existing NAS algorithms in different search spaces. Our NAS algorithm can discover an architecture with a 22.1% test error on the ImageNet-1k dataset within 4 GPU hours. All codes are publicly available at this https URL. 

**Abstract (ZH)**: 无监督神经架构搜索：一种基于加权响应相关性的新型估计代理方法 

---
# Domain-Adaptive Diagnosis of Lewy Body Disease with Transferability Aware Transformer 

**Title (ZH)**: 带有转移意识的变换器在莱氏体疾病领域自适应诊断中应用 

**Authors**: Xiaowei Yu, Jing Zhang, Tong Chen, Yan Zhuang, Minheng Chen, Chao Cao, Yanjun Lyu, Lu Zhang, Li Su, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08839)  

**Abstract**: Lewy Body Disease (LBD) is a common yet understudied form of dementia that imposes a significant burden on public health. It shares clinical similarities with Alzheimer's disease (AD), as both progress through stages of normal cognition, mild cognitive impairment, and dementia. A major obstacle in LBD diagnosis is data scarcity, which limits the effectiveness of deep learning. In contrast, AD datasets are more abundant, offering potential for knowledge transfer. However, LBD and AD data are typically collected from different sites using different machines and protocols, resulting in a distinct domain shift. To effectively leverage AD data while mitigating domain shift, we propose a Transferability Aware Transformer (TAT) that adapts knowledge from AD to enhance LBD diagnosis. Our method utilizes structural connectivity (SC) derived from structural MRI as training data. Built on the attention mechanism, TAT adaptively assigns greater weights to disease-transferable features while suppressing domain-specific ones, thereby reducing domain shift and improving diagnostic accuracy with limited LBD data. The experimental results demonstrate the effectiveness of TAT. To the best of our knowledge, this is the first study to explore domain adaptation from AD to LBD under conditions of data scarcity and domain shift, providing a promising framework for domain-adaptive diagnosis of rare diseases. 

**Abstract (ZH)**: Lewy 体疾病 (LBD) 是一种常见但研究不足的痴呆形式，对公共卫生造成重大负担。它在临床表现上与阿尔茨海默病 (AD) 相似，两者均经历正常认知、轻度认知 impairment 和痴呆等阶段。LBD 诊断的一大障碍是数据稀缺性，限制了深度学习的有效性。相比之下，AD 数据更为丰富，为知识迁移提供了潜在可能性。然而，LBD 和 AD 数据通常来自不同的地点，使用不同的设备和协议收集，导致了独特的领域偏移。为有效利用 AD 数据并减轻领域偏移，我们提出了一种aware Transformer (TAT)，以将 AD 中的知识应用于增强 LBD 诊断。该方法利用结构 MRI 提取的结构连接性 (SC) 作为训练数据。基于注意力机制，TAT 自适应地赋予可转移疾病特征更大的权重，同时抑制领域特异性特征，从而减少领域偏移并提高在有限 LBD 数据下的诊断准确性。实验结果证明了 TAT 的有效性。据我们所知，这是首次在数据稀缺性和领域偏移条件下探索从 AD 到 LBD 的领域适应性诊断的研究，为稀有疾病的领域适应性诊断提供了有 promise 的框架。 

---
# Representation learning with a transformer by contrastive learning for money laundering detection 

**Title (ZH)**: 基于对比学习的变压器表示学习在洗钱检测中的应用 

**Authors**: Harold Guéneau, Alain Celisse, Pascal Delange  

**Link**: [PDF](https://arxiv.org/pdf/2507.08835)  

**Abstract**: The present work tackles the money laundering detection problem. A new procedure is introduced which exploits structured time series of both qualitative and quantitative data by means of a transformer neural network. The first step of this procedure aims at learning representations of time series through contrastive learning (without any labels). The second step leverages these representations to generate a money laundering scoring of all observations. A two-thresholds approach is then introduced, which ensures a controlled false-positive rate by means of the Benjamini-Hochberg (BH) procedure. Experiments confirm that the transformer is able to produce general representations that succeed in exploiting money laundering patterns with minimal supervision from domain experts. It also illustrates the higher ability of the new procedure for detecting nonfraudsters as well as fraudsters, while keeping the false positive rate under control. This greatly contrasts with rule-based procedures or the ones based on LSTM architectures. 

**Abstract (ZH)**: 本研究解决了洗钱检测问题。提出了一种新的方法，通过变压器神经网络利用结构化的时间序列数据（既有定性数据也有定量数据）进行检测。该方法的第一步通过对比学习（无需标签）学习时间序列的表示。第二步利用这些表示为所有观察生成洗钱评分。引入了一种双阈值方法，通过贝杰曼-霍奇格（BH）程序确保了较低的假阳性率。实验结果证实，变压器能够生成通用表示，能够利用洗钱模式进行检测，同时需要较少的主题专家监督。此外，该方法在检测非欺诈者和欺诈者方面表现更优，同时将假阳性率控制在可接受范围内。这与基于规则的方法或基于LSTM的架构形成了鲜明对比。 

---
# Efficient Triple Modular Redundancy for Reliability Enhancement of DNNs Using Explainable AI 

**Title (ZH)**: 基于可解释人工智能的DNN可靠性增强的高效三模冗余方法 

**Authors**: Kimia Soroush, Nastaran Shirazi, Mohsen Raji  

**Link**: [PDF](https://arxiv.org/pdf/2507.08829)  

**Abstract**: Deep Neural Networks (DNNs) are widely employed in safety-critical domains, where ensuring their reliability is essential. Triple Modular Redundancy (TMR) is an effective technique to enhance the reliability of DNNs in the presence of bit-flip faults. In order to handle the significant overhead of TMR, it is applied selectively on the parameters and components with the highest contribution at the model output. Hence, the accuracy of the selection criterion plays the key role on the efficiency of TMR. This paper presents an efficient TMR approach to enhance the reliability of DNNs against bit-flip faults using an Explainable Artificial Intelligence (XAI) method. Since XAI can provide valuable insights about the importance of individual neurons and weights in the performance of the network, they can be applied as the selection metric in TMR techniques. The proposed method utilizes a low-cost, gradient-based XAI technique known as Layer-wise Relevance Propagation (LRP) to calculate importance scores for DNN parameters. These scores are then used to enhance the reliability of the model, with the most critical weights being protected by TMR. The proposed approach is evaluated on two DNN models, VGG16 and AlexNet, using datasets such as MNIST and CIFAR-10. The results demonstrate that the method can protect the AlexNet model at a bit error rate of 10-4, achieving over 60% reliability improvement while maintaining the same overhead as state-of-the-art methods. 

**Abstract (ZH)**: 深层神经网络（DNNs）广泛应用于安全关键领域，确保其可靠性至关重要。在存在位翻转故障的情况下，三模冗余（TMR）是一种有效的提高DNNs可靠性的技术。为了处理TMR的显著开销，它仅被有最高贡献的模型参数和组件上选择性地应用。因此，选择标准的准确性在TMR的效率中起着关键作用。本文提出了一种高效的基于可解释人工智能（XAI）的TMR方法，以提高DNNs在位翻转故障情况下的可靠性。由于XAI可以提供有关网络性能中 individual 神经元和权重重要性的宝贵见解，它们可以作为TMR技术中的选择指标。所提出的方法利用一种低成本的基于梯度的XAI技术——层相关性传播（LRP）来计算DNN参数的重要性分数。这些分数随后被用于增强模型的可靠性，最关键的权重通过TMR进行保护。所提出的方法在VGG16和AlexNet两个DNN模型上进行了评估，使用MNIST和CIFAR-10等数据集。结果表明，在位错误率为 \(10^{-4}\) 的情况下，该方法能够保护AlexNet模型，同时在保持与最新方法相同开销的同时，可靠性提高了超过60%。 

---
# Advancing network resilience theories with symbolized reinforcement learning 

**Title (ZH)**: 用符号化强化学习推进网络韧性理论 

**Authors**: Yu Zheng, Jingtao Ding, Depeng Jin, Jianxi Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08827)  

**Abstract**: Many complex networks display remarkable resilience under external perturbations, internal failures and environmental changes, yet they can swiftly deteriorate into dysfunction upon the removal of a few keystone nodes. Discovering theories that measure network resilience offers the potential to prevent catastrophic collapses--from species extinctions to financial crise--with profound implications for real-world systems. Current resilience theories address the problem from a single perspective of topology, neglecting the crucial role of system dynamics, due to the intrinsic complexity of the coupling between topology and dynamics which exceeds the capabilities of human analytical methods. Here, we report an automatic method for resilience theory discovery, which learns from how AI solves a complicated network dismantling problem and symbolizes its network attack strategies into theoretical formulas. This proposed self-inductive approach discovers the first resilience theory that accounts for both topology and dynamics, highlighting how the correlation between node degree and state shapes overall network resilience, and offering insights for designing early warning signals of systematic collapses. Additionally, our approach discovers formulas that refine existing well-established resilience theories with over 37.5% improvement in accuracy, significantly advancing human understanding of complex networks with AI. 

**Abstract (ZH)**: 自动发现同时考虑拓扑与动力学的网络韧性理论：从人工智能复杂网络拆解中学习并提炼网络攻击策略公式 

---
# Lightweight Cloud Masking Models for On-Board Inference in Hyperspectral Imaging 

**Title (ZH)**: 轻量级云遮蔽模型：用于高光谱成像的机载推理 

**Authors**: Mazen Ali, António Pereira, Fabio Gentile, Aser Cortines, Sam Mugel, Román Orús, Stelios P. Neophytides, Michalis Mavrovouniotis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08052)  

**Abstract**: Cloud and cloud shadow masking is a crucial preprocessing step in hyperspectral satellite imaging, enabling the extraction of high-quality, analysis-ready data. This study evaluates various machine learning approaches, including gradient boosting methods such as XGBoost and LightGBM as well as convolutional neural networks (CNNs). All boosting and CNN models achieved accuracies exceeding 93%. Among the investigated models, the CNN with feature reduction emerged as the most efficient, offering a balance of high accuracy, low storage requirements, and rapid inference times on both CPUs and GPUs. Variations of this version, with only up to 597 trainable parameters, demonstrated the best trade-off in terms of deployment feasibility, accuracy, and computational efficiency. These results demonstrate the potential of lightweight artificial intelligence (AI) models for real-time hyperspectral image processing, supporting the development of on-board satellite AI systems for space-based applications. 

**Abstract (ZH)**: 云和云影掩模是高光谱卫星成像中的一个关键预处理步骤，能够提取高质量的分析数据。本研究评估了多种机器学习方法，包括梯度提升方法如XGBoost和LightGBM以及卷积神经网络（CNNs）。所有增强和CNN模型的准确性均超过93%。在所研究的模型中，具有特征降维的CNN模型在准确率、存储需求和在CPU和GPU上的快速推理速度方面表现出最佳平衡。其变体版本仅具有最多597个可训练参数，在部署可行性、准确性和计算效率方面达到了最佳权衡。这些结果表明轻量级人工智能（AI）模型在实时高光谱图像处理中的潜力，支持空间基应用中机载AI系统的开发。 

---
# Principled Foundations for Preference Optimization 

**Title (ZH)**: 原则性的基础理论用于偏好优化 

**Authors**: Wenxuan Zhou, Shujian Zhang, Brice Magdalou, John Lambert, Ehsan Amid, Richard Nock, Andrew Hard  

**Link**: [PDF](https://arxiv.org/pdf/2507.07855)  

**Abstract**: In this paper, we show that direct preference optimization (DPO) is a very specific form of a connection between two major theories in the ML context of learning from preferences: loss functions (Savage) and stochastic choice (Doignon-Falmagne and Machina). The connection is established for all of Savage's losses and at this level of generality, (i) it includes support for abstention on the choice theory side, (ii) it includes support for non-convex objectives on the ML side, and (iii) it allows to frame for free some notable extensions of the DPO setting, including margins and corrections for length. Getting to understand how DPO operates from a general principled perspective is crucial because of the huge and diverse application landscape of models, because of the current momentum around DPO, but also -- and importantly -- because many state of the art variations on DPO definitely occupy a small region of the map that we cover. It also helps to understand the pitfalls of departing from this map, and figure out workarounds. 

**Abstract (ZH)**: 在本文中，我们展示了直接偏好优化（DPO）是连接机器学习（ML）中从偏好学习场景下的两类主要理论——损失函数（Savage）和随机选择（Doignon-Falmagne和Machina）——的非常具体的形式。该连接涵盖了Savage的所有损失函数，并在这一普遍性的水平上，（i）包括选择理论中的弃权支持，（ii）包括ML方面的非凸优化目标支持，（iii）允许免费框架一些DPO设置的显著扩展，包括边际和长度校正。从一般原则的角度理解DPO是如何运作的至关重要，这不仅是因为模型的应用场景极其广泛和多样，而且因为目前DPO研究的势头正盛，更重要的是——许多最先进的DPO变体肯定占据我们在覆盖范围内的一部分区域，这也有助于理解偏离这一框架的漏洞，并找出解决方案。 

---
# An Enhanced Classification Method Based on Adaptive Multi-Scale Fusion for Long-tailed Multispectral Point Clouds 

**Title (ZH)**: 基于自适应多尺度融合的长尾多光谱点云增强分类方法 

**Authors**: TianZhu Liu, BangYan Hu, YanFeng Gu, Xian Li, Aleksandra Pižurica  

**Link**: [PDF](https://arxiv.org/pdf/2412.11407)  

**Abstract**: Multispectral point cloud (MPC) captures 3D spatial-spectral information from the observed scene, which can be used for scene understanding and has a wide range of applications. However, most of the existing classification methods were extensively tested on indoor datasets, and when applied to outdoor datasets they still face problems including sparse labeled targets, differences in land-covers scales, and long-tailed distributions. To address the above issues, an enhanced classification method based on adaptive multi-scale fusion for MPCs with long-tailed distributions is proposed. In the training set generation stage, a grid-balanced sampling strategy is designed to reliably generate training samples from sparse labeled datasets. In the feature learning stage, a multi-scale feature fusion module is proposed to fuse shallow features of land-covers at different scales, addressing the issue of losing fine features due to scale variations in land-covers. In the classification stage, an adaptive hybrid loss module is devised to utilize multi-classification heads with adaptive weights to balance the learning ability of different classes, improving the classification performance of small classes due to various-scales and long-tailed distributions in land-covers. Experimental results on three MPC datasets demonstrate the effectiveness of the proposed method compared with the state-of-the-art methods. 

**Abstract (ZH)**: 基于自适应多尺度融合的宽尾分布多光谱点云增强分类方法 

---
