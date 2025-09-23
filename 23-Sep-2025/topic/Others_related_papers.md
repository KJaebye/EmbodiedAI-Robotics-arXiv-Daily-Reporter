# Guided Multi-Fidelity Bayesian Optimization for Data-driven Controller Tuning with Digital Twins 

**Title (ZH)**: 指导下的多保真度贝叶斯优化在数字孪生驱动的控制器调优中 

**Authors**: Mahdi Nobar, Jürg Keller, Alessandro Forino, John Lygeros, Alisa Rupenyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17952)  

**Abstract**: We propose a \textit{guided multi-fidelity Bayesian optimization} framework for data-efficient controller tuning that integrates corrected digital twin (DT) simulations with real-world measurements. The method targets closed-loop systems with limited-fidelity simulations or inexpensive approximations. To address model mismatch, we build a multi-fidelity surrogate with a learned correction model that refines DT estimates from real data. An adaptive cost-aware acquisition function balances expected improvement, fidelity, and sampling cost. Our method ensures adaptability as new measurements arrive. The accuracy of DTs is re-estimated, dynamically adapting both cross-source correlations and the acquisition function. This ensures that accurate DTs are used more frequently, while inaccurate DTs are appropriately downweighted. Experiments on robotic drive hardware and supporting numerical studies demonstrate that our method enhances tuning efficiency compared to standard Bayesian optimization (BO) and multi-fidelity methods. 

**Abstract (ZH)**: 指导多保真度贝叶斯优化框架：将校正数字孪生仿真与实时测量集成用于数据高效控制器调优 

---
# EigenSafe: A Spectral Framework for Learning-Based Stochastic Safety Filtering 

**Title (ZH)**: EigenSafe: 一种基于谱的方法进行学习驱动的随机安全性过滤 

**Authors**: Inkyu Jang, Jonghae Park, Chams E. Mballo, Sihyun Cho, Claire J. Tomlin, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17750)  

**Abstract**: We present EigenSafe, an operator-theoretic framework for learning-enabled safety-critical control for stochastic systems. In many robotic systems where dynamics are best modeled as stochastic systems due to factors such as sensing noise and environmental disturbances, it is challenging for conventional methods such as Hamilton-Jacobi reachability and control barrier functions to provide a holistic measure of safety. We derive a linear operator governing the dynamic programming principle for safety probability, and find that its dominant eigenpair provides information about safety for both individual states and the overall closed-loop system. The proposed learning framework, called EigenSafe, jointly learns this dominant eigenpair and a safe backup policy in an offline manner. The learned eigenfunction is then used to construct a safety filter that detects potentially unsafe situations and falls back to the backup policy. The framework is validated in three simulated stochastic safety-critical control tasks. 

**Abstract (ZH)**: EigenSafe：一种用于随机系统学习驱动的安全临界控制的操作理论框架 

---
# OpenGVL - Benchmarking Visual Temporal Progress for Data Curation 

**Title (ZH)**: OpenGVL - 数据编辑中的视觉时间进度基准评估 

**Authors**: Paweł Budzianowski, Emilia Wiśnios, Gracjan Góral, Igor Kulakov, Viktor Petrenko, Krzysztof Walas  

**Link**: [PDF](https://arxiv.org/pdf/2509.17321)  

**Abstract**: Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{this http URL}{OpenGVL}. 

**Abstract (ZH)**: 数据稀缺仍然是驱动机器人领域进展的最大限制因素之一。然而，野外可获取的机器人数据量正在以指数级增长，创造了大规模数据利用的新机会。可靠的阶段性任务完成预测可以帮助大规模自动标注和整理这些数据。最近提出了一种生成性价值学习（GVL）方法，利用视觉语言模型（VLMs）中的知识，从视觉观察中预测任务进度。在此基础上，我们提出了一种名为OpenGVL的基准测试，用于评估多样化挑战性操作任务中的任务进度，这些任务涉及机器人的和人类的实体。我们评估了公开可用的开源基础模型的能力，发现开源模型系列明显劣于闭源对应模型，仅在时间进度预测任务上达到闭源模型约70%的性能。此外，我们展示了OpenGVL如何作为自动化数据整理和过滤的实际工具，从而有效地评估大规模机器人数据集的质量。我们发布了基准测试及其完整的代码库，可在\href{this http URL}{OpenGVL}找到。 

---
# Learning and Optimization with 3D Orientations 

**Title (ZH)**: 学习与优化中的3D方向分析 

**Authors**: Alexandros Ntagkas, Constantinos Tsakonas, Chairi Kiourt, Konstantinos Chatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2509.17274)  

**Abstract**: There exist numerous ways of representing 3D orientations. Each representation has both limitations and unique features. Choosing the best representation for one task is often a difficult chore, and there exist conflicting opinions on which representation is better suited for a set of family of tasks. Even worse, when dealing with scenarios where we need to learn or optimize functions with orientations as inputs and/or outputs, the set of possibilities (representations, loss functions, etc.) is even larger and it is not easy to decide what is best for each scenario. In this paper, we attempt to a) present clearly, concisely and with unified notation all available representations, and "tricks" related to 3D orientations (including Lie Group algebra), and b) benchmark them in representative scenarios. The first part feels like it is missing from the robotics literature as one has to read many different textbooks and papers in order have a concise and clear understanding of all possibilities, while the benchmark is necessary in order to come up with recommendations based on empirical evidence. More precisely, we experiment with the following settings that attempt to cover most widely used scenarios in robotics: 1) direct optimization, 2) imitation/supervised learning with a neural network controller, 3) reinforcement learning, and 4) trajectory optimization using differential dynamic programming. We finally provide guidelines depending on the scenario, and make available a reference implementation of all the orientation math described. 

**Abstract (ZH)**: 存在多种表示3D方向的方法。每种表示方法都有其局限性和独特特征。选择最适合某一任务的表示方法往往是一项艰巨的任务，不同的人对于哪些表示方法更适合一组任务存在意见分歧。更糟糕的是，在需要以方向作为输入和/或输出来学习或优化函数的情况下，可供选择的可能性（包括表示方法、损失函数等）更多，很难决定每种情景的最佳选择。本文旨在：a) 用简洁统一的符号清晰地呈现所有可用的表示方法及相关技巧（包括李群代数），b) 在代表性情景中对这些表示方法进行基准测试。当前机器人学文献中缺乏这部分内容，通常需要阅读多本不同的教科书和论文才能获得对所有可能性的简洁清晰理解，而基准测试则是基于实证证据提出推荐的必要步骤。具体来说，我们实验了如下设置以涵盖大多数常用情景：1) 直接优化，2) 使用神经网络控制器的模仿/监督学习，3) 强化学习，4) 使用差分动态规划进行轨迹优化。最后，我们根据情景提供指导，并提供所有方向数学描述的参考实现。 

---
# Neural Network and ANFIS based auto-adaptive MPC for path tracking in autonomous vehicles 

**Title (ZH)**: 基于神经网络和ANFIS的自适应 MPC 在自主车辆路径跟踪中的应用 

**Authors**: Yassine Kebbati, Naima Ait-Oufroukh, Vincent Vigneron, Dalil Ichala  

**Link**: [PDF](https://arxiv.org/pdf/2509.17213)  

**Abstract**: Self-driving cars operate in constantly changing environments and are exposed to a variety of uncertainties and disturbances. These factors render classical controllers ineffective, especially for lateral control. Therefore, an adaptive MPC controller is designed in this paper for the path tracking task, tuned by an improved particle swarm optimization algorithm. Online parameter adaptation is performed using Neural Networks and ANFIS. The designed controller showed promising results compared to standard MPC in triple lane change and trajectory tracking scenarios. Code can be found here: this https URL 

**Abstract (ZH)**: 自动驾驶汽车在不断变化的环境中运行，并暴露于各种不确定性与干扰中。这些因素使得经典控制器无效，尤其是在横向控制方面。因此，本文设计了一种用于路径跟踪任务的自适应MPC控制器，该控制器通过改进的粒子群优化算法进行调整。在线参数适应使用神经网络和ANFIS实现。所设计的控制器在三车道变换和轨迹跟踪场景中表现出色，与标准MPC相比具有潜在优势。代码可在此处找到：this https URL 

---
# Certifiably Optimal Doppler Positioning using Opportunistic LEO Satellites 

**Title (ZH)**: 使用机会低轨卫星的可验证最优多普勒定位 

**Authors**: Baoshan Song, Weisong Wen, Qi Zhang, Bing Xu, Li-Ta Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17198)  

**Abstract**: To provide backup and augmentation to global navigation satellite system (GNSS), Doppler shift from Low Earth Orbit (LEO) satellites can be employed as signals of opportunity (SOP) for position, navigation and timing (PNT). Since the Doppler positioning problem is non-convex, local searching methods may produce two types of estimates: a global optimum without notice or a local optimum given an inexact initial estimate. As exact initialization is unavailable in some unknown environments, a guaranteed global optimization method in no need of initialization becomes necessary. To achieve this goal, we propose a certifiably optimal LEO Doppler positioning method by utilizing convex optimization. In this paper, the certifiable positioning method is implemented through a graduated weight approximation (GWA) algorithm and semidefinite programming (SDP) relaxation. To guarantee the optimality, we derive the necessary conditions for optimality in ideal noiseless cases and sufficient noise bounds conditions in noisy cases. Simulation and real tests are conducted to evaluate the effectiveness and robustness of the proposed method. Specially, the real test using Iridium-NEXT satellites shows that the proposed method estimates an certifiably optimal solution with an 3D positioning error of 140 m without initial estimates while Gauss-Newton and Dog-Leg are trapped in local optima when the initial point is equal or larger than 1000 km away from the ground truth. Moreover, the certifiable estimation can also be used as initialization in local searching methods to lower down the 3D positioning error to 130 m. 

**Abstract (ZH)**: 利用 convex 优化实现可认证的低地球轨道 Doppler 定位方法 

---
# SMART-3D: Three-Dimensional Self-Morphing Adaptive Replanning Tree 

**Title (ZH)**: SMART-3D: 三维自变形适应重规划树 

**Authors**: Priyanshu Agrawal, Shalabh Gupta, Zongyuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.16812)  

**Abstract**: This paper presents SMART-3D, an extension of the SMART algorithm to 3D environments. SMART-3D is a tree-based adaptive replanning algorithm for dynamic environments with fast moving obstacles. SMART-3D morphs the underlying tree to find a new path in real-time whenever the current path is blocked by obstacles. SMART-3D removed the grid decomposition requirement of the SMART algorithm by replacing the concept of hot-spots with that of hot-nodes, thus making it computationally efficient and scalable to 3D environments. The hot-nodes are nodes which allow for efficient reconnections to morph the existing tree to find a new safe and reliable path. The performance of SMART-3D is evaluated by extensive simulations in 2D and 3D environments populated with randomly moving dynamic obstacles. The results show that SMART-3D achieves high success rates and low replanning times, thus highlighting its suitability for real-time onboard applications. 

**Abstract (ZH)**: SMART-3D：SMART算法在3D环境中的扩展 

---
# ORN-CBF: Learning Observation-conditioned Residual Neural Control Barrier Functions via Hypernetworks 

**Title (ZH)**: ORN-CBF: 通过超网络学习观测条件下的残差神经控制屏障函数 

**Authors**: Bojan Derajić, Sebastian Bernhard, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2509.16614)  

**Abstract**: Control barrier functions (CBFs) have been demonstrated as an effective method for safety-critical control of autonomous systems. Although CBFs are simple to deploy, their design remains challenging, motivating the development of learning-based approaches. Yet, issues such as suboptimal safe sets, applicability in partially observable environments, and lack of rigorous safety guarantees persist. In this work, we propose observation-conditioned neural CBFs based on Hamilton-Jacobi (HJ) reachability analysis, which approximately recover the maximal safe sets. We exploit certain mathematical properties of the HJ value function, ensuring that the predicted safe set never intersects with the observed failure set. Moreover, we leverage a hypernetwork-based architecture that is particularly suitable for the design of observation-conditioned safety filters. The proposed method is examined both in simulation and hardware experiments for a ground robot and a quadcopter. The results show improved success rates and generalization to out-of-domain environments compared to the baselines. 

**Abstract (ZH)**: 基于哈密顿-雅可比可达性分析的观测条件神经控制壁垒函数 

---
# Trajectory Encryption Cooperative Salvo Guidance 

**Title (ZH)**: 轨迹加密协同齐射 guidance 

**Authors**: Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2509.17341)  

**Abstract**: This paper introduces the concept of trajectory encryption in cooperative simultaneous target interception, wherein heterogeneity in guidance principles across a team of unmanned autonomous systems is leveraged as a strategic design feature. By employing a mix of heterogeneous time-to-go formulations leading to a cooperative guidance strategy, the swarm of vehicles is able to generate diverse trajectory families. This diversity expands the feasible solution space for simultaneous target interception, enhances robustness under disturbances, and enables flexible time-to-go adjustments without predictable detouring. From an adversarial perspective, heterogeneity obscures the collective interception intent by preventing straightforward prediction of swarm dynamics, effectively acting as an encryption layer in the trajectory domain. Simulations demonstrate that the swarm of heterogeneous vehicles is able to intercept a moving target simultaneously from a diverse set of initial engagement configurations. 

**Abstract (ZH)**: 本文介绍了协同同时目标拦截中轨迹加密的概念，其中无人驾驶自主系统团队中的指导原则异质性被用作一种战略设计特征。通过采用多种异质性剩余时间形式，从而实现协同指导策略，车辆群能够生成多样化的轨迹家族。这种多样性扩展了同时目标拦截的可行解空间，增强了在干扰下的稳健性，并允许灵活调整剩余时间而不预示可预测的迂回行为。从敌对角度看，异质性模糊了集合拦截意图，防止对手简单预测群体动力学，从而在轨迹领域有效起到加密层的作用。模拟结果表明，异质性车辆群能够从多种初始交战配置中同时拦截移动目标。 

---
# Delay compensation of multi-input distinct delay nonlinear systems via neural operators 

**Title (ZH)**: 多输入不相同延迟非线性系统的时延补偿 via 神经运算器 

**Authors**: Filip Bajraktari, Luke Bhan, Miroslav Krstic, Yuanyuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.17131)  

**Abstract**: In this work, we present the first stability results for approximate predictors in multi-input non-linear systems with distinct actuation delays. We show that if the predictor approximation satisfies a uniform (in time) error bound, semi-global practical stability is correspondingly achieved. For such approximators, the required uniform error bound depends on the desired region of attraction and the number of control inputs in the system. The result is achieved through transforming the delay into a transport PDE and conducting analysis on the coupled ODE-PDE cascade. To highlight the viability of such error bounds, we demonstrate our results on a class of approximators - neural operators - showcasing sufficiency for satisfying such a universal bound both theoretically and in simulation on a mobile robot experiment. 

**Abstract (ZH)**: 本工作中，我们提出了多输入非线性系统中具有不同执行器延迟的近似预测器的第一稳定性结果。我们证明，如果预测器逼近满足时间一致的误差界，则相应地实现半全局实用稳定性。对于此类逼近器，所需的误差界依赖于期望的吸引域及其系统中的控制输入数量。我们通过将延迟转化为传输偏微分方程（PDE）并分析耦合的ODE-PDE级联来实现这一结果。为了突出此类误差界的可行性，我们在一类逼近器——神经算子——上展示了其充分性，并在移动机器人实验中通过理论和仿真验证了满足此类通用误差界的条件。 

---
# SLAM-Former: Putting SLAM into One Transformer 

**Title (ZH)**: SLAM-Former: 将SLAM融入一个变压器 

**Authors**: Yijun Yuan, Zhuoguang Chen, Kenan Li, Weibang Wang, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.16909)  

**Abstract**: We present SLAM-Former, a novel neural approach that integrates full SLAM capabilities into a single transformer. Similar to traditional SLAM systems, SLAM-Former comprises both a frontend and a backend that operate in tandem. The frontend processes sequential monocular images in real-time for incremental mapping and tracking, while the backend performs global refinement to ensure a geometrically consistent result. This alternating execution allows the frontend and backend to mutually promote one another, enhancing overall system performance. Comprehensive experimental results demonstrate that SLAM-Former achieves superior or highly competitive performance compared to state-of-the-art dense SLAM methods. 

**Abstract (ZH)**: SLAM-Former：一种将完整SLAM能力集成到单个Transformer中的新型神经方法 

---
# L2M-Reg: Building-level Uncertainty-aware Registration of Outdoor LiDAR Point Clouds and Semantic 3D City Models 

**Title (ZH)**: 基于L2M-正则化的建筑物级不确定性强景注册和语义三维城市模型构建 

**Authors**: Ziyang Xu, Benedikt Schwab, Yihui Yang, Thomas H. Kolbe, Christoph Holst  

**Link**: [PDF](https://arxiv.org/pdf/2509.16832)  

**Abstract**: Accurate registration between LiDAR (Light Detection and Ranging) point clouds and semantic 3D city models is a fundamental topic in urban digital twinning and a prerequisite for downstream tasks, such as digital construction, change detection and model refinement. However, achieving accurate LiDAR-to-Model registration at individual building level remains challenging, particularly due to the generalization uncertainty in semantic 3D city models at the Level of Detail 2 (LoD2). This paper addresses this gap by proposing L2M-Reg, a plane-based fine registration method that explicitly accounts for model uncertainty. L2M-Reg consists of three key steps: establishing reliable plane correspondence, building a pseudo-plane-constrained Gauss-Helmert model, and adaptively estimating vertical translation. Experiments on three real-world datasets demonstrate that L2M-Reg is both more accurate and computationally efficient than existing ICP-based and plane-based methods. Overall, L2M-Reg provides a novel building-level solution regarding LiDAR-to-Model registration when model uncertainty is present. 

**Abstract (ZH)**: 基于平面的精细注册方法L2M-Reg在模型不确定情况下的LiDAR与_semantic 3D城市模型之间准确定位 

---
# A Regularized Riccati Recursion for Interior-Point Optimal Control 

**Title (ZH)**: 正则化的里卡提递归算法在内部点最优控制中的应用 

**Authors**: João Sousa-Pinto, Dominique Orban  

**Link**: [PDF](https://arxiv.org/pdf/2509.16370)  

**Abstract**: We derive a closed-form extension of Riccati's recursion for solving regularized LQR problems. We also show how this can be used to solve general constrained, non-convex, discrete-time optimal control problems via a regularized interior point method, while guaranteeing that each step is a descent direction of an Augmented Barrier-Lagrangian merit function. We also provide MIT-licensed implementations of our method in C++ and JAX. 

**Abstract (ZH)**: 我们推导出了求解正则化LQR问题的Riccati递推的闭式扩展。同时，我们展示了如何通过正则化的内点法利用这一扩展来求解一般约束的非凸离散时间最优控制问题，并保证每一步都是增强屏障-拉格朗日歧函数下降方向。我们还在C++和JAX中提供了该方法的MIT许可实现。 

---
# On the Variational Costs of Changing Our Minds 

**Title (ZH)**: 关于改变我们想法的变分成本 

**Authors**: David Hyland, Mahault Albarracin  

**Link**: [PDF](https://arxiv.org/pdf/2509.17957)  

**Abstract**: The human mind is capable of extraordinary achievements, yet it often appears to work against itself. It actively defends its cherished beliefs even in the face of contradictory evidence, conveniently interprets information to conform to desired narratives, and selectively searches for or avoids information to suit its various purposes. Despite these behaviours deviating from common normative standards for belief updating, we argue that such 'biases' are not inherently cognitive flaws, but rather an adaptive response to the significant pragmatic and cognitive costs associated with revising one's beliefs. This paper introduces a formal framework that aims to model the influence of these costs on our belief updating mechanisms.
We treat belief updating as a motivated variational decision, where agents weigh the perceived 'utility' of a belief against the informational cost required to adopt a new belief state, quantified by the Kullback-Leibler divergence from the prior to the variational posterior. We perform computational experiments to demonstrate that simple instantiations of this resource-rational model can be used to qualitatively emulate commonplace human behaviours, including confirmation bias and attitude polarisation. In doing so, we suggest that this framework makes steps toward a more holistic account of the motivated Bayesian mechanics of belief change and provides practical insights for predicting, compensating for, and correcting deviations from desired belief updating processes. 

**Abstract (ZH)**: 人类思维能够取得非凡成就，但往往自我矛盾。它积极捍卫自身珍视的信念，即使面对矛盾的证据；方便地解释信息以符合期望的故事；并且根据不同的目的有选择地搜索或避免信息。尽管这些行为偏离了常见的规范性信念更新标准，我们认为这些所谓的“偏差”并非认知缺陷，而是对信念修订带来的重大实用和认知成本的一种适应性响应。本文引入了一个形式框架，旨在模型化这些成本对我们信念更新机制的影响。

我们将信念更新视为一种有动机的变分决策，其中代理权衡信念的“效用”感知与其采用新信念状态所需的信息成本之间的关系，后者通过从先验到变分后验的Kullback-Leibler发散度来量化。我们通过计算实验展示，这种资源理性模型的简单实例可以用于定性模拟常见的认知偏差，包括确认偏差和态度极化。在此过程中，我们建议该框架朝着对信念变化的动机贝叶斯机理提供更全面的解释迈进，并为预测、补偿和纠正理想的信念更新过程中的偏差提供了实际见解。 

---
# "I think this is fair'': Uncovering the Complexities of Stakeholder Decision-Making in AI Fairness Assessment 

**Title (ZH)**: “我认为这已经是公平的”：探索人工智能公平性评估中利益相关者决策的复杂性 

**Authors**: Lin Luo, Yuri Nakao, Mathieu Chollet, Hiroya Inakoshi, Simone Stumpf  

**Link**: [PDF](https://arxiv.org/pdf/2509.17956)  

**Abstract**: Assessing fairness in artificial intelligence (AI) typically involves AI experts who select protected features, fairness metrics, and set fairness thresholds. However, little is known about how stakeholders, particularly those affected by AI outcomes but lacking AI expertise, assess fairness. To address this gap, we conducted a qualitative study with 30 stakeholders without AI expertise, representing potential decision subjects in a credit rating scenario, to examine how they assess fairness when placed in the role of deciding on features with priority, metrics, and thresholds. We reveal that stakeholders' fairness decisions are more complex than typical AI expert practices: they considered features far beyond legally protected features, tailored metrics for specific contexts, set diverse yet stricter fairness thresholds, and even preferred designing customized fairness. Our results extend the understanding of how stakeholders can meaningfully contribute to AI fairness governance and mitigation, underscoring the importance of incorporating stakeholders' nuanced fairness judgments. 

**Abstract (ZH)**: 评估人工智能中的公平性通常涉及AI专家选择保护特征、公平性指标并设定公平性阈值。然而，对于受到AI结果影响但缺乏AI专业知识的利益相关者来说，他们如何评估公平性知之甚少。为了弥补这一差距，我们对30名缺乏AI专业知识的利益相关者进行了质性研究，他们代表了信用评级场景下的潜在决策主体，探讨了他们在决定优先特征、指标和阈值角色时如何评估公平性。研究发现，利益相关者的公平性决策比典型的AI专家做法更为复杂：他们考虑的特征远超法律保护的范畴，针对特定情境定制指标，设定多种且更为严格的公平性阈值，甚至偏好设计定制化的公平性。研究结果扩展了对利益相关者如何能够实质性地参与AI公平性治理和缓解的理解，强调了将利益相关者精细的公平性判断纳入考量的重要性。 

---
# MEF: A Systematic Evaluation Framework for Text-to-Image Models 

**Title (ZH)**: MEF：文本到图像模型系统的评估框架 

**Authors**: Xiaojing Dong, Weilin Huang, Liang Li, Yiying Li, Shu Liu, Tongtong Ou, Shuang Ouyang, Yu Tian, Fengxuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17907)  

**Abstract**: Rapid advances in text-to-image (T2I) generation have raised higher requirements for evaluation methodologies. Existing benchmarks center on objective capabilities and dimensions, but lack an application-scenario perspective, limiting external validity. Moreover, current evaluations typically rely on either ELO for overall ranking or MOS for dimension-specific scoring, yet both methods have inherent shortcomings and limited interpretability. Therefore, we introduce the Magic Evaluation Framework (MEF), a systematic and practical approach for evaluating T2I models. First, we propose a structured taxonomy encompassing user scenarios, elements, element compositions, and text expression forms to construct the Magic-Bench-377, which supports label-level assessment and ensures a balanced coverage of both user scenarios and capabilities. On this basis, we combine ELO and dimension-specific MOS to generate model rankings and fine-grained assessments respectively. This joint evaluation method further enables us to quantitatively analyze the contribution of each dimension to user satisfaction using multivariate logistic regression. By applying MEF to current T2I models, we obtain a leaderboard and key characteristics of the leading models. We release our evaluation framework and make Magic-Bench-377 fully open-source to advance research in the evaluation of visual generative models. 

**Abstract (ZH)**: 深度伪造生成模型的魔力评估框架：Magic Evaluation Framework (MEF)及其实现方法 

---
# Efficient & Correct Predictive Equivalence for Decision Trees 

**Title (ZH)**: 高效且正确的决策树预测等价性验证 

**Authors**: Joao Marques-Silva, Alexey Ignatiev  

**Link**: [PDF](https://arxiv.org/pdf/2509.17774)  

**Abstract**: The Rashomon set of decision trees (DTs) finds importance uses. Recent work showed that DTs computing the same classification function, i.e. predictive equivalent DTs, can represent a significant fraction of the Rashomon set. Such redundancy is undesirable. For example, feature importance based on the Rashomon set becomes inaccurate due the existence of predictive equivalent DTs, i.e. DTs with the same prediction for every possible input. In recent work, McTavish et al. proposed solutions for several computational problems related with DTs, including that of deciding predictive equivalent DTs. This approach, which this paper refers to as MBDSR, consists of applying the well-known method of Quine-McCluskey (QM) for obtaining minimum-size DNF (disjunctive normal form) representations of DTs, which are then used for comparing DTs for predictive equivalence. Furthermore, the minimum-size DNF representation was also applied to computing explanations for the predictions made by DTs, and to finding predictions in the presence of missing data. However, the problem of formula minimization is hard for the second level of the polynomial hierarchy, and the QM method may exhibit worst-case exponential running time and space. This paper first demonstrates that there exist decision trees that trigger the worst-case exponential running time and space of the QM method. Second, the paper shows that the MBDSR approach can produce incorrect results for the problem of deciding predictive equivalence. Third, the paper shows that any of the problems to which the minimum-size DNF representation has been applied to can in fact be solved in polynomial time, in the size of the DT. The experiments confirm that, for DTs for which the the worst-case of the QM method is triggered, the algorithms proposed in this paper are orders of magnitude faster than the ones proposed by McTavish et al. 

**Abstract (ZH)**: 决策树的Rashomon集及其应用：识别预测等价决策树的新方法 

---
# Virtual Arc Consistency for Linear Constraints inCost Function Networks 

**Title (ZH)**: 成本函数网络中线性约束的虚拟弧一致性 

**Authors**: Pierre Montalbano, Simon de Givry, George Katsirelos  

**Link**: [PDF](https://arxiv.org/pdf/2509.17706)  

**Abstract**: In Constraint Programming, solving discrete minimization problems with hard and soft constraints can be done either using (i) soft global constraints, (ii) a reformulation into a linear program, or (iii) a reformulation into local cost functions. Approach (i) benefits from a vast catalog of constraints. Each soft constraint propagator communicates with other soft constraints only through the variable domains, resulting in weak lower bounds. Conversely, the approach (ii) provides a global view with strong bounds, but the size of the reformulation can be problematic. We focus on approach (iii) in which soft arc consistency (SAC) algorithms produce bounds of intermediate quality. Recently, the introduction of linear constraints as local cost functions increases their modeling expressiveness. We adapt an existing SAC algorithm to handle linear constraints. We show that our algorithm significantly improves the lower bounds compared to the original algorithm on several benchmarks, reducing solving time in some cases. 

**Abstract (ZH)**: 在约束编程中，通过软全局约束、线性规划改革述或局部成本函数改革述解决带有硬约束和软约束的离散最小化问题，可以采用三种方法。第三种方法中，软弧一致性(SAC)算法产生中等质量的边界。最近，将线性约束作为局部成本函数的引入增强了其建模表达能力。我们调整了现有的SAC算法以处理线性约束。我们的算法在多个基准测试中显著提高了下界，并在某些情况下减少了求解时间。 

---
# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models 

**Title (ZH)**: 表2Latex-RL：通过强化多模态语言模型从表格图像生成高保真LaTeX代码 

**Authors**: Jun Ling, Yao Qi, Tao Huang, Shibo Zhou, Yanqin Huang, Jiang Yang, Ziqi Song, Ying Zhou, Yang Yang, Heng Tao Shen, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17589)  

**Abstract**: In this work, we address the task of table image to LaTeX code generation, with the goal of automating the reconstruction of high-quality, publication-ready tables from visual inputs. A central challenge of this task lies in accurately handling complex tables -- those with large sizes, deeply nested structures, and semantically rich or irregular cell content -- where existing methods often fail. We begin with a comprehensive analysis, identifying key challenges and highlighting the limitations of current evaluation protocols. To overcome these issues, we propose a reinforced multimodal large language model (MLLM) framework, where a pre-trained MLLM is fine-tuned on a large-scale table-to-LaTeX dataset. To further improve generation quality, we introduce a dual-reward reinforcement learning strategy based on Group Relative Policy Optimization (GRPO). Unlike standard approaches that optimize purely over text outputs, our method incorporates both a structure-level reward on LaTeX code and a visual fidelity reward computed from rendered outputs, enabling direct optimization of the visual output quality. We adopt a hybrid evaluation protocol combining TEDS-Structure and CW-SSIM, and show that our method achieves state-of-the-art performance, particularly on structurally complex tables, demonstrating the effectiveness and robustness of our approach. 

**Abstract (ZH)**: 本研究针对表格图像到LaTeX代码生成的任务，旨在自动化高质量、出版级表格的重建。该任务的核心挑战在于准确处理复杂表格——这些表格具有大尺寸、深层嵌套结构和语义丰富或不规则的单元格内容——现有方法在这些情况下往往表现不佳。我们从全面分析出发，识别关键挑战并强调当前评估协议的局限性。为解决这些问题，我们提出了一种强化多模态大型语言模型（MLLM）框架，其中预训练的MLLM在大规模表格到LaTeX数据集上进行微调。为提高生成质量，我们引入了一种基于Group Relative Policy Optimization (GRPO)的双奖励强化学习策略。不同于仅优化文本输出的标准方法，我们的方法在LaTeX代码的结构层面上和根据渲染输出计算的视觉保真度层面上都提供奖励，从而直接优化视觉输出质量。我们采用结合TEDS-Structure和CW-SSIM的混合评估协议，并展示了我们的方法在结构复杂表格上的最佳性能，证明了我们方法的有效性和鲁棒性。 

---
# Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem 

**Title (ZH)**: 它是绝对的Deepfake吗？检测与生成生态系统中的可靠性分析 

**Authors**: Neslihan Kose, Anthony Rhodes, Umur Aybars Ciftci, Ilke Demir  

**Link**: [PDF](https://arxiv.org/pdf/2509.17550)  

**Abstract**: As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection. 

**Abstract (ZH)**: 基于生成模型的合成内容不确定性分析：从检测器到生成器的系统研究 

---
# SPICED: A Synaptic Homeostasis-Inspired Framework for Unsupervised Continual EEG Decoding 

**Title (ZH)**: SPICED: 一种受突触稳态启发的无监督持续EEG解码框架 

**Authors**: Yangxuan Zhou, Sha Zhao, Jiquan Wang, Haiteng Jiang, Shijian Li, Tao Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17439)  

**Abstract**: Human brain achieves dynamic stability-plasticity balance through synaptic homeostasis. Inspired by this biological principle, we propose SPICED: a neuromorphic framework that integrates the synaptic homeostasis mechanism for unsupervised continual EEG decoding, particularly addressing practical scenarios where new individuals with inter-individual variability emerge continually. SPICED comprises a novel synaptic network that enables dynamic expansion during continual adaptation through three bio-inspired neural mechanisms: (1) critical memory reactivation; (2) synaptic consolidation and (3) synaptic renormalization. The interplay within synaptic homeostasis dynamically strengthens task-discriminative memory traces and weakens detrimental memories. By integrating these mechanisms with continual learning system, SPICED preferentially replays task-discriminative memory traces that exhibit strong associations with newly emerging individuals, thereby achieving robust adaptations. Meanwhile, SPICED effectively mitigates catastrophic forgetting by suppressing the replay prioritization of detrimental memories during long-term continual learning. Validated on three EEG datasets, SPICED show its effectiveness. 

**Abstract (ZH)**: 人类大脑通过突触稳态实现动态稳定-可塑性平衡。受此生物原理启发，我们提出了SPICED：一种结合突触稳态机制以实现无监督持续EEG解码的神经形态框架，特别适用于持续涌现具有个体差异的新个体的实际场景。SPICED包含一种新型突触网络，通过三种生物启发式的神经机制实现持续适应过程中的动态扩展：(1) 关键记忆重新激活；(2) 突触巩固；(3) 突触重规范化。突触稳态的相互作用动态加强任务区分性记忆痕迹并减弱不利记忆。通过将这些机制与持续学习系统集成，SPICED优先回放与新涌现个体有强关联的任务区分性记忆痕迹，从而实现稳健的适应。同时，SPICED有效减轻长期持续学习中的灾难性遗忘，通过抑制不利记忆的回放优先级实现这一目标。SPICED在三个EEG数据集中验证了其有效性。 

---
# Mind the Gap: Comparing Model- vs Agentic-Level Red Teaming with Action-Graph Observability on GPT-OSS-20B 

**Title (ZH)**: 注意差距：基于行动图可观测性比较模型级 vs 代理级红队演练 

**Authors**: Ilham Wicaksono, Zekun Wu, Rahul Patel, Theo King, Adriano Koshiyama, Philip Treleaven  

**Link**: [PDF](https://arxiv.org/pdf/2509.17259)  

**Abstract**: As the industry increasingly adopts agentic AI systems, understanding their unique vulnerabilities becomes critical. Prior research suggests that security flaws at the model level do not fully capture the risks present in agentic deployments, where models interact with tools and external environments. This paper investigates this gap by conducting a comparative red teaming analysis of GPT-OSS-20B, a 20-billion parameter open-source model. Using our observability framework AgentSeer to deconstruct agentic systems into granular actions and components, we apply iterative red teaming attacks with harmful objectives from HarmBench at two distinct levels: the standalone model and the model operating within an agentic loop. Our evaluation reveals fundamental differences between model level and agentic level vulnerability profiles. Critically, we discover the existence of agentic-only vulnerabilities, attack vectors that emerge exclusively within agentic execution contexts while remaining inert against standalone models. Agentic level iterative attacks successfully compromise objectives that completely failed at the model level, with tool-calling contexts showing 24\% higher vulnerability than non-tool contexts. Conversely, certain model-specific exploits work exclusively at the model level and fail when transferred to agentic contexts, demonstrating that standalone model vulnerabilities do not always generalize to deployed systems. 

**Abstract (ZH)**: 随着行业越来越多地采用代理型AI系统，理解其独特的脆弱性变得至关重要。本研究通过比较红队分析方法，研究GPT-OSS-20B（一个开源的200亿参数模型）在代理型部署中的漏洞差异，揭示了模型级与代理型级别漏洞特征的根本不同。研究发现，存在仅在代理型执行环境中出现的代理型专用漏洞，这些攻击向量在独立模型中无法生效。在代理型循环中的迭代攻击能够成功突破在独立模型中完全失败的目标，工具调用上下文的漏洞比非工具上下文高24%。同时，某些模型特定的利用方式仅在模型级别有效，在转移到代理型环境中时失败，这表明独立模型的漏洞并不总是能够推广到部署系统中。 

---
# Governing Automated Strategic Intelligence 

**Title (ZH)**: 自动化战略智能治理 

**Authors**: Nicholas Kruus, Madhavendra Thakur, Adam Khoja, Leonhard Nagel, Maximilian Nicholson, Abeer Sharma, Jason Hausenloy, Alberto KoTafoya, Aliya Mukhanova, Alli Katila-Miikkulainen, Harish Chandran, Ivan Zhang, Jessie Chen, Joel Raj, Jord Nguyen, Lai Hsien Hao, Neja Jayasundara, Soham Sen, Sophie Zhang, Ashley Dora Kokui Tamaklo, Bhavya Thakur, Henry Close, Janghee Lee, Nina Sefton, Raghavendra Thakur, Shiv Munagala, Yeeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17087)  

**Abstract**: Military and economic strategic competitiveness between nation-states will increasingly be defined by the capability and cost of their frontier artificial intelligence models. Among the first areas of geopolitical advantage granted by such systems will be in automating military intelligence. Much discussion has been devoted to AI systems enabling new military modalities, such as lethal autonomous weapons, or making strategic decisions. However, the ability of a country of "CIA analysts in a data-center" to synthesize diverse data at scale, and its implications, have been underexplored. Multimodal foundation models appear on track to automate strategic analysis previously done by humans. They will be able to fuse today's abundant satellite imagery, phone-location traces, social media records, and written documents into a single queryable system. We conduct a preliminary uplift study to empirically evaluate these capabilities, then propose a taxonomy of the kinds of ground truth questions these systems will answer, present a high-level model of the determinants of this system's AI capabilities, and provide recommendations for nation-states to remain strategically competitive within the new paradigm of automated intelligence. 

**Abstract (ZH)**: 国家之间军事和经济的战略竞争力将越来越多地由其前沿人工智能模型的能力和成本来定义。这类系统的首批地缘政治优势之一将是自动化军事情报。尽管已经对AI系统如何使军事新模式成为可能或进行战略决策进行了大量讨论，但“数据中心中的CIA分析师”如何规模化综合多种数据及其影响尚未得到充分探讨。多模态基础模型似乎正朝着自动化之前由人类完成的战略分析迈进。它们将能够将当今丰富的卫星 imagery、手机位置轨迹、社交媒体记录和书面文件融合到单一可查询系统中。我们进行初步提升研究，以实证评估这些能力，然后提出这些系统将回答的真理型问题分类，概述决定该系统人工智能能力的主要因素，并为国家在新的自动化智能范式中保持战略竞争力提供建议。 

---
# Intention-aware Hierarchical Diffusion Model for Long-term Trajectory Anomaly Detection 

**Title (ZH)**: 意图感知分层扩散模型在长期轨迹异常检测中的应用 

**Authors**: Chen Wang, Sarah Erfani, Tansu Alpcan, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2509.17068)  

**Abstract**: Long-term trajectory anomaly detection is a challenging problem due to the diversity and complex spatiotemporal dependencies in trajectory data. Existing trajectory anomaly detection methods fail to simultaneously consider both the high-level intentions of agents as well as the low-level details of the agent's navigation when analysing an agent's trajectories. This limits their ability to capture the full diversity of normal trajectories. In this paper, we propose an unsupervised trajectory anomaly detection method named Intention-aware Hierarchical Diffusion model (IHiD), which detects anomalies through both high-level intent evaluation and low-level sub-trajectory analysis. Our approach leverages Inverse Q Learning as the high-level model to assess whether a selected subgoal aligns with an agent's intention based on predicted Q-values. Meanwhile, a diffusion model serves as the low-level model to generate sub-trajectories conditioned on subgoal information, with anomaly detection based on reconstruction error. By integrating both models, IHiD effectively utilises subgoal transition knowledge and is designed to capture the diverse distribution of normal trajectories. Our experiments show that the proposed method IHiD achieves up to 30.2% improvement in anomaly detection performance in terms of F1 score over state-of-the-art baselines. 

**Abstract (ZH)**: 基于意图的层次扩散模型在长期轨迹异常检测中的应用 

---
# From domain-landmark graph learning to problem-landmark graph generation 

**Title (ZH)**: 从领域地标图学习到问题地标图生成 

**Authors**: Cristian Pérez-Corral, Antonio Garrido, Laura Sebastia  

**Link**: [PDF](https://arxiv.org/pdf/2509.17062)  

**Abstract**: Landmarks have long played a pivotal role in automated planning, serving as crucial elements for improving the planning algorithms. The main limitation of classical landmark extraction methods is their sensitivity to specific planning tasks. This results in landmarks fully tailored to individual instances, thereby limiting their applicability across other instances of the same planning domain. We propose a novel approach that learns landmark relationships from multiple planning tasks of a planning domain. This leads to the creation of a \textit{probabilistic lifted ordering graph}, as a structure that captures weighted abstractions of relationships between parameterized landmarks. Although these orderings are not 100\% true (they are probabilistic), they can still be very useful in planning. Next, given a new planning task for that domain, we instantiate the relationships from that graph to this particular instance. This instantiation operates in two phases. First, it generates two graphs: the former instantiating information from the initial state and the latter from the goal state. Second, it combines these two graphs into one unified graph by searching equivalences to extract landmark orderings. We evaluate the precision and recallof the information found by our approach over well-known planning domains. 

**Abstract (ZH)**: 基于多个规划任务学习概率提升排序图的地标关系 

---
# KAHAN: Knowledge-Augmented Hierarchical Analysis and Narration for Financial Data Narration 

**Title (ZH)**: KAHAN: 知识增强的层次化分析与叙述方法在财务数据叙述中的应用 

**Authors**: Yajing Yang, Tony Deng, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17037)  

**Abstract**: We propose KAHAN, a knowledge-augmented hierarchical framework that systematically extracts insights from raw tabular data at entity, pairwise, group, and system levels. KAHAN uniquely leverages LLMs as domain experts to drive the analysis. On DataTales financial reporting benchmark, KAHAN outperforms existing approaches by over 20% on narrative quality (GPT-4o), maintains 98.2% factuality, and demonstrates practical utility in human evaluation. Our results reveal that knowledge quality drives model performance through distillation, hierarchical analysis benefits vary with market complexity, and the framework transfers effectively to healthcare domains. The data and code are available at this https URL. 

**Abstract (ZH)**: 我们提出了一种知识增强的层次框架KAHAN，该框架系统地从实体、成对、组及系统层面的原始表格数据中提取洞察。KAHAN独特地利用大语言模型作为领域专家来驱动分析。在DataTales财务报告基准测试中，KAHAN在叙事质量上超过现有方法20%以上（GPT-4o），事实准确性保持在98.2%，并在人类评估中展示了实用价值。我们的研究结果表明，知识质量通过萃取过程影响模型性能，层次分析的优势随市场复杂度而变化，并且该框架在医疗健康领域中具有有效的转移能力。更多信息请参见此链接。 

---
# Quantum Abduction: A New Paradigm for Reasoning under Uncertainty 

**Title (ZH)**: 量子归约：一种新的不确定性推理范式 

**Authors**: Remo Pareschi  

**Link**: [PDF](https://arxiv.org/pdf/2509.16958)  

**Abstract**: Abductive reasoning - the search for plausible explanations - has long been central to human inquiry, from forensics to medicine and scientific discovery. Yet formal approaches in AI have largely reduced abduction to eliminative search: hypotheses are treated as mutually exclusive, evaluated against consistency constraints or probability updates, and pruned until a single "best" explanation remains. This reductionist framing overlooks the way human reasoners sustain multiple explanatory lines in suspension, navigate contradictions, and generate novel syntheses. This paper introduces quantum abduction, a non-classical paradigm that models hypotheses in superposition, allows them to interfere constructively or destructively, and collapses only when coherence with evidence is reached. Grounded in quantum cognition and implemented with modern NLP embeddings and generative AI, the framework supports dynamic synthesis rather than premature elimination. Case studies span historical mysteries (Ludwig II of Bavaria, the "Monster of Florence"), literary demonstrations ("Murder on the Orient Express"), medical diagnosis, and scientific theory change. Across these domains, quantum abduction proves more faithful to the constructive and multifaceted nature of human reasoning, while offering a pathway toward expressive and transparent AI reasoning systems. 

**Abstract (ZH)**: abduction推理——寻找合理的解释——长期以来一直是人类探究的核心，从法医学到医学和科学发现。然而，AI中的形式方法主要将 abduction 化约为排除性搜索：假设被视为互斥的，根据一致性约束或概率更新进行评估，并修剪直到剩下单一的“最佳”解释。这种还原主义框架忽视了人类推理者在悬置中维持多个解释线并在矛盾中导航以及生成新颖综合的方式。本文介绍了量子 abduction，这是一种非经典范式，它以叠加状态模型假设，使假设能够相互 constructive 或 destructive 干涉，并仅在与证据的共融性达到一致时才塌缩。该框架基于量子认知，并借助现代 NLP 向量表示和生成 AI 实现，支持动态综合而非过早排除。案例研究涵盖历史谜团（巴伐利亚国王路德维希二世、“佛罗伦萨怪物”）、文学示例（《东方快车谋杀案》）、医学诊断和科学理论变革。在这些领域中，量子 abduction 更加忠实地反映了人类推理的建设性和多面性，同时为表达性和透明的 AI 推理系统提供了途径。 

---
# Checking extracted rules in Neural Networks 

**Title (ZH)**: 检查神经网络中提取的规则 

**Authors**: Adrian Wurm  

**Link**: [PDF](https://arxiv.org/pdf/2509.16547)  

**Abstract**: In this paper we investigate formal verification of extracted rules for Neural Networks under a complexity theoretic point of view. A rule is a global property or a pattern concerning a large portion of the input space of a network. These rules are algorithmically extracted from networks in an effort to better understand their inner way of working. Here, three problems will be in the focus: Does a given set of rules apply to a given network? Is a given set of rules consistent or do the rules contradict themselves? Is a given set of rules exhaustive in the sense that for every input the output is determined? Finding algorithms that extract such rules out of networks has been investigated over the last 30 years, however, to the author's current knowledge, no attempt in verification was made until now. A lot of attempts of extracting rules use heuristics involving randomness and over-approximation, so it might be beneficial to know whether knowledge obtained in that way can actually be trusted.
We investigate the above questions for neural networks with ReLU-activation as well as for Boolean networks, each for several types of rules. We demonstrate how these problems can be reduced to each other and show that most of them are co-NP-complete. 

**Abstract (ZH)**: 本文从复杂性理论的角度研究了神经网络中提取规则的形式验证。我们关注三个问题：一组给定的规则是否适用于某个网络？一组给定的规则是否一致或相互矛盾？一组给定的规则是否详尽，即对每一个输入其输出都能确定？过去30年里，从网络中提取此类规则的算法已被研究，但据作者所知，至今仍未有人尝试验证这些规则。许多提取规则的方法使用涉及随机性和过度近似的启发式算法，因此了解通过这种方式获得的知识是否可靠是有益的。我们研究了ReLU激活函数神经网络和布尔网络中不同类型规则的问题，并展示了这些问题之间的相互归约，证明它们大多数是co-NP完全问题。 

---
# Proactive Statistical Process Control Using AI: A Time Series Forecasting Approach for Semiconductor Manufacturing 

**Title (ZH)**: 基于时间序列预测的 proactive 统计过程控制：半导体制造中的 AI 方法 

**Authors**: Mohammad Iqbal Rasul Seeam, Victor S. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.16431)  

**Abstract**: In the manufacturing industry, it is very important to keep machines and processes running smoothly and without unexpected problems. One of the most common tools used to check if everything is working properly is called Statistical Process Control (SPC). Traditional SPC methods work by checking whether recent measurements are within acceptable limits. However, they only react after a problem has already occurred. This can lead to wasted materials, machine downtime, and increased costs. In this paper, we present a smarter way to use SPC. Instead of just reacting to issues after they happen, our system can predict future problems before they occur. We use a machine learning tool called Facebook Prophet, which is designed to work with time-series data (data that changes over time). Prophet looks at past data and forecasts what the next value will be. Then, we use SPC rules to decide if the predicted value is in a Safe zone (no problem), a Warning zone (needs attention), or a Critical zone (may require shutting down the process). We applied this system to real data from a semiconductor manufacturing company. One of the challenges with this data is that the measurements are not taken at regular time intervals. This makes it harder to predict future values accurately. Despite this, our model was able to make strong predictions and correctly classify the risk level of future measurements. The main benefit of our system is that it gives engineers and technicians a chance to act early - before something goes wrong. This helps reduce unexpected failures and improves the overall stability and reliability of the production process. By combining machine learning with traditional SPC, we make quality control more proactive, accurate, and useful for modern industry. 

**Abstract (ZH)**: 在制造业中，确保机器和流程平稳运行且无意外问题至关重要。最常用的一种检查是否一切正常的方法被称为统计过程控制（SPC）。传统SPC方法通过检查最近的测量值是否在可接受的范围内来工作，但它们只能在问题发生后才作出反应。这可能导致材料浪费、机器停机时间和成本增加。在这篇文章中，我们提出了一种更智能的使用SPC的方法。我们的系统不仅能对已经发生的问题作出反应，还能预测未来的问题。我们使用了一种名为Facebook Prophet的机器学习工具，该工具适用于时间序列数据（随时间变化的数据）。Prophet会分析过去的数据并预测下一个值，然后，我们使用SPC规则来决定预测值是否处于安全区（无问题）、警告区（需关注）或危急区（可能需要停机处理）。我们把这个系统应用到了一家半导体制造公司的实际数据中。数据的一个挑战是测量值并非在固定的时间间隔内进行。这使得准确预测未来值变得更加困难。尽管如此，我们的模型依然能够做出强大预测，并正确分类未来测量的风险级别。我们系统的主要优点在于它给工程师和技术人员提供了提前行动的机会——在问题发生之前。这有助于减少意外故障，提高生产过程的整体稳定性和可靠性。通过将机器学习与传统SPC相结合，我们使质量控制更加主动、准确且适用于现代工业。 

---
# A Unified AI Approach for Continuous Monitoring of Human Health and Diseases from Intensive Care Unit to Home with Physiological Foundation Models (UNIPHY+) 

**Title (ZH)**: 基于生理基础模型的统一人工智能方法，从重症监护室到居家的持续人体健康与疾病监测（UNIPHY+） 

**Authors**: Minxiao Wang, Saurabh Kataria, Juntong Ni, Timothy G. Buchman, Jocelyn Grunwell, Mark Mai, Wei Jin, Matthew Clark, Stephanie Brown, Michael Fundora, Puneet Sharma, Tony Pan, Sam Khan, Timothy Ruchti, Naveen Muthu, Kevin Maher, Sivasubramanium V Bhavani, Xiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16348)  

**Abstract**: We present UNIPHY+, a unified physiological foundation model (physioFM) framework designed to enable continuous human health and diseases monitoring across care settings using ubiquitously obtainable physiological data. We propose novel strategies for incorporating contextual information during pretraining, fine-tuning, and lightweight model personalization via multi-modal learning, feature fusion-tuning, and knowledge distillation. We advocate testing UNIPHY+ with a broad set of use cases from intensive care to ambulatory monitoring in order to demonstrate that UNIPHY+ can empower generalizable, scalable, and personalized physiological AI to support both clinical decision-making and long-term health monitoring. 

**Abstract (ZH)**: UNIPHY+: 统一生理基础模型框架及其在多种临床和随访监测场景中的应用 

---
# On the Non-Uniqueness of Representation of $(U,N)$-Implications 

**Title (ZH)**: $(U,N)$-蕴涵表示的非唯一性 

**Authors**: Raquel Fernandez-Peralta, Andrea Mesiarová-Zemánková  

**Link**: [PDF](https://arxiv.org/pdf/2509.16299)  

**Abstract**: Fuzzy implication functions constitute fundamental operators in fuzzy logic systems, extending classical conditionals to manage uncertainty in logical inference. Among the extensive families of these operators, generalizations of the classical material implication have received considerable theoretical attention, particularly $(S,N)$-implications constructed from t-conorms and fuzzy negations, and their further generalizations to $(U,N)$-implications using disjunctive uninorms. Prior work has established characterization theorems for these families under the assumption that the fuzzy negation $N$ is continuous, ensuring uniqueness of representation. In this paper, we disprove this last fact for $(U,N)$-implications and we show that they do not necessarily possess a unique representation, even if the fuzzy negation is continuous. Further, we provide a comprehensive study of uniqueness conditions for both uninorms with continuous and non-continuous underlying functions. Our results offer important theoretical insights into the structural properties of these operators. 

**Abstract (ZH)**: 模糊蕴含函数是模糊逻辑系统中的基本运算符，扩展了经典条件命题以处理逻辑推理中的不确定性。在这众多运算符家族中，经典材料蕴含的一般化得到了较多理论关注，特别是由t-余运算子和模糊否定构造的$(S,N)$-蕴含，以及利用析取非结合运算子进一步一般化的$(U,N)$-蕴含。先前的研究在假设模糊否定$N$连续的情况下建立了这些家族的表征定理，确保了表示的唯一性。本文推翻了$(U,N)$-蕴含具有唯一表示形式这一事实，并展示了即使在模糊否定连续的情况下，它们也可能不具备唯一的表示形式。此外，我们对具有连续和非连续基础函数的非结合运算子的唯一性条件进行了全面研究。我们的结果为这些运算子的结构性质提供了重要的理论见解。 

---
# A global view of diverse construction methods of fuzzy implication functions rooted on F-chains 

**Title (ZH)**: 基于F-链的多样模糊蕴涵函数构造方法综述 

**Authors**: Raquel Fernandez-Peralta, Juan Vicente Riera  

**Link**: [PDF](https://arxiv.org/pdf/2509.16298)  

**Abstract**: Fuzzy implication functions are one of the most important operators used in the fuzzy logic framework. While their flexible definition allows for diverse families with distinct properties, this variety needs a deeper theoretical understanding of their structural relationships. In this work, we focus on the study of construction methods, which employ different techniques to generate new fuzzy implication functions from existing ones. Particularly, we generalize the $F$-chain-based construction, recently introduced by Mesiar et al. to extend a method for constructing aggregation functions to the context of fuzzy implication functions. Our generalization employs collections of fuzzy implication functions rather than single ones, and uses two different increasing functions instead of a unique $F$-chain. We analyze property preservation under this construction and establish sufficient conditions. Furthermore, we demonstrate that our generalized $F$-chain-based construction is a unifying framework for several existing methods. In particular, we show that various construction techniques, such as contraposition, aggregation, and generalized vertical/horizontal threshold methods, can be reformulated within our approach. This reveals structural similarities between seemingly distinct construction strategies and provides a cohesive perspective on fuzzy implication construction methods. 

**Abstract (ZH)**: 模糊蕴含函数是模糊逻辑框架中最重要的运算符之一。尽管其灵活的定义允许形成具有不同性质的各种类，但这一多样性需要对其结构关系进行更深入的理论理解。在本文中，我们专注于研究方法的探讨，这些方法利用不同的技术从现有的模糊蕴含函数生成新的模糊蕴含函数。特别地，我们将 Mesiar 等人最近引入的 $F$-链基于构建方法一般化，将其从聚合函数的构建方法扩展到模糊蕴含函数的上下文。我们的推广采用模糊蕴含函数的集合而非单一的模糊蕴含函数，并使用两个不同的递增函数而非唯一的 $F$-链。我们分析了在该构建方法下属性的保持情况，并建立了充分条件。此外，我们展示了我们的广义 $F$-链基于构建方法是一个多个现有方法的统一框架。特别地，我们证明了诸如反置、聚合以及广义垂直/水平阈值方法等各种构建技术可以在我们的方法内重新表述。这揭示了看似不同的构建策略之间的结构相似性，并提供了模糊蕴含函数构建方法的一致视角。 

---
# Identifying Critical Pathways in Coronary Heart Disease via Fuzzy Subgraph Connectivity 

**Title (ZH)**: 冠心病关键路径识别基于模糊子图连通性 

**Authors**: Shanookha Ali, Nitha Niralda P C  

**Link**: [PDF](https://arxiv.org/pdf/2509.16288)  

**Abstract**: Coronary heart disease (CHD) arises from complex interactions among uncontrollable factors, controllable lifestyle factors, and clinical indicators, where relationships are often uncertain. Fuzzy subgraph connectivity (FSC) provides a systematic tool to capture such imprecision by quantifying the strength of association between vertices and subgraphs in fuzzy graphs. In this work, a fuzzy CHD graph is constructed with vertices for uncontrollable, controllable, and indicator components, and edges weighted by fuzzy memberships. Using FSC, we evaluate connectivity to identify strongest diagnostic routes, dominant risk factors, and critical bridges. Results show that FSC highlights influential pathways, bounds connectivity between weakest and strongest correlations, and reveals critical edges whose removal reduces predictive strength. Thus, FSC offers an interpretable and robust framework for modeling uncertainty in CHD risk prediction and supporting clinical decision-making. 

**Abstract (ZH)**: 冠心病（CHD）源于不可控因素、可调控的生活方式因素及其临床指标之间的复杂交互作用，其中关系往往具有不确定性。模糊子图连通性（FSC）提供了一种系统工具，通过量化模糊图中顶点与子图间的关联强度来捕捉这种不确定性。在本研究中，构建了一个模糊CHD图，其中顶点代表不可控、可控和指标组成部分，边权重由模糊隶属度给出。利用FSC，评估连通性以识别最强的诊断路径、主导风险因素和关键桥梁。结果表明，FSC突显了有影响力的途径，界定了最弱和最强相关性之间的连通性上限，并揭示了移除这些边会降低预测强度的关键边。因此，FSC为建模CHD风险预测中的不确定性并支持临床决策提供了一种可解释且稳健的框架。 

---
# TMD-TTS: A Unified Tibetan Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation 

**Title (ZH)**: TMD-TTS： Unified Tibetan Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation 

**Authors**: Yutong Liu, Ziyue Zhang, Ban Ma-bao, Renzeng Duojie, Yuqing Cai, Yongbin Yu, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18060)  

**Abstract**: Tibetan is a low-resource language with limited parallel speech corpora spanning its three major dialects (Ü-Tsang, Amdo, and Kham), limiting progress in speech modeling. To address this issue, we propose TMD-TTS, a unified Tibetan multi-dialect text-to-speech (TTS) framework that synthesizes parallel dialectal speech from explicit dialect labels. Our method features a dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects. Extensive objective and subjective evaluations demonstrate that TMD-TTS significantly outperforms baselines in dialectal expressiveness. We further validate the quality and utility of the synthesized speech through a challenging Speech-to-Speech Dialect Conversion (S2SDC) task. 

**Abstract (ZH)**: 藏语是一种资源匮乏的语言，仅有有限的涵盖其三大方言（ü-tsang、Amdo和Kham）的平行语音语料库，这限制了语音建模的进步。为了解决这个问题，我们提出了TMD-TTS，这是一种统一的多方言藏语文本到语音(TTS)框架，能够从明确的方言标签中合成平行的方言语音。我们的方法包含一个方言融合模块和一种方言特定动态路由网络(DSDR-Net)，以捕捉方言间的细微声学和语言变化。详尽的客观和主观评估表明，TMD-TTS在方言表达能力方面明显优于baseline模型。我们还通过一项具有挑战性的语音到语音方言转换(S2SDC)任务进一步验证了合成语音的质量和实用性。 

---
# Hybrid Reputation Aggregation: A Robust Defense Mechanism for Adversarial Federated Learning in 5G and Edge Network Environments 

**Title (ZH)**: 混合声誉聚合：第五代和边缘网络环境中对抗联邦学习的稳健防御机制 

**Authors**: Saeid Sheikhi, Panos Kostakos, Lauri Loven  

**Link**: [PDF](https://arxiv.org/pdf/2509.18044)  

**Abstract**: Federated Learning (FL) in 5G and edge network environments face severe security threats from adversarial clients. Malicious participants can perform label flipping, inject backdoor triggers, or launch Sybil attacks to corrupt the global model. This paper introduces Hybrid Reputation Aggregation (HRA), a novel robust aggregation mechanism designed to defend against diverse adversarial behaviors in FL without prior knowledge of the attack type. HRA combines geometric anomaly detection with momentum-based reputation tracking of clients. In each round, it detects outlier model updates via distance-based geometric analysis while continuously updating a trust score for each client based on historical behavior. This hybrid approach enables adaptive filtering of suspicious updates and long-term penalization of unreliable clients, countering attacks ranging from backdoor insertions to random noise Byzantine failures. We evaluate HRA on a large-scale proprietary 5G network dataset (3M+ records) and the widely used NF-CSE-CIC-IDS2018 benchmark under diverse adversarial attack scenarios. Experimental results reveal that HRA achieves robust global model accuracy of up to 98.66% on the 5G dataset and 96.60% on NF-CSE-CIC-IDS2018, outperforming state-of-the-art aggregators such as Krum, Trimmed Mean, and Bulyan by significant margins. Our ablation studies further demonstrate that the full hybrid system achieves 98.66% accuracy, while the anomaly-only and reputation-only variants drop to 84.77% and 78.52%, respectively, validating the synergistic value of our dual-mechanism approach. This demonstrates HRA's enhanced resilience and robustness in 5G/edge federated learning deployments, even under significant adversarial conditions. 

**Abstract (ZH)**: 联邦学习（FL）在5G和边缘网络环境中的安全威胁来自恶意客户端，包括标签翻转、后门植入和Sybil攻击。本文介绍了混合声誉聚合（HRA），这是一种新型的鲁棒聚合机制，能够在无需了解攻击类型的情况下防御多样的恶意行为。HRA结合了几何异常检测和基于动量的客户端声誉追踪。在每一轮中，通过基于距离的几何分析检测异常模型更新，同时根据历史行为持续更新每个客户端的信任评分。这种混合方法能够适应性地过滤可疑更新，并长期惩罚不可靠客户端，从而抵御从后门植入到随机噪音拜占庭故障等多种攻击。我们在一个大型 proprietary 5G 网络数据集（3M+ 记录）和广泛使用的 NF-CSE-CIC-IDS2018 基准上，对 HRA 在多种恶意攻击场景下的性能进行了评估。实验结果表明，HRA 在 5G 数据集上的全局模型准确率达到98.66%，在 NF-CSE-CIC-IDS2018 上达到96.60%，显著优于现有的聚合器如 Krum、截断均值和 Bulyan。我们的消融研究进一步证实，完整的混合系统准确率达到98.66%，而仅使用异常检测和仅使用声誉追踪的变体分别降至84.77%和78.52%，证明了双机制方法的协同价值。这表明，在显著的恶意攻击条件下，HRA 在 5G/边缘联邦学习部署中展现出增强的鲁棒性和抗攻击能力。 

---
# Cross-Attention is Half Explanation in Speech-to-Text Models 

**Title (ZH)**: 跨注意力机制在语音转文本模型中占一半解释作用 

**Authors**: Sara Papi, Dennis Fucci, Marco Gaido, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2509.18010)  

**Abstract**: Cross-attention is a core mechanism in encoder-decoder architectures, widespread in many fields, including speech-to-text (S2T) processing. Its scores have been repurposed for various downstream applications--such as timestamp estimation and audio-text alignment--under the assumption that they reflect the dependencies between input speech representation and the generated text. While the explanatory nature of attention mechanisms has been widely debated in the broader NLP literature, this assumption remains largely unexplored within the speech domain. To address this gap, we assess the explanatory power of cross-attention in S2T models by comparing its scores to input saliency maps derived from feature attribution. Our analysis spans monolingual and multilingual, single-task and multi-task models at multiple scales, and shows that attention scores moderately to strongly align with saliency-based explanations, particularly when aggregated across heads and layers. However, it also shows that cross-attention captures only about 50% of the input relevance and, in the best case, only partially reflects how the decoder attends to the encoder's representations--accounting for just 52-75% of the saliency. These findings uncover fundamental limitations in interpreting cross-attention as an explanatory proxy, suggesting that it offers an informative yet incomplete view of the factors driving predictions in S2T models. 

**Abstract (ZH)**: 跨注意力是编码器-解码器架构中的核心机制，广泛应用于包括语音转文本（S2T）处理在内的多个领域。其分数已被重新利用于各种下游应用——如时间戳估计和语音-文本对齐——并在假设它们反映了输入语音表示与生成文本之间的依赖性的前提下。虽然在更广泛的自然语言处理（NLP）文献中对注意力机制的解释性性质已有广泛争论，但在语音领域内这一假设尚未被充分探讨。为填补这一空白，我们通过将跨注意力的分数与源自特征归因的输入显著图进行比较，评估了跨注意力在S2T模型中的解释能力。我们的分析涵盖了单语和多语、单任务和多任务模型，并在多个尺度上进行了，结果显示跨注意力分数与基于显著性的解释适度到强烈地对齐，特别是在跨多个头和层聚合时。然而，研究还表明，跨注意力仅捕捉到输入相关性的大约50%，在最佳情况下，也只能部分反映了解码器如何关注编码器的表示——仅解释了显著性解释的52-75%。这些发现揭示了将跨注意力视为解释性代理的基本局限性，表明它提供了有关驱动S2T模型预测的因素的富有信息但不完全的观点。 

---
# Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise 

**Title (ZH)**: 通过随机梯度噪声的结构揭露m-锐度 

**Authors**: Haocheng Luo, Mehrtash Harandi, Dinh Phung, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2509.18001)  

**Abstract**: Sharpness-aware minimization (SAM) has emerged as a highly effective technique for improving model generalization, but its underlying principles are not fully understood. We investigated the phenomenon known as m-sharpness, where the performance of SAM improves monotonically as the micro-batch size for computing perturbations decreases. Leveraging an extended Stochastic Differential Equation (SDE) framework, combined with an analysis of the structure of stochastic gradient noise (SGN), we precisely characterize the dynamics of various SAM variants. Our findings reveal that the stochastic noise introduced during SAM perturbations inherently induces a variance-based sharpness regularization effect. Motivated by our theoretical insights, we introduce Reweighted SAM, which employs sharpness-weighted sampling to mimic the generalization benefits of m-SAM while remaining parallelizable. Comprehensive experiments validate the effectiveness of our theoretical analysis and proposed method. 

**Abstract (ZH)**: Sharpness-aware minimization (SAM) 已成为提高模型泛化能力的有效技术，但其背后的原理尚未完全明了。我们研究了被称为 m-尖锐度的现象，其中在计算扰动时微批处理大小减小时 SAM 的性能单调提高。借助扩展的随机微分方程（SDE）框架，并结合对随机梯度噪声（SGN）结构的分析，我们精确刻画了各种 SAM 变体的动力学。我们的研究发现，在 SAM 扰动期间引入的随机噪声本质上产生了基于方差的尖锐度正则化效应。受理论洞察的启发，我们引入了重加权 SAM，它通过尖锐度加权抽样来模拟 m-SAM 的泛化益处，同时保持并行化。全面的实验验证了我们理论分析和提出方法的有效性。 

---
# ReDepress: A Cognitive Framework for Detecting Depression Relapse from Social Media 

**Title (ZH)**: ReDepress: 一种从社交媒体检测抑郁复发的认知框架 

**Authors**: Aakash Kumar Agarwal, Saprativa Bhattacharjee, Mauli Rastogi, Jemima S. Jacob, Biplab Banerjee, Rashmi Gupta, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2509.17991)  

**Abstract**: Almost 50% depression patients face the risk of going into relapse. The risk increases to 80% after the second episode of depression. Although, depression detection from social media has attained considerable attention, depression relapse detection has remained largely unexplored due to the lack of curated datasets and the difficulty of distinguishing relapse and non-relapse users. In this work, we present ReDepress, the first clinically validated social media dataset focused on relapse, comprising 204 Reddit users annotated by mental health professionals. Unlike prior approaches, our framework draws on cognitive theories of depression, incorporating constructs such as attention bias, interpretation bias, memory bias and rumination into both annotation and modeling. Through statistical analyses and machine learning experiments, we demonstrate that cognitive markers significantly differentiate relapse and non-relapse groups, and that models enriched with these features achieve competitive performance, with transformer-based temporal models attaining an F1 of 0.86. Our findings validate psychological theories in real-world textual data and underscore the potential of cognitive-informed computational methods for early relapse detection, paving the way for scalable, low-cost interventions in mental healthcare. 

**Abstract (ZH)**: 近50%的抑郁患者面临复发的风险。第二次抑郁发作后，这一风险增加到80%。尽管从社交媒体中检测抑郁已获得相当的关注，但由于缺乏经过整理的数据集以及区分复发与非复发用户的困难，抑郁复发检测仍基本未被探索。在本文中，我们介绍了ReDepress，这是第一个专注于复发的临床验证社交媒体数据集，包含204位标注有心理健康专业人员标注的Reddit用户。与先前的方法不同，我们的框架借鉴了抑郁的认知理论，将注意力偏差、解释偏差、记忆偏差和 rumination 等构造纳入标注和建模过程。通过统计分析和机器学习实验，我们证明了认知标志物显著区分复发和非复发组，并且融合这些特征的模型表现优异，基于转换器的时间模型达到F1值为0.86。我们的研究结果在真实世界文本数据中验证了心理理论，并突显了基于认知的计算方法在早期复发检测中的潜力，为可扩展的低成本精神卫生干预铺平了道路。 

---
# Intra-Cluster Mixup: An Effective Data Augmentation Technique for Complementary-Label Learning 

**Title (ZH)**: 簇内 Mixup：一种有效的互补标签学习数据增强技术 

**Authors**: Tan-Ha Mai, Hsuan-Tien Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.17971)  

**Abstract**: In this paper, we investigate the challenges of complementary-label learning (CLL), a specialized form of weakly-supervised learning (WSL) where models are trained with labels indicating classes to which instances do not belong, rather than standard ordinary labels. This alternative supervision is appealing because collecting complementary labels is generally cheaper and less labor-intensive. Although most existing research in CLL emphasizes the development of novel loss functions, the potential of data augmentation in this domain remains largely underexplored. In this work, we uncover that the widely-used Mixup data augmentation technique is ineffective when directly applied to CLL. Through in-depth analysis, we identify that the complementary-label noise generated by Mixup negatively impacts the performance of CLL models. We then propose an improved technique called Intra-Cluster Mixup (ICM), which only synthesizes augmented data from nearby examples, to mitigate the noise effect. ICM carries the benefits of encouraging complementary label sharing of nearby examples, and leads to substantial performance improvements across synthetic and real-world labeled datasets. In particular, our wide spectrum of experimental results on both balanced and imbalanced CLL settings justifies the potential of ICM in allying with state-of-the-art CLL algorithms, achieving significant accuracy increases of 30% and 10% on MNIST and CIFAR datasets, respectively. 

**Abstract (ZH)**: 在这项研究中，我们探究了补充标签学习（CLL）的挑战，这是一种特殊形式的弱监督学习（WSL），模型通过指示实例不属于哪一类的标签进行训练，而非使用标准的普通标签。虽然现有大多数关于CLL的研究侧重于开发新的损失函数，但该领域中数据增强的潜力尚未得到充分探索。在这项工作中，我们发现广泛使用的Mixup数据增强技术直接应用于CLL是无效的。通过深入分析，我们发现Mixup生成的补充标签噪声会负面影响CLL模型的性能。我们随后提出了一种改进的技术，称为Intra-Cluster Mixup（ICM），该技术仅从邻近样本生成增强数据，以减轻噪声的影响。ICM鼓励邻近样本之间的补充标签共享，且在合成数据集和真实世界标注数据集上均带来了显著的性能提升。特别是，我们在均衡和不平衡CLL设置下的广泛实验结果证明了ICM与当前最先进的CLL算法相结合的潜力，在MNIST和CIFAR数据集上分别实现了30%和10%的显著准确性提升。 

---
# Joint Optimization of Memory Frequency, Computing Frequency, Transmission Power and Task Offloading for Energy-efficient DNN Inference 

**Title (ZH)**: 针对能效优化的DNN推理中内存频率、计算频率、传输功率和任务卸载联合优化 

**Authors**: Yunchu Han, Zhaojun Nan, Sheng Zhou, Zhisheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17970)  

**Abstract**: Deep neural networks (DNNs) have been widely applied in diverse applications, but the problems of high latency and energy overhead are inevitable on resource-constrained devices. To address this challenge, most researchers focus on the dynamic voltage and frequency scaling (DVFS) technique to balance the latency and energy consumption by changing the computing frequency of processors. However, the adjustment of memory frequency is usually ignored and not fully utilized to achieve efficient DNN inference, which also plays a significant role in the inference time and energy consumption. In this paper, we first investigate the impact of joint memory frequency and computing frequency scaling on the inference time and energy consumption with a model-based and data-driven method. Then by combining with the fitting parameters of different DNN models, we give a preliminary analysis for the proposed model to see the effects of adjusting memory frequency and computing frequency simultaneously. Finally, simulation results in local inference and cooperative inference cases further validate the effectiveness of jointly scaling the memory frequency and computing frequency to reduce the energy consumption of devices. 

**Abstract (ZH)**: 基于联合记忆频率和计算频率调节的深度神经网络高效推理研究 

---
# StefaLand: An Efficient Geoscience Foundation Model That Improves Dynamic Land-Surface Predictions 

**Title (ZH)**: StefaLand：一种高效的地学基础模型，用于改善动态地表预测 

**Authors**: Nicholas Kraabel, Jiangtao Liu, Yuchen Bian, Daniel Kifer, Chaopeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17942)  

**Abstract**: Stewarding natural resources, mitigating floods, droughts, wildfires, and landslides, and meeting growing demands require models that can predict climate-driven land-surface responses and human feedback with high accuracy. Traditional impact models, whether process-based, statistical, or machine learning, struggle with spatial generalization due to limited observations and concept drift. Recently proposed vision foundation models trained on satellite imagery demand massive compute and are ill-suited for dynamic land-surface prediction. We introduce StefaLand, a generative spatiotemporal earth foundation model centered on landscape interactions. StefaLand improves predictions on three tasks and four datasets: streamflow, soil moisture, and soil composition, compared to prior state-of-the-art. Results highlight its ability to generalize across diverse, data-scarce regions and support broad land-surface applications. The model builds on a masked autoencoder backbone that learns deep joint representations of landscape attributes, with a location-aware architecture fusing static and time-series inputs, attribute-based representations that drastically reduce compute, and residual fine-tuning adapters that enhance transfer. While inspired by prior methods, their alignment with geoscience and integration in one model enables robust performance on dynamic land-surface tasks. StefaLand can be pretrained and finetuned on academic compute yet outperforms state-of-the-art baselines and even fine-tuned vision foundation models. To our knowledge, this is the first geoscience land-surface foundation model that demonstrably improves dynamic land-surface interaction predictions and supports diverse downstream applications. 

**Abstract (ZH)**: StefaLand：一种基于景观交互的生成时空地球基础模型 

---
# Transformer-Encoder Trees for Efficient Multilingual Machine Translation and Speech Translation 

**Title (ZH)**: Transformer-Encoder 树结构用于高效的多语言机器翻译和语音翻译 

**Authors**: Yiwen Guan, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2509.17930)  

**Abstract**: Multilingual translation faces challenges of computational redundancy and limited accuracy for low-resource languages, especially in speech translation. To address this, we propose a novel hierarchical Transformer Encoder Tree (TET) combined with non-autoregressive encoder-only models trained with Connectionist Temporal Classification for multilingual translation. By sharing intermediate representations among linguistically similar target languages, TET can improve accuracy on low-resource languages, reduce computational redundancy, and allow generating all target languages in a single forward pass, thus eliminating sequential bottlenecks and improving parallelism. For speech translation, combining TET with a non-autoregressive speech recognition backbone (wav2vec2) shows promising results in terms of translation quality compared to autoregressive systems while being 7-14 times faster. 

**Abstract (ZH)**: 多语言翻译面临着低资源语言计算冗余和准确性有限的挑战，特别是在语音翻译中。为应对这一挑战，我们提出了一种结合非自回归编码器模型和连接主义时序分类训练的新型分层Transformer编码树（TET）方法，以实现多语言翻译。通过在语义相似的目标语言之间共享中间表示，TET可以提高低资源语言的准确性、减少计算冗余，并允许在单次前向传播中生成所有目标语言，从而消除顺序瓶颈并提高并行性。对于语音翻译，将TET与非自回归语音识别骨干模型（wav2vec2）结合使用，在翻译质量方面显示出有希望的结果，同时比自回归系统快7-14倍。 

---
# Confidence-gated training for efficient early-exit neural networks 

**Title (ZH)**: 高效早期退出神经网络的置信门控训练 

**Authors**: Saad Mokssit, Ouassim Karrakchou, Alejandro Mousist, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2509.17885)  

**Abstract**: Early-exit neural networks reduce inference cost by enabling confident predictions at intermediate layers. However, joint training often leads to gradient interference, with deeper classifiers dominating optimization. We propose Confidence-Gated Training (CGT), a paradigm that conditionally propagates gradients from deeper exits only when preceding exits fail. This encourages shallow classifiers to act as primary decision points while reserving deeper layers for harder inputs. By aligning training with the inference-time policy, CGT mitigates overthinking, improves early-exit accuracy, and preserves efficiency. Experiments on the Indian Pines and Fashion-MNIST benchmarks show that CGT lowers average inference cost while improving overall accuracy, offering a practical solution for deploying deep models in resource-constrained environments. 

**Abstract (ZH)**: Confidence-Gated Training for Early-exit Neural Networks 

---
# From Documents to Database: Failure Modes for Industrial Assets 

**Title (ZH)**: 从文档到数据库：工业资产的失效模式 

**Authors**: Duygu Kabakci-Zorlu, Fabio Lorenzi, John Sheehan, Karol Lynch, Bradley Eck  

**Link**: [PDF](https://arxiv.org/pdf/2509.17834)  

**Abstract**: We propose an interactive system using foundation models and user-provided technical documents to generate Failure Mode and Effects Analyses (FMEA) for industrial equipment. Our system aggregates unstructured content across documents to generate an FMEA and stores it in a relational database. Leveraging this tool, the time required for creation of this knowledge-intensive content is reduced, outperforming traditional manual approaches. This demonstration showcases the potential of foundation models to facilitate the creation of specialized structured content for enterprise asset management systems. 

**Abstract (ZH)**: 我们提出一种使用基础模型和用户提供的技术文档的交互系统，以生成工业设备的故障模式和效果分析（FMEA）。该系统汇总文档中的非结构化内容生成FMEA，并将其存储在关系数据库中。借助该工具，创建这种知识密集型内容所需的时间缩短，优于传统的手动方法。本演示展示了基础模型在促进企业资产管理系统专用结构化内容创建方面潜力。 

---
# Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation 

**Title (ZH)**: 基于句级分割的AI生成文本细粒度检测 

**Authors**: Lekkala Sai Teja, Annepaka Yadagiri, and Partha Pakray, Chukhu Chunka, Mangadoddi Srikar Vardhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17830)  

**Abstract**: Generation of Artificial Intelligence (AI) texts in important works has become a common practice that can be used to misuse and abuse AI at various levels. Traditional AI detectors often rely on document-level classification, which struggles to identify AI content in hybrid or slightly edited texts designed to avoid detection, leading to concerns about the model's efficiency, which makes it hard to distinguish between human-written and AI-generated texts. A sentence-level sequence labeling model proposed to detect transitions between human- and AI-generated text, leveraging nuanced linguistic signals overlooked by document-level classifiers. By this method, detecting and segmenting AI and human-written text within a single document at the token-level granularity is achieved. Our model combines the state-of-the-art pre-trained Transformer models, incorporating Neural Networks (NN) and Conditional Random Fields (CRFs). This approach extends the power of transformers to extract semantic and syntactic patterns, and the neural network component to capture enhanced sequence-level representations, thereby improving the boundary predictions by the CRF layer, which enhances sequence recognition and further identification of the partition between Human- and AI-generated texts. The evaluation is performed on two publicly available benchmark datasets containing collaborative human and AI-generated texts. Our experimental comparisons are with zero-shot detectors and the existing state-of-the-art models, along with rigorous ablation studies to justify that this approach, in particular, can accurately detect the spans of AI texts in a completely collaborative text. All our source code and the processed datasets are available in our GitHub repository. 

**Abstract (ZH)**: 生成人工智能（AI）文本在重要作品中的应用已成为一种常见做法，可能会在多个层面被滥用和误用。传统的AI检测器通常依赖于文档级别的分类，难以识别旨在规避检测的混合文本或轻微编辑的AI内容，这引发了关于模型效率的担忧，使得区分人类撰写的和AI生成的文本变得困难。为了解决这一问题，提出了基于句子级别的序列标注模型，利用文档级别分类器忽略的细腻语言信号来检测人类撰写的和AI生成的文本之间的转换。通过这种方法，在单个文档中实现了按词元级别粒度检测和分割人类撰写的和AI生成的文本。我们的模型结合了最先进的预训练Transformer模型，并引入了神经网络（NN）和条件随机场（CRFs）。该方法将Transformer的能力扩展到提取语义和句法模式，并利用神经网络部分捕捉增强的序列级表示，从而通过CRF层提高边界预测能力，增强序列识别，并进一步识别人类撰写的和AI生成的文本之间的分隔。我们在包含合作人类和AI生成文本的两个公开基准数据集上进行了评估。我们的实验比较了零样本检测器和现有的先进模型，并进行了严格的消融研究，以证明这种方法特别能够准确检测完全合作文本中的AI文本跨度。我们所有的源代码和处理后的数据集均已在GitHub仓库中提供。 

---
# One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts 

**Title (ZH)**: 一 Agent 服务于所有：面向千万多风格官方账号的轻适应风格化AI助理 

**Authors**: Xingyu Fan, Feifei Li, Wenhui Que, Hailong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17788)  

**Abstract**: Conversational agents deployed in industrial-scale official account platforms must generate responses that are both contextually grounded and stylistically aligned-requirements that existing methods struggle to meet. Chain-of-thought (CoT) prompting induces significant latency due to multi-turn reasoning; per-account fine-tuning is computationally prohibitive; and long prompt-based methods degrade the model's ability to grasp injected context and style. In this paper, we propose WeStar, a lite-adaptive framework for stylized contextual question answering that scales to millions of official accounts. WeStar combines context-grounded generation via RAG with style-aware generation using Parametric RAG (PRAG), where LoRA modules are dynamically activated per style cluster. Our contributions are fourfold: (1) We introduce WeStar, a unified framework capable of serving large volumes of official accounts with minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter sharing scheme that enables compact style representation while preserving stylistic diversity. (3) We develop a style-enhanced Direct Preference Optimization (SeDPO) method to optimize each style cluster's parameters for improved generation quality. (4) Experiments on a large-scale industrial dataset validate the effectiveness and efficiency of WeStar, underscoring its pracitical value in real-world deployment. 

**Abstract (ZH)**: 大规模官方账号平台部署的对话代理必须生成既具上下文相关性又具风格一致性的响应——现有方法难以同时满足这些要求。WeStar：一种适用于数百万官方账号的轻量级自适应风格化上下文问答框架 

---
# Accurate and Efficient Low-Rank Model Merging in Core Space 

**Title (ZH)**: 核心空间中准确高效的小秩模型融合 

**Authors**: Aniello Panariello, Daniel Marczak, Simone Magistri, Angelo Porrello, Bartłomiej Twardowski, Andrew D. Bagdanov, Simone Calderara, Joost van de Weijer  

**Link**: [PDF](https://arxiv.org/pdf/2509.17786)  

**Abstract**: In this paper, we address the challenges associated with merging low-rank adaptations of large neural networks. With the rise of parameter-efficient adaptation techniques, such as Low-Rank Adaptation (LoRA), model fine-tuning has become more accessible. While fine-tuning models with LoRA is highly efficient, existing merging methods often sacrifice this efficiency by merging fully-sized weight matrices. We propose the Core Space merging framework, which enables the merging of LoRA-adapted models within a common alignment basis, thereby preserving the efficiency of low-rank adaptation while substantially improving accuracy across tasks. We further provide a formal proof that projection into Core Space ensures no loss of information and provide a complexity analysis showing the efficiency gains. Extensive empirical results demonstrate that Core Space significantly improves existing merging techniques and achieves state-of-the-art results on both vision and language tasks while utilizing a fraction of the computational resources. Codebase is available at this https URL. 

**Abstract (ZH)**: 在本文中，我们addresses了将低秩适应的大神经网络合并所面临的挑战。随着参数高效适应技术（如Low-Rank Adaptation, LoRA）的兴起，模型微调变得更加可行。虽然使用LoRA进行微调非常高效，但现有合并方法往往通过合并全尺寸权重矩阵来牺牲这种效率。我们提出了Core Space合并框架，该框架能够在共同的对齐基上合并LoRA适应模型，从而保持低秩适应的效率同时显著提高任务准确性。我们进一步提供了形式化证明，表明投影到Core Space不会丢失信息，并提供了复杂性分析以展示效率增益。广泛的实证结果表明，Core Space显着改善了现有合并技术，在视觉和语言任务上利用少量计算资源的同时达到了最先进的性能。代码库可在以下网址获取。 

---
# DIVERS-Bench: Evaluating Language Identification Across Domain Shifts and Code-Switching 

**Title (ZH)**: DIVERS-Bench: 评估跨领域变化和码切换的语言识别 

**Authors**: Jessica Ojo, Zina Kamel, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2509.17768)  

**Abstract**: Language Identification (LID) is a core task in multilingual NLP, yet current systems often overfit to clean, monolingual data. This work introduces DIVERS-BENCH, a comprehensive evaluation of state-of-the-art LID models across diverse domains, including speech transcripts, web text, social media texts, children's stories, and code-switched text. Our findings reveal that while models achieve high accuracy on curated datasets, performance degrades sharply on noisy and informal inputs. We also introduce DIVERS-CS, a diverse code-switching benchmark dataset spanning 10 language pairs, and show that existing models struggle to detect multiple languages within the same sentence. These results highlight the need for more robust and inclusive LID systems in real-world settings. 

**Abstract (ZH)**: 语言识别（LID）是多语言自然语言处理中的核心任务，但当前系统往往过度拟合于干净的单语言数据。本文介绍了DIVERS-BENCH，这是一个在多样领域对最先进语言识别模型的全面评估，包括语音转录、网页文本、社交媒体文本、儿童故事和混合语言文本。我们的研究发现，虽然模型在精心制作的数据集上表现出高准确性，但在嘈杂和非正式的输入上性能急剧下降。我们还引入了DIVERS-CS，这是一个跨越10种语言对的多样化混合语言基准数据集，并展示了现有模型在单句内检测多种语言方面的困难。这些结果突显了在实际应用场景中需要更加稳健和包容的语言识别系统。 

---
# GEM-T: Generative Tabular Data via Fitting Moments 

**Title (ZH)**: Gem-T: 生成表格数据 via 时刻匹配 

**Authors**: Miao Li, Phuc Nguyen, Christopher Tam, Alexandra Morgan, Kenneth Ge, Rahul Bansal, Linzi Yu, Rima Arnaout, Ramy Arnaout  

**Link**: [PDF](https://arxiv.org/pdf/2509.17752)  

**Abstract**: Tabular data dominates data science but poses challenges for generative models, especially when the data is limited or sensitive. We present a novel approach to generating synthetic tabular data based on the principle of maximum entropy -- MaxEnt -- called GEM-T, for ``generative entropy maximization for tables.'' GEM-T directly captures nth-order interactions -- pairwise, third-order, etc. -- among columns of training data. In extensive testing, GEM-T matches or exceeds deep neural network approaches previously regarded as state-of-the-art in 23 of 34 publicly available datasets representing diverse subject domains (68\%). Notably, GEM-T involves orders-of-magnitude fewer trainable parameters, demonstrating that much of the information in real-world data resides in low-dimensional, potentially human-interpretable correlations, provided that the input data is appropriately transformed first. Furthermore, MaxEnt better handles heterogeneous data types (continuous vs. discrete vs. categorical), lack of local structure, and other features of tabular data. GEM-T represents a promising direction for light-weight high-performance generative models for structured data. 

**Abstract (ZH)**: 基于最大熵原理的生成模型GEM-T：表数据的生成熵最大化方法 

---
# Dual-View Alignment Learning with Hierarchical-Prompt for Class-Imbalance Multi-Label Classification 

**Title (ZH)**: 基于层次提示的双视图对齐学习方法及其在类别不平衡多标签分类中的应用 

**Authors**: Sheng Huang, Jiexuan Yan, Beiyan Liu, Bo Liu, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.17747)  

**Abstract**: Real-world datasets often exhibit class imbalance across multiple categories, manifesting as long-tailed distributions and few-shot scenarios. This is especially challenging in Class-Imbalanced Multi-Label Image Classification (CI-MLIC) tasks, where data imbalance and multi-object recognition present significant obstacles. To address these challenges, we propose a novel method termed Dual-View Alignment Learning with Hierarchical Prompt (HP-DVAL), which leverages multi-modal knowledge from vision-language pretrained (VLP) models to mitigate the class-imbalance problem in multi-label settings. Specifically, HP-DVAL employs dual-view alignment learning to transfer the powerful feature representation capabilities from VLP models by extracting complementary features for accurate image-text alignment. To better adapt VLP models for CI-MLIC tasks, we introduce a hierarchical prompt-tuning strategy that utilizes global and local prompts to learn task-specific and context-related prior knowledge. Additionally, we design a semantic consistency loss during prompt tuning to prevent learned prompts from deviating from general knowledge embedded in VLP models. The effectiveness of our approach is validated on two CI-MLIC benchmarks: MS-COCO and VOC2007. Extensive experimental results demonstrate the superiority of our method over SOTA approaches, achieving mAP improvements of 10.0\% and 5.2\% on the long-tailed multi-label image classification task, and 6.8\% and 2.9\% on the multi-label few-shot image classification task. 

**Abstract (ZH)**: 多视图对齐学习与层次提示在类不平衡多标签图像分类中的应用（基于视觉-语言预训练模型的层次提示-DUAL-VALEN 方法） 

---
# Cluster Workload Allocation: A Predictive Approach Leveraging Machine Learning Efficiency 

**Title (ZH)**: 基于机器学习效率的聚类工作负载分配预测方法 

**Authors**: Leszek Sliwko  

**Link**: [PDF](https://arxiv.org/pdf/2509.17695)  

**Abstract**: This research investigates how Machine Learning (ML) algorithms can assist in workload allocation strategies by detecting tasks with node affinity operators (referred to as constraint operators), which constrain their execution to a limited number of nodes. Using real-world Google Cluster Data (GCD) workload traces and the AGOCS framework, the study extracts node attributes and task constraints, then analyses them to identify suitable node-task pairings. It focuses on tasks that can be executed on either a single node or fewer than a thousand out of 12.5k nodes in the analysed GCD cluster. Task constraint operators are compacted, pre-processed with one-hot encoding, and used as features in a training dataset. Various ML classifiers, including Artificial Neural Networks, K-Nearest Neighbours, Decision Trees, Naive Bayes, Ridge Regression, Adaptive Boosting, and Bagging, are fine-tuned and assessed for accuracy and F1-scores. The final ensemble voting classifier model achieved 98% accuracy and a 1.5-1.8% misclassification rate for tasks with a single suitable node. 

**Abstract (ZH)**: 本研究 investigates 如何通过检测具有节点亲和力操作符的任务（称为约束操作符）来利用机器学习（ML）算法辅助工作负载分配策略。使用现实世界的 Google Cluster Data (GCD) 工作负载跟踪数据和 AGOCS 框架，研究提取节点属性和任务约束，然后分析这些信息以识别合适的节点-任务配对。研究侧重于可以在单个节点或 12500 个节点中的不到 1000 个节点上执行的任务。任务约束操作符被压缩并使用一对一独热编码预处理后作为训练数据集的特征。多种机器学习分类器，包括人工神经网络、K-最近邻、决策树、朴素贝叶斯、岭回归、自适应提升和袋装方法，被调整并评估其准确率和F1分数。最终的集成投票分类器模型实现了98%的准确率，并且对于单个适宜节点的任务，错误分类率为1.5%-1.8%。 

---
# SeqBattNet: A Discrete-State Physics-Informed Neural Network with Aging Adaptation for Battery Modeling 

**Title (ZH)**: SeqBattNet: 一种带有老化适应的离散状态物理知情神经网络的电池建模方法 

**Authors**: Khoa Tran, Hung-Cuong Trinh, Vy-Rin Nguyen, T. Nguyen-Thoi, Vin Nguyen-Thai  

**Link**: [PDF](https://arxiv.org/pdf/2509.17621)  

**Abstract**: Accurate battery modeling is essential for reliable state estimation in modern applications, such as predicting the remaining discharge time and remaining discharge energy in battery management systems. Existing approaches face several limitations: model-based methods require a large number of parameters; data-driven methods rely heavily on labeled datasets; and current physics-informed neural networks (PINNs) often lack aging adaptation, or still depend on many parameters, or continuously regenerate states. In this work, we propose SeqBattNet, a discrete-state PINN with built-in aging adaptation for battery modeling, to predict terminal voltage during the discharge process. SeqBattNet consists of two components: (i) an encoder, implemented as the proposed HRM-GRU deep learning module, which generates cycle-specific aging adaptation parameters; and (ii) a decoder, based on the equivalent circuit model (ECM) combined with deep learning, which uses these parameters together with the input current to predict voltage. The model requires only three basic battery parameters and, when trained on data from a single cell, still achieves robust performance. Extensive evaluations across three benchmark datasets (TRI, RT-Batt, and NASA) demonstrate that SeqBattNet significantly outperforms classical sequence models and PINN baselines, achieving consistently lower RMSE while maintaining computational efficiency. 

**Abstract (ZH)**: 准确的电池建模对于现代应用中的可靠状态估计至关重要，如电池管理系统中的剩余放电时间和剩余放电能量预测。现有方法面临诸多局限性：基于模型的方法需要大量参数；数据驱动的方法高度依赖标记数据集；当前的物理感知神经网络（PINN）通常缺乏老化适应性，或者仍然依赖大量参数，或者持续再生状态。为解决这些问题，我们提出SeqBattNet，这是一种内置老化适应性的离散状态PINN，用于预测放电过程中的终态电压。SeqBattNet由两个组件组成：（i）一个编码器，采用提出的HRM-GRU深度学习模块实现，生成特定循环的老化适应参数；（ii）一个解码器，基于等效电路模型（ECM）结合深度学习，使用这些参数以及输入电流来预测电压。该模型仅需三个基本电池参数，在单个电池数据上训练仍能实现稳健性能。针对三个基准数据集（TRI、RT-Batt和NASA）进行的广泛评估表明，SeqBattNet显著优于经典序列模型和PINN基准，具有更低的RMSE且保持计算效率。 

---
# AutiHero: Leveraging Generative AI in Social Narratives to Engage Parents in Story-Driven Behavioral Guidance for Autistic Children 

**Title (ZH)**: AutiHero: 利用生成式AI在社会叙事中促进父母参与以故事为导向的 autistic 孩童行为指导 

**Authors**: Jungeun Lee, Kyungah Lee, Inseok Hwang, SoHyun Park, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17608)  

**Abstract**: Social narratives are known to help autistic children understand and navigate social situations through stories. To ensure effectiveness, however, the materials need to be customized to reflect each child's unique behavioral context, requiring considerable time and effort for parents to practice at home. We present AutiHero, a generative AI-based social narrative system for behavioral guidance, which supports parents to create personalized stories for their autistic children and read them together. AutiHero generates text and visual illustrations that reflect their children's interests, target behaviors, and everyday contexts. In a two-week deployment study with 16 autistic child-parent dyads, parents created 218 stories and read an average of 4.25 stories per day, demonstrating a high level of engagement. AutiHero also provided an effective, low-demanding means to guide children's social behaviors, encouraging positive change. We discuss the implications of generative AI-infused tools to empower parents in guiding their children's behaviors, fostering their social learning. 

**Abstract (ZH)**: 基于生成AI的社会叙事系统AutiHero：个性化指导自闭症儿童的社会行为 

---
# Evaluating the Energy Efficiency of NPU-Accelerated Machine Learning Inference on Embedded Microcontrollers 

**Title (ZH)**: 评价NPU加速的机器学习推理在嵌入式微控制器上的能效 

**Authors**: Anastasios Fanariotis, Theofanis Orphanoudakis, Vasilis Fotopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.17533)  

**Abstract**: The deployment of machine learning (ML) models on microcontrollers (MCUs) is constrained by strict energy, latency, and memory requirements, particularly in battery-operated and real-time edge devices. While software-level optimizations such as quantization and pruning reduce model size and computation, hardware acceleration has emerged as a decisive enabler for efficient embedded inference. This paper evaluates the impact of Neural Processing Units (NPUs) on MCU-based ML execution, using the ARM Cortex-M55 core combined with the Ethos-U55 NPU on the Alif Semiconductor Ensemble E7 development board as a representative platform. A rigorous measurement methodology was employed, incorporating per-inference net energy accounting via GPIO-triggered high-resolution digital multimeter synchronization and idle-state subtraction, ensuring accurate attribution of energy costs. Experimental results across six representative ML models -including MiniResNet, MobileNetV2, FD-MobileNet, MNIST, TinyYolo, and SSD-MobileNet- demonstrate substantial efficiency gains when inference is offloaded to the NPU. For moderate to large networks, latency improvements ranged from 7x to over 125x, with per-inference net energy reductions up to 143x. Notably, the NPU enabled execution of models unsupported on CPU-only paths, such as SSD-MobileNet, highlighting its functional as well as efficiency advantages. These findings establish NPUs as a cornerstone of energy-aware embedded AI, enabling real-time, power-constrained ML inference at the MCU level. 

**Abstract (ZH)**: 基于微控制器的机器学习模型部署受严格的能量、延迟和内存要求限制，尤其是在电池供电和实时边缘设备中。虽然软件层面的优化如量化和剪枝减小了模型大小和计算量，但硬件加速已成为高效嵌入式推理的关键使能技术。本文使用Alif Semiconductor Ensemble E7开发板上的ARM Cortex-M55核心结合Ethos-U55 NPU，评估了神经处理单元（NPUs）对微控制器（MCUs）上机器学习执行的影响。通过GPIO触发的高分辨率数字多用表同步和空闲状态减法进行精确的能量计费，采用严格的测量方法，确保能量成本的准确归因。实验结果表明，将推理卸载到NPU上时，六种代表性机器学习模型（包括MiniResNet、MobileNetV2、FD-MobileNet、MNIST、TinyYolo和SSD-MobileNet）均实现了显著的效率提升。对于中到大型网络，延迟改进范围从7倍到超过125倍，每推理一次的净能量减少高达143倍。值得注意的是，NPU使CPU独占路径上无法运行的模型（如SSD-MobileNet）能够执行，突显了其功能和效率优势。这些发现确立了NPUs作为能量感知嵌入式AI的基础，并使MCU级别的实时、功率受限的机器学习推理成为可能。 

---
# ChartHal: A Fine-grained Framework Evaluating Hallucination of Large Vision Language Models in Chart Understanding 

**Title (ZH)**: ChartHal: 一种细粒度框架，用于图表理解中大型视觉语言模型的幻觉评估 

**Authors**: Xingqi Wang, Yiming Cui, Xin Yao, Shijin Wang, Guoping Hu, Xiaoyu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.17481)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently demonstrated remarkable progress, yet hallucination remains a critical barrier, particularly in chart understanding, which requires sophisticated perceptual and cognitive abilities as well as rigorous factual accuracy. While prior work has investigated hallucinations and chart comprehension independently, their intersection remains largely unexplored. To address this gap, we present ChartHal, a benchmark that features a fine-grained taxonomy of hallucination scenarios in chart understanding, along with a human-validated dataset of 1,062 samples. Our evaluation shows that state-of-the-art LVLMs suffer from severe hallucinations on ChartHal, including proprietary models such as GPT-5 and o4-mini, which achieve only 34.46% and 22.79% accuracy, respectively. Further analysis reveals that questions involving information absent from or contradictory to charts are especially likely to trigger hallucinations, underscoring the urgent need for more robust mitigation strategies. Code and data are available at this https URL . 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs) 在图表理解中的幻觉问题仍是一个关键障碍，ChartHal：一个细粒度幻觉场景分类的基准及其人类验证数据集 

---
# Transformer-Gather, Fuzzy-Reconsider: A Scalable Hybrid Framework for Entity Resolution 

**Title (ZH)**: Transformer-Gather, Fuzzy-Reconsider: 一种可扩展的实体解析混合框架 

**Authors**: Mohammadreza Sharifi, Danial Ahmadzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.17470)  

**Abstract**: Entity resolution plays a significant role in enterprise systems where data integrity must be rigorously maintained. Traditional methods often struggle with handling noisy data or semantic understanding, while modern methods suffer from computational costs or the excessive need for parallel computation. In this study, we introduce a scalable hybrid framework, which is designed to address several important problems, including scalability, noise robustness, and reliable results. We utilized a pre-trained language model to encode each structured data into corresponding semantic embedding vectors. Subsequently, after retrieving a semantically relevant subset of candidates, we apply a syntactic verification stage using fuzzy string matching techniques to refine classification on the unlabeled data. This approach was applied to a real-world entity resolution task, which exposed a linkage between a central user management database and numerous shared hosting server records. Compared to other methods, this approach exhibits an outstanding performance in terms of both processing time and robustness, making it a reliable solution for a server-side product. Crucially, this efficiency does not compromise results, as the system maintains a high retrieval recall of approximately 0.97. The scalability of the framework makes it deployable on standard CPU-based infrastructure, offering a practical and effective solution for enterprise-level data integrity auditing. 

**Abstract (ZH)**: 实体解析在企业系统中发挥着重要作用，必须严格维护数据完整性。传统方法往往难以处理嘈杂数据或语义理解问题，而现代方法则受到计算成本或过度需要并行计算的限制。在本研究中，我们提出了一种可扩展的混合框架，旨在解决包括可扩展性、抗噪性和可靠结果在内的多个重要问题。我们利用预训练的语言模型将每条结构化数据编码为相应的语义嵌入向量。随后，在检索到一个语义相关候选子集后，我们使用模糊字符串匹配技术应用句法验证阶段，以细化未标记数据的分类。该方法被应用于一个实际的实体解析任务，将中央用户管理数据库与众多共享主机服务器记录进行了关联。与其它方法相比，该方法在处理时间和鲁棒性方面表现出色，是服务器端产品的可靠解决方案。最关键的是，这种效率不会损害结果，系统保持了约0.97的高检索召回率。框架的可扩展性使其能够在标准CPU基础设施上部署，提供了一种适用于企业级数据完整性审计的实际和有效解决方案。 

---
# Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling 

**Title (ZH)**: Autiverse: 通过AI引导的多模态日记 eliciting 自闭症青少年的日常叙事 

**Authors**: Migyeong Yang, Kyungah Lee, Jinyoung Han, SoHyun Park, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17466)  

**Abstract**: Journaling can potentially serve as an effective method for autistic adolescents to improve narrative skills. However, its text-centric nature and high executive functioning demands present barriers to practice. We present Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds storytelling through conversational prompts and visual supports. Autiverse elicits key details through a stepwise dialogue with peer-like, customizable AI and composes them into an editable four-panel comic strip. Through a two-week deployment study with 10 autistic adolescent-parent dyads, we examine how Autiverse supports autistic adolescents to organize their daily experience and emotion. Autiverse helped them construct coherent narratives, while enabling parents to learn additional details of their child's events and emotions. The customized AI peer created a comfortable space for sharing, fostering enjoyment and a strong sense of agency. We discuss the implications of designing technologies that complement autistic adolescents' strengths while ensuring their autonomy and safety in sharing experiences. 

**Abstract (ZH)**: 日记可以作为一种有效的手段帮助 autistic 青少年提高叙事能力。然而，其文本中心的性质和高度的执行功能需求为其实践设置了障碍。我们介绍了一款名为 Autiverse 的 AI 引导式多模态日记应用，该应用适用于平板设备，通过对话式提示和支持性可视化工具辅助叙事。Autiverse 通过逐步对话与类似同伴的可定制 AI 获取关键细节，并将它们组合成可编辑的四格连环画。通过一项为期两周的研究，涉及 10 对 autistic 青少年及其家长，我们探讨了 Autiverse 如何帮助 autistic 青少年组织他们的日常生活经验和情感。Autiverse 帮助他们构建连贯的叙事，同时让家长了解到孩子事件和情感的更多细节。可定制的 AI 同伴创建了一个舒适的分享空间，促进了乐趣并增强了自主感。我们讨论了设计技术的含义，这些技术能够补充 autistic 青少年的优势，同时确保他们在分享经验时的自主权和安全。 

---
# Explainable AI for Analyzing Person-Specific Patterns in Facial Recognition Tasks 

**Title (ZH)**: 可解释的人工智能在面部识别任务中分析个人特异性模式 

**Authors**: Paweł Jakub Borsukiewicz, Jordan Samhi, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2509.17457)  

**Abstract**: The proliferation of facial recognition systems presents major privacy risks, driving the need for effective countermeasures. Current adversarial techniques apply generalized methods rather than adapting to individual facial characteristics, limiting their effectiveness and inconspicuousness. In this work, we introduce Layer Embedding Activation Mapping (LEAM), a novel technique that identifies which facial areas contribute most to recognition at an individual level. Unlike adversarial attack methods that aim to fool recognition systems, LEAM is an explainability technique designed to understand how these systems work, providing insights that could inform future privacy protection research. We integrate LEAM with a face parser to analyze data from 1000 individuals across 9 pre-trained facial recognition models.
Our analysis reveals that while different layers within facial recognition models vary significantly in their focus areas, these models generally prioritize similar facial regions across architectures when considering their overall activation patterns, which show significantly higher similarity between images of the same individual (Bhattacharyya Coefficient: 0.32-0.57) vs. different individuals (0.04-0.13), validating the existence of person-specific recognition patterns. Our results show that facial recognition models prioritize the central region of face images (with nose areas accounting for 18.9-29.7% of critical recognition regions), while still distributing attention across multiple facial fragments. Proper selection of relevant facial areas was confirmed using validation occlusions, based on just 1% of the most relevant, LEAM-identified, image pixels, which proved to be transferable across different models. Our findings establish the foundation for future individually tailored privacy protection systems centered around LEAM's choice of areas to be perturbed. 

**Abstract (ZH)**: 面部识别系统普及带来的隐私风险及其有效的应对措施：Layer Embedding Activation Mapping (LEAM) 的引入与分析 

---
# Codifying Natural Langauge Tasks 

**Title (ZH)**: 自然语言任务编码化 

**Authors**: Haoyang Chen, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2509.17455)  

**Abstract**: We explore the applicability of text-to-code to solve real-world problems that are typically solved in natural language, such as legal judgment and medical QA. Unlike previous works, our approach leverages the explicit reasoning provided by program generation. We present ICRAG, a framework that transforms natural language into executable programs through iterative refinement using external knowledge from domain resources and GitHub. Across 13 benchmarks, ICRAG achieves up to 161.1\% relative improvement. We provide a detailed analysis of the generated code and the impact of external knowledge, and we discuss the limitations of applying text-to-code approaches to real-world natural language tasks. 

**Abstract (ZH)**: 我们将文本生成代码的应用扩展到法律判决和医疗 QA 等通常用自然语言解决的实际问题。不同于以往的工作，我们的方法利用了程序生成提供的显式推理。我们提出了一种名为ICRAG的框架，通过迭代细化和利用域资源以及GitHub的外部知识将自然语言转换为可执行程序。在13个基准测试中，ICRAG实现了高达161.1%的相对改进。我们详细分析了生成的代码以及外部知识的影响，并讨论了将文本生成代码的方法应用于实际自然语言任务的局限性。 

---
# Training-Free Label Space Alignment for Universal Domain Adaptation 

**Title (ZH)**: 无训练的标签空间对齐以实现通用领域适应 

**Authors**: Dujin Lee, Sojung An, Jungmyung Wi, Kuniaki Saito, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17452)  

**Abstract**: Universal domain adaptation (UniDA) transfers knowledge from a labeled source domain to an unlabeled target domain, where label spaces may differ and the target domain may contain private classes. Previous UniDA methods primarily focused on visual space alignment but often struggled with visual ambiguities due to content differences, which limited their robustness and generalizability. To overcome this, we introduce a novel approach that leverages the strong \textit{zero-shot capabilities} of recent vision-language foundation models (VLMs) like CLIP, concentrating solely on label space alignment to enhance adaptation stability. CLIP can generate task-specific classifiers based only on label names. However, adapting CLIP to UniDA is challenging because the label space is not fully known in advance. In this study, we first utilize generative vision-language models to identify unknown categories in the target domain. Noise and semantic ambiguities in the discovered labels -- such as those similar to source labels (e.g., synonyms, hypernyms, hyponyms) -- complicate label alignment. To address this, we propose a training-free label-space alignment method for UniDA (\ours). Our method aligns label spaces instead of visual spaces by filtering and refining noisy labels between the domains. We then construct a \textit{universal classifier} that integrates both shared knowledge and target-private class information, thereby improving generalizability under domain shifts. The results reveal that the proposed method considerably outperforms existing UniDA techniques across key DomainBed benchmarks, delivering an average improvement of \textcolor{blue}{+7.9\%}in H-score and \textcolor{blue}{+6.1\%} in H$^3$-score. Furthermore, incorporating self-training further enhances performance and achieves an additional (\textcolor{blue}{+1.6\%}) increment in both H- and H$^3$-scores. 

**Abstract (ZH)**: 通用领域适应（UniDA）将标记的源领域知识 transfers 给未标记的目标领域，在目标领域可能包含私有类别的同时，标签空间可能不同。先前的 UniDA 方法主要集中在视觉空间对齐上，但常常由于内容差异造成的视觉模糊性而难以应对，这限制了它们的稳定性和泛化能力。为克服这一问题，我们引入了一种新型方法，利用近期视觉-语言基础模型（如CLIP）的强零样本能力，专注于标签空间对齐以增强适应稳定性。CLIP可以根据仅标签名称生成任务特定的分类器。然而，将CLIP适应于UniDA具有挑战性，因为标签空间不可能提前完全已知。在本研究中，我们首先使用生成性视觉-语言模型来识别目标领域的未知类别。发现的标签中的噪声和语义模糊性——例如与源标签相似的标签（如同义词、上位词、下位词）——使得标签对齐复杂化。为此，我们提出了一种无需训练的通用领域适应中的标签空间对齐方法（\ours）。该方法通过过滤和精炼领域间的噪声标签，而不是视觉空间，来对齐标签空间。然后，我们构建了一个综合共享知识和目标私有类信息的“通用分类器”，从而在领域转换下提高泛化能力。实验结果表明，所提出的方法在关键的DomainBed基准测试中显著优于现有UniDA技术，平均提升H-score多达\textcolor{blue}{+7.9\%}和H$^3$-score多达\textcolor{blue}{+6.1\%}。进一步结合自训练还能进一步提高性能，在H-和H$^3$-score上分别额外获得\textcolor{blue}{+1.6\%}的提升。 

---
# Distributionally Robust Safety Verification of Neural Networks via Worst-Case CVaR 

**Title (ZH)**: 基于最坏情况CVaR的神经网络安全性验证的分布鲁棒方法 

**Authors**: Masako Kishida  

**Link**: [PDF](https://arxiv.org/pdf/2509.17413)  

**Abstract**: Ensuring the safety of neural networks under input uncertainty is a fundamental challenge in safety-critical applications. This paper builds on and expands Fazlyab's quadratic-constraint (QC) and semidefinite-programming (SDP) framework for neural network verification to a distributionally robust and tail-risk-aware setting by integrating worst-case Conditional Value-at-Risk (WC-CVaR) over a moment-based ambiguity set with fixed mean and covariance. The resulting conditions remain SDP-checkable and explicitly account for tail risk. This integration broadens input-uncertainty geometry-covering ellipsoids, polytopes, and hyperplanes-and extends applicability to safety-critical domains where tail-event severity matters. Applications to closed-loop reachability of control systems and classification are demonstrated through numerical experiments, illustrating how the risk level $\varepsilon$ trades conservatism for tolerance to tail events-while preserving the computational structure of prior QC/SDP methods for neural network verification and robustness analysis. 

**Abstract (ZH)**: 在输入不确定性下的神经网络安全性确保是安全关键应用中的一个基本挑战。本文在Fazlyab的二次约束（QC）和半定规划（SDP）框架基础上，通过将最坏情况条件值风险（WC-CVaR）最坏情况条件值风险（WC-CVaR）集成到基于矩的含固定均值和协方差的不确定集合中，将其扩展到分布鲁棒且重尾风险意识的设置中。由此产生的条件仍保持SDP可检验证，并明确考虑重尾风险。这种集成扩大了输入不确定性几何覆盖的椭球体、多面体和超平面，并将适用范围扩展到尾事件严重性至关重要的安全关键领域。通过数值实验将该方法应用于控制系统的闭环可达性和分类，展示了风险水平$\varepsilon$如何在保证计算结构的同时，在保守性和对尾事件的容忍度之间进行折衷。 

---
# SongPrep: A Preprocessing Framework and End-to-end Model for Full-song Structure Parsing and Lyrics Transcription 

**Title (ZH)**: SongPrep: 一首歌结构解析和歌词转录的预处理框架及端到端模型 

**Authors**: Wei Tan, Shun Lei, Huaicheng Zhang, Guangzheng Li, Yixuan Zhang, Hangting Chen, Jianwei Yu, Rongzhi Gu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17404)  

**Abstract**: Artificial Intelligence Generated Content (AIGC) is currently a popular research area. Among its various branches, song generation has attracted growing interest. Despite the abundance of available songs, effective data preparation remains a significant challenge. Converting these songs into training-ready datasets typically requires extensive manual labeling, which is both time consuming and costly. To address this issue, we propose SongPrep, an automated preprocessing pipeline designed specifically for song data. This framework streamlines key processes such as source separation, structure analysis, and lyric recognition, producing structured data that can be directly used to train song generation models. Furthermore, we introduce SongPrepE2E, an end-to-end structured lyrics recognition model based on pretrained language models. Without the need for additional source separation, SongPrepE2E is able to analyze the structure and lyrics of entire songs and provide precise timestamps. By leveraging context from the whole song alongside pretrained semantic knowledge, SongPrepE2E achieves low Diarization Error Rate (DER) and Word Error Rate (WER) on the proposed SSLD-200 dataset. Downstream tasks demonstrate that training song generation models with the data output by SongPrepE2E enables the generated songs to closely resemble those produced by humans. 

**Abstract (ZH)**: 人工智能生成内容（AIGC）中的歌曲生成：自动预处理管道SongPrep及端到端结构化歌词识别模型SongPrepE2E 

---
# SeqUDA-Rec: Sequential User Behavior Enhanced Recommendation via Global Unsupervised Data Augmentation for Personalized Content Marketing 

**Title (ZH)**: SeqUDA-Rec：基于全局无监督数据增强的序列用户行为增强推荐方法及其在个性化内容营销中的应用 

**Authors**: Ruihan Luo, Xuanjing Chen, Ziyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.17361)  

**Abstract**: Personalized content marketing has become a crucial strategy for digital platforms, aiming to deliver tailored advertisements and recommendations that match user preferences. Traditional recommendation systems often suffer from two limitations: (1) reliance on limited supervised signals derived from explicit user feedback, and (2) vulnerability to noisy or unintentional interactions. To address these challenges, we propose SeqUDA-Rec, a novel deep learning framework that integrates user behavior sequences with global unsupervised data augmentation to enhance recommendation accuracy and robustness. Our approach first constructs a Global User-Item Interaction Graph (GUIG) from all user behavior sequences, capturing both local and global item associations. Then, a graph contrastive learning module is applied to generate robust embeddings, while a sequential Transformer-based encoder models users' evolving preferences. To further enhance diversity and counteract sparse supervised labels, we employ a GAN-based augmentation strategy, generating plausible interaction patterns and supplementing training data. Extensive experiments on two real-world marketing datasets (Amazon Ads and TikTok Ad Clicks) demonstrate that SeqUDA-Rec significantly outperforms state-of-the-art baselines such as SASRec, BERT4Rec, and GCL4SR. Our model achieves a 6.7% improvement in NDCG@10 and 11.3% improvement in HR@10, proving its effectiveness in personalized advertising and intelligent content recommendation. 

**Abstract (ZH)**: 个性化内容营销已成为数字平台的关键策略，旨在提供符合用户偏好的个性化广告和推荐。传统的推荐系统通常存在两种局限性：（1）依赖于从明确用户反馈中得出的有限监督信号，以及（2）容易受到噪声或无意交互的影响。为了解决这些问题，我们提出了SeqUDA-Rec，这是一种新颖的深度学习框架，将用户行为序列与全局无监督数据增强相结合，以提高推荐的准确性和鲁棒性。我们的方法首先从所有用户行为序列中构建全局用户-项目交互图（GUIG），捕获局部和全局项目关联。然后应用图对比学习模块生成稳健的嵌入表示，同时使用基于Transformers的序列编码器建模用户不断变化的偏好。为了进一步增强多样性并抵消稀疏的监督标签，我们采用了基于GAN的数据增强策略，生成合理的交互模式并补充训练数据。在两个真实世界营销数据集（亚马逊广告和抖音广告点击）上的广泛实验表明，SeqUDA-Rec 显著优于最新的基准模型，如SASRec、BERT4Rec 和 GCL4SR。我们的模型在NDCG@10上的改进率为6.7%，在HR@10上的改进率为11.3%，证明其在个性化广告和智能内容推荐方面的有效性。 

---
# Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation 

**Title (ZH)**: 姗姗来迟也胜于不来：同时发表演讲识别翻译中延迟度量的评估 

**Authors**: Peter Polák, Sara Papi, Luisa Bentivogli, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2509.17349)  

**Abstract**: Simultaneous speech-to-text translation (SimulST) systems have to balance translation quality with latency--the delay between speech input and the translated output. While quality evaluation is well established, accurate latency measurement remains a challenge. Existing metrics often produce inconsistent or misleading results, especially in the widely used short-form setting, where speech is artificially presegmented. In this paper, we present the first comprehensive analysis of SimulST latency metrics across language pairs, systems, and both short- and long-form regimes. We uncover a structural bias in current metrics related to segmentation that undermines fair and meaningful comparisons. To address this, we introduce YAAL (Yet Another Average Lagging), a refined latency metric that delivers more accurate evaluations in the short-form regime. We extend YAAL to LongYAAL for unsegmented audio and propose SoftSegmenter, a novel resegmentation tool based on word-level alignment. Our experiments show that YAAL and LongYAAL outperform popular latency metrics, while SoftSegmenter enhances alignment quality in long-form evaluation, together enabling more reliable assessments of SimulST systems. 

**Abstract (ZH)**: 同时同传翻译系统的时延评价：跨语言对、系统及短、长形式评价的全面分析 

---
# Explainability matters: The effect of liability rules on the healthcare sector 

**Title (ZH)**: 可解释性很重要：责任规则对医疗卫生产业的影响 

**Authors**: Jiawen Wei, Elena Verona, Andrea Bertolini, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2509.17334)  

**Abstract**: Explainability, the capability of an artificial intelligence system (AIS) to explain its outcomes in a manner that is comprehensible to human beings at an acceptable level, has been deemed essential for critical sectors, such as healthcare. Is it really the case? In this perspective, we consider two extreme cases, ``Oracle'' (without explainability) versus ``AI Colleague'' (with explainability) for a thorough analysis. We discuss how the level of automation and explainability of AIS can affect the determination of liability among the medical practitioner/facility and manufacturer of AIS. We argue that explainability plays a crucial role in setting a responsibility framework in healthcare, from a legal standpoint, to shape the behavior of all involved parties and mitigate the risk of potential defensive medicine practices. 

**Abstract (ZH)**: 可解释性：人工智能系统在医疗保健等关键领域的可解释能力对于确定相关责任至关重要，从法律角度来看，可解释性在构建责任框架、塑造各方行为并减少潜在防御性医疗实践的风险方面起着关键作用。 

---
# Scaling, Simplification, and Adaptation: Lessons from Pretraining on Machine-Translated Text 

**Title (ZH)**: 规模扩展、简化与适应：机器翻译文本预训练的教训 

**Authors**: Dan John Velasco, Matthew Theodore Roque  

**Link**: [PDF](https://arxiv.org/pdf/2509.17317)  

**Abstract**: Most languages lack sufficient data for large-scale monolingual pretraining, creating a "data wall." Multilingual pretraining helps but is limited by language imbalance and the "curse of multilinguality." An alternative is to translate high-resource text with machine translation (MT), which raises three questions: (1) How does MT-derived data scale with model capacity? (2) Can source-side transformations (e.g., simplifying English with an LLM) improve generalization to native text? (3) How well do models pretrained on MT-derived data adapt when continually trained on limited native text? We investigate these questions by translating English into Indonesian and Tamil--two typologically distant, lower-resource languages--and pretraining GPT-2 models (124M-774M) on native or MT-derived corpora from raw and LLM-simplified English. We evaluate cross-entropy loss on native text, along with accuracy on syntactic probes and downstream tasks. Our results show that (1) MT-pretrained models benefit from scaling; (2) source-side simplification harms generalization to native text; and (3) adapting MT-pretrained models on native text often yields better performance than native-only models, even with less native data. However, tasks requiring cultural nuance (e.g., toxicity detection) demand more exposure to native data. 

**Abstract (ZH)**: 大多数语言缺乏足够的数据进行大规模单语预训练，形成了“数据墙”。多语言预训练有所帮助，但受到语言不平衡和“多语种困境”的限制。一种替代方法是使用机器翻译（MT）翻译高资源文本，这提出了三个问题：（1）MT来源的数据如何随模型容量扩展？（2）源侧变换（例如，用LLM简化英语）能否改善对原生文本的泛化能力？（3）预训练于MT来源数据的模型，在持续使用有限的原生文本训练时，如何适应？我们通过将英语翻译成两种类型学上距离较大、资源较少的语言（印尼语和泰米尔语），并用原始英语和LLM简化后的英语训练或MT来源的语料库进行GPT-2模型（从124M到774M）的预训练，来研究这些问题。我们用交叉熵损失评估原生文本，以及对句法探针和下游任务的准确性进行评估。结果显示，（1）MT预训练模型从扩展中受益；（2）源侧简化损害了对原生文本的泛化能力；（3）在有限原生文本上微调MT预训练模型通常比仅使用原生数据的模型表现更好，即使使用较少的原生数据也是如此。然而，需要文化细微差别的任务（例如，检测毒性）需要更多接触原生数据。 

---
# Training the next generation of physicians for artificial intelligence-assisted clinical neuroradiology: ASNR MICCAI Brain Tumor Segmentation (BraTS) 2025 Lighthouse Challenge education platform 

**Title (ZH)**: 培养人工智能辅助临床神经影像学领域的下一代医师：ASNR MICCAI脑肿瘤分割（BraTS）2025灯塔挑战教育平台 

**Authors**: Raisa Amiruddin, Nikolay Y. Yordanov, Nazanin Maleki, Pascal Fehringer, Athanasios Gkampenis, Anastasia Janas, Kiril Krantchev, Ahmed Moawad, Fabian Umeh, Salma Abosabie, Sara Abosabie, Albara Alotaibi, Mohamed Ghonim, Mohanad Ghonim, Sedra Abou Ali Mhana, Nathan Page, Marko Jakovljevic, Yasaman Sharifi, Prisha Bhatia, Amirreza Manteghinejad, Melisa Guelen, Michael Veronesi, Virginia Hill, Tiffany So, Mark Krycia, Bojan Petrovic, Fatima Memon, Justin Cramer, Elizabeth Schrickel, Vilma Kosovic, Lorenna Vidal, Gerard Thompson, Ichiro Ikuta, Basimah Albalooshy, Ali Nabavizadeh, Nourel Hoda Tahon, Karuna Shekdar, Aashim Bhatia, Claudia Kirsch, Gennaro D'Anna, Philipp Lohmann, Amal Saleh Nour, Andriy Myronenko, Adam Goldman-Yassen, Janet R. Reid, Sanjay Aneja, Spyridon Bakas, Mariam Aboian  

**Link**: [PDF](https://arxiv.org/pdf/2509.17281)  

**Abstract**: High-quality reference standard image data creation by neuroradiology experts for automated clinical tools can be a powerful tool for neuroradiology & artificial intelligence education. We developed a multimodal educational approach for students and trainees during the MICCAI Brain Tumor Segmentation Lighthouse Challenge 2025, a landmark initiative to develop accurate brain tumor segmentation algorithms. Fifty-six medical students & radiology trainees volunteered to annotate brain tumor MR images for the BraTS challenges of 2023 & 2024, guided by faculty-led didactics on neuropathology MRI. Among the 56 annotators, 14 select volunteers were then paired with neuroradiology faculty for guided one-on-one annotation sessions for BraTS 2025. Lectures on neuroanatomy, pathology & AI, journal clubs & data scientist-led workshops were organized online. Annotators & audience members completed surveys on their perceived knowledge before & after annotations & lectures respectively. Fourteen coordinators, each paired with a neuroradiologist, completed the data annotation process, averaging 1322.9+/-760.7 hours per dataset per pair and 1200 segmentations in total. On a scale of 1-10, annotation coordinators reported significant increase in familiarity with image segmentation software pre- and post-annotation, moving from initial average of 6+/-2.9 to final average of 8.9+/-1.1, and significant increase in familiarity with brain tumor features pre- and post-annotation, moving from initial average of 6.2+/-2.4 to final average of 8.1+/-1.2. We demonstrate an innovative offering for providing neuroradiology & AI education through an image segmentation challenge to enhance understanding of algorithm development, reinforce the concept of data reference standard, and diversify opportunities for AI-driven image analysis among future physicians. 

**Abstract (ZH)**: 神经放射学专家创建的高 calidad参考标准图像数据用于自动临床工具可以成为神经放射学与人工智能教育的强大工具。我们开发了一种多模态教育方法，用于MICCAI脑肿瘤分割灯塔挑战2025期间的学生和培训生，这是一个旨在开发准确脑肿瘤分割算法的重大倡议。56名医学学生和放射学培训生自愿为2023年和2024年的BraTS挑战批注脑肿瘤MR图像，并在神经病理MRI指导下进行教学。在56名批注者中，14名选定的志愿者与神经放射学教授配对，进行指导下的单独批注会话，用于BraTS 2025。组织了关于神经解剖学、病理学和AI的讲座、期刊俱乐部和数据科学家主导的工作坊。批注者和听众分别在批注前和讲座后的知识测试中完成了问卷调查。14名协调员，每人都与一名神经放射学家配对，完成了数据标注过程，每组平均花费1322.9±760.7小时/数据集/组，总计1200个分割。在1-10的评分中，标注协调员报告在标注前后对图像分割软件的熟悉程度显著增加，从最初的平均6±2.9提高到最终的8.9±1.1，并且对脑肿瘤特征的熟悉程度也显著增加，从最初的平均6.2±2.4提高到最终的8.1±1.2。我们展示了一种创新的方法，通过图像分割挑战提供神经放射学与人工智能教育，以增强对算法开发的理解，强化数据参考标准的概念，并为未来的医生提供多样的AI驱动图像分析机会。 

---
# From Prediction to Understanding: Will AI Foundation Models Transform Brain Science? 

**Title (ZH)**: 从预测到理解：AI基础模型将如何 transform 神经科学？ 

**Authors**: Thomas Serre, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2509.17280)  

**Abstract**: Generative pretraining (the "GPT" in ChatGPT) enables language models to learn from vast amounts of internet text without human supervision. This approach has driven breakthroughs across AI by allowing deep neural networks to learn from massive, unstructured datasets. We use the term foundation models to refer to large pretrained systems that can be adapted to a wide range of tasks within and across domains, and these models are increasingly applied beyond language to the brain sciences. These models achieve strong predictive accuracy, raising hopes that they might illuminate computational principles. But predictive success alone does not guarantee scientific understanding. Here, we outline how foundation models can be productively integrated into the brain sciences, highlighting both their promise and their limitations. The central challenge is to move from prediction to explanation: linking model computations to mechanisms underlying neural activity and cognition. 

**Abstract (ZH)**: 生成预训练（如ChatGPT中的“GPT”）使语言模型能够在无需人类监督的情况下从海量互联网文本中学习。这种方法通过允许深层神经网络从大规模的非结构化数据集中学习，推动了人工智能领域的突破性进展。我们使用“基础模型”一词来指代那些可以适应广泛任务的基础大型预训练系统，并且这些模型正越来越多地应用于语言之外的脑科学领域。这些模型实现了强大的预测准确性，引发了它们能否揭示计算原理的希望。然而，单纯的预测成功并不保证科学理解。在这里，我们概述了如何将基础模型有成效地整合到脑科学中，既强调其潜力也指出其局限性。关键挑战是从预测转向解释：将模型计算与神经活动和认知背后的机制联系起来。 

---
# Point-RTD: Replaced Token Denoising for Pretraining Transformer Models on Point Clouds 

**Title (ZH)**: 点云替代表征去噪预训练Transformer模型 

**Authors**: Gunner Stone, Youngsook Choi, Alireza Tavakkoli, Ankita Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2509.17207)  

**Abstract**: Pre-training strategies play a critical role in advancing the performance of transformer-based models for 3D point cloud tasks. In this paper, we introduce Point-RTD (Replaced Token Denoising), a novel pretraining strategy designed to improve token robustness through a corruption-reconstruction framework. Unlike traditional mask-based reconstruction tasks that hide data segments for later prediction, Point-RTD corrupts point cloud tokens and leverages a discriminator-generator architecture for denoising. This shift enables more effective learning of structural priors and significantly enhances model performance and efficiency. On the ShapeNet dataset, Point-RTD reduces reconstruction error by over 93% compared to PointMAE, and achieves more than 14x lower Chamfer Distance on the test set. Our method also converges faster and yields higher classification accuracy on ShapeNet, ModelNet10, and ModelNet40 benchmarks, clearly outperforming the baseline Point-MAE framework in every case. 

**Abstract (ZH)**: 基于预训练策略在提升Transformer模型处理3D点云任务性能中的关键作用。本文介绍了一种新型的预训练策略Point-RTD（替换标记去噪），该策略通过 corruption-reconstruction 框架提高标记的稳健性。与传统的基于掩码的重建任务隐藏数据段以供后续预测不同，Point-RTD 对点云标记进行破坏并利用判别器-生成器架构进行去噪。这一转变使得更有效地学习结构先验知识，显著提升模型性能和效率。在ShapeNet数据集中，与PointMAE相比，Point-RTD 的重建误差降低超过93%，在测试集上的Chamfer距离低14倍以上。我们的方法在ShapeNet、ModelNet10和ModelNet40基准测试中收敛速度更快，分类准确率更高，始终优于baseline Point-MAE框架。 

---
# Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling 

**Title (ZH)**: 树突共振-放电神经元用于有效的长序列建模 

**Authors**: Dehao Zhang, Malu Zhang, Shuai Wang, Jingya Wang, Wenjie Wei, Zeyu Ma, Guoqing Wang, Yang Yang, HaiZhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17186)  

**Abstract**: The explosive growth in sequence length has intensified the demand for effective and efficient long sequence modeling. Benefiting from intrinsic oscillatory membrane dynamics, Resonate-and-Fire (RF) neurons can efficiently extract frequency components from input signals and encode them into spatiotemporal spike trains, making them well-suited for long sequence modeling. However, RF neurons exhibit limited effective memory capacity and a trade-off between energy efficiency and training speed on complex temporal tasks. Inspired by the dendritic structure of biological neurons, we propose a Dendritic Resonate-and-Fire (D-RF) model, which explicitly incorporates a multi-dendritic and soma architecture. Each dendritic branch encodes specific frequency bands by utilizing the intrinsic oscillatory dynamics of RF neurons, thereby collectively achieving comprehensive frequency representation. Furthermore, we introduce an adaptive threshold mechanism into the soma structure that adjusts the threshold based on historical spiking activity, reducing redundant spikes while maintaining training efficiency in long sequence tasks. Extensive experiments demonstrate that our method maintains competitive accuracy while substantially ensuring sparse spikes without compromising computational efficiency during training. These results underscore its potential as an effective and efficient solution for long sequence modeling on edge platforms. 

**Abstract (ZH)**: 长序列建模对序列长度爆炸性增长的需求引发了高效有效的长期序列建模方法的渴求。借助内在振荡膜动力学，共振-放电（RF）神经元能够高效地从输入信号中提取频率成分并将其编码为空间时间尖锋 trains，使其非常适合长期序列建模。然而，RF 神经元表现出有限的有效记忆容量，并且在复杂的时间任务上，能量效率和训练速度之间存在权衡。受生物神经元树突结构的启发，我们提出了一种树突共振-放电（D-RF）模型，该模型明确包含了多树突和 soma 架构。每个树突分支通过利用 RF 神经元的内在振荡动力学来编码特定的频率带宽，从而实现全面的频率表示。此外，我们在 soma 结构中引入了一种自适应阈值机制，该机制根据历史尖锋活动调整阈值，从而减少冗余尖锋并在长序列任务中保持训练效率。 extensive 实验表明，我们的方法在保持竞争力的同时，显著减少了尖锋数量，而不牺牲训练过程中的计算效率。这些结果表明，D-RF 模型在边缘平台上的长期序列建模中具备有效且高效的潜力。 

---
# Time Series Forecasting Using a Hybrid Deep Learning Method: A Bi-LSTM Embedding Denoising Auto Encoder Transformer 

**Title (ZH)**: 基于混合深度学习方法的时序预测：双向LSTM嵌入去噪自编码变换器 

**Authors**: Sahar Koohfar, Wubeshet Woldemariam  

**Link**: [PDF](https://arxiv.org/pdf/2509.17165)  

**Abstract**: Time series data is a prevalent form of data found in various fields. It consists of a series of measurements taken over time. Forecasting is a crucial application of time series models, where future values are predicted based on historical data. Accurate forecasting is essential for making well-informed decisions across industries. When it comes to electric vehicles (EVs), precise predictions play a key role in planning infrastructure development, load balancing, and energy management. This study introduces a BI-LSTM embedding denoising autoencoder model (BDM) designed to address time series problems, focusing on short-term EV charging load prediction. The performance of the proposed model is evaluated by comparing it with benchmark models like Transformer, CNN, RNN, LSTM, and GRU. Based on the results of the study, the proposed model outperforms the benchmark models in four of the five-time steps, demonstrating its effectiveness for time series forecasting. This research makes a significant contribution to enhancing time series forecasting, thereby improving decision-making processes. 

**Abstract (ZH)**: 时间序列数据是一种在各个领域中常见的数据形式，由时间序列上的测量值组成。时间序列模型的预报应用是基于历史数据预测未来值的关键所在，准确的预报对于跨行业的决策至关重要。对于电动汽车（EVs）而言，精确的预测在规划基础设施发展、负载平衡和能源管理方面起着关键作用。本研究提出了一种双向长短期记忆嵌入去噪自编码模型（BDM），旨在解决时间序列问题，并集中于短期EV充电负载预测。通过与Transformer、CNN、RNN、LSTM和GRU等基准模型进行比较，评估所提出模型的性能。根据研究结果，所提出的模型在五个时间步中的四个时间步上优于基准模型，展示了其在时间序列预报中的有效性。本研究对提高时间序列预报能力和优化决策过程作出了重要贡献。 

---
# Flow-Induced Diagonal Gaussian Processes 

**Title (ZH)**: 流诱导对角高斯过程 

**Authors**: Moule Lin, Andrea Patane, Weipeng Jing, Shuhao Guan, Goetz Botterweck  

**Link**: [PDF](https://arxiv.org/pdf/2509.17153)  

**Abstract**: We present Flow-Induced Diagonal Gaussian Processes (FiD-GP), a compression framework that incorporates a compact inducing weight matrix to project a neural network's weight uncertainty into a lower-dimensional subspace. Critically, FiD-GP relies on normalising-flow priors and spectral regularisations to augment its expressiveness and align the inducing subspace with feature-gradient geometry through a numerically stable projection mechanism objective. Furthermore, we demonstrate how the prediction framework in FiD-GP can help to design a single-pass projection for Out-of-Distribution (OoD) detection. Our analysis shows that FiD-GP improves uncertainty estimation ability on various tasks compared with SVGP-based baselines, satisfies tight spectral residual bounds with theoretically guaranteed OoD detection, and significantly compresses the neural network's storage requirements at the cost of increased inference computation dependent on the number of inducing weights employed. Specifically, in a comprehensive empirical study spanning regression, image classification, semantic segmentation, and out-of-distribution detection benchmarks, it cuts Bayesian training cost by several orders of magnitude, compresses parameters by roughly 51%, reduces model size by about 75%, and matches state-of-the-art accuracy and uncertainty estimation. 

**Abstract (ZH)**: Flow-诱导对角高斯过程（FiD-GP）：一种压缩框架 

---
# MaskVCT: Masked Voice Codec Transformer for Zero-Shot Voice Conversion With Increased Controllability via Multiple Guidances 

**Title (ZH)**: MaskVCT: 带有多重引导以提高可控性的掩码语音编码变换器在零样本语音转换中的应用 

**Authors**: Junhyeok Lee, Helin Wang, Yaohan Guan, Thomas Thebaud, Laureano Moro-Velazquez, Jesús Villalba, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2509.17143)  

**Abstract**: We introduce MaskVCT, a zero-shot voice conversion (VC) model that offers multi-factor controllability through multiple classifier-free guidances (CFGs). While previous VC models rely on a fixed conditioning scheme, MaskVCT integrates diverse conditions in a single model. To further enhance robustness and control, the model can leverage continuous or quantized linguistic features to enhance intellgibility and speaker similarity, and can use or omit pitch contour to control prosody. These choices allow users to seamlessly balance speaker identity, linguistic content, and prosodic factors in a zero-shot VC setting. Extensive experiments demonstrate that MaskVCT achieves the best target speaker and accent similarities while obtaining competitive word and character error rates compared to existing baselines. Audio samples are available at this https URL. 

**Abstract (ZH)**: MaskVCT：一种通过多重分类器自由引导（CFGs）实现多因素可控性的零样本voice转换模型 

---
# ScenGAN: Attention-Intensive Generative Model for Uncertainty-Aware Renewable Scenario Forecasting 

**Title (ZH)**: ScenGAN：面向不确定性 Awareness 可再生能源场景预测的注意力密集型生成模型 

**Authors**: Yifei Wu, Bo Wang, Jingshi Cui, Pei-chun Lin, Junzo Watada  

**Link**: [PDF](https://arxiv.org/pdf/2509.17119)  

**Abstract**: To address the intermittency of renewable energy source (RES) generation, scenario forecasting offers a series of stochastic realizations for predictive objects with superior flexibility and direct views. Based on a long time-series perspective, this paper explores uncertainties in the realms of renewable power and deep learning. Then, an uncertainty-aware model is meticulously designed for renewable scenario forecasting, which leverages an attention mechanism and generative adversarial networks (GANs) to precisely capture complex spatial-temporal dynamics. To improve the interpretability of uncertain behavior in RES generation, Bayesian deep learning and adaptive instance normalization (AdaIN) are incorporated to simulate typical patterns and variations. Additionally, the integration of meteorological information, forecasts, and historical trajectories in the processing layer improves the synergistic forecasting capability for multiscale periodic regularities. Numerical experiments and case analyses demonstrate that the proposed approach provides an appropriate interpretation for renewable uncertainty representation, including both aleatoric and epistemic uncertainties, and shows superior performance over state-of-the-art methods. 

**Abstract (ZH)**: 基于可再生能源生成间歇性的区间预测：利用注意力机制和生成对抗网络探索可再生能源和深度学习中的不确定性 

---
# Ultra-short-term solar power forecasting by deep learning and data reconstruction 

**Title (ZH)**: 基于深度学习和数据重构的 ultra-short-term 太阳能功率预测 

**Authors**: Jinbao Wang, Jun Liu, Shiliang Zhang, Xuehui Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.17095)  

**Abstract**: The integration of solar power has been increasing as the green energy transition rolls out. The penetration of solar power challenges the grid stability and energy scheduling, due to its intermittent energy generation. Accurate and near real-time solar power prediction is of critical importance to tolerant and support the permeation of distributed and volatile solar power production in the energy system. In this paper, we propose a deep-learning based ultra-short-term solar power prediction with data reconstruction. We decompose the data for the prediction to facilitate extensive exploration of the spatial and temporal dependencies within the data. Particularly, we reconstruct the data into low- and high-frequency components, using ensemble empirical model decomposition with adaptive noise (CEEMDAN). We integrate meteorological data with those two components, and employ deep-learning models to capture long- and short-term dependencies towards the target prediction period. In this way, we excessively exploit the features in historical data in predicting a ultra-short-term solar power production. Furthermore, as ultra-short-term prediction is vulnerable to local optima, we modify the optimization in our deep-learning training by penalizing long prediction intervals. Numerical experiments with diverse settings demonstrate that, compared to baseline models, the proposed method achieves improved generalization in data reconstruction and higher prediction accuracy for ultra-short-term solar power production. 

**Abstract (ZH)**: 基于数据重建的深度学习超短期太阳能发电预测 

---
# $\texttt{DiffSyn}$: A Generative Diffusion Approach to Materials Synthesis Planning 

**Title (ZH)**: DiffSyn: 材料合成规划的生成扩散方法 

**Authors**: Elton Pan, Soonhyoung Kwon, Sulin Liu, Mingrou Xie, Alexander J. Hoffman, Yifei Duan, Thorben Prein, Killian Sheriff, Yuriy Roman-Leshkov, Manuel Moliner, Rafael Gomez-Bombarelli, Elsa Olivetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.17094)  

**Abstract**: The synthesis of crystalline materials, such as zeolites, remains a significant challenge due to a high-dimensional synthesis space, intricate structure-synthesis relationships and time-consuming experiments. Considering the one-to-many relationship between structure and synthesis, we propose $\texttt{DiffSyn}$, a generative diffusion model trained on over 23,000 synthesis recipes spanning 50 years of literature. $\texttt{DiffSyn}$ generates probable synthesis routes conditioned on a desired zeolite structure and an organic template. $\texttt{DiffSyn}$ achieves state-of-the-art performance by capturing the multi-modal nature of structure-synthesis relationships. We apply $\texttt{DiffSyn}$ to differentiate among competing phases and generate optimal synthesis routes. As a proof of concept, we synthesize a UFI material using $\texttt{DiffSyn}$-generated synthesis routes. These routes, rationalized by density functional theory binding energies, resulted in the successful synthesis of a UFI material with a high Si/Al$_{\text{ICP}}$ of 19.0, which is expected to improve thermal stability and is higher than that of any previously recorded. 

**Abstract (ZH)**: 结晶材料合成，如沸石的合成，依然面临巨大挑战，由于高维合成空间、复杂的结构-合成关系以及耗时的实验。考虑到结构与合成之间的一对多关系，我们提出了一种基于超过23,000个合成配方训练的生成扩散模型$\texttt{DiffSyn}$，这些配方涵盖了近50年的文献。$\texttt{DiffSyn}$根据所需的沸石结构和有机模板生成可能的合成路径。通过捕捉结构-合成关系的多模态性质，$\texttt{DiffSyn}$实现了当前最先进的性能。我们应用$\texttt{DiffSyn}$来区分竞争性相态并生成最优合成路径。作为概念验证，我们使用$\texttt{DiffSyn}$生成的合成路径合成了一个UFI材料，并通过密度泛函理论结合能合理化，成功合成了Si/Al$_{\text{ICP}}$为19.0的UFI材料，其热稳定性预期得到提高，并超过了所有已记录的值。 

---
# A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All Histopathology Categories 

**Title (ZH)**: 涵盖所有组织病理学类别的乳腺超声 chain-of-thought 推理数据集 

**Authors**: Haojun Yu, Youcheng Li, Zihan Niu, Nan Zhang, Xuantong Gong, Huan Li, Zhiying Zou, Haifeng Qi, Zhenxiao Cao, Zijie Lan, Xingjian Yuan, Jiating He, Haokai Zhang, Shengtao Zhang, Zicheng Wang, Dong Wang, Ziwei Zhao, Congying Chen, Yong Wang, Wangyan Qin, Qingli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17046)  

**Abstract**: Breast ultrasound (BUS) is an essential tool for diagnosing breast lesions, with millions of examinations per year. However, publicly available high-quality BUS benchmarks for AI development are limited in data scale and annotation richness. In this work, we present BUS-CoT, a BUS dataset for chain-of-thought (CoT) reasoning analysis, which contains 11,439 images of 10,019 lesions from 4,838 patients and covers all 99 histopathology types. To facilitate research on incentivizing CoT reasoning, we construct the reasoning processes based on observation, feature, diagnosis and pathology labels, annotated and verified by experienced experts. Moreover, by covering lesions of all histopathology types, we aim to facilitate robust AI systems in rare cases, which can be error-prone in clinical practice. 

**Abstract (ZH)**: 乳腺超声（BUS）是诊断乳腺病变的重要工具，每年进行数百万人次的检查。然而，公开的高质量BUS基准数据集用于AI开发的数据规模和注释丰富度有限。在此工作中，我们呈现了BUS-CoT数据集，用于链式推理（CoT）分析，包含4,838名患者的10,019个病变的11,439张图像，涵盖了所有99种病理类型。为了促进激励链式推理的研究，我们基于观察、特征、诊断和病理标签构建推理过程，并由经验丰富的专家注释和验证。此外，通过涵盖所有病理类型的病变，我们旨在促进在罕见病例中鲁棒的AI系统，这些病例在临床实践中可能容易出错。 

---
# From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning 

**Title (ZH)**: 从易到难：渐进交织多图像推理的MIR基准 

**Authors**: Hang Du, Jiayang Zhang, Guoshun Nan, Wendi Deng, Zhenyan Chen, Chenyang Zhang, Wang Xiao, Shan Huang, Yuqi Pan, Tao Qi, Sicong Leng  

**Link**: [PDF](https://arxiv.org/pdf/2509.17040)  

**Abstract**: Multi-image Interleaved Reasoning aims to improve Multi-modal Large Language Models (MLLMs) ability to jointly comprehend and reason across multiple images and their associated textual contexts, introducing unique challenges beyond single-image or non-interleaved multi-image tasks. While current multi-image benchmarks overlook interleaved textual contexts and neglect distinct relationships between individual images and their associated texts, enabling models to reason over multi-image interleaved data may significantly enhance their comprehension of complex scenes and better capture cross-modal correlations. To bridge this gap, we introduce a novel benchmark MIR, requiring joint reasoning over multiple images accompanied by interleaved textual contexts to accurately associate image regions with corresponding texts and logically connect information across images. To enhance MLLMs ability to comprehend multi-image interleaved data, we introduce reasoning steps for each instance within the benchmark and propose a stage-wise curriculum learning strategy. This strategy follows an "easy to hard" approach, progressively guiding models from simple to complex scenarios, thereby enhancing their ability to handle challenging tasks. Extensive experiments benchmarking multiple MLLMs demonstrate that our method significantly enhances models reasoning performance on MIR and other established benchmarks. We believe that MIR will encourage further research into multi-image interleaved reasoning, facilitating advancements in MLLMs capability to handle complex inter-modal this http URL code and dataset are available at this https URL. 

**Abstract (ZH)**: 多图交织推理旨在提高多模大型语言模型（MLLMs）在跨多个图像及其相关文本上下文联合理解与推理方面的能力，引入了超出单图或非交织多图任务的独特挑战。当前的多图基准忽视了交织的文本上下文，并忽略了单个图像与其相关文本之间的独特关系，使模型能够推理多图交织数据可能显著增强其对复杂场景的理解并更好地捕捉跨模态相关性。为弥补这一差距，我们引入了一个新的基准MIR，要求对多个伴随交织文本上下文的图像进行联合推理，以准确关联图像区域与相应的文本，并逻辑地在图像间连接信息。为增强MLLMs对多图交织数据的理解能力，我们为基准中的每个实例引入了推理步骤，并提出了一种阶段性的递进式教学策略。该策略遵循“从易到难”的方式，逐步引导模型从简单场景过渡到复杂场景，从而增强其处理挑战性任务的能力。广泛的实验表明，我们的方法显著提升了模型在MIR和其它现有基准上的推理性能。我们认为，MIR将激发对多图交织推理的进一步研究，促进MLLMs处理复杂跨模态任务能力的提升。相关代码和数据集可在以下链接获取：[代码链接]和[数据集链接]。 

---
# Leveraging Multiple Speech Enhancers for Non-Intrusive Intelligibility Prediction for Hearing-Impaired Listeners 

**Title (ZH)**: 利用多种语音增强器进行非侵入性可懂度预测以供听力受损听者使用 

**Authors**: Boxuan Cao, Linkai Li, Hanlin Yu, Changgeng Mo, Haoshuai Zhou, Shan Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16979)  

**Abstract**: Speech intelligibility evaluation for hearing-impaired (HI) listeners is essential for assessing hearing aid performance, traditionally relying on listening tests or intrusive methods like HASPI. However, these methods require clean reference signals, which are often unavailable in real-world conditions, creating a gap between lab-based and real-world assessments. To address this, we propose a non-intrusive intelligibility prediction framework that leverages speech enhancers to provide a parallel enhanced-signal pathway, enabling robust predictions without reference signals. We evaluate three state-of-the-art enhancers and demonstrate that prediction performance depends on the choice of enhancer, with ensembles of strong enhancers yielding the best results. To improve cross-dataset generalization, we introduce a 2-clips augmentation strategy that enhances listener-specific variability, boosting robustness on unseen datasets. Our approach consistently outperforms the non-intrusive baseline, CPC2 Champion across multiple datasets, highlighting the potential of enhancer-guided non-intrusive intelligibility prediction for real-world applications. 

**Abstract (ZH)**: 听损（HI）听众的语音清晰度评估对于评估助听器性能至关重要，传统上依赖于听力测试或侵入性方法如HASPI。然而，这些方法需要干净的参考信号，在真实世界条件下常常不可获得，从而在基于实验室和实际应用的评估之间造成差距。为了解决这一问题，我们提出了一种非侵入性的可理解性预测框架，利用语音增强器提供并行的增强信号路径，从而在无需参考信号的情况下进行稳健的预测。我们评估了三种最先进的增强器，并证明预测性能取决于增强器的选择，强增强器的组合能获得最佳结果。为提高跨数据集的一般化能力，我们引入了2段剪辑增强策略，增强听者特定的变异性，提高对未见数据集的鲁棒性。我们的方法在多个数据集上均优于非侵入性基线CPC2冠军，突显了增强器引导的非侵入性可理解性预测在实际应用中的潜力。 

---
# Gradient Interference-Aware Graph Coloring for Multitask Learning 

**Title (ZH)**: 基于梯度干扰的图着色多任务学习 

**Authors**: Santosh Patapati, Trisanth Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16959)  

**Abstract**: When different objectives conflict with each other in multi-task learning, gradients begin to interfere and slow convergence, thereby reducing the final model's performance. To address this, we introduce a scheduler that computes gradient interference, constructs an interference graph, and then applies greedy graph-coloring to partition tasks into groups that align well with each other. At each training step, only one group (color class) of tasks are activated. The grouping partition is constantly recomputed as task relationships evolve throughout training. By ensuring that each mini-batch contains only tasks that pull the model in the same direction, our method improves the effectiveness of any underlying multi-task learning optimizer without additional tuning. Since tasks within these groups will update in compatible directions, model performance will be improved rather than impeded. Empirical results on six different datasets show that this interference-aware graph-coloring approach consistently outperforms baselines and state-of-the-art multi-task optimizers. 

**Abstract (ZH)**: 当多任务学习中的不同目标相互冲突时，梯度开始相互干扰，从而减缓收敛速度并降低最终模型的性能。为此，我们引入了一种调度器，该调度器计算梯度干扰、构建干扰图，并通过贪婪图着色将任务划分为彼此兼容的任务组。在每次训练步中，仅激活一个任务组（颜色类别）。随着训练过程中任务关系的变化，分组划分会不断重新计算。通过确保每个小批量仅包含推进模型沿相同方向的任务，我们的方法能够提升任何底层多任务学习优化器的效果，而无需额外调整。由于这些组内的任务将以兼容的方向更新，因此模型性能将得到提升而非阻碍。实验结果表明，这种意识梯度干扰的图着色方法在六个不同数据集上始终优于基线和最先进的多任务优化器。 

---
# AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation 

**Title (ZH)**: AirQA：面向AI研究的实例级评估综合问答数据集 

**Authors**: Tiancheng Huang, Ruisheng Cao, Yuxin Zhang, Zhangyi Kang, Zijian Wang, Chenrun Wang, Yijie Luo, Hang Zheng, Lirong Qian, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16952)  

**Abstract**: The growing volume of academic papers has made it increasingly difficult for researchers to efficiently extract key information. While large language models (LLMs) based agents are capable of automating question answering (QA) workflows for scientific papers, there still lacks a comprehensive and realistic benchmark to evaluate their capabilities. Moreover, training an interactive agent for this specific task is hindered by the shortage of high-quality interaction trajectories. In this work, we propose AirQA, a human-annotated comprehensive paper QA dataset in the field of artificial intelligence (AI), with 13,948 papers and 1,246 questions, that encompasses multi-task, multi-modal and instance-level evaluation. Furthermore, we propose ExTrActor, an automated framework for instruction data synthesis. With three LLM-based agents, ExTrActor can perform example generation and trajectory collection without human intervention. Evaluations of multiple open-source and proprietary models show that most models underperform on AirQA, demonstrating the quality of our dataset. Extensive experiments confirm that ExTrActor consistently improves the multi-turn tool-use capability of small models, enabling them to achieve performance comparable to larger ones. 

**Abstract (ZH)**: 随着学术论文数量的增长，研究人员越来越难以高效地提取关键信息。虽然基于大型语言模型的代理能够自动化科学论文的问答（QA）工作流，但仍缺乏一个全面且实际的基准来评估它们的能力。此外，由于高质量交互轨迹的短缺，为这一特定任务训练互动代理也受到了限制。在本文中，我们提出了AirQA，这是一个由人工标注的全面的人工智能（AI）领域论文问答数据集，包含13,948篇论文和1,246个问题，涵盖了多任务、多模态和实例级评估。进一步地，我们提出了ExTrActor，这是一个自动化指令数据合成框架。通过三个基于大型语言模型的代理，ExTrActor可以在不需要人工干预的情况下执行示例生成和轨迹收集。对于多个开源和专有模型的评估显示，大多数模型在AirQA上的表现不佳，证明了我们数据集的质量。广泛实验表明，ExTrActor始终能够提高小型模型的多轮工具使用能力，使其达到大型模型的性能。 

---
# Equip Pre-ranking with Target Attention by Residual Quantization 

**Title (ZH)**: 通过残差量化配备目标注意力的预排名 

**Authors**: Yutong Li, Yu Zhu, Yichen Qiao, Ziyu Guan, Lv Shao, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.16931)  

**Abstract**: The pre-ranking stage in industrial recommendation systems faces a fundamental conflict between efficiency and effectiveness. While powerful models like Target Attention (TA) excel at capturing complex feature interactions in the ranking stage, their high computational cost makes them infeasible for pre-ranking, which often relies on simplistic vector-product models. This disparity creates a significant performance bottleneck for the entire system. To bridge this gap, we propose TARQ, a novel pre-ranking framework. Inspired by generative models, TARQ's key innovation is to equip pre-ranking with an architecture approximate to TA by Residual Quantization. This allows us to bring the modeling power of TA into the latency-critical pre-ranking stage for the first time, establishing a new state-of-the-art trade-off between accuracy and efficiency. Extensive offline experiments and large-scale online A/B tests at Taobao demonstrate TARQ's significant improvements in ranking performance. Consequently, our model has been fully deployed in production, serving tens of millions of daily active users and yielding substantial business improvements. 

**Abstract (ZH)**: 工业推荐系统中的预排名阶段面临效率与效果之间的根本冲突。虽然目标注意（TA）等强大模型在排序阶段能够有效地捕捉复杂的特征交互，但其高昂的计算成本使其不适合预排名，而预排名通常依赖于简单的向量积模型。这种差异为整个系统的性能瓶颈带来了重大影响。为了弥合这一差距，我们提出了TARQ，一种新型的预排名框架。受生成模型的启发，TARQ的核心创新是通过残差量化为预排名提供一个接近TA的架构。这使我们首次能够在关键的延迟阶段为预排名带来TA的建模能力，从而建立了准确性和效率之间的一种新的最优 trade-off。淘宝的大量离线实验和大规模在线 A/B 测试表明，TARQ 在排名性能上取得了显著改进。因此，我们的模型已全面部署在生产环境中，服务于数千万活跃用户并带来了显著的业务改进。 

---
# Cross-Attention with Confidence Weighting for Multi-Channel Audio Alignment 

**Title (ZH)**: 基于置信加权的跨注意力多通道音频对齐 

**Authors**: Ragib Amin Nihal, Benjamin Yen, Takeshi Ashizawa, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2509.16926)  

**Abstract**: Multi-channel audio alignment is a key requirement in bioacoustic monitoring, spatial audio systems, and acoustic localization. However, existing methods often struggle to address nonlinear clock drift and lack mechanisms for quantifying uncertainty. Traditional methods like Cross-correlation and Dynamic Time Warping assume simple drift patterns and provide no reliability measures. Meanwhile, recent deep learning models typically treat alignment as a binary classification task, overlooking inter-channel dependencies and uncertainty estimation. We introduce a method that combines cross-attention mechanisms with confidence-weighted scoring to improve multi-channel audio synchronization. We extend BEATs encoders with cross-attention layers to model temporal relationships between channels. We also develop a confidence-weighted scoring function that uses the full prediction distribution instead of binary thresholding. Our method achieved first place in the BioDCASE 2025 Task 1 challenge with 0.30 MSE average across test datasets, compared to 0.58 for the deep learning baseline. On individual datasets, we achieved 0.14 MSE on ARU data (77% reduction) and 0.45 MSE on zebra finch data (18% reduction). The framework supports probabilistic temporal alignment, moving beyond point estimates. While validated in a bioacoustic context, the approach is applicable to a broader range of multi-channel audio tasks where alignment confidence is critical. Code available on: this https URL 

**Abstract (ZH)**: 多通道音频对齐是生物声学监测、空间音频系统和声源定位的关键要求。然而，现有的方法往往难以解决非线性时钟漂移问题，并缺乏不确定性量化机制。传统的交叉相关和动态时间规整方法假设简单的漂移模式，并不提供可靠性度量。同时，最近的深度学习模型通常将对齐视为二元分类任务，忽视了通道间依赖性和不确定性估计。我们提出了一种结合交叉注意力机制与置信加权评分的方法，以提高多通道音频同步性能。我们扩展了BEATs编码器，加入交叉注意力层以建模通道间的时序关系。我们还开发了一种置信加权评分函数，使用完整的预测分布而非二元阈值。该方法在BioDCASE 2025任务1挑战中获得第一名，平均MSE为0.30，而深度学习基线为0.58。在单个数据集中，我们在ARU数据上实现了0.14 MSE（77%的降幅），在斑马雀数据上实现了0.45 MSE（18%的降幅）。该框架支持概率时序对齐，超越了点估计。虽然在生物声学领域得到了验证，但该方法适用于更广泛的需要对齐置信度的多通道音频任务。代码可从以下链接获取：this https URL。 

---
# FedEL: Federated Elastic Learning for Heterogeneous Devices 

**Title (ZH)**: FedEL：异构设备上的联邦弹性学习 

**Authors**: Letian Zhang, Bo Chen, Jieming Bian, Lei Wang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16902)  

**Abstract**: Federated learning (FL) enables distributed devices to collaboratively train machine learning models while maintaining data privacy. However, the heterogeneous hardware capabilities of devices often result in significant training delays, as straggler clients with limited resources prolong the aggregation process. Existing solutions such as client selection, asynchronous FL, and partial training partially address these challenges but encounter issues such as reduced accuracy, stale updates, and compromised model performance due to inconsistent training contributions. To overcome these limitations, we propose FedEL, a federated elastic learning framework that enhances training efficiency while maintaining model accuracy. FedEL introduces a novel window-based training process, sliding the window to locate the training part of the model and dynamically selecting important tensors for training within a coordinated runtime budget. This approach ensures progressive and balanced training across all clients, including stragglers. Additionally, FedEL employs a tensor importance adjustment module, harmonizing local and global tensor importance to mitigate biases caused by data heterogeneity. The experiment results show that FedEL achieves up to 3.87x improvement in time-to-accuracy compared to baselines while maintaining or exceeding final test accuracy. 

**Abstract (ZH)**: 联邦弹性学习框架（FedEL）：在保持模型准确性的前提下提升训练效率 

---
# Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation 

**Title (ZH)**: 动态专家专业化：走向无灾难遗忘多领域MoE适应 

**Authors**: Junzhuo Li, Bo Wang, Xiuze Zhou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16882)  

**Abstract**: Mixture-of-Experts (MoE) models offer immense capacity via sparsely gated expert subnetworks, yet adapting them to multiple domains without catastrophic forgetting remains an open challenge. Existing approaches either incur prohibitive computation, suffer cross-domain interference, or require separate runs per domain. We propose DES-MoE, a dynamic expert specialization framework for multi-domain adaptation of Mixture-of-Experts models. DES-MoE addresses catastrophic forgetting through three innovations: (1) an adaptive router balancing pre-trained knowledge retention and task-specific updates via distillation, (2) real-time expert-domain correlation mapping to isolate domain-specific gradients, and (3) a three-phase adaptive fine-tuning schedule that progressively freezes non-specialized parameters. Evaluated on six domains (math, code, law, etc.), DES-MoE matches single-domain ESFT performance while training one unified model, reduces forgetting by 89% compared to full fine-tuning as domains scale from 2 to 6, and achieves 68% faster convergence than conventional methods. Our work establishes dynamic expert isolation as a scalable paradigm for multi-task MoE adaptation. 

**Abstract (ZH)**: Mixture-of-Experts (MoE)模型通过稀疏门控专家子网络提供了巨大的容量，但在多个领域中的适应性应用中避免灾难性遗忘仍然是一个开放的挑战。现有的方法要么计算成本高昂，要么跨领域干扰严重，或者需要为每个领域单独运行。我们提出了DES-MoE，一种动态专家专业化框架，用于Mixture-of-Experts模型的多领域适应。DES-MoE通过以下三项创新解决灾难性遗忘问题：(1)自适应路由器通过蒸馏平衡预训练知识保留和任务特定更新，(2)实时专家-领域相关映射以隔离领域特定梯度，(3)一个分三阶段的自适应微调计划，逐步冻结非专业化参数。在六个领域（数学、代码、法律等）上评估，DES-MoE在培训一个统一模型的同时达到单领域ESFT的表现，随着领域从2增加到6，与全微调相比减少遗忘89%，并且比传统方法快68%的收敛时间。我们的工作确立了动态专家隔离作为多任务MoE适应的可扩展范式。 

---
# ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching 

**Title (ZH)**: ShadowServe: 无干扰的KV缓存获取方法用于分布式前缀缓存 

**Authors**: Xingyu Xiang, Raj Joshi, Yuhan Liu, Jiayi Yao, Chenxingyu Zhao, Junchen Jiang, Yang Zhou, Eddie Kohler, Minlan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16857)  

**Abstract**: Distributed prefix caching accelerates long-context LLM serving by reusing KV cache entries for common context prefixes. However, KV cache fetches can become a bottleneck when network bandwidth is limited. Compression mitigates the bandwidth issue, but can degrade overall performance when decompression interferes with model computation.
We present ShadowServe, the first SmartNIC-accelerated, interference-free prefix caching system for LLM serving. ShadowServe separates a control plane on the host and a data plane fully offloaded to the SmartNIC, which eliminates interference to both host GPU and CPU. To overcome the SmartNIC's limited compute and memory resources, we design a chunked pipeline that parallelizes data plane operations across the SmartNIC's compute resources, and a minimal-copy memory management scheme that reduces memory pressure on the SmartNIC. Compared to state-of-the-art solutions, ShadowServe achieves up to 2.2x lower loaded time-per-output-token (TPOT), and reduces time-to-first-token (TTFT) by up to 1.38x in low-bandwidth scenarios (<= 20 Gbps), translating to up to 1.35x higher throughput. 

**Abstract (ZH)**: 智能网卡加速的无干扰前缀缓存系统ShadowServe：提升LLM服务性能 

---
# Semantic-Driven Topic Modeling for Analyzing Creativity in Virtual Brainstorming 

**Title (ZH)**: 基于语义驱动的主题建模：分析虚拟brainstorming中的 creativity 

**Authors**: Melkamu Abay Mersha, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2509.16835)  

**Abstract**: Virtual brainstorming sessions have become a central component of collaborative problem solving, yet the large volume and uneven distribution of ideas often make it difficult to extract valuable insights efficiently. Manual coding of ideas is time-consuming and subjective, underscoring the need for automated approaches to support the evaluation of group creativity. In this study, we propose a semantic-driven topic modeling framework that integrates four modular components: transformer-based embeddings (Sentence-BERT), dimensionality reduction (UMAP), clustering (HDBSCAN), and topic extraction with refinement. The framework captures semantic similarity at the sentence level, enabling the discovery of coherent themes from brainstorming transcripts while filtering noise and identifying outliers. We evaluate our approach on structured Zoom brainstorming sessions involving student groups tasked with improving their university. Results demonstrate that our model achieves higher topic coherence compared to established methods such as LDA, ETM, and BERTopic, with an average coherence score of 0.687 (CV), outperforming baselines by a significant margin. Beyond improved performance, the model provides interpretable insights into the depth and diversity of topics explored, supporting both convergent and divergent dimensions of group creativity. This work highlights the potential of embedding-based topic modeling for analyzing collaborative ideation and contributes an efficient and scalable framework for studying creativity in synchronous virtual meetings. 

**Abstract (ZH)**: 基于语义的主题建模框架：用于虚拟头脑风暴会议的模块化组件集成 

---
# KANO: Kolmogorov-Arnold Neural Operator 

**Title (ZH)**: KANO:科莫哥洛夫-阿诺尔德神经算子 

**Authors**: Jin Lee, Ziming Liu, Xinling Yu, Yixuan Wang, Haewon Jeong, Murphy Yuezhen Niu, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16825)  

**Abstract**: We introduce Kolmogorov--Arnold Neural Operator (KANO), a dual-domain neural operator jointly parameterized by both spectral and spatial bases with intrinsic symbolic interpretability. We theoretically demonstrate that KANO overcomes the pure-spectral bottleneck of Fourier Neural Operator (FNO): KANO remains expressive over generic position-dependent dynamics for any physical input, whereas FNO stays practical only for spectrally sparse operators and strictly imposes a fast-decaying input Fourier tail. We verify our claims empirically on position-dependent differential operators, for which KANO robustly generalizes but FNO fails to. In the quantum Hamiltonian learning benchmark, KANO reconstructs ground-truth Hamiltonians in closed-form symbolic representations accurate to the fourth decimal place in coefficients and attains $\approx 6\times10^{-6}$ state infidelity from projective measurement data, substantially outperforming that of the FNO trained with ideal full wave function data, $\approx 1.5\times10^{-2}$, by orders of magnitude. 

**Abstract (ZH)**: Kolmogorov--Arnold神经操作符（KANO）：兼具谱域和空域基的双重域神经操作符及其内在符号可解释性 

---
# KuBERT: Central Kurdish BERT Model and Its Application for Sentiment Analysis 

**Title (ZH)**: KuBERT：库尔德语BERT模型及其在情感分析中的应用 

**Authors**: Kozhin muhealddin Awlla, Hadi Veisi, Abdulhady Abas Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2509.16804)  

**Abstract**: This paper enhances the study of sentiment analysis for the Central Kurdish language by integrating the Bidirectional Encoder Representations from Transformers (BERT) into Natural Language Processing techniques. Kurdish is a low-resourced language, having a high level of linguistic diversity with minimal computational resources, making sentiment analysis somewhat challenging. Earlier, this was done using a traditional word embedding model, such as Word2Vec, but with the emergence of new language models, specifically BERT, there is hope for improvements. The better word embedding capabilities of BERT lend to this study, aiding in the capturing of the nuanced semantic pool and the contextual intricacies of the language under study, the Kurdish language, thus setting a new benchmark for sentiment analysis in low-resource languages. 

**Abstract (ZH)**: 本文通过将双向编码器表示从变换器（BERT）集成到自然语言处理技术中，增强了对中央库尔德语情感分析的研究。 

---
# Domain-Adaptive Pre-Training for Arabic Aspect-Based Sentiment Analysis: A Comparative Study of Domain Adaptation and Fine-Tuning Strategies 

**Title (ZH)**: 基于 domains 调适应预训练的阿拉伯语方面情感分析：域适应和微调策略的比较研究 

**Authors**: Salha Alyami, Amani Jamal, Areej Alhothali  

**Link**: [PDF](https://arxiv.org/pdf/2509.16788)  

**Abstract**: Aspect-based sentiment analysis (ABSA) in natural language processing enables organizations to understand customer opinions on specific product aspects. While deep learning models are widely used for English ABSA, their application in Arabic is limited due to the scarcity of labeled data. Researchers have attempted to tackle this issue by using pre-trained contextualized language models such as BERT. However, these models are often based on fact-based data, which can introduce bias in domain-specific tasks like ABSA. To our knowledge, no studies have applied adaptive pre-training with Arabic contextualized models for ABSA. This research proposes a novel approach using domain-adaptive pre-training for aspect-sentiment classification (ASC) and opinion target expression (OTE) extraction. We examine fine-tuning strategies - feature extraction, full fine-tuning, and adapter-based methods - to enhance performance and efficiency, utilizing multiple adaptation corpora and contextualized models. Our results show that in-domain adaptive pre-training yields modest improvements. Adapter-based fine-tuning is a computationally efficient method that achieves competitive results. However, error analyses reveal issues with model predictions and dataset labeling. In ASC, common problems include incorrect sentiment labeling, misinterpretation of contrastive markers, positivity bias for early terms, and challenges with conflicting opinions and subword tokenization. For OTE, issues involve mislabeling targets, confusion over syntactic roles, difficulty with multi-word expressions, and reliance on shallow heuristics. These findings underscore the need for syntax- and semantics-aware models, such as graph convolutional networks, to more effectively capture long-distance relations and complex aspect-based opinion alignments. 

**Abstract (ZH)**: 基于领域自适应预训练的阿拉伯语方面情感分析 

---
# Geometric Mixture Classifier (GMC): A Discriminative Per-Class Mixture of Hyperplanes 

**Title (ZH)**: 几何混合分类器（GMC）：具有区分性的各类别超平面混合模型 

**Authors**: Prasanth K K, Shubham Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.16769)  

**Abstract**: Many real world categories are multimodal, with single classes occupying disjoint regions in feature space. Classical linear models (logistic regression, linear SVM) use a single global hyperplane and perform poorly on such data, while high-capacity methods (kernel SVMs, deep nets) fit multimodal structure but at the expense of interpretability, heavier tuning, and higher computational cost. We propose the Geometric Mixture Classifier (GMC), a discriminative model that represents each class as a mixture of hyperplanes. Within each class, GMC combines plane scores via a temperature-controlled soft-OR (log-sum-exp), smoothly approximating the max; across classes, standard softmax yields probabilistic posteriors. GMC optionally uses Random Fourier Features (RFF) for nonlinear mappings while keeping inference linear in the number of planes and features. Our practical training recipe: geometry-aware k-means initialization, silhouette-based plane budgeting, alpha annealing, usage-aware L2 regularization, label smoothing, and early stopping, makes GMC plug-and-play. Across synthetic multimodal datasets (moons, circles, blobs, spirals) and tabular/image benchmarks (iris, wine, WDBC, digits), GMC consistently outperforms linear baselines and k-NN, is competitive with RBF-SVM, Random Forests, and small MLPs, and provides geometric introspection via per-plane and class responsibility visualizations. Inference scales linearly in planes and features, making GMC CPU-friendly, with single-digit microsecond latency per example, often faster than RBF-SVM and compact MLPs. Post-hoc temperature scaling reduces ECE from about 0.06 to 0.02. GMC thus strikes a favorable balance of accuracy, interpretability, and efficiency: it is more expressive than linear models and lighter, more transparent, and faster than kernel or deep models. 

**Abstract (ZH)**: 几何混合分类器：兼具准确性和可解释性的多模态分类方法 

---
# CAMBench-QR : A Structure-Aware Benchmark for Post-Hoc Explanations with QR Understanding 

**Title (ZH)**: CAMBench-QR：一种基于结构的认知后验解释基准 

**Authors**: Ritabrata Chakraborty, Avijit Dasgupta, Sandeep Chaurasia  

**Link**: [PDF](https://arxiv.org/pdf/2509.16745)  

**Abstract**: Visual explanations are often plausible but not structurally faithful. We introduce CAMBench-QR, a structure-aware benchmark that leverages the canonical geometry of QR codes (finder patterns, timing lines, module grid) to test whether CAM methods place saliency on requisite substructures while avoiding background. CAMBench-QR synthesizes QR/non-QR data with exact masks and controlled distortions, and reports structure-aware metrics (Finder/Timing Mass Ratios, Background Leakage, coverage AUCs, Distance-to-Structure) alongside causal occlusion, insertion/deletion faithfulness, robustness, and latency. We benchmark representative, efficient CAMs (LayerCAM, EigenGrad-CAM, XGrad-CAM) under two practical regimes of zero-shot and last-block fine-tuning. The benchmark, metrics, and training recipes provide a simple, reproducible yardstick for structure-aware evaluation of visual explanations. Hence we propose that CAMBENCH-QR can be used as a litmus test of whether visual explanations are truly structure-aware. 

**Abstract (ZH)**: 视觉解释通常合情合理但不一定结构忠实。我们引入CAMBench-QR，这是一个结构意识基准，利用QR码的 canonical 几何结构（寻址图案、计时线、模块网格）来测试CAM方法是否将显著性置于必要子结构上并避免背景。CAMBench-QR 合成 QR/非 QR 数据，并采用精确掩模和可控失真，同时报告结构意识度量（寻址模式/计时机群质量比、背景泄漏、覆盖 AUC、结构距离）以及因果遮挡、插入/删除的忠实性、鲁棒性和延迟。我们在零样本和最后一层微调两种实际模式下对代表性高效 CAM（LayerCAM、EigenGrad-CAM、XGrad-CAM）进行基准测试。该基准、度量和训练配方提供了一个简单可复制的标准，用于评估视觉解释的结构意识。因此我们提出，CAMBENCH-QR 可以作为检验视觉解释是否真正具有结构意识的试金石。 

---
# A Hybrid PCA-PR-Seq2Seq-Adam-LSTM Framework for Time-Series Power Outage Prediction 

**Title (ZH)**: 基于PCA-PR-Seq2Seq-Adam-LSTM的混合时序停电预测框架 

**Authors**: Subhabrata Das, Bodruzzaman Khan, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16743)  

**Abstract**: Accurately forecasting power outages is a complex task influenced by diverse factors such as weather conditions [1], vegetation, wildlife, and load fluctuations. These factors introduce substantial variability and noise into outage data, making reliable prediction challenging. Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), are particularly effective for modeling nonlinear and dynamic time-series data, with proven applications in stock price forecasting [2], energy demand prediction, demand response [3], and traffic flow management [4]. This paper introduces a hybrid deep learning framework, termed PCA-PR-Seq2Seq-Adam-LSTM, that integrates Principal Component Analysis (PCA), Poisson Regression (PR), a Sequence-to-Sequence (Seq2Seq) architecture, and an Adam-optimized LSTM. PCA is employed to reduce dimensionality and stabilize data variance, while Poisson Regression effectively models discrete outage events. The Seq2Seq-Adam-LSTM component enhances temporal feature learning through efficient gradient optimization and long-term dependency capture. The framework is evaluated using real-world outage records from Michigan, and results indicate that the proposed approach significantly improves forecasting accuracy and robustness compared to existing methods. 

**Abstract (ZH)**: 准确预测电力中断是一项受到天气条件、植被、野生动物和负载波动等多种因素影响的复杂任务。这些因素引入了显著的变量和噪声，使可靠预测具有挑战性。长短期记忆（LSTM）网络，作为一种循环神经网络（RNN），特别适合 modeling 非线性和动态时间序列数据，已在股票价格预测、能源需求预测、需求响应以及交通流管理等方面得到了证明。本文介绍了一种名为PCA-PR-Seq2Seq-Adam-LSTM的混合深度学习框架，该框架结合了主成分分析（PCA）、泊松回归（PR）、序列到序列（Seq2Seq）架构以及Adam优化的LSTM。PCA用于降低维度并稳定数据方差，泊松回归有效地建模了离散的中断事件。Seq2Seq-Adam-LSTM组件通过有效的梯度优化和长期依赖性捕获增强了时间特征学习。该框架使用来自密歇根州的实际中断记录进行了评估，结果表明，与现有方法相比，提出的这种方法在预测准确性和稳健性方面显著改进。 

---
# Exploring AI Capabilities in Participatory Budgeting within Smart Cities: The Case of Sao Paulo 

**Title (ZH)**: 探索智能城市中参与性预算中人工智能的能力：以圣保罗为例 

**Authors**: Italo Alberto Sousa, Mariana Carvalho da Silva, Jorge Machado, José Carlos Vaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.16724)  

**Abstract**: This research examines how Artificial Intelligence (AI) can improve participatory budgeting processes within smart cities. In response to challenges like declining civic participation and resource allocation conflicts, the study explores how online political participation can be improved by AI. It investigates the state capacity governments need to implement AI-enhanced participatory tools, considering technological dependencies and vulnerabilities. It analyzes technological and administrative structures, actors, interests, and strategies to understand the dynamics of online political participation technologies in the case of Sao Paulo, Brazil. The study contributes to understanding how technological advancements can reshape participatory budgeting processes. In a broader sense, the research highlights how AI can transform participatory institutions by offering new tools for citizens and also for government officials in charge of participatory processes within smart cities. 

**Abstract (ZH)**: 本研究探讨了人工智能（AI）如何改善智能城市中参与式预算过程。针对如公民参与下降和资源配置冲突等挑战，研究调查了AI如何提高在线政治参与。研究考察了政府实施增强型参与工具所需要的状态能力，考虑了技术的依赖性和脆弱性。研究分析了技术与行政结构、行为者、利益和策略，以理解巴西圣保罗市在线政治参与技术的动态。研究为理解技术进步如何重塑参与式预算过程提供了见解。更广泛地说，研究突出了AI如何通过为市民和负责智能城市中参与过程的政府官员提供新的工具来转变参与式机构。 

---
# Governed By Agents: A Survey On The Role Of Agentic AI In Future Computing Environments 

**Title (ZH)**: 由代理驱动：关于未来计算环境中文本处理中代理型AI作用的综述 

**Authors**: Nauman Ali Murad, Safia Baloch  

**Link**: [PDF](https://arxiv.org/pdf/2509.16676)  

**Abstract**: The emergence of agentic Artificial Intelligence (AI), which can operate autonomously, demonstrate goal-directed behavior, and adaptively learn, indicates the onset of a massive change in today's computing infrastructure. This study investigates how agentic AI models' multiple characteristics may impact the architecture, governance, and operation under which computing environments function. Agentic AI has the potential to reduce reliance on extremely large (public) cloud environments due to resource efficiency, especially with processing and/or storage. The aforementioned characteristics provide us with an opportunity to canvas the likelihood of strategic migration in computing infrastructures away from massive public cloud services, towards more locally distributed architectures: edge computing and on-premises computing infrastructures. Many of these likely migrations will be spurred by factors like on-premises processing needs, diminished data consumption footprints, and cost savings. This study examines how a solution for implementing AI's autonomy could result in a re-architecture of the systems and model a departure from today's governance models to help us manage these increasingly autonomous agents, and an operational overhaul of processes over a very diverse computing systems landscape that bring together computing via cloud, edge, and on-premises computing solutions. To enable us to explore these intertwined decisions, it will be fundamentally important to understand how to best position agentic AI, and to navigate the future state of computing infrastructures. 

**Abstract (ZH)**: 代理性人工智能的兴起及其对计算基础设施架构、治理和运营的影响探究 

---
# On the de-duplication of the Lakh MIDI dataset 

**Title (ZH)**: Lakh MIDI 数据集的去重研究 

**Authors**: Eunjin Choi, Hyerin Kim, Jiwoo Ryu, Juhan Nam, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2509.16662)  

**Abstract**: A large-scale dataset is essential for training a well-generalized deep-learning model. Most such datasets are collected via scraping from various internet sources, inevitably introducing duplicated data. In the symbolic music domain, these duplicates often come from multiple user arrangements and metadata changes after simple editing. However, despite critical issues such as unreliable training evaluation from data leakage during random splitting, dataset duplication has not been extensively addressed in the MIR community. This study investigates the dataset duplication issues regarding Lakh MIDI Dataset (LMD), one of the largest publicly available sources in the symbolic music domain. To find and evaluate the best retrieval method for duplicated data, we employed the Clean MIDI subset of the LMD as a benchmark test set, in which different versions of the same songs are grouped together. We first evaluated rule-based approaches and previous symbolic music retrieval models for de-duplication and also investigated with a contrastive learning-based BERT model with various augmentations to find duplicate files. As a result, we propose three different versions of the filtered list of LMD, which filters out at least 38,134 samples in the most conservative settings among 178,561 files. 

**Abstract (ZH)**: 大规模数据集对于训练泛化良好的深度学习模型至关重要。大多数此类数据集是通过从各种互联网来源抓取收集的，不可避免地引入了重复数据。在象征性音乐领域，这些重复数据通常来自多个用户的编排以及简单的编辑后的元数据更改。尽管数据泄露导致随机分割时的训练评估不可靠等问题，但在MIR社区中，数据集重复问题并未得到广泛解决。本研究调查了Lakh MIDI数据集（LMD）的重复数据问题，LMD是象征性音乐领域最大的公开数据来源之一。为了找出和评估最有效的重复数据检索方法，我们使用LMD的Clean MIDI子集作为基准测试集，在该集中将相同歌曲的不同版本分组在一起。我们首先评估了基于规则的方法和先前的象征性音乐检索模型的去重效果，并通过对比学习基于BERT的模型及其各种增强方法来查找重复文件。最终，我们在最保守的设置中提出了LMD去重后的三个不同版本，其中至少过滤掉了178,561个文件中的38,134个样本。 

---
# Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model 

**Title (ZH)**: 基于细调地理空间基础模型的城市热岛检测与模拟 

**Authors**: David Kreismann  

**Link**: [PDF](https://arxiv.org/pdf/2509.16617)  

**Abstract**: As urbanization and climate change progress, urban heat island effects are becoming more frequent and severe. To formulate effective mitigation plans, cities require detailed air temperature data. However, predictive analytics methods based on conventional machine learning models and limited data infrastructure often provide inaccurate predictions, especially in underserved areas. In this context, geospatial foundation models trained on unstructured global data demonstrate strong generalization and require minimal fine-tuning, offering an alternative for predictions where traditional approaches are limited. This study fine-tunes a geospatial foundation model to predict urban land surface temperatures under future climate scenarios and explores its response to land cover changes using simulated vegetation strategies. The fine-tuned model achieved pixel-wise downscaling errors below 1.74 °C and aligned with ground truth patterns, demonstrating an extrapolation capacity up to 3.62 °C. 

**Abstract (ZH)**: 随着城市化进程和气候变化的推进，城市热岛效应越来越频繁和严重。为了制定有效的缓解计划，城市需要详细的空气温度数据。然而，基于传统机器学习模型和有限数据基础设施的预测分析方法往往提供不准确的预测，特别是在未得到充分服务的地区。在此背景下，训练于全球未结构化数据的地表空间基础模型展示出强大的泛化能力，并且只需最少的微调，为传统方法受限的预测提供了一个替代方案。本研究对地表空间基础模型进行微调，以预测未来气候情景下的城市地表温度，并利用模拟植被策略探索其对土地覆盖变化的响应。微调后的模型实现了像素级下标误差低于1.74°C，并与地面真实模式一致，展示了高达3.62°C的外推能力。 

---
# FakeChain: Exposing Shallow Cues in Multi-Step Deepfake Detection 

**Title (ZH)**: 假象链：多步深度伪影检测中的浅层线索揭露 

**Authors**: Minji Heo, Simon S. Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.16602)  

**Abstract**: Multi-step or hybrid deepfakes, created by sequentially applying different deepfake creation methods such as Face-Swapping, GAN-based generation, and Diffusion methods, can pose an emerging and unforseen technical challenge for detection models trained on single-step forgeries. While prior studies have mainly focused on detecting isolated single manipulation, little is known about the detection model behavior under such compositional, hybrid, and complex manipulation pipelines. In this work, we introduce \textbf{FakeChain}, a large-scale benchmark comprising 1-, 2-, and 3-Step forgeries synthesized using five state-of-the-art representative generators. Using this approach, we analyze detection performance and spectral properties across hybrid manipulation at different step, along with varying generator combinations and quality settings. Surprisingly, our findings reveal that detection performance highly depends on the final manipulation type, with F1-score dropping by up to \textbf{58.83\%} when it differs from training distribution. This clearly demonstrates that detectors rely on last-stage artifacts rather than cumulative manipulation traces, limiting generalization. Such findings highlight the need for detection models to explicitly consider manipulation history and sequences. Our results highlight the importance of benchmarks such as FakeChain, reflecting growing synthesis complexity and diversity in real-world scenarios. Our sample code is available here\footnote{this https URL}. 

**Abstract (ZH)**: 多步或混合深度伪造：由不同的深度伪造生成方法（如面部互换、基于GAN的生成和扩散方法）依次应用所创建的伪造内容，对于仅针对单步伪造进行训练的检测模型构成了一个新兴且不可预见的技术挑战。虽然先前的研究主要集中在检测孤立的单步操作上，但对这样的组合性、混合性和复杂操作管道下检测模型的行为了解甚少。在本文中，我们引入了FakeChain这一大规模基准，包含使用五种最新的代表性生成器合成的1步、2步和3步伪造。通过这种方法，我们分析了不同步骤的混合操作下的检测性能和频谱特性，以及不同生成器组合和质量设置下的变化。令人意外的是，我们的研究发现检测性能高度依赖于最终的伪造类型，当伪造类型与训练分布不同步时，F1分数可下降高达58.83%。这清楚地表明，检测器依赖于最终阶段的伪迹而非累计的操作痕迹，从而限制了泛化能力。这些发现强调了检测模型需要明确考虑操作历史和序列的必要性。我们的结果突显了基准如FakeChain的重要性，反映了合成复杂性和多样性在现实世界中的增长。我们的示例代码可在此获取。 

---
# V-CECE: Visual Counterfactual Explanations via Conceptual Edits 

**Title (ZH)**: V-CECE：基于概念编辑的视觉反事实解释 

**Authors**: Nikolaos Spanos, Maria Lymperaiou, Giorgos Filandrianos, Konstantinos Thomas, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2509.16567)  

**Abstract**: Recent black-box counterfactual generation frameworks fail to take into account the semantic content of the proposed edits, while relying heavily on training to guide the generation process. We propose a novel, plug-and-play black-box counterfactual generation framework, which suggests step-by-step edits based on theoretical guarantees of optimal edits to produce human-level counterfactual explanations with zero training. Our framework utilizes a pre-trained image editing diffusion model, and operates without access to the internals of the classifier, leading to an explainable counterfactual generation process. Throughout our experimentation, we showcase the explanatory gap between human reasoning and neural model behavior by utilizing both Convolutional Neural Network (CNN), Vision Transformer (ViT) and Large Vision Language Model (LVLM) classifiers, substantiated through a comprehensive human evaluation. 

**Abstract (ZH)**: Recent黑箱反事实生成框架未能考虑所提编辑的语义内容，而是高度依赖训练来指导生成过程。我们提出了一种新颖的即插即用黑箱反事实生成框架，该框架基于最优编辑的理论保证，提供逐步编辑建议，以生成零训练的人类级别反事实解释。我们的框架利用了预训练的图像编辑扩散模型，并在不访问分类器内部结构的情况下运行，导致了可解释的反事实生成过程。在我们的实验中，我们通过使用卷积神经网络（CNN）、视觉变换器（ViT）和大视觉语言模型（LVLM）分类器，并通过全面的人类评估来体现人类推理与神经模型行为之间的解释差距。 

---
# Train to Defend: First Defense Against Cryptanalytic Neural Network Parameter Extraction Attacks 

**Title (ZH)**: Train to Defend: 首次针对密码分析神经网络参数提取攻击的防御尝试 

**Authors**: Ashley Kurian, Aydin Aysu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16546)  

**Abstract**: Neural networks are valuable intellectual property due to the significant computational cost, expert labor, and proprietary data involved in their development. Consequently, protecting their parameters is critical not only for maintaining a competitive advantage but also for enhancing the model's security and privacy. Prior works have demonstrated the growing capability of cryptanalytic attacks to scale to deeper models. In this paper, we present the first defense mechanism against cryptanalytic parameter extraction attacks. Our key insight is to eliminate the neuron uniqueness necessary for these attacks to succeed. We achieve this by a novel, extraction-aware training method. Specifically, we augment the standard loss function with an additional regularization term that minimizes the distance between neuron weights within a layer. Therefore, the proposed defense has zero area-delay overhead during inference. We evaluate the effectiveness of our approach in mitigating extraction attacks while analyzing the model accuracy across different architectures and datasets. When re-trained with the same model architecture, the results show that our defense incurs a marginal accuracy change of less than 1% with the modified loss function. Moreover, we present a theoretical framework to quantify the success probability of the attack. When tested comprehensively with prior attack settings, our defense demonstrated empirical success for sustained periods of extraction, whereas unprotected networks are extracted between 14 minutes to 4 hours. 

**Abstract (ZH)**: 神经网络由于其开发过程中涉及的重要计算成本、专家劳动力和专有数据，成为有价值的知识产权。因此，保护其参数对于保持竞争力和提高模型的安全性和隐私性至关重要。以往研究显示，密码分析性攻击的能力正在逐渐扩大以针对更深的模型。在本文中，我们提出了首个针对密码分析性参数提取攻击的防御机制。我们的核心见解是消除这些攻击成功所必需的神经元唯一性。我们通过一种新颖的、提取感知的训练方法实现了这一点。具体而言，我们通过将标准损失函数与一个额外的正则化项相结合，以最小化层内神经元权重之间的距离来实现这一目标。因此，所提出的防御机制在推断过程中没有额外的时间延迟开销。我们通过评估在不同架构和数据集上的模型准确率来验证该方法对提取攻击的缓解效果。当使用相同的模型架构重新训练时，结果显示在修改后的损失函数下，我们的防御措施对模型准确率的影响不到1%。此外，我们提出了一种理论框架来量化攻击的成功概率。在全面测试中，采用先前攻击设置时，我们的防御措施在持续提取期间显示出实际成功，而未受保护的网络则在14分钟到4小时内被提取。 

---
# Causal Fuzzing for Verifying Machine Unlearning 

**Title (ZH)**: 因果模糊测试以验证机器遗忘 

**Authors**: Anna Mazhar, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2509.16525)  

**Abstract**: As machine learning models become increasingly embedded in decision-making systems, the ability to "unlearn" targeted data or features is crucial for enhancing model adaptability, fairness, and privacy in models which involves expensive training. To effectively guide machine unlearning, a thorough testing is essential. Existing methods for verification of machine unlearning provide limited insights, often failing in scenarios where the influence is indirect. In this work, we propose CAFÉ, a new causality based framework that unifies datapoint- and feature-level unlearning for verification of black-box ML models. CAFÉ evaluates both direct and indirect effects of unlearning targets through causal dependencies, providing actionable insights with fine-grained analysis. Our evaluation across five datasets and three model architectures demonstrates that CAFÉ successfully detects residual influence missed by baselines while maintaining computational efficiency. 

**Abstract (ZH)**: 随着机器学习模型在决策系统中的应用越来越广泛，能够在不.Library中的“Library”看起来像是被误截断了，请提供完整的句子以便于翻译。请确认您是希望翻译完整句子的后半部分，还是整个句子。 

---
# Entropic Causal Inference: Graph Identifiability 

**Title (ZH)**: 熵因果推断：图的可识别性 

**Authors**: Spencer Compton, Kristjan Greenewald, Dmitriy Katz, Murat Kocaoglu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16463)  

**Abstract**: Entropic causal inference is a recent framework for learning the causal graph between two variables from observational data by finding the information-theoretically simplest structural explanation of the data, i.e., the model with smallest entropy. In our work, we first extend the causal graph identifiability result in the two-variable setting under relaxed assumptions. We then show the first identifiability result using the entropic approach for learning causal graphs with more than two nodes. Our approach utilizes the property that ancestrality between a source node and its descendants can be determined using the bivariate entropic tests. We provide a sound sequential peeling algorithm for general graphs that relies on this property. We also propose a heuristic algorithm for small graphs that shows strong empirical performance. We rigorously evaluate the performance of our algorithms on synthetic data generated from a variety of models, observing improvement over prior work. Finally we test our algorithms on real-world datasets. 

**Abstract (ZH)**: 熵驱动因果推断是近年来从观察数据中学习两个变量之间因果图的一种框架，通过寻找数据的信息论上最简单的结构解释，即熵最小的模型。在我们的工作中，我们首先在放宽假设的情况下扩展了两个变量设置中的因果图可识别性结果。随后，我们展示了使用熵驱动方法学习具有多个节点的因果图的第一个可识别性结果。我们的方法利用了祖先性这一属性，即使用双变量熵测试可以确定源头节点与其后代之间的祖先关系。我们提供了一种基于这一属性的通用图的稳健序列剥离算法。我们还提出了一种用于小图的启发式算法，显示了强大的实证性能。我们严格评估了算法在从多种模型生成的合成数据上的性能，观察到相对于先前工作的改进。最后，我们在实际数据集上测试了我们的算法。 

---
# A Generative AI System for Biomedical Data Discovery with Grammar-Based Visualizations 

**Title (ZH)**: 基于语法导向可视化的一种生物医学数据发现生成式人工智能系统 

**Authors**: Devin Lange, Shanghua Gao, Pengwei Sui, Austen Money, Priya Misner, Marinka Zitnik, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2509.16454)  

**Abstract**: We explore the potential for combining generative AI with grammar-based visualizations for biomedical data discovery. In our prototype, we use a multi-agent system to generate visualization specifications and apply filters. These visualizations are linked together, resulting in an interactive dashboard that is progressively constructed. Our system leverages the strengths of natural language while maintaining the utility of traditional user interfaces. Furthermore, we utilize generated interactive widgets enabling user adjustment. Finally, we demonstrate the potential utility of this system for biomedical data discovery with a case study. 

**Abstract (ZH)**: 我们探索将生成性AI与基于语法的可视化结合用于生物医学数据发现的潜力。在我们的原型系统中，我们使用多代理系统生成可视化规范并应用过滤器。这些可视化结果相互链接，形成一个逐步构建的交互式仪表板。该系统利用自然语言的优势同时保持传统用户界面的实用性。此外，我们利用生成的交互式控件使用户能够进行调整。最后，通过一个案例研究展示了该系统在生物医学数据发现方面的潜在应用价值。 

---
# PersonaMatrix: A Recipe for Persona-Aware Evaluation of Legal Summarization 

**Title (ZH)**: PersonaMatrix：一种面向人物意识的法律摘要评估方法 

**Authors**: Tsz Fung Pang, Maryam Berijanian, Thomas Orth, Breanna Shi, Charlotte S. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2509.16449)  

**Abstract**: Legal documents are often long, dense, and difficult to comprehend, not only for laypeople but also for legal experts. While automated document summarization has great potential to improve access to legal knowledge, prevailing task-based evaluators overlook divergent user and stakeholder needs. Tool development is needed to encompass the technicality of a case summary for a litigator yet be accessible for a self-help public researching for their lawsuit. We introduce PersonaMatrix, a persona-by-criterion evaluation framework that scores summaries through the lens of six personas, including legal and non-legal users. We also introduce a controlled dimension-shifted pilot dataset of U.S. civil rights case summaries that varies along depth, accessibility, and procedural detail as well as Diversity-Coverage Index (DCI) to expose divergent optima of legal summary between persona-aware and persona-agnostic judges. This work enables refinement of legal AI summarization systems for both expert and non-expert users, with the potential to increase access to legal knowledge. The code base and data are publicly available in GitHub. 

**Abstract (ZH)**: Legal文档经常冗长、密集且难以理解，不仅对普通大众，对法律专家也是如此。尽管自动化文档总结具有提高法律知识获取潜力的前景，但主流基于任务的评估者往往忽略了不同用户和利益相关者的多样化需求。需要开发工具以涵盖对诉讼律师具有技术性的案件摘要，同时使进行法律诉讼研究的自助公众能够理解。我们介绍了一种基于人物的标准评价框架PersonaMatrix，通过六个人物的视角对摘要进行评分，包括法律专业人士和非专业人士。我们还介绍了一个沿深度、易用性和程序细节变化的受控维度转换试点数据集，以及多样性覆盖指数（DCI），以揭示面向人物和非面向人物的法官在法律摘要方面的异质最优解。本工作有助于完善针对专家和非专家用户都适用的法律AI总结系统，从而增加法律知识的获取机会。代码库和数据已在GitHub上公开。 

---
# SENSE-7: Taxonomy and Dataset for Measuring User Perceptions of Empathy in Sustained Human-AI Conversations 

**Title (ZH)**: SENSE-7: 用于测量用户在持续人机对话中同理心感知的分类和数据集 

**Authors**: Jina Suh, Lindy Le, Erfan Shayegani, Gonzalo Ramos, Judith Amores, Desmond C. Ong, Mary Czerwinski, Javier Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2509.16437)  

**Abstract**: Empathy is increasingly recognized as a key factor in human-AI communication, yet conventional approaches to "digital empathy" often focus on simulating internal, human-like emotional states while overlooking the inherently subjective, contextual, and relational facets of empathy as perceived by users. In this work, we propose a human-centered taxonomy that emphasizes observable empathic behaviors and introduce a new dataset, Sense-7, of real-world conversations between information workers and Large Language Models (LLMs), which includes per-turn empathy annotations directly from the users, along with user characteristics, and contextual details, offering a more user-grounded representation of empathy. Analysis of 695 conversations from 109 participants reveals that empathy judgments are highly individualized, context-sensitive, and vulnerable to disruption when conversational continuity fails or user expectations go unmet. To promote further research, we provide a subset of 672 anonymized conversation and provide exploratory classification analysis, showing that an LLM-based classifier can recognize 5 levels of empathy with an encouraging average Spearman $\rho$=0.369 and Accuracy=0.487 over this set. Overall, our findings underscore the need for AI designs that dynamically tailor empathic behaviors to user contexts and goals, offering a roadmap for future research and practical development of socially attuned, human-centered artificial agents. 

**Abstract (ZH)**: 同理心日益被视为人机沟通中的关键因素，但传统的“数字同理心”方法往往侧重于模拟内部的人类情感状态，而忽视了用户感知到的同理心的基本主观性、情境性和关系性特征。在这项工作中，我们提出了一种以用户为中心的分类体系，强调可观察的同理行为，并引入了一个新的数据集Sense-7，其中包括来自用户在与大型语言模型（LLMs）的对话中的逐轮同理心注释，以及用户特征和情境细节，提供了一个更具用户基础的同理心表征。对109名参与者进行的695场对话的分析表明，同理心判断高度个体化、情境敏感，当对话连续性中断或用户期望未得到满足时，会受到影响。为了促进进一步的研究，我们提供了一组672个匿名对话的子集，并进行了探索性分类分析，显示基于LLM的分类器在该组数据上可以识别5级同理心，平均Spearman $\rho$=0.369，准确率为0.487。总体而言，我们的发现强调了AI设计需要动态地根据用户情境和目标调整同理行为的必要性，为未来研究和社会适应性、用户中心的人工智能代理的实际开发提供了 roadmap。 

---
# LenslessMic: Audio Encryption and Authentication via Lensless Computational Imaging 

**Title (ZH)**: LenslessMic: 无透镜计算成像实现音频加密和身份验证 

**Authors**: Petr Grinberg, Eric Bezzam, Paolo Prandoni, Martin Vetterli  

**Link**: [PDF](https://arxiv.org/pdf/2509.16418)  

**Abstract**: With society's increasing reliance on digital data sharing, the protection of sensitive information has become critical. Encryption serves as one of the privacy-preserving methods; however, its realization in the audio domain predominantly relies on signal processing or software methods embedded into hardware. In this paper, we introduce LenslessMic, a hybrid optical hardware-based encryption method that utilizes a lensless camera as a physical layer of security applicable to multiple types of audio. We show that LenslessMic enables (1) robust authentication of audio recordings and (2) encryption strength that can rival the search space of 256-bit digital standards, while maintaining high-quality signals and minimal loss of content information. The approach is validated with a low-cost Raspberry Pi prototype and is open-sourced together with datasets to facilitate research in the area. 

**Abstract (ZH)**: 随着社会对数字数据共享的依赖不断增加，敏感信息的保护变得至关重要。加密作为一种隐私保护方法起到了关键作用；然而，其在音频领域的实现主要依赖于嵌入硬件中的信号处理或软件方法。本文介绍了一种名为LenslessMic的混合光学硬件加密方法，该方法利用无镜头相机作为安全物理层，适用于多种类型的音频。我们展示了LenslessMic能够实现（1）稳健的音频记录认证及（2）可与256位数字标准的搜索空间相媲美的加密强度，同时保持高质量的信号和最小的内容信息损失。该方法使用低成本的树莓派原型进行验证，并一并开源相关数据集，以促进该领域的研究。 

---
# GRID: Graph-based Reasoning for Intervention and Discovery in Built Environments 

**Title (ZH)**: 基于图的推理在建筑环境中的干预与发现 

**Authors**: Taqiya Ehsan, Shuren Xia, Jorge Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2509.16397)  

**Abstract**: Manual HVAC fault diagnosis in commercial buildings takes 8-12 hours per incident and achieves only 60 percent diagnostic accuracy, reflecting analytics that stop at correlation instead of causation. To close this gap, we present GRID (Graph-based Reasoning for Intervention and Discovery), a three-stage causal discovery pipeline that combines constraint-based search, neural structural equation modeling, and language model priors to recover directed acyclic graphs from building sensor data. Across six benchmarks: synthetic rooms, EnergyPlus simulation, the ASHRAE Great Energy Predictor III dataset, and a live office testbed, GRID achieves F1 scores ranging from 0.65 to 1.00, with exact recovery (F1 = 1.00) in three controlled environments (Base, Hidden, Physical) and strong performance on real-world data (F1 = 0.89 on ASHRAE, 0.86 in noisy conditions). The method outperforms ten baseline approaches across all evaluation scenarios. Intervention scheduling achieves low operational impact in most scenarios (cost <= 0.026) while reducing risk metrics compared to baseline approaches. The framework integrates constraint-based methods, neural architectures, and domain-specific language model prompts to address the observational-causal gap in building analytics. 

**Abstract (ZH)**: 基于图的因果推理方法GRID在商业建筑中的手动暖通空调故障诊断耗时8-12小时，准确率仅为60%，反映了一种仅停留在相关性分析而非因果性分析的现状。为了弥合这一差距，我们提出了GRID（基于图的推理与发现），这是一种包含基于约束搜索、神经结构方程建模和语言模型先验的三阶段因果发现管道，用于从建筑传感器数据中恢复有向无环图。在六个基准测试中，GRID实现了从0.65到1.00的F1分数，在三种受控环境（基线、隐藏、物理环境）中实现了完全恢复（F1 = 1.00），并在实际数据上表现强劲（ASHRAE数据F1 = 0.89，多噪声条件下F1 = 0.86）。该方法在所有评估场景中均优于十种基线方法。干预调度在多数场景中实现了较低的操作影响（成本<=0.026），并减少了与基线方法相比的风险指标。该框架整合了基于约束的方法、神经架构和领域特定语言模型提示，以解决建筑分析中的观察-因果差距。 

---
# CoUn: Empowering Machine Unlearning via Contrastive Learning 

**Title (ZH)**: CoUn: 通过对比学习增强机器遗忘能力 

**Authors**: Yasser H. Khalil, Mehdi Setayesh, Hongliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16391)  

**Abstract**: Machine unlearning (MU) aims to remove the influence of specific "forget" data from a trained model while preserving its knowledge of the remaining "retain" data. Existing MU methods based on label manipulation or model weight perturbations often achieve limited unlearning effectiveness. To address this, we introduce CoUn, a novel MU framework inspired by the observation that a model retrained from scratch using only retain data classifies forget data based on their semantic similarity to the retain data. CoUn emulates this behavior by adjusting learned data representations through contrastive learning (CL) and supervised learning, applied exclusively to retain data. Specifically, CoUn (1) leverages semantic similarity between data samples to indirectly adjust forget representations using CL, and (2) maintains retain representations within their respective clusters through supervised learning. Extensive experiments across various datasets and model architectures show that CoUn consistently outperforms state-of-the-art MU baselines in unlearning effectiveness. Additionally, integrating our CL module into existing baselines empowers their unlearning effectiveness. 

**Abstract (ZH)**: 机器遗忘（Machine Unlearning，MU）旨在从训练模型中去除特定“遗忘”数据的影响，同时保持对剩余“保留”数据的知识。现有的基于标签操纵或模型权重扰动的MU方法往往实现有限的遗忘效果。为了解决这一问题，我们引入了CoUn，这是一种新颖的MU框架，该框架受到重新训练模型仅使用保留数据类别化遗忘数据时基于其与保留数据的语义相似性进行分类这种行为的启发。CoUn通过对比学习（CL）和仅应用于保留数据的监督学习来调整学习到的数据表示，以此模拟这种行为。（1）利用数据样本之间的语义相似性间接调整遗忘表示，（2）通过监督学习保持保留表示在其各自的簇内。在各种数据集和模型架构上进行的大量实验表明，CoUn在遗忘效果上始终优于最先进的MU基线。此外，将我们的CL模块集成到现有基线中能够增强它们的遗忘效果。 

---
# Enhancing Financial RAG with Agentic AI and Multi-HyDE: A Novel Approach to Knowledge Retrieval and Hallucination Reduction 

**Title (ZH)**: 提升金融RAG的方法：基于行动AI和多HyDE的新知识检索与幻觉减少 Approach 

**Authors**: Akshay Govind Srinivasan, Ryan Jacob George, Jayden Koshy Joe, Hrushikesh Kant, Harshith M R, Sachin Sundar, Sudharshan Suresh, Rahul Vimalkanth, Vijayavallabh  

**Link**: [PDF](https://arxiv.org/pdf/2509.16369)  

**Abstract**: Accurate and reliable knowledge retrieval is vital for financial question-answering, where continually updated data sources and complex, high-stakes contexts demand precision. Traditional retrieval systems rely on a single database and retriever, but financial applications require more sophisticated approaches to handle intricate regulatory filings, market analyses, and extensive multi-year reports. We introduce a framework for financial Retrieval Augmented Generation (RAG) that leverages agentic AI and the Multi-HyDE system, an approach that generates multiple, nonequivalent queries to boost the effectiveness and coverage of retrieval from large, structured financial corpora. Our pipeline is optimized for token efficiency and multi-step financial reasoning, and we demonstrate that their combination improves accuracy by 11.2% and reduces hallucinations by 15%. Our method is evaluated on standard financial QA benchmarks, showing that integrating domain-specific retrieval mechanisms such as Multi-HyDE with robust toolsets, including keyword and table-based retrieval, significantly enhances both the accuracy and reliability of answers. This research not only delivers a modular, adaptable retrieval framework for finance but also highlights the importance of structured agent workflows and multi-perspective retrieval for trustworthy deployment of AI in high-stakes financial applications. 

**Abstract (ZH)**: 准确可靠的知识检索对于金融问答至关重要，其中不断更新的数据来源和复杂的高风险背景对精准性有严格要求。传统检索系统依赖单一数据库和检索器，但金融应用需要更复杂的处理方式来应对复杂的监管文件、市场分析和广泛的多年度报告。我们提出了一种金融检索增强生成（RAG）框架，利用自主人工智能和Multi-HyDE系统，这种方法生成多个非等价查询以提高从大规模结构化金融语料中检索的效率和覆盖面。我们的管道优化了标记效率和多步金融推理，实验结果表明，其组合将准确性提高11.2%，幻觉减少15%。该方法在标准金融问答基准上进行了评估，表明集成特定领域检索机制（如Multi-HyDE）与稳健的工具集（包括关键词和表格检索）显著提高了答案的准确性和可靠性。此研究不仅提供了可模块化和适应的金融检索框架，还强调了结构化代理工作流和多视角检索对于在高风险金融应用中可靠部署AI的重要性。 

---
# Secure Confidential Business Information When Sharing Machine Learning Models 

**Title (ZH)**: 安全共享机器学习模型时保护企业机密信息 

**Authors**: Yunfan Yang, Jiarong Xu, Hongzhe Zhang, Xiao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16352)  

**Abstract**: Model-sharing offers significant business value by enabling firms with well-established Machine Learning (ML) models to monetize and share their models with others who lack the resources to develop ML models from scratch. However, concerns over data confidentiality remain a significant barrier to model-sharing adoption, as Confidential Property Inference (CPI) attacks can exploit shared ML models to uncover confidential properties of the model provider's private model training data. Existing defenses often assume that CPI attacks are non-adaptive to the specific ML model they are targeting. This assumption overlooks a key characteristic of real-world adversaries: their responsiveness, i.e., adversaries' ability to dynamically adjust their attack models based on the information of the target and its defenses. To overcome this limitation, we propose a novel defense method that explicitly accounts for the responsive nature of real-world adversaries via two methodological innovations: a novel Responsive CPI attack and an attack-defense arms race framework. The former emulates the responsive behaviors of adversaries in the real world, and the latter iteratively enhances both the target and attack models, ultimately producing a secure ML model that is robust against responsive CPI attacks. Furthermore, we propose and integrate a novel approximate strategy into our defense, which addresses a critical computational bottleneck of defense methods and improves defense efficiency. Through extensive empirical evaluations across various realistic model-sharing scenarios, we demonstrate that our method outperforms existing defenses by more effectively defending against CPI attacks, preserving ML model utility, and reducing computational overhead. 

**Abstract (ZH)**: 模型共享通过使拥有成熟机器学习模型的公司能够 monetize 和与缺乏从零开发机器学习模型资源的公司共享模型，提供了显著的商业价值。然而，数据保密性方面的担忧仍然是模型共享采用的重要障碍，因为保密属性推断（CPI）攻击可以利用共享的机器学习模型来揭露模型提供者私有模型训练数据中的保密属性。现有防御通常假设CPI攻击对目标特定的机器学习模型是非适应性的。这一假设忽略了现实世界对手的关键特征：他们的响应能力，即对手根据目标及其防御信息动态调整攻击模型的能力。为了克服这一限制，我们提出了一种新型防御方法，通过两种方法创新explicitly 考虑现实世界对手的响应性：一种新型响应式CPI攻击和攻防军备竞赛框架。前者模拟现实世界对手的响应行为，后者则迭代增强目标模型和攻击模型，最终生成一种针对响应式CPI攻击具有鲁棒性的安全机器学习模型。此外，我们提出了并整合了一种新型近似策略到我们的防御方案中，该策略解决了防御方法中的关键计算瓶颈，提高了防御效率。通过对各种现实模型共享场景进行广泛的实证评估，我们证明了我们的方法比现有防御方法更有效地抵御CPI攻击，保持了机器学习模型的实用性和降低了计算开销。 

---
# QUINTA: Reflexive Sensibility For Responsible AI Research and Data-Driven Processes 

**Title (ZH)**: QUINTA: 反身性感知为负责任的人工智能研究和数据驱动过程服务 

**Authors**: Alicia E. Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2509.16347)  

**Abstract**: As the field of artificial intelligence (AI) and machine learning (ML) continues to prioritize fairness and the concern for historically marginalized communities, the importance of intersectionality in AI research has gained significant recognition. However, few studies provide practical guidance on how researchers can effectively incorporate intersectionality into critical praxis. In response, this paper presents a comprehensive framework grounded in critical reflexivity as intersectional praxis. Operationalizing intersectionality within the AI/DS (Artificial Intelligence/Data Science) pipeline, Quantitative Intersectional Data (QUINTA) is introduced as a methodological paradigm that challenges conventional and superficial research habits, particularly in data-centric processes, to identify and mitigate negative impacts such as the inadvertent marginalization caused by these practices. The framework centers researcher reflexivity to call attention to the AI researchers' power in creating and analyzing AI/DS artifacts through data-centric approaches. To illustrate the effectiveness of QUINTA, we provide a reflexive AI/DS researcher demonstration utilizing the \#metoo movement as a case study. Note: This paper was accepted as a poster presentation at Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO) Conference in 2023. 

**Abstract (ZH)**: 随着人工智能（AI）和机器学习（ML）领域继续强调公平及对历史上边缘化社区的关注，交叠性在AI研究中的重要性获得了显著认可。然而，很少有研究提供实用指导，说明研究人员如何有效将交叠性融入关键实践。为应对这一挑战，本文提出了一种基于批判性反思的综合框架，作为交叠实践的理论基础。通过将交叠性应用于AI/DS（人工智能/数据科学）管道中，我们引入了定量交叠数据（QUINTA）作为方法论范式，挑战传统且表面的研究习惯，特别是在数据为中心的过程中，以识别并缓解由这些实践造成的无意中边缘化的负面影响。该框架强调研究者的反思性，以引起对通过数据为中心的方法创建和分析AI/DS制品的权力的认识。为了说明QUINTA的有效性，我们通过#me太运动案例研究，提供了一种批判性AI/DS研究者的示范。 

---
# Estimating Clinical Lab Test Result Trajectories from PPG using Physiological Foundation Model and Patient-Aware State Space Model -- a UNIPHY+ Approach 

**Title (ZH)**: 一种基于生理基础模型和患者感知状态空间模型的UNIPHY+方法：从PPG估计临床实验室检测结果轨迹 

**Authors**: Minxiao Wang, Runze Yan, Carol Li, Saurabh Kataria, Xiao Hu, Matthew Clark, Timothy Ruchti, Timothy G. Buchman, Sivasubramanium V Bhavani, Randall J. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.16345)  

**Abstract**: Clinical laboratory tests provide essential biochemical measurements for diagnosis and treatment, but are limited by intermittent and invasive sampling. In contrast, photoplethysmogram (PPG) is a non-invasive, continuously recorded signal in intensive care units (ICUs) that reflects cardiovascular dynamics and can serve as a proxy for latent physiological changes. We propose UNIPHY+Lab, a framework that combines a large-scale PPG foundation model for local waveform encoding with a patient-aware Mamba model for long-range temporal modeling. Our architecture addresses three challenges: (1) capturing extended temporal trends in laboratory values, (2) accounting for patient-specific baseline variation via FiLM-modulated initial states, and (3) performing multi-task estimation for interrelated biomarkers. We evaluate our method on the two ICU datasets for predicting the five key laboratory tests. The results show substantial improvements over the LSTM and carry-forward baselines in MAE, RMSE, and $R^2$ among most of the estimation targets. This work demonstrates the feasibility of continuous, personalized lab value estimation from routine PPG monitoring, offering a pathway toward non-invasive biochemical surveillance in critical care. 

**Abstract (ZH)**: UNIPHY+Lab：结合大规模PPG基础模型和患者感知Mamba模型的连续个性化实验室值估计框架 

---
# Highly Imbalanced Regression with Tabular Data in SEP and Other Applications 

**Title (ZH)**: 高不平衡回归在表格数据中的应用及其它领域 

**Authors**: Josias K. Moukpe, Philip K. Chan, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16339)  

**Abstract**: We investigate imbalanced regression with tabular data that have an imbalance ratio larger than 1,000 ("highly imbalanced"). Accurately estimating the target values of rare instances is important in applications such as forecasting the intensity of rare harmful Solar Energetic Particle (SEP) events. For regression, the MSE loss does not consider the correlation between predicted and actual values. Typical inverse importance functions allow only convex functions. Uniform sampling might yield mini-batches that do not have rare instances. We propose CISIR that incorporates correlation, Monotonically Decreasing Involution (MDI) importance, and stratified sampling. Based on five datasets, our experimental results indicate that CISIR can achieve lower error and higher correlation than some recent methods. Also, adding our correlation component to other recent methods can improve their performance. Lastly, MDI importance can outperform other importance functions. Our code can be found in this https URL. 

**Abstract (ZH)**: 我们研究了样本不均衡比大于1000的表格数据的不均衡回归问题。准确估计稀有实例的目标值在预测罕见有害太阳 energetic粒子（SEP）事件的强度等应用中非常重要。对于回归任务，均方误差损失没有考虑预测值和实际值之间的相关性。典型的逆重要性函数仅允许凸函数。均匀抽样可能会导致mini-batches中没有稀有实例。我们提出了CISIR，它结合了相关性、单调递减卷积（MDI）重要性以及分层抽样。基于五个数据集，我们的实验结果表明，CISIR可以比一些最近的方法实现更低的误差和更高的相关性。此外，将我们的相关性组件添加到其他最近的方法可以改善它们的性能。最后，MDI重要性可以优于其他重要性函数。我们的代码可以在以下链接找到：this https URL。 

---
# Patterns in the Transition From Founder-Leadership to Community Governance of Open Source 

**Title (ZH)**: 创始人领导向开源社区治理过渡的模式 

**Authors**: Mobina Noori, Mahasweta Chakraborti, Amy X Zhang, Seth Frey  

**Link**: [PDF](https://arxiv.org/pdf/2509.16295)  

**Abstract**: Open digital public infrastructure needs community management to ensure accountability, sustainability, and robustness. Yet open-source projects often rely on centralized decision-making, and the determinants of successful community management remain unclear. We analyze 637 GitHub repositories to trace transitions from founder-led to shared governance. Specifically, we document trajectories to community governance by extracting institutional roles, actions, and deontic cues from version-controlled project constitutions this http URL. With a semantic parsing pipeline, we cluster elements into broader role and action types. We find roles and actions grow, and regulation becomes more balanced, reflecting increases in governance scope and differentiation over time. Rather than shifting tone, communities grow by layering and refining responsibilities. As transitions to community management mature, projects increasingly regulate ecosystem-level relationships and add definition to project oversight roles. Overall, this work offers a scalable pipeline for tracking the growth and development of community governance regimes from open-source software's familiar default of founder-ownership. 

**Abstract (ZH)**: 开放数字公共基础设施需要社区管理以确保问责制、可持续性和稳健性。然而，开源项目往往依赖集中决策，成功社区管理的决定因素仍然不清楚。我们分析了637个GitHub仓库，追踪从创始人主导到共享治理的转变。具体地，我们通过提取受版本控制的项目宪法中的机构角色、行动和义务线索，记录社区治理的发展轨迹。使用语义解析流水线，我们将元素聚类为更广泛的角色和行动类型。我们发现角色和行动在增长，调节变得更加平衡，反映了治理范围和分化随时间的增长。社区不是通过改变语气来增长，而是通过分层和细化职责来发展。随着向社区管理的过渡成熟，项目越来越多地规范生态系统层面的关系，并为项目监督角色增加定义。总体而言，本研究提供了从开源软件熟悉的创始人所有权默认模式追踪社区治理制度成长和发展的可扩展框架。 

---
# Energy Equity, Infrastructure and Demographic Analysis with XAI Methods 

**Title (ZH)**: 能源公平性、基础设施和人口统计分析的解释性人工智能方法 

**Authors**: Sarahana Shrestha, Aparna S. Varde, Pankaj Lal  

**Link**: [PDF](https://arxiv.org/pdf/2509.16279)  

**Abstract**: This study deploys methods in explainable artificial intelligence (XAI), e.g. decision trees and Pearson's correlation coefficient (PCC), to investigate electricity usage in multiple locales. It addresses the vital issue of energy burden, i.e. total amount spent on energy divided by median household income. Socio-demographic data is analyzed with energy features, especially using decision trees and PCC, providing explainable predictors on factors affecting energy burden. Based on the results of the analysis, a pilot energy equity web portal is designed along with a novel energy burden calculator. Leveraging XAI, this portal (with its calculator) serves as a prototype information system that can offer tailored actionable advice to multiple energy stakeholders. The ultimate goal of this study is to promote greater energy equity through the adaptation of XAI methods for energy-related analysis with suitable recommendations. 

**Abstract (ZH)**: 本研究运用可解释人工智能（XAI）方法，如决策树和皮尔逊相关系数（PCC），探究多个区域的用电情况。该研究探讨了能源负担这一关键问题，即总能源支出除以中位户收入的比例。通过分析社会人口统计数据和能源特征，特别是利用决策树和PCC，提供了可解释的预测因子，影响因素对能源负担的影响。基于分析结果，设计了一个试点能源公平网络门户，并开发了一个新型能源负担计算器。借助XAI，该门户（及其计算器）作为一个原型信息系统，可以为多个能源利益相关者提供量身定制的行动建议。本研究的最终目标是通过适应XAI方法进行能源相关分析并提出适当建议，促进能源公平。 

---
# Comparative Analysis of STEM and non-STEM Teachers' Needs for Integrating AI into Educational Environments 

**Title (ZH)**: STEM与非STEM教师整合人工智能于教育环境中的需求比较研究 

**Authors**: Bahare Riahi, Veronica Catete  

**Link**: [PDF](https://arxiv.org/pdf/2509.16276)  

**Abstract**: There is an increasing imperative to integrate programming platforms within AI frameworks to enhance educational tasks for both teachers and students. However, commonly used platforms such as this http URL, Scratch, and Snap fall short of providing the desired AI features and lack adaptability for interdisciplinary applications. This study explores how educational platforms can be improved by incorporating AI and analytics features to create more effective learning environments across various subjects and domains. We interviewed 8 K-12 teachers and asked their practices and needs while using any block-based programming (BBP) platform in their classes. We asked for their approaches in assessment, course development and expansion of resources, and student monitoring in their classes. Thematic analysis of the interview transcripts revealed both commonalities and differences in the AI tools needed between the STEM and non-STEM groups. Our results indicated advanced AI features that could promote BBP platforms. Both groups stressed the need for integrity and plagiarism checks, AI adaptability, customized rubrics, and detailed feedback in assessments. Non-STEM teachers also emphasized the importance of creative assignments and qualitative assessments. Regarding resource development, both AI tools desired for updating curricula, tutoring libraries, and generative AI features. Non-STEM teachers were particularly interested in supporting creative endeavors, such as art simulations. For student monitoring, both groups prioritized desktop control, daily tracking, behavior monitoring, and distraction prevention tools. Our findings identify specific AI-enhanced features needed by K-12 teachers across various disciplines and lay the foundation for creating more efficient, personalized, and engaging educational experiences. 

**Abstract (ZH)**: 将编程平台整合到AI框架中以增强教育任务：探索AI和分析特征在教育平台中的应用以创建跨学科的有效学习环境 

---
# SubDyve: Subgraph-Driven Dynamic Propagation for Virtual Screening Enhancement Controlling False Positive 

**Title (ZH)**: SubDyve: 以子图驱动的动力传播方法以提高虚拟筛选并控制假阳性 

**Authors**: Jungseob Yi, Seoyoung Choi, Sun Kim, Sangseon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.16273)  

**Abstract**: Virtual screening (VS) aims to identify bioactive compounds from vast chemical libraries, but remains difficult in low-label regimes where only a few actives are known. Existing methods largely rely on general-purpose molecular fingerprints and overlook class-discriminative substructures critical to bioactivity. Moreover, they consider molecules independently, limiting effectiveness in low-label regimes. We introduce SubDyve, a network-based VS framework that constructs a subgraph-aware similarity network and propagates activity signals from a small known actives. When few active compounds are available, SubDyve performs iterative seed refinement, incrementally promoting new candidates based on local false discovery rate. This strategy expands the seed set with promising candidates while controlling false positives from topological bias and overexpansion. We evaluate SubDyve on ten DUD-E targets under zero-shot conditions and on the CDK7 target with a 10-million-compound ZINC dataset. SubDyve consistently outperforms existing fingerprint or embedding-based approaches, achieving margins of up to +34.0 on the BEDROC and +24.6 on the EF1% metric. 

**Abstract (ZH)**: 基于网络的虚拟筛选框架SubDyve：识别小分子生物活性化合物 

---
# HausaMovieReview: A Benchmark Dataset for Sentiment Analysis in Low-Resource African Language 

**Title (ZH)**: HausaMovieReview：低资源非洲语言情感分析基准数据集 

**Authors**: Asiya Ibrahim Zanga, Salisu Mamman Abdulrahman, Abubakar Ado, Abdulkadir Abubakar Bichi, Lukman Aliyu Jibril, Abdulmajid Babangida Umar, Alhassan Adamu, Shamsuddeen Hassan Muhammad, Bashir Salisu Abubakar  

**Link**: [PDF](https://arxiv.org/pdf/2509.16256)  

**Abstract**: The development of Natural Language Processing (NLP) tools for low-resource languages is critically hindered by the scarcity of annotated datasets. This paper addresses this fundamental challenge by introducing HausaMovieReview, a novel benchmark dataset comprising 5,000 YouTube comments in Hausa and code-switched English. The dataset was meticulously annotated by three independent annotators, demonstrating a robust agreement with a Fleiss' Kappa score of 0.85 between annotators. We used this dataset to conduct a comparative analysis of classical models (Logistic Regression, Decision Tree, K-Nearest Neighbors) and fine-tuned transformer models (BERT and RoBERTa). Our results reveal a key finding: the Decision Tree classifier, with an accuracy and F1-score 89.72% and 89.60% respectively, significantly outperformed the deep learning models. Our findings also provide a robust baseline, demonstrating that effective feature engineering can enable classical models to achieve state-of-the-art performance in low-resource contexts, thereby laying a solid foundation for future research.
Keywords: Hausa, Kannywood, Low-Resource Languages, NLP, Sentiment Analysis 

**Abstract (ZH)**: 自然语言处理（NLP）工具在低资源语言的发展受到标注数据稀缺的严重影响。本文通过引入一个包含5000条豪萨语和混合英语YouTube评论的新颖基准数据集HausaMovieReview，来应对这一基本挑战。该数据集由三位独立注释者仔细标注，注释者间的一致性通过Fleiss' Kappa系数为0.85得到验证。我们使用该数据集对经典模型（逻辑回归、决策树、K最近邻）和微调的变压器模型（BERT和RoBERTa）进行了比较分析。研究结果表明，决策树分类器在准确率和F1分数方面分别达到了89.72%和89.60%，显著优于深度学习模型。我们的研究结果还提供了一个稳健的基准，展示了有效的特征工程可以使经典模型在低资源环境中达到最先进的性能，从而为未来的研究奠定了坚实的基础。

关键词：豪萨语，卡南伍德，低资源语言，自然语言处理，情感分析 

---
# R-Net: A Reliable and Resource-Efficient CNN for Colorectal Cancer Detection with XAI Integration 

**Title (ZH)**: R-Net：一种可靠的资源高效CNN在XAI集成的结直肠癌检测中 

**Authors**: Rokonozzaman Ayon, Md Taimur Ahad, Bo Song, Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16251)  

**Abstract**: State-of-the-art (SOTA) Convolutional Neural Networks (CNNs) are criticized for their extensive computational power, long training times, and large datasets. To overcome this limitation, we propose a reasonable network (R-Net), a lightweight CNN only to detect and classify colorectal cancer (CRC) using the Enteroscope Biopsy Histopathological Hematoxylin and Eosin Image Dataset (EBHI). Furthermore, six SOTA CNNs, including Multipath-based CNNs (DenseNet121, ResNet50), Depth-based CNNs (InceptionV3), width-based multi-connection CNNs (Xception), depth-wise separable convolutions (MobileNetV2), spatial exploitation-based CNNs (VGG16), Transfer learning, and two ensemble models are also tested on the same dataset. The ensemble models are a multipath-depth-width combination (DenseNet121-InceptionV3-Xception) and a multipath-depth-spatial combination (ResNet18-InceptionV3-VGG16). However, the proposed R-Net lightweight achieved 99.37% accuracy, outperforming MobileNet (95.83%) and ResNet50 (96.94%). Most importantly, to understand the decision-making of R-Net, Explainable AI such as SHAP, LIME, and Grad-CAM are integrated to visualize which parts of the EBHI image contribute to the detection and classification process of R-Net. The main novelty of this research lies in building a reliable, lightweight CNN R-Net that requires fewer computing resources yet maintains strong prediction results. SOTA CNNs, transfer learning, and ensemble models also extend our knowledge on CRC classification and detection. XAI functionality and the impact of pixel intensity on correct and incorrect classification images are also some novelties in CRC detection and classification. 

**Abstract (ZH)**: 最先进的卷积神经网络（SOTA CNNs）因耗用大量计算资源、训练时间长和对大数据集的需求而受到批评。为克服这一限制，我们提出了一种合理的网络（R-Net），这是一种仅用于检测和分类结直肠癌（CRC）的轻量级CNN，使用的是Enteroscope Biopsy Histopathological Hematoxylin和Eosin图像数据集（EBHI）。此外，还在同一数据集上测试了六种SOTA CNNs，包括基于多路径的CNNs（DenseNet121、ResNet50）、基于深度的CNNs（InceptionV3）、基于宽度的多连接CNNs（Xception）、深度可分离卷积（MobileNetV2）、基于空间利用的CNNs（VGG16）、迁移学习以及两种集成模型。集成模型包括多路径-深度-宽度组合（DenseNet121-InceptionV3-Xception）和多路径-深度-空间分辨率组合（ResNet18-InceptionV3-VGG16）。然而，提出的R-Net轻量级网络实现了99.37%的准确率，优于MobileNet（95.83%）和ResNet50（96.94%）。最重要的是，为了理解R-Net的决策过程，结合了可解释的人工智能技术（如SHAP、LIME和Grad-CAM），可视化了EBHI图像中哪些部分对R-Net的检测和分类过程有贡献。该研究的主要创新在于构建了一种可靠的轻量级CNN R-Net，其计算资源需求较少但仍能保持强大的预测性能。SOTA CNNs、迁移学习和集成模型拓展了我们对CRC分类和检测的知识。R-Net的可解释性和像素强度对正确和错误分类图像的影响也是CRC检测和分类中的创新点。 

---
# REAMS: Reasoning Enhanced Algorithm for Maths Solving 

**Title (ZH)**: REAMS: 增强推理算法 for 数学求解 

**Authors**: Eishkaran Singh, Tanav Singh Bajaj, Siddharth Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2509.16241)  

**Abstract**: The challenges of solving complex university-level mathematics problems, particularly those from MIT, and Columbia University courses, and selected tasks from the MATH dataset, remain a significant obstacle in the field of artificial intelligence. Conventional methods have consistently fallen short in this domain, highlighting the need for more advanced approaches. In this paper, we introduce a language-based solution that leverages zero-shot learning and mathematical reasoning to effectively solve, explain, and generate solutions for these advanced math problems. By integrating program synthesis, our method reduces reliance on large-scale training data while significantly improving problem-solving accuracy. Our approach achieves an accuracy of 90.15%, representing a substantial improvement over the previous benchmark of 81% and setting a new standard in automated mathematical problem-solving. These findings highlight the significant potential of advanced AI methodologies to address and overcome the challenges presented by some of the most complex mathematical courses and datasets. 

**Abstract (ZH)**: 解决麻省理工学院、哥伦比亚大学高等数学课程以及MATH数据集中选定任务的复杂大学级数学问题的挑战仍然在人工智能领域构成重大障碍。传统的解决方法在此领域一直未能满足需求，凸显了需要更高级方法的重要性。在本文中，我们引入了一种基于语言的解决方案，利用零样本学习和数学推理来有效解决、解释和生成这些高级数学问题的答案。通过结合程序合成，我们的方法减少了对大规模训练数据的依赖，并显著提高了问题解决的准确性。我们的方法实现了90.15%的准确率，比之前的基准81%有显著改进，并在自动化数学问题解决中确立了新标准。这些发现突显了高级人工智能方法在解决和克服一些最复杂数学课程和数据集带来的挑战方面的巨大潜力。 

---
# Discovering Software Parallelization Points Using Deep Neural Networks 

**Title (ZH)**: 使用深度神经网络发现软件并行化点 

**Authors**: Izavan dos S. Correia, Henrique C. T. Santos, Tiago A. E. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2509.16215)  

**Abstract**: This study proposes a deep learning-based approach for discovering loops in programming code according to their potential for parallelization. Two genetic algorithm-based code generators were developed to produce two distinct types of code: (i) independent loops, which are parallelizable, and (ii) ambiguous loops, whose dependencies are unclear, making them impossible to define if the loop is parallelizable or not. The generated code snippets were tokenized and preprocessed to ensure a robust dataset. Two deep learning models - a Deep Neural Network (DNN) and a Convolutional Neural Network (CNN) - were implemented to perform the classification. Based on 30 independent runs, a robust statistical analysis was employed to verify the expected performance of both models, DNN and CNN. The CNN showed a slightly higher mean performance, but the two models had a similar variability. Experiments with varying dataset sizes highlighted the importance of data diversity for model performance. These results demonstrate the feasibility of using deep learning to automate the identification of parallelizable structures in code, offering a promising tool for software optimization and performance improvement. 

**Abstract (ZH)**: 基于深度学习的编程代码并行化循环发现方法研究：两种遗传算法生成器用于生成独立循环和模糊循环的代码片段，并通过统计分析验证深度神经网络和卷积神经网络模型的性能，强调数据多样性对模型性能的重要性，展示了使用深度学习自动化识别可并行化代码结构的可行性，为软件优化和性能提升提供有前景的工具。 

---
# DarwinWafer: A Wafer-Scale Neuromorphic Chip 

**Title (ZH)**: 达尔文圆片：一种晶圆尺度的类脑芯片 

**Authors**: Xiaolei Zhu, Xiaofei Jin, Ziyang Kang, Chonghui Sun, Junjie Feng, Dingwen Hu, Zengyi Wang, Hanyue Zhuang, Qian Zheng, Huajin Tang, Shi Gu, Xin Du, De Ma, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16213)  

**Abstract**: Neuromorphic computing promises brain-like efficiency, yet today's multi-chip systems scale over PCBs and incur orders-of-magnitude penalties in bandwidth, latency, and energy, undermining biological algorithms and system efficiency. We present DarwinWafer, a hyperscale system-on-wafer that replaces off-chip interconnects with wafer-scale, high-density integration of 64 Darwin3 chiplets on a 300 mm silicon interposer. A GALS NoC within each chiplet and an AER-based asynchronous wafer fabric with hierarchical time-step synchronization provide low-latency, coherent operation across the wafer. Each chiplet implements 2.35 M neurons and 0.1 B synapses, yielding 0.15 B neurons and 6.4 B synapses per this http URL 333 MHz and 0.8 V, DarwinWafer consumes ~100 W and achieves 4.9 pJ/SOP, with 64 TSOPS peak throughput (0.64 TSOPS/W). Realization is enabled by a holistic chiplet-interposer co-design flow (including an in-house interposer-bump planner with early SI/PI and electro-thermal closure) and a warpage-tolerant assembly that fans out I/O via PCBlets and compliant pogo-pin connections, enabling robust, demountable wafer-to-board integration. Measurements confirm 10 mV supply droop and a uniform thermal profile (34-36 °C) under ~100 W. Application studies demonstrate whole-brain simulations: two zebrafish brains per chiplet with high connectivity fidelity (Spearman r = 0.896) and a mouse brain mapped across 32 chiplets (r = 0.645). To our knowledge, DarwinWafer represents a pioneering demonstration of wafer-scale neuromorphic computing, establishing a viable and scalable path toward large-scale, brain-like computation on silicon by replacing PCB-level interconnects with high-density, on-wafer integration. 

**Abstract (ZH)**: Neuromorphic Computing on a Hyperscale System-on-Wafer: DarwinWafer实现脑-like 效率的亚微米级集成 

---
# EPIC: Generative AI Platform for Accelerating HPC Operational Data Analytics 

**Title (ZH)**: EPIC：加速HPC运营数据解析的生成式AI平台 

**Authors**: Ahmad Maroof Karimi, Woong Shin, Jesse Hines, Tirthankar Ghosal, Naw Safrin Sattar, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16212)  

**Abstract**: We present EPIC, an AI-driven platform designed to augment operational data analytics. EPIC employs a hierarchical multi-agent architecture where a top-level large language model provides query processing, reasoning and synthesis capabilities. These capabilities orchestrate three specialized low-level agents for information retrieval, descriptive analytics, and predictive analytics. This architecture enables EPIC to perform HPC operational analytics on multi-modal data, including text, images, and tabular formats, dynamically and iteratively. EPIC addresses the limitations of existing HPC operational analytics approaches, which rely on static methods that struggle to adapt to evolving analytics tasks and stakeholder demands.
Through extensive evaluations on the Frontier HPC system, we demonstrate that EPIC effectively handles complex queries. Using descriptive analytics as a use case, fine-tuned smaller models outperform large state-of-the-art foundation models, achieving up to 26% higher accuracy. Additionally, we achieved 19x savings in LLM operational costs compared to proprietary solutions by employing a hybrid approach that combines large foundational models with fine-tuned local open-weight models. 

**Abstract (ZH)**: 我们呈现了EPIC，一个基于AI的平台，旨在增强操作数据 analytics。EPIC采用分层多代理架构，其中高层的大语言模型提供查询处理、推理和综合能力。这些能力协调三个专门的低层代理，分别用于信息检索、描述性分析和预测性分析。该架构使EPIC能够动态和迭代地在多模态数据（包括文本、图像和表格格式）上执行HPC操作分析。EPIC解决了现有HPC操作分析方法的局限性，这些方法依赖于静态方法，难以适应不断变化的分析任务和利益相关者的需求。

通过在Frontier HPC系统上的广泛评估，我们证明EPIC能够有效处理复杂的查询。以描述性分析为例，微调的小模型优于大型最先进的基础模型，准确率高出26%。此外，通过结合大型基础模型和微调的本地开源模型的混合方法，我们实现了与专有解决方案相比高达19倍的LLM运营成本节省。 

---
# Breast Cancer Classification Using Gradient Boosting Algorithms Focusing on Reducing the False Negative and SHAP for Explainability 

**Title (ZH)**: 基于减少假阴性的渐增梯度算法的乳腺癌分类及其可解释性分析 

**Authors**: João Manoel Herrera Pinheiro, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2403.09548)  

**Abstract**: Cancer is one of the diseases that kill the most women in the world, with breast cancer being responsible for the highest number of cancer cases and consequently deaths. However, it can be prevented by early detection and, consequently, early treatment. Any development for detection or perdition this kind of cancer is important for a better healthy life. Many studies focus on a model with high accuracy in cancer prediction, but sometimes accuracy alone may not always be a reliable metric. This study implies an investigative approach to studying the performance of different machine learning algorithms based on boosting to predict breast cancer focusing on the recall metric. Boosting machine learning algorithms has been proven to be an effective tool for detecting medical diseases. The dataset of the University of California, Irvine (UCI) repository has been utilized to train and test the model classifier that contains their attributes. The main objective of this study is to use state-of-the-art boosting algorithms such as AdaBoost, XGBoost, CatBoost and LightGBM to predict and diagnose breast cancer and to find the most effective metric regarding recall, ROC-AUC, and confusion matrix. Furthermore, our study is the first to use these four boosting algorithms with Optuna, a library for hyperparameter optimization, and the SHAP method to improve the interpretability of our model, which can be used as a support to identify and predict breast cancer. We were able to improve AUC or recall for all the models and reduce the False Negative for AdaBoost and LigthGBM the final AUC were more than 99.41\% for all models. 

**Abstract (ZH)**: 乳腺癌是导致全球女性死亡的主要疾病之一，其中乳腺癌是最常见的癌症类型，也是导致癌症死亡的主要原因。然而，通过早期检测和早期治疗可以预防乳腺癌。对于这种癌症的任何检测或预防方法的发展都对于提高健康生活至关重要。许多研究侧重于高精度的癌症预测模型，但有时仅依赖准确性可能并不是一个可靠的指标。本研究采用基于提升的方法调查了不同机器学习算法在预测乳腺癌方面的性能，重点关注召回率指标。提升机器学习算法已被证明是检测医学疾病的有效工具。利用加州大学欧文分校（UCI）数据集训练和测试了包含其特征的模型分类器。本研究的主要目标是使用先进的提升算法如AdaBoost、XGBoost、CatBoost和LightGBM预测和诊断乳腺癌，并找到最有效的召回率、ROC-AUC和混淆矩阵指标。此外，本研究是首次使用这四种提升算法结合Optuna（超参数优化库）和SHAP方法提高模型的可解释性，可作为识别和预测乳腺癌的支持工具。所有模型的AUC或召回率均有所提高，最终AUC均超过99.41%，对于AdaBoost和LightGBM减少了假阴性。 

---
