# Multi-Mapcher: Loop Closure Detection-Free Heterogeneous LiDAR Multi-Session SLAM Leveraging Outlier-Robust Registration for Autonomous Vehicles 

**Title (ZH)**: 多地图匹配器：基于离群值鲁棒注册的自主车辆异构LiDAR多会话SLAM 

**Authors**: Hyungtae Lim, Daebeom Kim, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2511.00635)  

**Abstract**: As various 3D light detection and ranging (LiDAR) sensors have been introduced to the market, research on multi-session simultaneous localization and mapping (MSS) using heterogeneous LiDAR sensors has been actively conducted. Existing MSS methods mostly rely on loop closure detection for inter-session alignment; however, the performance of loop closure detection can be potentially degraded owing to the differences in the density and field of view (FoV) of the sensors used in different sessions. In this study, we challenge the existing paradigm that relies heavily on loop detection modules and propose a novel MSS framework, called Multi-Mapcher, that employs large-scale map-to-map registration to perform inter-session initial alignment, which is commonly assumed to be infeasible, by leveraging outlier-robust 3D point cloud registration. Next, after finding inter-session loops by radius search based on the assumption that the inter-session initial alignment is sufficiently precise, anchor node-based robust pose graph optimization is employed to build a consistent global map. As demonstrated in our experiments, our approach shows substantially better MSS performance for various LiDAR sensors used to capture the sessions and is faster than state-of-the-art approaches. Our code is available at this https URL. 

**Abstract (ZH)**: 各种3D激光雷达（LiDAR）传感器引入市场后，关于使用异构LiDAR传感器进行多会话同时定位与建图（MSS）的研究一直十分活跃。现有MSS方法大多依赖环路闭合检测进行跨会话对准，但由于不同会话中使用的传感器密度和视场（FoV）差异，环路闭合检测的性能可能会被潜在地削弱。本研究挑战了高度依赖环路检测模块的现有范式，提出了一种名为Multi-Mapcher的新MSS框架，通过利用鲁棒的3D点云对准进行跨会话初始对准，从而实现通常认为不可行的大规模地图到地图注册。接着，在初始对准充分精确的假设下通过基于半径搜索的方法找到跨会话环路，并采用锚节点基于的鲁棒位姿图优化构建一致的全局地图。实验结果表明，我们的方法对于用于捕捉会话的各种LiDAR传感器显示出显著更好的MSS性能，并且比目前最先进的方法更快。我们的代码可访问于此网址。 

---
# Runge-Kutta Approximations for Direct Coning Compensation Applying Lie Theory 

**Title (ZH)**: Runge-Kutta 近似在李群理论应用于直接会流修正中的应用 

**Authors**: John A. Christian, Michael R. Walker II, Wyatt Bridgman, Michael J. Sparapany  

**Link**: [PDF](https://arxiv.org/pdf/2511.00412)  

**Abstract**: The integration of gyroscope measurements is an essential task for most navigation systems. Modern vehicles typically use strapdown systems, such that gyro integration requires coning compensation to account for the sensor's rotation during the integration. Many coning compensation algorithms have been developed and a few are reviewed. This work introduces a new class of coning correction algorithm built directly from the classical Runge-Kutta integration routines. A simple case is shown to collapse to one of the most popular coning algorithms and a clear procedure for generating higher-order algorithms is presented. 

**Abstract (ZH)**: 基于经典Runge-Kutta积分方法的新型俯仰修正算法的研究 

---
# SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping 

**Title (ZH)**: SonarSweep：融合声纳与视觉实现稳健的平面扫描三维重建 

**Authors**: Lingpeng Chen, Jiakun Tang, Apple Pui-Yi Chui, Ziyang Hong, Junfeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00392)  

**Abstract**: Accurate 3D reconstruction in visually-degraded underwater environments remains a formidable challenge. Single-modality approaches are insufficient: vision-based methods fail due to poor visibility and geometric constraints, while sonar is crippled by inherent elevation ambiguity and low resolution. Consequently, prior fusion technique relies on heuristics and flawed geometric assumptions, leading to significant artifacts and an inability to model complex scenes. In this paper, we introduce SonarSweep, a novel, end-to-end deep learning framework that overcomes these limitations by adapting the principled plane sweep algorithm for cross-modal fusion between sonar and visual data. Extensive experiments in both high-fidelity simulation and real-world environments demonstrate that SonarSweep consistently generates dense and accurate depth maps, significantly outperforming state-of-the-art methods across challenging conditions, particularly in high turbidity. To foster further research, we will publicly release our code and a novel dataset featuring synchronized stereo-camera and sonar data, the first of its kind. 

**Abstract (ZH)**: 精确的无视觉降级水下环境三维重建仍然是一个严峻的挑战。单一模态方法不足：基于视觉的方法由于能见度差和几何约束而失效，而声纳因固有的视高模糊和低分辨率而受限。因此，先前的融合技术依赖于启发式方法和不准确的几何假设，导致显著的伪影且无法建模复杂场景。在本文中，我们提出了SonarSweep，一种新颖的端到端深度学习框架，通过将原理上的平面扫描算法适应应用于声纳和视觉数据之间的跨模态融合，从而克服了这些限制。在高保真仿真和真实环境中的广泛实验表明，SonarSweep 一致生成密集且准确的深度图，在多种条件下尤其是高浑浊度情况下显著优于现有最佳方法。为了促进进一步研究，我们将公开发布我们的代码和一个新的数据集，该数据集包含同步立体相机和声纳数据，是此类数据集中的第一个。 

---
# FGO MythBusters: Explaining how Kalman Filter variants achieve the same performance as FGO in navigation applications 

**Title (ZH)**: FGO神话破除：解释卡尔曼滤波器变体在导航应用中如何达到与FGO相同性能的方法 

**Authors**: Baoshan Song, Ruijie Xu, Li-Ta Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00306)  

**Abstract**: Sliding window-factor graph optimization (SW-FGO) has gained more and more attention in navigation research due to its robust approximation to non-Gaussian noises and nonlinearity of measuring models. There are lots of works focusing on its application performance compared to extended Kalman filter (EKF) but there is still a myth at the theoretical relationship between the SW-FGO and EKF. In this paper, we find the necessarily fair condition to connect SW-FGO and Kalman filter variants (KFV) (e.g., EKF, iterative EKF (IEKF), robust EKF (REKF) and robust iterative EKF (RIEKF)). Based on the conditions, we propose a recursive FGO (Re-FGO) framework to represent KFV under SW-FGO formulation. Under explicit conditions (Markov assumption, Gaussian noise with L2 loss, and a one-state window), Re-FGO regenerates exactly to EKF/IEKF/REKF/RIEKF, while SW-FGO shows measurable benefits in nonlinear, non-Gaussian regimes at a predictable compute cost. Finally, after clarifying the connection between them, we highlight the unique advantages of SW-FGO in practical phases, especially on numerical estimation and deep learning integration. The code and data used in this work is open sourced at this https URL. 

**Abstract (ZH)**: 滑动窗口-因子图优化（SW-FGO）因其对非高斯噪声和测量模型非线性的稳健近似而在导航研究中逐渐获得关注。尽管有许多工作关注其应用性能与扩展卡尔曼滤波器（EKF）的对比，但SW-FGO与EKF之间的理论关系仍然存在争议。本文找到了将SW-FGO与卡尔曼滤波器变体（KFV，如EKF、迭代EKF（IEKF）、稳健EKF（REKF）和稳健迭代EKF（RIEKF））相连接的必要公平条件。基于这些条件，我们提出了一种递归因子图优化（Re-FGO）框架，以SW-FGO形式表示KFV。在显式条件下（马尔可夫假设、L2损失的高斯噪声和单状态窗口），Re-FGO可以再生为EKF/IEKF/REKF/RIEKF，而SW-FGO则在非线性和非高斯环境中表现出可预测计算成本的优势。最后，在明确了两者之间的关系后，我们强调了SW-FGO在实际应用中的独特优势，特别是在数值估计和深度学习整合方面。本文使用的代码和数据已开源于此网址：此 https URL。 

---
# Fractional Diffusion Bridge Models 

**Title (ZH)**: 分数阶扩散桥模型 

**Authors**: Gabriel Nobis, Maximilian Springenberg, Arina Belova, Rembert Daems, Christoph Knochenhauer, Manfred Opper, Tolga Birdal, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2511.01795)  

**Abstract**: We present Fractional Diffusion Bridge Models (FDBM), a novel generative diffusion bridge framework driven by an approximation of the rich and non-Markovian fractional Brownian motion (fBM). Real stochastic processes exhibit a degree of memory effects (correlations in time), long-range dependencies, roughness and anomalous diffusion phenomena that are not captured in standard diffusion or bridge modeling due to the use of Brownian motion (BM). As a remedy, leveraging a recent Markovian approximation of fBM (MA-fBM), we construct FDBM that enable tractable inference while preserving the non-Markovian nature of fBM. We prove the existence of a coupling-preserving generative diffusion bridge and leverage it for future state prediction from paired training data. We then extend our formulation to the Schrödinger bridge problem and derive a principled loss function to learn the unpaired data translation. We evaluate FDBM on both tasks: predicting future protein conformations from aligned data, and unpaired image translation. In both settings, FDBM achieves superior performance compared to the Brownian baselines, yielding lower root mean squared deviation (RMSD) of C$_\alpha$ atomic positions in protein structure prediction and lower Fréchet Inception Distance (FID) in unpaired image translation. 

**Abstract (ZH)**: Fractional Diffusion Bridge Models：基于富集且非马尔可夫分数布朗运动的生成性扩散桥框架 

---
# Lyapunov Stability Learning with Nonlinear Control via Inductive Biases 

**Title (ZH)**: 非线性控制下的李雅普unov稳定性学习与归纳偏置 

**Authors**: Yupu Lu, Shijie Lin, Hao Xu, Zeqing Zhang, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01283)  

**Abstract**: Finding a control Lyapunov function (CLF) in a dynamical system with a controller is an effective way to guarantee stability, which is a crucial issue in safety-concerned applications. Recently, deep learning models representing CLFs have been applied into a learner-verifier framework to identify satisfiable candidates. However, the learner treats Lyapunov conditions as complex constraints for optimisation, which is hard to achieve global convergence. It is also too complicated to implement these Lyapunov conditions for verification. To improve this framework, we treat Lyapunov conditions as inductive biases and design a neural CLF and a CLF-based controller guided by this knowledge. This design enables a stable optimisation process with limited constraints, and allows end-to-end learning of both the CLF and the controller. Our approach achieves a higher convergence rate and larger region of attraction (ROA) in learning the CLF compared to existing methods among abundant experiment cases. We also thoroughly reveal why the success rate decreases with previous methods during learning. 

**Abstract (ZH)**: 在动态系统控制器中寻找控制李雅普诺夫函数（CLF）是保证稳定性的一种有效方法，这对于安全关切的应用至关重要。近年来，深度学习模型被用于CLF的学习-验证框架中以识别满足条件的候选者。然而，学习器将李雅普诺夫条件视为优化的复杂约束，难以实现全局收敛。同时，验证时实施这些李雅普诺夫条件也过于复杂。为了改进此框架，我们将李雅普诺夫条件视为归纳偏置，并设计了一个神经CLF以及基于CLF的控制器，该设计允许在有限约束下稳定优化过程，并能够端到端学习CLF和控制器。在多种实验案例中，我们的方法在学习CLF时实现了更高的收敛率和更宽的研究吸引域（ROA），并且还详细揭示了为什么在学习过程中先前方法的成功率会降低。 

---
# Saliency-Guided Domain Adaptation for Left-Hand Driving in Autonomous Steering 

**Title (ZH)**: 基于显著性引导的域适应方法用于自主转向的左驾驾驶场景äß
user
基于显著性引导的域适应方法用于自主驾驶左转向场景 

**Authors**: Zahra Mehraban, Sebastien Glaser, Michael Milford, Ronald Schroeter  

**Link**: [PDF](https://arxiv.org/pdf/2511.01223)  

**Abstract**: Domain adaptation is required for automated driving models to generalize well across diverse road conditions. This paper explores a training method for domain adaptation to adapt PilotNet, an end-to-end deep learning-based model, for left-hand driving conditions using real-world Australian highway data. Four training methods were evaluated: (1) a baseline model trained on U.S. right-hand driving data, (2) a model trained on flipped U.S. data, (3) a model pretrained on U.S. data and then fine-tuned on Australian highways, and (4) a model pretrained on flipped U.S. data and then finetuned on Australian highways. This setup examines whether incorporating flipped data enhances the model adaptation by providing an initial left-hand driving alignment. The paper compares model performance regarding steering prediction accuracy and attention, using saliency-based analysis to measure attention shifts across significant road regions. Results show that pretraining on flipped data alone worsens prediction stability due to misaligned feature representations, but significantly improves adaptation when followed by fine-tuning, leading to lower prediction error and stronger focus on left-side cues. To validate this approach across different architectures, the same experiments were done on ResNet, which confirmed similar adaptation trends. These findings emphasize the importance of preprocessing techniques, such as flipped-data pretraining, followed by fine-tuning to improve model adaptation with minimal retraining requirements. 

**Abstract (ZH)**: 自动化驾驶模型跨多样道路条件泛化的领域自适应训练方法探究：以澳大利亚高速公路数据适配左驾条件的PilotNet模型为例 

---
# pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis 

**Title (ZH)**: pacSTL: 基于数据驱动可达性分析的PAC有界信号时序逻辑 

**Authors**: Elizabeth Dietrich, Hanna Krasowski, Emir Cem Gezer, Roger Skjetne, Asgeir Johan Sørensen, Murat Arcak  

**Link**: [PDF](https://arxiv.org/pdf/2511.00934)  

**Abstract**: Real-world robotic systems must comply with safety requirements in the presence of uncertainty. To define and measure requirement adherence, Signal Temporal Logic (STL) offers a mathematically rigorous and expressive language. However, standard STL cannot account for uncertainty. We address this problem by presenting pacSTL, a framework that combines Probably Approximately Correct (PAC) bounded set predictions with an interval extension of STL through optimization problems on the atomic proposition level. pacSTL provides PAC-bounded robustness intervals on the specification level that can be utilized in monitoring. We demonstrate the effectiveness of this approach through maritime navigation and analyze the efficiency and scalability of pacSTL through simulation and real-world experimentation on model vessels. 

**Abstract (ZH)**: 真实世界中的机器人系统在不确定性存在的情况下必须遵守安全要求。为定义和衡量要求的遵守程度，时序逻辑信号（STL）提供了一种数学严谨且表达能力较强的语言。然而，标准STL无法处理不确定性问题。为解决这一问题，我们提出了pacSTL框架，该框架结合了Probably Approximately Correct（PAC）有界集预测和STL的区间扩展，并在原子命题层面通过优化问题实现。pacSTL在规范层面提供了PAC有界稳健性区间，可用于监控。通过海上导航实例展示了该方法的有效性，并通过模拟和实际船舶实验分析了pacSTL的效率和可扩展性。 

---
# X-TRACK: Physics-Aware xLSTM for Realistic Vehicle Trajectory Prediction 

**Title (ZH)**: X-TRACK：物理意识的xLSTM车辆轨迹预测 

**Authors**: Aanchal Rajesh Chugh, Marion Neumeier, Sebastian Dorn  

**Link**: [PDF](https://arxiv.org/pdf/2511.00266)  

**Abstract**: Recent advancements in Recurrent Neural Network (RNN) architectures, particularly the Extended Long Short Term Memory (xLSTM), have addressed the limitations of traditional Long Short Term Memory (LSTM) networks by introducing exponential gating and enhanced memory structures. These improvements make xLSTM suitable for time-series prediction tasks as they exhibit the ability to model long-term temporal dependencies better than LSTMs. Despite their potential, these xLSTM-based models remain largely unexplored in the context of vehicle trajectory prediction. Therefore, this paper introduces a novel xLSTM-based vehicle trajectory prediction framework, X-TRAJ, and its physics-aware variant, X-TRACK (eXtended LSTM for TRAjectory prediction Constraint by Kinematics), which explicitly integrates vehicle motion kinematics into the model learning process. By introducing physical constraints, the proposed model generates realistic and feasible trajectories. A comprehensive evaluation on the highD and NGSIM datasets demonstrates that X-TRACK outperforms state-of-the-art baselines. 

**Abstract (ZH)**: Recent advancements in Recurrent Neural Network (RNN) architectures, particularly the Extended Long Short Term Memory (xLSTM), have addressed the limitations of traditional Long Short Term Memory (LSTM) networks by introducing exponential gating and enhanced memory structures. These improvements make xLSTM suitable for time-series prediction tasks as they exhibit the ability to model long-term temporal dependencies better than LSTMs. Despite their potential, these xLSTM-based models remain largely unexplored in the context of vehicle trajectory prediction. Therefore, this paper introduces a novel xLSTM-based vehicle trajectory prediction framework, X-TRAJ, and its physics-aware variant, X-TRACK (eXtended LSTM for TRAjectory prediction Constraint by Kinematics), which explicitly integrates vehicle motion kinematics into the model learning process. By introducing physical constraints, the proposed model generates realistic and feasible trajectories. A comprehensive evaluation on the highD and NGSIM datasets demonstrates that X-TRACK outperforms state-of-the-art baselines。

翻译后的标题：

基于扩展长短期记忆网络（xLSTM）的车辆轨迹预测框架X-TRACK及其动力学aware变体的研究 

---
# Which LiDAR scanning pattern is better for roadside perception: Repetitive or Non-repetitive? 

**Title (ZH)**: 路边感知中哪种LiDAR扫描模式更优：重复扫描还是非重复扫描？ 

**Authors**: Zhiqi Qi, Runxin Zhao, Hanyang Zhuang, Chunxiang Wang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00060)  

**Abstract**: LiDAR-based roadside perception is a cornerstone of advanced Intelligent Transportation Systems (ITS). While considerable research has addressed optimal LiDAR placement for infrastructure, the profound impact of differing LiDAR scanning patterns on perceptual performance remains comparatively under-investigated. The inherent nature of various scanning modes - such as traditional repetitive (mechanical/solid-state) versus emerging non-repetitive (e.g. prism-based) systems - leads to distinct point cloud distributions at varying distances, critically dictating the efficacy of object detection and overall environmental understanding. To systematically investigate these differences in infrastructure-based contexts, we introduce the "InfraLiDARs' Benchmark," a novel dataset meticulously collected in the CARLA simulation environment using concurrently operating infrastructure-based LiDARs exhibiting both scanning paradigms. Leveraging this benchmark, we conduct a comprehensive statistical analysis of the respective LiDAR scanning abilities and evaluate the impact of these distinct patterns on the performance of various leading 3D object detection algorithms. Our findings reveal that non-repetitive scanning LiDAR and the 128-line repetitive LiDAR were found to exhibit comparable detection performance across various scenarios. Despite non-repetitive LiDAR's limited perception range, it's a cost-effective option considering its low price. Ultimately, this study provides insights for setting up roadside perception system with optimal LiDAR scanning patterns and compatible algorithms for diverse roadside applications, and publicly releases the "InfraLiDARs' Benchmark" dataset to foster further research. 

**Abstract (ZH)**: 基于LiDAR的道路侧感知是先进智能运输系统（ITS）的基石。尽管已有大量研究关注基础设施的最优LiDAR布设，但不同LiDAR扫描模式对感知性能的深远影响仍相对较少被探究。各种扫描模式的固有特性——如传统的重复扫描（机械式/固态）与新兴的非重复扫描（如棱镜式）系统——造成了在不同距离处点云分布的差异，直接影响到对象检测效果及整体环境理解的效度。为系统性地探讨这些差异，我们引入了“InfraLiDARs’ Benchmark”，一个在CARLA仿真环境中采用同时运行的、展示两种扫描模式的基础设施基部署LiDAR精心收集的新颖数据集。利用这一基准，我们进行了全面的统计分析，评估了不同LiDAR扫描模式对各种领先三维物体检测算法性能的影响。研究结果表明，非重复扫描LiDAR与128线重复扫描LiDAR在多种场景下检测性能相近。尽管非重复扫描LiDAR的感知范围有限，但其低廉的价格使其成为一种经济有效的选择。最终，本研究为设计具有最优LiDAR扫描模式和兼容算法的道路侧感知系统提供了见解，并公开发布了“InfraLiDARs’ Benchmark”数据集，以促进进一步的研究。 

---
# Hybrid Retrieval-Augmented Generation Agent for Trustworthy Legal Question Answering in Judicial Forensics 

**Title (ZH)**: 司法鉴定中可信赖法律问答的混合检索增强生成代理 

**Authors**: Yueqing Xi, Yifan Bai, Huasen Luo, Weiliang Wen, Hui Liu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01668)  

**Abstract**: As artificial intelligence permeates judicial forensics, ensuring the veracity and traceability of legal question answering (QA) has become critical. Conventional large language models (LLMs) are prone to hallucination, risking misleading guidance in legal consultation, while static knowledge bases struggle to keep pace with frequently updated statutes and case law. We present a hybrid legal QA agent tailored for judicial settings that integrates retrieval-augmented generation (RAG) with multi-model ensembling to deliver reliable, auditable, and continuously updatable counsel. The system prioritizes retrieval over generation: when a trusted legal repository yields relevant evidence, answers are produced via RAG; otherwise, multiple LLMs generate candidates that are scored by a specialized selector, with the top-ranked answer returned. High-quality outputs then undergo human review before being written back to the repository, enabling dynamic knowledge evolution and provenance tracking. Experiments on the Law\_QA dataset show that our hybrid approach significantly outperforms both a single-model baseline and a vanilla RAG pipeline on F1, ROUGE-L, and an LLM-as-a-Judge metric. Ablations confirm the complementary contributions of retrieval prioritization, model ensembling, and the human-in-the-loop update mechanism. The proposed system demonstrably reduces hallucination while improving answer quality and legal compliance, advancing the practical landing of media forensics technologies in judicial scenarios. 

**Abstract (ZH)**: 随着人工智能渗透到司法取证领域，确保法律问答的准确性和可追溯性变得至关重要。传统的大型语言模型容易产生幻觉，这可能在法律咨询中误导指导，而静态知识库则难以跟上频繁更新的法律法规。我们提出了一种针对司法环境设计的混合法律问答代理，该代理结合了检索增强生成（RAG）与多模型集成，以提供可靠、可审计并可持续更新的咨询。该系统优先考虑检索：当可信赖的法律 repository 提供相关证据时，使用 RAG 生成答案；否则，多个大型语言模型生成候选答案，由专门的筛选器评估，排名第一的候选答案被返回。高质量的输出随后经过人工审核并重新写入 repository，以实现动态知识演化和溯源跟踪。在 Law_QA 数据集上的实验显示，我们的混合方法在 F1、ROUGE-L 和 LLM-as-a-Judge 指标上均显著优于单模型基线和标准 RAG 管道。消融实验确认了优先检索、模型集成和人工在环更新机制的互补贡献。所提出的系统显着减少了幻觉，提高了答案质量和法律合规性，推动了媒体取证技术在司法场景中的实际应用。 

---
# IVGAE-TAMA-BO: A novel temporal dynamic variational graph model for link prediction in global food trade networks with momentum structural memory and Bayesian optimization 

**Title (ZH)**: IVGAE-TAMA-BO：一种具有动量结构记忆和贝叶斯优化的新型时间动态变分图模型，用于全球食品贸易网络中的链接预测 

**Authors**: Sicheng Wang, Shuhao Chen, Jingran Zhou, Chengyi Tu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01639)  

**Abstract**: Global food trade plays a crucial role in ensuring food security and maintaining supply chain stability. However, its network structure evolves dynamically under the influence of geopolitical, economic, and environmental factors, making it challenging to model and predict future trade links. Effectively capturing temporal patterns in food trade networks is therefore essential for improving the accuracy and robustness of link prediction. This study introduces IVGAE-TAMA-BO, a novel dynamic graph neural network designed to model evolving trade structures and predict future links in global food trade networks. To the best of our knowledge, this is the first work to apply dynamic graph neural networks to this domain, significantly enhancing predictive performance. Building upon the original IVGAE framework, the proposed model incorporates a Trade-Aware Momentum Aggregator (TAMA) to capture the temporal evolution of trade networks, jointly modeling short-term fluctuations and long-term structural dependencies. A momentum-based structural memory mechanism further improves predictive stability and performance. In addition, Bayesian optimization is used to automatically tune key hyperparameters, enhancing generalization across diverse trade scenarios. Extensive experiments on five crop-specific datasets demonstrate that IVGAE-TAMA substantially outperforms the static IVGAE and other dynamic baselines by effectively modeling temporal dependencies, while Bayesian optimization further boosts performance in IVGAE-TAMA-BO. These results highlight the proposed framework as a robust and scalable solution for structural prediction in global trade networks, with strong potential for applications in food security monitoring and policy decision support. 

**Abstract (ZH)**: 全球粮食贸易在网络结构动态变化背景下对确保粮食安全和维持供应链稳定性起着关键作用。然而，其网络结构受到地缘政治、经济和环境因素的影响而动态变化，使得对其进行建模和预测具有挑战性。有效捕捉粮食贸易网络中的时间模式对于提高预测链接的准确性和稳健性至关重要。本研究引入了IVGAE-TAMA-BO，这是一种新型动态图神经网络，用于建模演变中的贸易结构并预测全球粮食贸易网络中的未来链接。据我们所知，这是首次将动态图神经网络应用于该领域，显著提升了预测性能。该模型在原始IVGAE框架的基础上，引入了贸易感知动量聚合器（TAMA），以捕捉贸易网络的时间演化，并同时建模短期波动和长期结构依赖性。基于动量的结构记忆机制进一步提高了预测稳定性和性能。此外，使用贝叶斯优化自动调整关键超参数，从而增强了对不同贸易情景的一般化能力。通过对五种作物特定数据集的广泛实验表明，IVGAE-TAMA在有效建模时间依赖性方面显著优于静态IVGAE和其他动态基线，而贝叶斯优化进一步提高了IVGAE-TAMA-BO的性能。这些结果突显了所提出的框架作为全球贸易网络结构预测的稳健且可扩展的解决方案，具有在粮食安全监测和政策决策支持方面的强大应用潜力。 

---
# Analyzing Sustainability Messaging in Large-Scale Corporate Social Media 

**Title (ZH)**: 分析大型企业社交媒体中的可持续性信息传播 

**Authors**: Ujjwal Sharma, Stevan Rudinac, Ana Mićković, Willemijn van Dolen, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2511.01550)  

**Abstract**: In this work, we introduce a multimodal analysis pipeline that leverages large foundation models in vision and language to analyze corporate social media content, with a focus on sustainability-related communication. Addressing the challenges of evolving, multimodal, and often ambiguous corporate messaging on platforms such as X (formerly Twitter), we employ an ensemble of large language models (LLMs) to annotate a large corpus of corporate tweets on their topical alignment with the 17 Sustainable Development Goals (SDGs). This approach avoids the need for costly, task-specific annotations and explores the potential of such models as ad-hoc annotators for social media data that can efficiently capture both explicit and implicit references to sustainability themes in a scalable manner. Complementing this textual analysis, we utilize vision-language models (VLMs), within a visual understanding framework that uses semantic clusters to uncover patterns in visual sustainability communication. This integrated approach reveals sectoral differences in SDG engagement, temporal trends, and associations between corporate messaging, environmental, social, governance (ESG) risks, and consumer engagement. Our methods-automatic label generation and semantic visual clustering-are broadly applicable to other domains and offer a flexible framework for large-scale social media analysis. 

**Abstract (ZH)**: 本研究引入了一种结合视觉和语言大型基础模型的多模态分析pipeline，专注于分析 CORPORATE 社交媒体内容中的可持续性相关沟通。面对 X 平台（原 Twitter）上不断变化、多模态且常含模糊性的企业信息传递挑战，我们采用了大型语言模型（LLM）的集成方法，对大量企业的推文进行注释，标注其与17个可持续发展目标（SDGs）的专题契合度。该方法避免了昂贵且针对特定任务的注释需求，探索了此类模型作为社交媒体数据的即用型注释工具的潜力，能够高效地捕捉显性和隐性的可持续性主题参考，并以可扩展的方式进行分析。结合文本分析，我们利用视觉语言模型（VLM）在使用语义簇进行视觉可持续性沟通模式发现的框架内进行分析。这种综合方法揭示了不同行业在SDG参与方面的差异、时间趋势，以及企业信息传递、环境、社会、治理体系（ESG）风险和消费者参与之间的关联。我们的方法——自动标签生成和语义视觉聚类——广泛适用于其他领域，提供了一个灵活的框架，用于大规模社交媒体分析。 

---
# From Passive to Proactive: A Multi-Agent System with Dynamic Task Orchestration for Intelligent Medical Pre-Consultation 

**Title (ZH)**: 从被动到主动：一种基于动态任务 orchestration 的智能医疗预咨询多-agent 系统 

**Authors**: ChengZhang Yu, YingRu He, Hongyan Cheng, nuo Cheng, Zhixing Liu, Dongxu Mu, Zhangrui Shen, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01445)  

**Abstract**: Global healthcare systems face critical challenges from increasing patient volumes and limited consultation times, with primary care visits averaging under 5 minutes in many countries. While pre-consultation processes encompassing triage and structured history-taking offer potential solutions, they remain limited by passive interaction paradigms and context management challenges in existing AI systems. This study introduces a hierarchical multi-agent framework that transforms passive medical AI systems into proactive inquiry agents through autonomous task orchestration. We developed an eight-agent architecture with centralized control mechanisms that decomposes pre-consultation into four primary tasks: Triage ($T_1$), History of Present Illness collection ($T_2$), Past History collection ($T_3$), and Chief Complaint generation ($T_4$), with $T_1$--$T_3$ further divided into 13 domain-specific subtasks. Evaluated on 1,372 validated electronic health records from a Chinese medical platform across multiple foundation models (GPT-OSS 20B, Qwen3-8B, Phi4-14B), the framework achieved 87.0% accuracy for primary department triage and 80.5% for secondary department classification, with task completion rates reaching 98.2% using agent-driven scheduling versus 93.1% with sequential processing. Clinical quality scores from 18 physicians averaged 4.56 for Chief Complaints, 4.48 for History of Present Illness, and 4.69 for Past History on a 5-point scale, with consultations completed within 12.7 rounds for $T_2$ and 16.9 rounds for $T_3$. The model-agnostic architecture maintained high performance across different foundation models while preserving data privacy through local deployment, demonstrating the potential for autonomous AI systems to enhance pre-consultation efficiency and quality in clinical settings. 

**Abstract (ZH)**: 全球医疗保健系统面临日益增长的患者量和有限的问诊时间的严峻挑战，许多国家的首诊平均时间不足5分钟。虽然包含分诊和结构化病史采集的预问诊过程提供了潜在的解决方案，但现有的人工智能系统仍受限于被动交互范式和上下文管理挑战。本研究引入了一个分层多 agent 框架，通过自主任务协调将被动的医疗 AI 系统转变为积极的查询代理。我们开发了一种由八个 agent 构成的架构，并配备了集中控制机制，将预问诊分解为四个主要任务：分诊 ($T_1$)、现病史收集 ($T_2$)、既往史收集 ($T_3$) 和主诉生成 ($T_4$)，其中 $T_1$ 到 $T_3$ 进一步细分为 13 个特定领域的子任务。该框架在一家中国医疗平台上的 1,372 份经过验证的电子病历数据中，针对多种基础模型（GPT-OSS 20B、Qwen3-8B、Phi4-14B）进行了评估，首诊部门分诊的准确率为 87.0%，二级部门分类的准确率为 80.5%，相较顺序处理其任务完成率达到了 98.2%，而人为调度为 93.1%。18 位临床医师对主诉、现病史和既往史的临床质量评分分别为 4.56、4.48 和 4.69（满分 5 分），$T_2$ 在 12.7 个轮次内完成了会诊，$T_3$ 则在 16.9 个轮次内完成。该无模型架构在不同基础模型中保持了高性能，通过本地部署保护了数据隐私，展示了自主人工智能系统在临床预问诊效率和质量方面具有增强潜力。 

---
# Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering 

**Title (ZH)**: 在Cluster-DAGs中放松分区可接受性：任意变量聚类的因果算子 

**Authors**: Clément Yvernes, Emilie Devijver, Adèle H. Ribeiro, Marianne Clausel--Lesourd, Éric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2511.01396)  

**Abstract**: Cluster DAGs (C-DAGs) provide an abstraction of causal graphs in which nodes represent clusters of variables, and edges encode both cluster-level causal relationships and dependencies arisen from unobserved confounding. C-DAGs define an equivalence class of acyclic causal graphs that agree on cluster-level relationships, enabling causal reasoning at a higher level of abstraction. However, when the chosen clustering induces cycles in the resulting C-DAG, the partition is deemed inadmissible under conventional C-DAG semantics. In this work, we extend the C-DAG framework to support arbitrary variable clusterings by relaxing the partition admissibility constraint, thereby allowing cyclic C-DAG representations. We extend the notions of d-separation and causal calculus to this setting, significantly broadening the scope of causal reasoning across clusters and enabling the application of C-DAGs in previously intractable scenarios. Our calculus is both sound and atomically complete with respect to the do-calculus: all valid interventional queries at the cluster level can be derived using our rules, each corresponding to a primitive do-calculus step. 

**Abstract (ZH)**: C-DAGs提供了一种因果图的抽象，在其中节点表示变量的聚类，边既编码了聚类水平的因果关系，也编码了由未观测混杂因素引起的依赖关系。C-DAG定义了一个在聚类水平关系一致的有向无环因果图等价类，促进了更高层次抽象上的因果推理。然而，当选定的聚类在C-DAG中引入环路时，该划分被认为不符合传统的C-DAG语义。在本工作中，我们扩展了C-DAG框架，通过放宽分区的可接受性约束，从而支持任意变量聚类，并允许循环C-DAG表示。我们将d-分离和因果推理的概念扩展到这一框架下，显著地拓宽了跨聚类的因果推理范围，并使C-DAG能够在先前难以处理的场景中应用。我们的因果推理规则相对于do-因果推理是既有效又原子完整的：所有有效的聚类水平的干预查询都可以通过我们的规则推导出来，每条规则对应于一个基本的do-因果推理步骤。 

---
# Unbiased Platform-Level Causal Estimation for Search Systems: A Competitive Isolation PSM-DID Framework 

**Title (ZH)**: 搜索系统级无偏因果估计：一种竞争隔离倾向评分差值框架 

**Authors**: Ying Song, Yijing Wang, Hui Yang, Weihan Jin, Jun Xiong, Congyi Zhou, Jialin Zhu, Xiang Gao, Rong Chen, HuaGuang Deng, Ying Dai, Fei Xiao, Haihong Tang, Bo Zheng, KaiFu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01329)  

**Abstract**: Evaluating platform-level interventions in search-based two-sided marketplaces is fundamentally challenged by systemic effects such as spillovers and network interference. While widely used for causal inference, the PSM (Propensity Score Matching) - DID (Difference-in-Differences) framework remains susceptible to selection bias and cross-unit interference from unaccounted spillovers. In this paper, we introduced Competitive Isolation PSM-DID, a novel causal framework that integrates propensity score matching with competitive isolation to enable platform-level effect measurement (e.g., order volume, GMV) instead of item-level metrics in search systems.
Our approach provides theoretically guaranteed unbiased estimation under mutual exclusion conditions, with an open dataset released to support reproducible research on marketplace interference (this http URL). Extensive experiments demonstrate significant reductions in interference effects and estimation variance compared to baseline methods. Successful deployment in a large-scale marketplace confirms the framework's practical utility for platform-level causal inference. 

**Abstract (ZH)**: 基于搜索的双边市场中评估平台级干预措施从根本上受到系统效应如溢出效应和网络干扰的挑战。尽管广泛用于因果推断，PSM-DID（倾向得分匹配-差异差异）框架仍然容易受到未考虑的溢出效应引起的选择偏差和跨单位干扰的影响。本文介绍了竞争隔离PSM-DID，这是一种新颖的因果框架，将倾向得分匹配与竞争隔离相结合，以实现对平台级效果（如订单量、GMV）的测量，而不是项目级指标。 

---
# Graph Neural Network-Based Semi-Supervised Open-Set Fault Diagnosis for Marine Machinery Systems 

**Title (ZH)**: 基于图神经网络的半监督开集故障诊断方法及其在船舶机械系统中的应用 

**Authors**: Chuyue Lou, M. Amine Atoui  

**Link**: [PDF](https://arxiv.org/pdf/2511.01258)  

**Abstract**: Recently, fault diagnosis methods for marine machinery systems based on deep learning models have attracted considerable attention in the shipping industry. Most existing studies assume fault classes are consistent and known between the training and test datasets, and these methods perform well under controlled environment. In practice, however, previously unseen or unknown fault types (i.e., out-of-distribution or open-set observations not present during training) can occur, causing such methods to fail and posing a significant challenge to their widespread industrial deployment. To address this challenge, this paper proposes a semi-supervised open-set fault diagnosis (SOFD) framework that enhances and extends the applicability of deep learning models in open-set fault diagnosis scenarios. The framework includes a reliability subset construction process, which uses a multi-layer fusion feature representation extracted by a supervised feature learning model to select an unlabeled test subset. The labeled training set and pseudo-labeled test subset are then fed into a semi-supervised diagnosis model to learn discriminative features for each class, enabling accurate classification of known faults and effective detection of unknown samples. Experimental results on a public maritime benchmark dataset demonstrate the effectiveness and superiority of the proposed SOFD framework. 

**Abstract (ZH)**: 基于半监督开放集故障诊断框架的海洋机械系统故障诊断方法研究 

---
# MiRAGE: Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion 

**Title (ZH)**: MiRAGE：基于检索引导多阶段推理和集成融合的误区检测方法 

**Authors**: Cuong Van Duc, Thai Tran Quoc, Minh Nguyen Dinh Tuan, Tam Vu Duc, Son Nguyen Van, Hanh Nguyen Thi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01182)  

**Abstract**: Detecting student misconceptions in open-ended responses is a longstanding challenge, demanding semantic precision and logical reasoning. We propose MiRAGE - Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion, a novel framework for automated misconception detection in mathematics. MiRAGE operates in three stages: (1) a Retrieval module narrows a large candidate pool to a semantically relevant subset; (2) a Reasoning module employs chain-of-thought generation to expose logical inconsistencies in student solutions; and (3) a Reranking module refines predictions by aligning them with the reasoning. These components are unified through an ensemble-fusion strategy that enhances robustness and interpretability. On mathematics datasets, MiRAGE achieves Mean Average Precision scores of 0.82/0.92/0.93 at levels 1/3/5, consistently outperforming individual modules. By coupling retrieval guidance with multi-stage reasoning, MiRAGE reduces dependence on large-scale language models while delivering a scalable and effective solution for educational assessment. 

**Abstract (ZH)**: 基于检索引导多阶段推理和ensemble融合的学生误解检测：数学中的自动误解检测 

---
# AI for pRedicting Exacerbations in KIDs with aSthma (AIRE-KIDS) 

**Title (ZH)**: AI在KIDs哮喘加重预测中的应用：AIRE-KIDS研究 

**Authors**: Hui-Lee Ooi, Nicholas Mitsakakis, Margerie Huet Dastarac, Roger Zemek, Amy C. Plint, Jeff Gilchrist, Khaled El Emam, Dhenuka Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01018)  

**Abstract**: Recurrent exacerbations remain a common yet preventable outcome for many children with asthma. Machine learning (ML) algorithms using electronic medical records (EMR) could allow accurate identification of children at risk for exacerbations and facilitate referral for preventative comprehensive care to avoid this morbidity. We developed ML algorithms to predict repeat severe exacerbations (i.e. asthma-related emergency department (ED) visits or future hospital admissions) for children with a prior asthma ED visit at a tertiary care children's hospital.
Retrospective pre-COVID19 (Feb 2017 - Feb 2019, N=2716) Epic EMR data from the Children's Hospital of Eastern Ontario (CHEO) linked with environmental pollutant exposure and neighbourhood marginalization information was used to train various ML models. We used boosted trees (LGBM, XGB) and 3 open-source large language model (LLM) approaches (DistilGPT2, Llama 3.2 1B and Llama-8b-UltraMedical). Models were tuned and calibrated then validated in a second retrospective post-COVID19 dataset (Jul 2022 - Apr 2023, N=1237) from CHEO. Models were compared using the area under the curve (AUC) and F1 scores, with SHAP values used to determine the most predictive features.
The LGBM ML model performed best with the most predictive features in the final AIRE-KIDS_ED model including prior asthma ED visit, the Canadian triage acuity scale, medical complexity, food allergy, prior ED visits for non-asthma respiratory diagnoses, and age for an AUC of 0.712, and F1 score of 0.51. This is a nontrivial improvement over the current decision rule which has F1=0.334. While the most predictive features in the AIRE-KIDS_HOSP model included medical complexity, prior asthma ED visit, average wait time in the ED, the pediatric respiratory assessment measure score at triage and food allergy. 

**Abstract (ZH)**: 反复加重仍然是许多哮喘儿童常见且可预防的结局。通过电子医疗记录（EMR）的机器学习（ML）算法可以准确识别出有加重风险的儿童，并促进预防性全面护理的转介，以避免这种并发症。我们开发了ML算法来预测有既往哮喘急诊就诊记录的儿童在三级儿童医院发生重复严重加重（即哮喘相关的急诊就诊或未来住院）的风险。 

---
# Lifted Successor Generation in Numeric Planning 

**Title (ZH)**: 数值规划中的提升后继生成 

**Authors**: Dominik Drexler  

**Link**: [PDF](https://arxiv.org/pdf/2511.00673)  

**Abstract**: Most planners ground numeric planning tasks, given in a first-order-like language, into a ground task representation. However, this can lead to an exponential blowup in task representation size, which occurs in practice for hard-to-ground tasks. We extend a state-of-the-art lifted successor generator for classical planning to support numeric precondition applicability. The method enumerates maximum cliques in a substitution consistency graph. Each maximum clique represents a substitution for the variables of the action schema, yielding a ground action. We augment this graph with numeric action preconditions and prove the successor generator is exact under formally specified conditions. When the conditions fail, our generator may list inapplicable ground actions; a final applicability check filters these without affecting completeness. However, this cannot happen in 23 of 25 benchmark domains, and it occurs only in 1 domain. To the authors' knowledge, no other lifted successor generator supports numeric action preconditions. This enables future research on lifted planning for a very rich planning fragment. 

**Abstract (ZH)**: 一种支持数值先决条件的提升 successors 生成器扩展研究：一种丰富规划片段的提升规划方向 

---
# Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting 

**Title (ZH)**: 利用多智能体系统(MAS)和微调的小型语言模型(SLMs)实现自动化电信网络故障排查 

**Authors**: Chenhua Shi, Bhavika Jalli, Gregor Macdonald, John Zou, Wanlu Lei, Mridul Jain, Joji Philip  

**Link**: [PDF](https://arxiv.org/pdf/2511.00651)  

**Abstract**: Telecom networks are rapidly growing in scale and complexity, making effective management, operation, and optimization increasingly challenging. Although Artificial Intelligence (AI) has been applied to many telecom tasks, existing models are often narrow in scope, require large amounts of labeled data, and struggle to generalize across heterogeneous deployments. Consequently, network troubleshooting continues to rely heavily on Subject Matter Experts (SMEs) to manually correlate various data sources to identify root causes and corrective actions. To address these limitations, we propose a Multi-Agent System (MAS) that employs an agentic workflow, with Large Language Models (LLMs) coordinating multiple specialized tools for fully automated network troubleshooting. Once faults are detected by AI/ML-based monitors, the framework dynamically activates agents such as an orchestrator, solution planner, executor, data retriever, and root-cause analyzer to diagnose issues and recommend remediation strategies within a short time frame. A key component of this system is the solution planner, which generates appropriate remediation plans based on internal documentation. To enable this, we fine-tuned a Small Language Model (SLM) on proprietary troubleshooting documents to produce domain-grounded solution plans. Experimental results demonstrate that the proposed framework significantly accelerates troubleshooting automation across both Radio Access Network (RAN) and Core network domains. 

**Abstract (ZH)**: 电信网络正在迅速增长在规模和复杂性方面，有效管理、运行和优化变得越来越具挑战性。虽然人工智能（AI）已被应用到许多电信任务中，但现有模型往往范围狭窄，需要大量的标注数据，并且难以在异构部署中泛化。因此，网络故障排查仍然很大程度上依赖于领域专家（SMEs）手动关联各种数据源以识别根本原因和纠正措施。为了解决这些限制，我们提出了一种多智能体系统（MAS），该系统采用基于任务的流程，通过大型语言模型（LLMs）协调多个专业工具，实现完全自动化的网络故障排查。一旦AI/ML基于的监视器检测到故障，该框架将动态激活协调器、解决方案规划器、执行器、数据检索器和根本原因分析器等智能体，在短时间内诊断问题并推荐补救策略。该系统的关键组成部分是解决方案规划器，它可以基于内部文档生成适当的补救计划。为了实现这一点，我们对一个小语言模型（SLM）进行了微调，以生成领域相关的解决方案计划。实验结果表明，提出的框架显著加速了无线接入网络（RAN）和核心网络领域中的故障排查自动化。 

---
# PreferThinker: Reasoning-based Personalized Image Preference Assessment 

**Title (ZH)**: 基于推理的个性化图像偏好评估 

**Authors**: Shengqi Xu, Xinpeng Zhou, Yabo Zhang, Ming Liu, Tao Liang, Tianyu Zhang, Yalong Bai, Zuxuan Wu, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00609)  

**Abstract**: Personalized image preference assessment aims to evaluate an individual user's image preferences by relying only on a small set of reference images as prior information. Existing methods mainly focus on general preference assessment, training models with large-scale data to tackle well-defined tasks such as text-image alignment. However, these approaches struggle to handle personalized preference because user-specific data are scarce and not easily scalable, and individual tastes are often diverse and complex. To overcome these challenges, we introduce a common preference profile that serves as a bridge across users, allowing large-scale user data to be leveraged for training profile prediction and capturing complex personalized preferences. Building on this idea, we propose a reasoning-based personalized image preference assessment framework that follows a \textit{predict-then-assess} paradigm: it first predicts a user's preference profile from reference images, and then provides interpretable, multi-dimensional scores and assessments of candidate images based on the predicted profile. To support this, we first construct a large-scale Chain-of-Thought (CoT)-style personalized assessment dataset annotated with diverse user preference profiles and high-quality CoT-style reasoning, enabling explicit supervision of structured reasoning. Next, we adopt a two-stage training strategy: a cold-start supervised fine-tuning phase to empower the model with structured reasoning capabilities, followed by reinforcement learning to incentivize the model to explore more reasonable assessment paths and enhance generalization. Furthermore, we propose a similarity-aware prediction reward to encourage better prediction of the user's preference profile, which facilitates more reasonable assessments exploration. Extensive experiments demonstrate the superiority of the proposed method. 

**Abstract (ZH)**: 个性化图像偏好评估旨在通过少量参考图像作为先验信息来评估个体用户的图像偏好。现有方法主要侧重于通用偏好评估，通过大规模数据训练模型以应对诸如图文对齐等明确定义的任务。然而，这些方法在处理个性化偏好方面存在困难，因为用户特定的数据稀缺且难以规模化，而个人品味往往多样化且复杂。为克服这些挑战，我们引入了一个通用偏好概况，作为用户之间的桥梁，使得大规模用户数据可以用于训练概况预测并捕捉复杂的个性化偏好。基于这一理念，我们提出了一种基于推理的个性化图像偏好评估框架，遵循“预测-评估”范式：首先从参考图像预测用户的偏好概况，然后基于预测的概况提供可解释的多维度候选图像评分和评估。为了支持这一点，我们首先构建了一个大规模的带有多样化用户偏好概况和高质量推理注释的Chain-of-Thought风格个性化评估数据集，为结构化推理提供了明确监督。接着我们采用两阶段训练策略：冷启动监督微调阶段以赋予模型结构化推理能力，随后是强化学习阶段以激励模型探索更合理的评估路径并增强泛化能力。此外，我们提出了相似性意识预测奖励以鼓励更好地预测用户偏好概况，这促进了更合理评估路径的探索。广泛实验表明所提出方法的优越性。 

---
# Single-agent Reinforcement Learning Model for Regional Adaptive Traffic Signal Control 

**Title (ZH)**: 单代理强化学习模型在区域自适应交通信号控制中的应用 

**Authors**: Qiang Li, Ningjing Zeng, Lina Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00551)  

**Abstract**: Several studies have employed reinforcement learning (RL) to address the challenges of regional adaptive traffic signal control (ATSC) and achieved promising results. In this field, existing research predominantly adopts multi-agent frameworks. However, the adoption of multi-agent frameworks presents challenges for scalability. Instead, the Traffic signal control (TSC) problem necessitates a single-agent framework. TSC inherently relies on centralized management by a single control center, which can monitor traffic conditions across all roads in the study area and coordinate the control of all intersections. This work proposes a single-agent RL-based regional ATSC model compatible with probe vehicle technology. Key components of the RL design include state, action, and reward function definitions. To facilitate learning and manage congestion, both state and reward functions are defined based on queue length, with action designed to regulate queue dynamics. The queue length definition used in this study differs slightly from conventional definitions but is closely correlated with congestion states. More importantly, it allows for reliable estimation using link travel time data from probe vehicles. With probe vehicle data already covering most urban roads, this feature enhances the proposed method's potential for widespread deployment. The method was comprehensively evaluated using the SUMO simulation platform. Experimental results demonstrate that the proposed model effectively mitigates large-scale regional congestion levels via coordinated multi-intersection control. 

**Abstract (ZH)**: 基于探针车辆技术的单智能体区域自适应交通信号控制模型 

---
# Efficient Generation of Binary Magic Squares 

**Title (ZH)**: 高效生成二元魔方阵 

**Authors**: Alain Riou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00547)  

**Abstract**: We propose a simple algorithm for generating Binary Magic Squares (BMS), i.e., square binary matrices where the sum of all rows and all columns are equal. We show by induction that our algorithm always returns valid BMS with optimal theoretical complexity. We then extend our study to non-square Binary Magic Squares, formalize conditions on the sum of rows and columns for these BMS to exist, and show that a slight variant of our first algorithm can generate provably generate them. Finally, we publicly release two implementations of our algorithm as Python packages, including one that can generate several BMS in parallel using GPU acceleration. 

**Abstract (ZH)**: 我们提出了一种生成二元魔方矩阵(BMS)的简单算法，即行和列的和相等的二元方阵。我们通过归纳证明，该算法总是返回有效的BMS，并具有最优的理论复杂度。然后我们将研究扩展到非方二元魔方矩阵，正式化这些BMS的行和列的和存在条件，并证明该算法的一个轻微变体可以生成可验证的此类BMS。最后，我们公开发布了两种算法的实现，作为Python包，包括一个利用GPU加速并行生成多个BMS的版本。 

---
# Advancing AI Challenges for the United States Department of the Air Force 

**Title (ZH)**: 美国空军部面临的AI挑战进展 

**Authors**: Christian Prothmann, Vijay Gadepally, Jeremy Kepner, Koley Borchard, Luca Carlone, Zachary Folcik, J. Daniel Grith, Michael Houle, Jonathan P. How, Nathan Hughes, Ifueko Igbinedion, Hayden Jananthan, Tejas Jayashankar, Michael Jones, Sertac Karaman, Binoy G. Kurien, Alejandro Lancho, Giovanni Lavezzi, Gary C. F. Lee, Charles E. Leiserson, Richard Linares, Lindsey McEvoy, Peter Michaleas, Chasen Milner, Alex Pentland, Yury Polyanskiy, Jovan Popovich, Jeffrey Price, Tim W. Reid, Stephanie Riley, Siddharth Samsi, Peter Saunders, Olga Simek, Mark S. Veillette, Amir Weiss, Gregory W. Wornell, Daniela Rus, Scott T. Ruppel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00267)  

**Abstract**: The DAF-MIT AI Accelerator is a collaboration between the United States Department of the Air Force (DAF) and the Massachusetts Institute of Technology (MIT). This program pioneers fundamental advances in artificial intelligence (AI) to expand the competitive advantage of the United States in the defense and civilian sectors. In recent years, AI Accelerator projects have developed and launched public challenge problems aimed at advancing AI research in priority areas. Hallmarks of AI Accelerator challenges include large, publicly available, and AI-ready datasets to stimulate open-source solutions and engage the wider academic and private sector AI ecosystem. This article supplements our previous publication, which introduced AI Accelerator challenges. We provide an update on how ongoing and new challenges have successfully contributed to AI research and applications of AI technologies. 

**Abstract (ZH)**: DAF-MIT AI加速器：美国国防部与麻省理工学院合作推动人工智能前沿进展及应用更新 

---
# Incremental Selection of Most-Filtering Conjectures and Proofs of the Selected Conjectures 

**Title (ZH)**: 增量选择最具过滤性的猜想及其所选猜想的证明选择 

**Authors**: Jovial Cheukam Ngouonou, Ramiz Gindullin, Claude-Guy Quimper, Nicolas Beldiceanu, Remi Douence  

**Link**: [PDF](https://arxiv.org/pdf/2511.00194)  

**Abstract**: We present an improved incremental selection algorithm of the selection algorithm presented in [1] and prove all the selected conjectures. 

**Abstract (ZH)**: 我们改进了参考文献[1]中提出的选择算法的增量选择算法，并证明了所有选定的猜想。 

---
# ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus 

**Title (ZH)**: ARC-GEN: 一个用于抽象与推理语料库的拟合程序基准生成器 

**Authors**: Michael D. Moffitt  

**Link**: [PDF](https://arxiv.org/pdf/2511.00162)  

**Abstract**: The Abstraction and Reasoning Corpus remains one of the most compelling and challenging benchmarks for tracking progress toward achieving Artificial General Intelligence. In contrast to other evaluation datasets designed to assess an agent's task-specific skills or accumulated knowledge, the ARC-AGI suite is specifically targeted at measuring skill acquisition efficiency, a trait that has (so far) been lacking in even the most sophisticated machine learning systems. For algorithms that require extensive intra-task exemplars, a significant constraint imposed by ARC-AGI is the modest cardinality of its demonstration set, comprising a small number of $\langle$ input, output $\rangle$ grids per task specifying the corresponding transformation. To embellish the space of viable sample pairs, this paper introduces ARC-GEN, an open-source procedural generator aimed at extending the original ARC-AGI training dataset as faithfully as possible. Unlike prior efforts, our generator is both exhaustive (covering all four-hundred tasks) and mimetic (more closely honoring the distributional properties and characteristics embodied in the initial ARC-AGI-1 release). We also discuss the use of this generator in establishing a static benchmark suite to verify the correctness of programs submitted to the 2025 Google Code Golf Championship. 

**Abstract (ZH)**: 抽象与推理语料库仍然是跟踪实现人工通用智能进展最具说服力和挑战性的基准之一。与旨在评估智能体任务特定技能或累积知识的其他评估数据集不同，ARC-AGI 集合特别针对测量技能获取效率，这是一个迄今为止在最先进的机器学习系统中缺乏的特征。对于需要大量任务内示例的算法而言，ARC-AGI 对其演示集的适度基数（每个任务包含少量的 $\langle$ 输入，输出 $\rangle$ 网格，规定相应的变换）构成了一个显著的约束。为了扩充可行样本对的空间，本文引入了 ARC-GEN，这是一个开源的程序生成器，旨在尽可能忠实地扩展原始 ARC-AGI 训练数据集。与先前的努力不同，我们的生成器是全面的（覆盖所有四百个任务）且模仿性的（更接近初始 ARC-AGI-1 发行版中体现的分布特性）。我们还讨论了该生成器在建立静态基准集中的应用，以验证提交给 2025 年 Google 代码高尔夫冠军赛的程序的正确性。 

---
# Engineering.ai: A Platform for Teams of AI Engineers in Computational Design 

**Title (ZH)**: Engineering.ai：计算设计领域中AI工程师团队的平台 

**Authors**: Ran Xu, Yupeng Qi, Jingsen Feng, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00122)  

**Abstract**: In modern engineering practice, human engineers collaborate in specialized teams to design complex products, with each expert completing their respective tasks while communicating and exchanging results and data with one another. While this division of expertise is essential for managing multidisciplinary complexity, it demands substantial development time and cost. Recently, we introduced OpenFOAMGPT (1.0, 2.0), which functions as an autonomous AI engineer for computational fluid dynamics, and this http URL, which can conduct end-to-end research in fluid mechanics draft publications and PhD theses. Building upon these foundations, we present this http URL, a platform for teams of AI engineers in computational design. The framework employs a hierarchical multi-agent architecture where a Chief Engineer coordinates specialized agents consisting of Aerodynamics, Structural, Acoustic, and Optimization Engineers, each powered by LLM with domain-specific knowledge. Agent-agent collaboration is achieved through file-mediated communication for data provenance and reproducibility, while a comprehensive memory system maintains project context, execution history, and retrieval-augmented domain knowledge to ensure reliable decision-making across the workflow. The system integrates FreeCAD, Gmsh, OpenFOAM, CalculiX, and BPM acoustic analysis, enabling parallel multidisciplinary simulations while maintaining computational accuracy. The framework is validated through UAV wing optimization. This work demonstrates that agentic-AI-enabled AI engineers has the potential to perform complex engineering tasks autonomously. Remarkably, the automated workflow achieved a 100% success rate across over 400 parametric configurations, with zero mesh generation failures, solver convergence issues, or manual interventions required, validating that the framework is trustworthy. 

**Abstract (ZH)**: 在现代工程实践中，人类工程师在专业化团队中合作设计复杂产品，每位专家完成各自的任务并相互沟通和交换结果及数据。虽然这种专业分工对于管理多学科复杂性至关重要，但它需要大量的开发时间和成本。最近，我们引入了OpenFOAMGPT（1.0, 2.0），这是一款用于计算流体力学的自主AI工程师，并开发了该平台，可以进行流体力学研究和博士论文的端到端撰写。在此基础上，我们介绍了一个计算设计领域AI工程师团队的平台。该框架采用分层多智能体架构，其中首席工程师协调空气动力学、结构学、声学和优化工程师等专业化智能体，这些智能体由领域特定的大语言模型驱动。智能体之间的协作通过文件传输实现数据溯源和可重现性，同时综合记忆系统维护项目上下文、执行历史和检索增强的领域知识，以确保跨流程可靠的决策。该系统集成了FreeCAD、Gmsh、OpenFOAM、CalculiX和BPM声学分析，支持并行多学科仿真，同时保持计算准确性。该框架通过无人驾驶航空器机翼优化进行了验证。这项工作表明，具备代理AI能力的AI工程师有可能自主完成复杂的工程任务。令人惊讶的是，自动化工作流在超过400个参数配置中实现了100%的成功率，没有出现网格生成失败、求解器收敛问题或需要手动干预的情况，验证了该框架的可靠性。 

---
# GEPOC Parameters - Open Source Parametrisation and Validation for Austria, Version 2.0 

**Title (ZH)**: GEPOC参数 - 奥地利的开源参数化与验证，版本2.0 

**Authors**: Martin Bicher, Maximilian Viehauser, Daniele Giannandrea, Hannah Kastinger, Dominik Brunmeir, Claire Rippinger, Christoph Urach, Niki Popper  

**Link**: [PDF](https://arxiv.org/pdf/2511.00048)  

**Abstract**: GEPOC, short for Generic Population Concept, is a collection of models and methods for analysing population-level research questions. For the valid application of the models for a specific country or region, stable and reproducible data processes are necessary, which provide valid and ready-to-use model parameters. This work contains a complete description of the data-processing methods for computation of model parameters for Austria, based exclusively on freely and publicly accessible data. In addition to the description of the source data used, this includes all algorithms used for aggregation, disaggregation, fusion, cleansing or scaling of the data, as well as a description of the resulting parameter files. The document places particular emphasis on the computation of parameters for the most important GEPOC model, GEPOC ABM, a continuous-time agent-based population model. An extensive validation study using this particular model was made and is presented at the end of this work. 

**Abstract (ZH)**: GEPOC（通用人口概念）是一种用于分析人口层面研究问题的模型和方法的集合。为了使这些模型在特定国家或地区有效应用，需要稳定且可重复的数据处理过程，从而提供有效的并可直接使用的模型参数。本文包含了基于完全开放和公开数据计算奥地利模型参数的完整数据处理方法描述。除了数据来源的描述外，还包括所有用于聚合、拆分、融合、清洗或缩放数据的算法，以及参数文件的描述。文件特别强调了计算GEPOC ABM（一个连续时间的基于代理的人口模型）最重要的模型参数的计算方法。通过对该模型进行了详尽的验证研究，并在本文末尾进行了呈现。 

---
# Graph-Attentive MAPPO for Dynamic Retail Pricing 

**Title (ZH)**: 基于图注意力的动态零售定价MAPPO算法 

**Authors**: Krishna Kumar Neelakanta Pillai Santha Kumari Amma  

**Link**: [PDF](https://arxiv.org/pdf/2511.00039)  

**Abstract**: Dynamic pricing in retail requires policies that adapt to shifting demand while coordinating decisions across related products. We present a systematic empirical study of multi-agent reinforcement learning for retail price optimization, comparing a strong MAPPO baseline with a graph-attention-augmented variant (MAPPO+GAT) that leverages learned interactions among products. Using a simulated pricing environment derived from real transaction data, we evaluate profit, stability across random seeds, fairness across products, and training efficiency under a standardized evaluation protocol. The results indicate that MAPPO provides a robust and reproducible foundation for portfolio-level price control, and that MAPPO+GAT further enhances performance by sharing information over the product graph without inducing excessive price volatility. These results indicate that graph-integrated MARL provides a more scalable and stable solution than independent learners for dynamic retail pricing, offering practical advantages in multi-product decision-making. 

**Abstract (ZH)**: 零售中的动态定价要求适应变化的需求并协调相关产品的决策。我们系统地研究了多代理 reinforcement learning 在零售价格优化中的应用，将强 MAPPO 基线与利用产品间学习到的交互的图注意力增强变体（MAPPO+GAT）进行比较。通过基于真实交易数据构建的仿真定价环境，我们在标准化评估协议下评估了利润、随机种子下的稳定性、产品间的公平性以及训练效率。结果表明，MAPPO 提供了一个稳健且可重复的产品组合级价格控制基础，而 MAPPO+GAT 通过在产品图中共享信息进一步提升了性能，同时避免了过度的价格波动。这些结果表明，集成图的 MARL 提供了一种比独立学习者更可扩展和稳定的选择，适用于多产品动态定价中的实际决策。 

---
# Trove: A Flexible Toolkit for Dense Retrieval 

**Title (ZH)**: Trove: 一种灵活的密集检索工具包 

**Authors**: Reza Esfandiarpoor, Max Zuo, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2511.01857)  

**Abstract**: We introduce Trove, an easy-to-use open-source retrieval toolkit that simplifies research experiments without sacrificing flexibility or speed. For the first time, we introduce efficient data management features that load and process (filter, select, transform, and combine) retrieval datasets on the fly, with just a few lines of code. This gives users the flexibility to easily experiment with different dataset configurations without the need to compute and store multiple copies of large datasets. Trove is highly customizable: in addition to many built-in options, it allows users to freely modify existing components or replace them entirely with user-defined objects. It also provides a low-code and unified pipeline for evaluation and hard negative mining, which supports multi-node execution without any code changes. Trove's data management features reduce memory consumption by a factor of 2.6. Moreover, Trove's easy-to-use inference pipeline incurs no overhead, and inference times decrease linearly with the number of available nodes. Most importantly, we demonstrate how Trove simplifies retrieval experiments and allows for arbitrary customizations, thus facilitating exploratory research. 

**Abstract (ZH)**: Trove：一个易用的开源检索工具包，简化研究实验同时保持灵活性和速度 

---
# Towards Robust Mathematical Reasoning 

**Title (ZH)**: 面向稳健的数学推理 

**Authors**: Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, Junehyuk Jung  

**Link**: [PDF](https://arxiv.org/pdf/2511.01846)  

**Abstract**: Finding the right north-star metrics is highly critical for advancing the mathematical reasoning capabilities of foundation models, especially given that existing evaluations are either too easy or only focus on getting correct short answers. To address these issues, we present IMO-Bench, a suite of advanced reasoning benchmarks, vetted by a panel of top specialists and that specifically targets the level of the International Mathematical Olympiad (IMO), the most prestigious venue for young mathematicians. IMO-AnswerBench first tests models on 400 diverse Olympiad problems with verifiable short answers. IMO-Proof Bench is the next-level evaluation for proof-writing capabilities, which includes both basic and advanced IMO level problems as well as detailed grading guidelines to facilitate automatic grading. These benchmarks played a crucial role in our historic achievement of the gold-level performance at IMO 2025 with Gemini Deep Think (Luong and Lockhart, 2025). Our model achieved 80.0% on IMO-AnswerBench and 65.7% on the advanced IMO-Proof Bench, surpassing the best non-Gemini models by large margins of 6.9% and 42.4% respectively. We also showed that autograders built with Gemini reasoning correlate well with human evaluations and construct IMO-GradingBench, with 1000 human gradings on proofs, to enable further progress in automatic evaluation of long-form answers. We hope that IMO-Bench will help the community towards advancing robust mathematical reasoning and release it at this https URL. 

**Abstract (ZH)**: 找到合适的北极星指标对于推进基础模型的数学推理能力至关重要，尤其是当现有评估要么过于简单，要么仅仅关注正确短答案的获得时。为应对这些问题，我们提出了IMO-Bench，这是一种由顶级专家审定的高级推理基准，特别针对国际数学奥林匹克（IMO）的水平，这是年轻数学家最 prestigious的比赛场地。IMO-AnswerBench 首先测试模型在400个具有可验证简短答案的奥林匹克问题上。IMO-Proof Bench 是针对证明写作能力的更高层次评估，包括基础和高级IMO级别的问题，并提供详细的评分指南以促进自动评分。这些基准对我们以Gemini Deep Think（Luong和Lockhart, 2025）在IMO 2025中获得金奖的历史性成就起到了关键作用。我们的模型在IMO-AnswerBench 上获得了80.0%的成绩，在高级IMO-Proof Bench 上获得了65.7%的成绩，分别比最佳非Gemini模型高出6.9%和42.4%。我们还展示了使用Gemini推理构建的自动评分系统与人工评估的相关性，并建立了IMO-GradingBench，包含1000份人工对证明的评分，以促进长篇文章自动评估的进一步进步。我们希望IMO-Bench 能够帮助社区提高稳健的数学推理能力，并在此处 https://this.url-release 它。 

---
# Efficient Vector Symbolic Architectures from Histogram Recovery 

**Title (ZH)**: 高效的直方图恢复向量符号架构 

**Authors**: Zirui Deng, Netanel Raviv  

**Link**: [PDF](https://arxiv.org/pdf/2511.01838)  

**Abstract**: Vector symbolic architectures (VSAs) are a family of information representation techniques which enable composition, i.e., creating complex information structures from atomic vectors via binding and superposition, and have recently found wide ranging applications in various neurosymbolic artificial intelligence (AI) systems. Recently, Raviv proposed the use of random linear codes in VSAs, suggesting that their subcode structure enables efficient binding, while preserving the quasi-orthogonality that is necessary for neural processing. Yet, random linear codes are difficult to decode under noise, which severely limits the resulting VSA's ability to support recovery, i.e., the retrieval of information objects and their attributes from a noisy compositional representation.
In this work we bridge this gap by utilizing coding theoretic tools. First, we argue that the concatenation of Reed-Solomon and Hadamard codes is suitable for VSA, due to the mutual quasi-orthogonality of the resulting codewords (a folklore result). Second, we show that recovery of the resulting compositional representations can be done by solving a problem we call histogram recovery. In histogram recovery, a collection of $N$ histograms over a finite field is given as input, and one must find a collection of Reed-Solomon codewords of length $N$ whose entry-wise symbol frequencies obey those histograms. We present an optimal solution to the histogram recovery problem by using algorithms related to list-decoding, and analyze the resulting noise resilience. Our results give rise to a noise-resilient VSA with formal guarantees regarding efficient encoding, quasi-orthogonality, and recovery, without relying on any heuristics or training, and while operating at improved parameters relative to similar solutions such as the Hadamard code. 

**Abstract (ZH)**: 基于编码理论工具的矢量象征架构改进 

---
# Machine and Deep Learning for Indoor UWB Jammer Localization 

**Title (ZH)**: 基于机器学习和深度学习的室内UWB Jammer定位技术 

**Authors**: Hamed Fard, Mahsa Kholghi, Benedikt Groß, Gerhard Wunder  

**Link**: [PDF](https://arxiv.org/pdf/2511.01819)  

**Abstract**: Ultra-wideband (UWB) localization delivers centimeter-scale accuracy but is vulnerable to jamming attacks, creating security risks for asset tracking and intrusion detection in smart buildings. Although machine learning (ML) and deep learning (DL) methods have improved tag localization, localizing malicious jammers within a single room and across changing indoor layouts remains largely unexplored. Two novel UWB datasets, collected under original and modified room configurations, are introduced to establish comprehensive ML/DL baselines. Performance is rigorously evaluated using a variety of classification and regression metrics. On the source dataset with the collected UWB features, Random Forest achieves the highest F1-macro score of 0.95 and XGBoost achieves the lowest mean Euclidean error of 20.16 cm. However, deploying these source-trained models in the modified room layout led to severe performance degradation, with XGBoost's mean Euclidean error increasing tenfold to 207.99 cm, demonstrating significant domain shift. To mitigate this degradation, a domain-adversarial ConvNeXt autoencoder (A-CNT) is proposed that leverages a gradient-reversal layer to align CIR-derived features across domains. The A-CNT framework restores localization performance by reducing the mean Euclidean error to 34.67 cm. This represents a 77 percent improvement over non-adversarial transfer learning and an 83 percent improvement over the best baseline, restoring the fraction of samples within 30 cm to 0.56. Overall, the results demonstrate that adversarial feature alignment enables robust and transferable indoor jammer localization despite environmental changes. Code and dataset available at this https URL 

**Abstract (ZH)**: 超宽带（UWB）定位提供厘米级精度，但易受干扰攻击影响，为智能建筑中资产跟踪和入侵检测带来安全风险。尽管机器学习（ML）和深度学习（DL）方法提高了标签定位精度，但在单个房间内定位恶意干扰器以及在不断变化的室内布局之间定位仍是一个未探索的领域。介绍了两种新的UWB数据集，分别在原始和修改的房间配置下收集，以建立全面的ML/DL baselines。使用多种分类和回归指标严格评估性能。在包含收集的UWB特征的源数据集上，随机森林实现最高的宏F1分数0.95，而XGBoost实现最低的平均欧氏误差20.16 cm。然而，在修改的房间布局中部署这些源训练模型导致性能严重下降，XGBoost的平均欧氏误差增加十倍至207.99 cm，显示出显著的领域偏移。为缓解这一下降，提出了一种域对抗ConvNeXt自编码器（A-CNT）框架，利用梯度反转层在不同域之间对载波相移键控（CIR）衍生特征进行对齐。A-CNT框架通过将平均欧氏误差降至34.67 cm，恢复了定位性能。这分别比非对抗性迁移学习提高了77%，比最好基线提高了83%，并将30 cm以内样本的比例恢复至0.56。总体而言，结果表明对抗性特征对齐能够即使在环境变化的情况下仍实现稳健且可转移的室内干扰器定位。代码和数据集可在如下链接获取。 

---
# SM-based Semantics for Answer Set Programs Containing Conditional Literals and Arithmetic 

**Title (ZH)**: 基于SM的包含条件文字和算术表达式的回答集程序语义学 

**Authors**: Zachary Hansen, Yuliya Lierler  

**Link**: [PDF](https://arxiv.org/pdf/2511.01753)  

**Abstract**: Modern answer set programming solvers such as CLINGO support advanced language constructs that improve the expressivity and conciseness of logic programs. Conditional literals are one such construct. They form "subformulas" that behave as nested implications within the bodies of logic rules. Their inclusion brings the form of rules closer to the less restrictive syntax of first-order logic. These qualities make conditional literals useful tools for knowledge representation. In this paper, we propose a semantics for logic programs with conditional literals and arithmetic based on the SM operator. These semantics do not require grounding, unlike the established semantics for such programs that relies on a translation to infinitary propositional logic. The main result of this paper establishes the precise correspondence between the proposed and existing semantics. 

**Abstract (ZH)**: 基于SM运算符的含条件文字与算术逻辑程序语义 

---
# An Open-Access Benchmark of Statistical and Machine-Learning Anomaly Detection Methods for Battery Applications 

**Title (ZH)**: 开放获取基准：统计和机器学习异常检测方法在电池应用中的评估 

**Authors**: Mei-Chin Pang, Suraj Adhikari, Takuma Kasahara, Nagihiro Haba, Saneyuki Ohno  

**Link**: [PDF](https://arxiv.org/pdf/2511.01745)  

**Abstract**: Battery safety is critical in applications ranging from consumer electronics to electric vehicles and aircraft, where undetected anomalies could trigger safety hazards or costly downtime. In this study, we present OSBAD as an open-source benchmark for anomaly detection frameworks in battery applications. By benchmarking 15 diverse algorithms encompassing statistical, distance-based, and unsupervised machine-learning methods, OSBAD enables a systematic comparison of anomaly detection methods across heterogeneous datasets. In addition, we demonstrate how a physics- and statistics-informed feature transformation workflow enhances anomaly separability by decomposing collective anomalies into point anomalies. To address a major bottleneck in unsupervised anomaly detection due to incomplete labels, we propose a Bayesian optimization pipeline that facilitates automated hyperparameter tuning based on transfer-learning and regression proxies. Through validation on datasets covering both liquid and solid-state chemistries, we further demonstrate the cross-chemistry generalization capability of OSBAD to identify irregularities across different electrochemical systems. By making benchmarking database with open-source reproducible anomaly detection workflows available to the community, OSBAD establishes a unified foundation for developing safe, scalable, and transferable anomaly detection tools in battery analytics. This research underscores the significance of physics- and statistics-informed feature engineering as well as model selection with probabilistic hyperparameter tuning, in advancing trustworthy, data-driven diagnostics for safety-critical energy systems. 

**Abstract (ZH)**: 电池安全在从消费电子产品到电动汽车和航空器的应用中至关重要，未检测到的异常可能导致安全风险或昂贵的停机时间。本文介绍OSBAD作为一种开放源代码基准，用于电池应用中的异常检测框架。通过评估涵盖统计方法、基于距离的方法和无监督机器学习方法的15种不同算法，OSBAD实现了异构数据集上异常检测方法的系统比较。此外，我们展示了如何通过将集体异常分解为点异常来增强异常可分性的一种物理和统计导向的特征转换工作流程。为了应对无监督异常检测中由于标签不完整而导致的主要瓶颈，我们提出了一种贝叶斯优化管道，该管道基于迁移学习和回归代理实现自动化超参数调整。通过在涵盖液态和固态化学组成的数据集上的验证，我们进一步展示了OSBAD在不同电化学系统中识别异常的一致泛化能力。通过向社区提供开放源代码可重复的异常检测工作流程基准数据库，OSBAD建立了用于电池分析中发展安全可靠的可迁移异常检测工具的统一基础。这项研究强调了物理和统计导向的特征工程以及基于概率超参数调整的模型选择对于推进可信的数据驱动诊断在关键能源系统中的应用的重要性。 

---
# Towards Efficient Federated Learning of Networked Mixture-of-Experts for Mobile Edge Computing 

**Title (ZH)**: 面向移动边缘计算的网络化混合专家高效联邦学习 

**Authors**: Song Gao, Shusen Jing, Shuai Zhang, Yue Wang, Xiangwei Zhou, Songyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01743)  

**Abstract**: Recent advancements in large artificial intelligence models (LAMs) are driving significant innovations in mobile edge computing within next-generation wireless networks. However, the substantial demands for computational resources and large-scale training data required to train LAMs conflict with the limited storage and computational capacity of edge devices, posing significant challenges to training and deploying LAMs at the edge. In this work, we introduce the Networked Mixture-of-Experts (NMoE) system, in which clients infer collaboratively by distributing tasks to suitable neighbors based on their expertise and aggregate the returned results. For training the NMoE, we propose a federated learning framework that integrates both supervised and self-supervised learning to balance personalization and generalization, while preserving communication efficiency and data privacy. We conduct extensive experiments to demonstrate the efficacy of the proposed NMoE system, providing insights and benchmarks for the NMoE training algorithms. 

**Abstract (ZH)**: Recent advancements in大型人工智能模型（LAMs）正推动下一代无线网络中移动边缘计算的显著创新。然而，训练LAMs所需的大量计算资源和大规模训练数据与边缘设备有限的存储和计算能力之间存在冲突，对在边缘培训和部署LAMs构成重大挑战。在此工作中，我们引入了网络混合专家（NMoE）系统，在该系统中，客户端通过将任务分配给基于其专长的合适邻居并汇总返回的结果来进行协作推理。为了训练NMoE，我们提出了一种结合有监督学习和自我监督学习的联邦学习框架，以平衡个性化和泛化能力，同时保持通信效率和数据隐私。我们进行了 extensive 实验以证明所提出 NMoE 系统的有效性，并为 NMoE 训练算法提供了见解和基准。 

---
# A Proof of Learning Rate Transfer under $μ$P 

**Title (ZH)**: μP条件下学习率迁移的证明 

**Authors**: Soufiane Hayou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01734)  

**Abstract**: We provide the first proof of learning rate transfer with width in a linear multi-layer perceptron (MLP) parametrized with $\mu$P, a neural network parameterization designed to ``maximize'' feature learning in the infinite-width limit. We show that under $\mu P$, the optimal learning rate converges to a \emph{non-zero constant} as width goes to infinity, providing a theoretical explanation to learning rate transfer. In contrast, we show that this property fails to hold under alternative parametrizations such as Standard Parametrization (SP) and Neural Tangent Parametrization (NTP). We provide intuitive proofs and support the theoretical findings with extensive empirical results. 

**Abstract (ZH)**: 我们提供了在μP参数化下的线性多层感知机（MLP）中学习率转移与宽度关系的第一个证明，μP是一种神经网络参数化方法，旨在在无限宽度极限下“最大化”特征学习。我们证明，在μP参数化下，最优学习率在宽度趋向无穷大时收敛于一个非零常数，从而为学习率转移提供了一种理论解释。相比之下，我们证明了这一性质在标准参数化（SP）和神经态参数化（NTP）等替代参数化方案下并不成立。我们提供了直观的证明，并通过广泛的实验证据支持了理论发现。 

---
# Solution Space Topology Guides CMTS Search 

**Title (ZH)**: 拓扑引导的CMTS搜索解决方案空间探索 

**Authors**: Mirco A. Mannucci  

**Link**: [PDF](https://arxiv.org/pdf/2511.01701)  

**Abstract**: A fundamental question in search-guided AI: what topology should guide Monte Carlo Tree Search (MCTS) in puzzle solving? Prior work applied topological features to guide MCTS in ARC-style tasks using grid topology -- the Laplacian spectral properties of cell connectivity -- and found no benefit. We identify the root cause: grid topology is constant across all instances. We propose measuring \emph{solution space topology} instead: the structure of valid color assignments constrained by detected pattern rules. We build this via compatibility graphs where nodes are $(cell, color)$ pairs and edges represent compatible assignments under pattern constraints.
Our method: (1) detect pattern rules automatically with 100\% accuracy on 5 types, (2) construct compatibility graphs encoding solution space structure, (3) extract topological features (algebraic connectivity, rigidity, color structure) that vary with task difficulty, (4) integrate these features into MCTS node selection via sibling-normalized scores.
We provide formal definitions, a rigorous selection formula, and comprehensive ablations showing that algebraic connectivity is the dominant signal. The work demonstrates that topology matters for search -- but only the \emph{right} topology. For puzzle solving, this is solution space structure, not problem space structure. 

**Abstract (ZH)**: 搜索导向型AI中的基础问题：拼图解决中蒙特卡洛树搜索(MCTS)应由何种拓扑结构引导？先前的研究使用网格拓扑——单元格连接的拉普拉斯谱特性——引导ARC风格任务中的MCTS，并未发现益处。我们识别出根本原因：网格拓扑在所有实例中是恒定的。我们建议测量解空间拓扑：由检测到的模式规则约束的有效颜色分配结构。我们通过兼容图构建这一结构，其中节点表示（单元格，颜色）对，边表示在模式约束下的兼容分配。我们的方法包括：(1) 自动检测五种模式规则，准确率100%；(2) 构建编码解空间结构的兼容图；(3) 提取随着任务难度变化的拓扑特征（代数连接性、刚性、颜色结构）；(4) 通过兄弟归一化得分将这些特征整合到MCTS节点选择中。我们提供了正式定义、严格的特征选择公式，并进行了全面的消融实验，表明代数连接性是最重要的信号。工作表明，拓扑结构对于搜索很重要——但只有合适的拓扑结构。对于拼图解决，这应该是解空间结构，而不是问题空间结构。 

---
# Student Engagement in AI Assisted Complex Problem Solving: A Pilot Study of Human AI Rubik's Cube Collaboration 

**Title (ZH)**: 基于人工智协作的复杂问题解决中学生的参与：一项人类与AI协作解决魔方试点研究 

**Authors**: Kirk Vanacore, Jaclyn Ocumpaugh, Forest Agostinelli, Dezhi Wu, Sai Vuruma, Matt Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01683)  

**Abstract**: Games and puzzles play important pedagogical roles in STEM learning. New AI algorithms that can solve complex problems offer opportunities for scaffolded instruction in puzzle solving. This paper presents the ALLURE system, which uses an AI algorithm (DeepCubeA) to guide students in solving a common first step of the Rubik's Cube (i.e., the white cross). Using data from a pilot study we present preliminary findings about students' behaviors in the system, how these behaviors are associated with STEM skills - including spatial reasoning, critical thinking and algorithmic thinking. We discuss how data from ALLURE can be used in future educational data mining to understand how students benefit from AI assistance and collaboration when solving complex problems. 

**Abstract (ZH)**: 游戏和谜题在STEM学习中发挥着重要的教学作用。新的AI算法能够解决复杂问题，为谜题解决的支架式教学提供了机会。本文介绍了ALLURE系统，该系统使用AI算法（DeepCubeA）指导学生解决魔方的第一个常见步骤（即白色十字架）。通过一项试点研究的数据，我们呈现了学生在系统中的行为及其与STEM技能（包括空间推理、批判性思维和算法思维）之间的关系。我们讨论了ALLURE数据在未来的教育数据挖掘中如何用于理解学生在解决复杂问题时如何从AI辅助和协作中受益。 

---
# Spin-Adapted Neural Network Wavefunctions in Real Space 

**Title (ZH)**: 实空间中的自旋适配神经网络波函数 

**Authors**: Ruichen Li, Yuzhi Liu, Du Jiang, Yixiao Chen, Xuelan Wen, Wenrui Li, Di He, Liwei Wang, Ji Chen, Weiluo Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.01671)  

**Abstract**: Spin plays a fundamental role in understanding electronic structure, yet many real-space wavefunction methods fail to adequately consider it. We introduce the Spin-Adapted Antisymmetrization Method (SAAM), a general procedure that enforces exact total spin symmetry for antisymmetric many-electron wavefunctions in real space. In the context of neural network-based quantum Monte Carlo (NNQMC), SAAM leverages the expressiveness of deep neural networks to capture electron correlation while enforcing exact spin adaptation via group representation theory. This framework provides a principled route to embed physical priors into otherwise black-box neural network wavefunctions, yielding a compact representation of correlated system with neural network orbitals. Compared with existing treatments of spin in NNQMC, SAAM is more accurate and efficient, achieving exact spin purity without any additional tunable hyperparameters. To demonstrate its effectiveness, we apply SAAM to study the spin ladder of iron-sulfur clusters, a long-standing challenge for many-body methods due to their dense spectrum of nearly degenerate spin states. Our results reveal accurate resolution of low-lying spin states and spin gaps in [Fe$_2$S$_2$] and [Fe$_4$S$_4$] clusters, offering new insights into their electronic structures. In sum, these findings establish SAAM as a robust, hyperparameter-free standard for spin-adapted NNQMC, particularly for strongly correlated systems. 

**Abstract (ZH)**: Spin-适应化反对称化方法（SAAM）在实空间中精确维护多电子波函数的总自旋对称性，在神经网络量子蒙特卡洛（NNQMC）框架下的应用及效果分析 

---
# EngChain: A Symbolic Benchmark for Verifiable Multi-Step Reasoning in Engineering 

**Title (ZH)**: EngChain: 一种用于工程中可验证多步推理的符号基准 

**Authors**: Ayesha Gull, Muhammad Usman Safder, Rania Elbadry, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.01650)  

**Abstract**: Large Language Models (LLMs) are increasingly being applied to specialized, high-stakes domains like engineering, which demands rigorous evaluation of their complex reasoning capabilities. While current benchmarks assess language understanding, factual recall, mathematics or code generation, none capture the integrative reasoning central to engineering where scientific principles, quantitative modeling and practical constraints must converge. To address this gap, we introduce EngChain, a benchmark for verifiable multi-step engineering problem-solving. EngChain contains 90 problems spanning three engineering branches, organized into 9 domains and 20 distinct areas. The problems are generated from symbolic templates with a high degree of randomization to ensure diversity and eliminate the risk of contamination. With this benchmark, we move beyond final answer accuracy with a two-stage evaluation: we first quantitatively verify the numerical and semantic validity of each reasoning step and then introduce LLM-As-A-Judge, an automated system to qualitatively categorize the identified reasoning errors. 

**Abstract (ZH)**: 大型语言模型（LLMs）在工程等高风险专领域中的应用要求对其复杂的推理能力进行严格的评估。当前的基准测试评估语言理解、事实记忆、数学或代码生成，但未能捕捉到工程领域中科学原理、定量建模和实践约束必须综合运用的核心推理能力。为填补这一空白，我们引入了EngChain，这是一个用于可验证的多步工程问题解决的基准测试。EngChain包含90个跨越三个工程分支的问题，分为9个领域和20个不同的研究方向。这些问题是根据符号模板生成的，并且具有高度的随机化，以确保多样性和消除污染的风险。借助这一基准测试，我们提出了两阶段评估方法：首先定量验证每一个推理步骤的数值和语义有效性，然后引入LLM-As-A-Judge，一个自动系统，用于对识别出的推理错误进行定性分类。 

---
# Federated Cyber Defense: Privacy-Preserving Ransomware Detection Across Distributed Systems 

**Title (ZH)**: 跨分布式系统中的隐私保护型勒索软件检测：联邦网络防御 

**Authors**: Daniel M. Jimenez-Gutierrez, Enrique Zuazua, Joaquin Del Rio, Oleksii Sliusarenko, Xabi Uribe-Etxebarria  

**Link**: [PDF](https://arxiv.org/pdf/2511.01583)  

**Abstract**: Detecting malware, especially ransomware, is essential to securing today's interconnected ecosystems, including cloud storage, enterprise file-sharing, and database services. Training high-performing artificial intelligence (AI) detectors requires diverse datasets, which are often distributed across multiple organizations, making centralization necessary. However, centralized learning is often impractical due to security, privacy regulations, data ownership issues, and legal barriers to cross-organizational sharing. Compounding this challenge, ransomware evolves rapidly, demanding models that are both robust and adaptable.
In this paper, we evaluate Federated Learning (FL) using the this http URL FL platform, which enables multiple organizations to collaboratively train a ransomware detection model while keeping raw data local and secure. This paradigm is particularly relevant for cybersecurity companies (including both software and hardware vendors) that deploy ransomware detection or firewall systems across millions of endpoints. In such environments, data cannot be transferred outside the customer's device due to strict security, privacy, or regulatory constraints. Although FL applies broadly to malware threats, we validate the approach using the Ransomware Storage Access Patterns (RanSAP) dataset.
Our experiments demonstrate that FL improves ransomware detection accuracy by a relative 9% over server-local models and achieves performance comparable to centralized training. These results indicate that FL offers a scalable, high-performing, and privacy-preserving framework for proactive ransomware detection across organizational and regulatory boundaries. 

**Abstract (ZH)**: 使用Federated Learning进行分布式 ransomware检测：基于this http URL平台的评估 

---
# Real-time Continual Learning on Intel Loihi 2 

**Title (ZH)**: Intel Loihi 2上的实时连续学习 

**Authors**: Elvin Hajizada, Danielle Rager, Timothy Shea, Leobardo Campos-Macias, Andreas Wild, Eyke Hüllermeier, Yulia Sandamirskaya, Mike Davies  

**Link**: [PDF](https://arxiv.org/pdf/2511.01553)  

**Abstract**: AI systems on edge devices face a critical challenge in open-world environments: adapting when data distributions shift and novel classes emerge. While offline training dominates current paradigms, online continual learning (OCL)--where models learn incrementally from non-stationary streams without catastrophic forgetting--remains challenging in power-constrained settings. We present a neuromorphic solution called CLP-SNN: a spiking neural network architecture for Continually Learning Prototypes and its implementation on Intel's Loihi 2 chip. Our approach introduces three innovations: (1) event-driven and spatiotemporally sparse local learning, (2) a self-normalizing three-factor learning rule maintaining weight normalization, and (3) integrated neurogenesis and metaplasticity for capacity expansion and forgetting mitigation. On OpenLORIS few-shot learning experiments, CLP-SNN achieves accuracy competitive with replay methods while being rehearsal-free. CLP-SNN delivers transformative efficiency gains: 70\times faster (0.33ms vs 23.2ms), and 5,600\times more energy efficient (0.05mJ vs 281mJ) than the best alternative OCL on edge GPU. This demonstrates that co-designed brain-inspired algorithms and neuromorphic hardware can break traditional accuracy-efficiency trade-offs for future edge AI systems. 

**Abstract (ZH)**: 边缘设备上的AI系统在开放世界环境中面临关键挑战：适应数据分布变化和新类别的出现。虽然离线训练目前占主导地位，但在功率受限的环境中，无灾难性遗忘的持续在线学习（OCL）仍然具有挑战性。我们提出了一种名为CLP-SNN的类神经解决方案：一种用于持续学习原型的突触神经网络架构及其在英特尔Loihi 2芯片上的实现。我们的方法引入了三项创新：（1）事件驱动和时空稀疏的局部学习，（2）一种自规范的三因素学习规则保持权重规范化，以及（3）集成的神经发生和元塑性以扩展容量并减轻遗忘。在OpenLORIS少样本学习实验中，CLP-SNN在无需回放的情况下达到了与回放方法竞争力的准确率。CLP-SNN实现了革命性的效率提升：速度提高了70倍（0.33ms vs 23.2ms），能效提高了5600倍（0.05mJ vs 281mJ），超越了边缘GPU上最佳替代持续在线学习方案。这表明，联合设计的仿脑算法和类神经硬件可以打破未来边缘AI系统的准确性和效率trade-offs。 

---
# DAMBench: A Multi-Modal Benchmark for Deep Learning-based Atmospheric Data Assimilation 

**Title (ZH)**: DAMBench：基于深度学习的大气数据同化多模态基准 

**Authors**: Hao Wang, Zixuan Weng, Jindong Han, Wei Fan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01468)  

**Abstract**: Data Assimilation is a cornerstone of atmospheric system modeling, tasked with reconstructing system states by integrating sparse, noisy observations with prior estimation. While traditional approaches like variational and ensemble Kalman filtering have proven effective, recent advances in deep learning offer more scalable, efficient, and flexible alternatives better suited for complex, real-world data assimilation involving large-scale and multi-modal observations. However, existing deep learning-based DA research suffers from two critical limitations: (1) reliance on oversimplified scenarios with synthetically perturbed observations, and (2) the absence of standardized benchmarks for fair model comparison. To address these gaps, in this work, we introduce DAMBench, the first large-scale multi-modal benchmark designed to evaluate data-driven DA models under realistic atmospheric conditions. DAMBench integrates high-quality background states from state-of-the-art forecasting systems and real-world multi-modal observations (i.e., real-world weather stations and satellite imagery). All data are resampled to a common grid and temporally aligned to support systematic training, validation, and testing. We provide unified evaluation protocols and benchmark representative data assimilation approaches, including latent generative models and neural process frameworks. Additionally, we propose a lightweight multi-modal plugin to demonstrate how integrating realistic observations can enhance even simple baselines. Through comprehensive experiments, DAMBench establishes a rigorous foundation for future research, promoting reproducibility, fair comparison, and extensibility to real-world multi-modal scenarios. Our dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 大规模多模态数据同化基准：DAMBench 

---
# When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA 

**Title (ZH)**: 何时信任答案：与问题对齐的语义最近邻熵在更安全的手术VQA中的应用 

**Authors**: Dennis Pierantozzi, Luca Carlini, Mauro Orazio Drago, Chiara Lena, Cesare Hassan, Elena De Momi, Danail Stoyanov, Sophia Bano, Mobarak I. Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2511.01458)  

**Abstract**: Safety and reliability are essential for deploying Visual Question Answering (VQA) in surgery, where incorrect or ambiguous responses can harm the patient. Most surgical VQA research focuses on accuracy or linguistic quality while overlooking safety behaviors such as ambiguity awareness, referral to human experts, or triggering a second opinion. Inspired by Automatic Failure Detection (AFD), we study uncertainty estimation as a key enabler of safer decision making. We introduce Question Aligned Semantic Nearest Neighbor Entropy (QA-SNNE), a black box uncertainty estimator that incorporates question semantics into prediction confidence. It measures semantic entropy by comparing generated answers with nearest neighbors in a medical text embedding space, conditioned on the question. We evaluate five models, including domain specific Parameter-Efficient Fine-Tuned (PEFT) models and zero-shot Large Vision-Language Models (LVLMs), on EndoVis18-VQA and PitVQA. PEFT models degrade under mild paraphrasing, while LVLMs are more resilient. Across three LVLMs and two PEFT baselines, QA-SNNE improves AUROC in most in-template settings and enhances hallucination detection. The Area Under the ROC Curve (AUROC) increases by 15-38% for zero-shot models, with gains maintained under out-of-template stress. QA-SNNE offers a practical and interpretable step toward AFD in surgical VQA by linking semantic uncertainty to question context. Combining LVLM backbones with question aligned uncertainty estimation can improve safety and clinician trust. The code and model are available at this https URL 

**Abstract (ZH)**: 视觉问答在手术中的安全性和可靠性至关重要，因为错误或模棱两可的回答可能会危害患者。大多数手术视觉问答研究集中在准确性或语言质量上，而忽视了诸如察觉模棱两可、咨询人类专家或触发二次意见等安全行为。受自动故障检测(Auto Failure Detection, AFD)的启发，我们研究不确定性估计作为实现更安全决策的关键使能器。我们引入了问题对齐语义最近邻熵(Question Aligned Semantic Nearest Neighbor Entropy, QA-SNNE)，这是一种黑盒不确定性估计器，将问题语义融入到预测置信度中。它通过在条件化于问题的医学文本嵌入空间中比较生成的答案与最近邻居来度量语义熵。我们在EndoVis18-VQA和PitVQA上评估了五个模型，包括特定领域的参数高效微调(Parameter-Efficient Fine-Tuned, PEFT)模型和零样本大型视觉-语言模型(Large Vision-Language Models, LVLMs)。PEFT模型在轻度改写下退化，而LVLMs更具抗性。在三个LVLMs和两个PEFT基线下，QA-SNNE在大多数套用模板设置中提高了Area Under the ROC Curve (AUROC)，并增强了幻觉检测能力。零样本模型的AUROC提高了15-38%，并在超出模板的压力环境下保持了收益。QA-SNNE通过将语义不确定性与问题上下文关联起来，为手术视觉问答中的自动故障检测提供了一个实用且可解释的步骤。结合LVLM骨干和问题对齐不确定度估计可以提高安全性和临床医生的信任度。代码和模型可在以下链接获取。 

---
# Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction 

**Title (ZH)**: 隐私保护的层次元学习与VLMs在细粒度水果品质预测中的序数学习 

**Authors**: Riddhi Jain, Manasi Patwardhan, Aayush Mishra, Parijat Deshpande, Beena Rai  

**Link**: [PDF](https://arxiv.org/pdf/2511.01449)  

**Abstract**: To effectively manage the wastage of perishable fruits, it is crucial to accurately predict their freshness or shelf life using non-invasive methods that rely on visual data. In this regard, deep learning techniques can offer a viable solution. However, obtaining fine-grained fruit freshness labels from experts is costly, leading to a scarcity of data. Closed proprietary Vision Language Models (VLMs), such as Gemini, have demonstrated strong performance in fruit freshness detection task in both zero-shot and few-shot settings. Nonetheless, food retail organizations are unable to utilize these proprietary models due to concerns related to data privacy, while existing open-source VLMs yield sub-optimal performance for the task. Fine-tuning these open-source models with limited data fails to achieve the performance levels of proprietary models. In this work, we introduce a Model-Agnostic Ordinal Meta-Learning (MAOML) algorithm, designed to train smaller VLMs. This approach utilizes meta-learning to address data sparsity and leverages label ordinality, thereby achieving state-of-the-art performance in the fruit freshness classification task under both zero-shot and few-shot settings. Our method achieves an industry-standard accuracy of 92.71%, averaged across all fruits.
Keywords: Fruit Quality Prediction, Vision Language Models, Meta Learning, Ordinal Regression 

**Abstract (ZH)**: 有效地管理 perishable 水果的损耗需要使用非侵入式方法准确预测其新鲜度或保质期。在这方面，深度学习技术可以提供可行的解决方案。然而，从专家获取精细的新鲜度标签成本高昂，导致数据稀缺。闭源产权视觉语言模型（VLMs），如 Gemini，在零样本和少量样本设置下的水果新鲜度检测任务中表现出色。然而，食品零售组织由于数据隐私问题无法使用这些专用模型，而现有的开源 VLMs 在任务中表现不佳。使用有限数据微调这些开源模型无法达到专用模型的性能水平。在这项工作中，我们提出了一种模型无关的序列表 Vlad 动力学习（MAOML）算法，旨在训练较小的 VLMs。该方法利用元学习解决数据稀疏性问题，并利用标签序数性，从而在零样本和少量样本设置下的水果新鲜度分类任务中实现了最先进的性能。我们的方法在所有水果上的平均准确率达到了行业标准的 92.71%。关键词：水果品质预测，视觉语言模型，元学习，序数回归。 

---
# RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets 

**Title (ZH)**: RAGSmith：跨数据集寻优检索增强生成方法组成框架 

**Authors**: Muhammed Yusuf Kartal, Suha Kagan Kose, Korhan Sevinç, Burak Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2511.01386)  

**Abstract**: Retrieval-Augmented Generation (RAG) quality depends on many interacting choices across retrieval, ranking, augmentation, prompting, and generation, so optimizing modules in isolation is brittle. We introduce RAGSmith, a modular framework that treats RAG design as an end-to-end architecture search over nine technique families and 46{,}080 feasible pipeline configurations. A genetic search optimizes a scalar objective that jointly aggregates retrieval metrics (recall@k, mAP, nDCG, MRR) and generation metrics (LLM-Judge and semantic similarity). We evaluate on six Wikipedia-derived domains (Mathematics, Law, Finance, Medicine, Defense Industry, Computer Science), each with 100 questions spanning factual, interpretation, and long-answer types. RAGSmith finds configurations that consistently outperform naive RAG baseline by +3.8\% on average (range +1.2\% to +6.9\% across domains), with gains up to +12.5\% in retrieval and +7.5\% in generation. The search typically explores $\approx 0.2\%$ of the space ($\sim 100$ candidates) and discovers a robust backbone -- vector retrieval plus post-generation reflection/revision -- augmented by domain-dependent choices in expansion, reranking, augmentation, and prompt reordering; passage compression is never selected. Improvement magnitude correlates with question type, with larger gains on factual/long-answer mixes than interpretation-heavy sets. These results provide practical, domain-aware guidance for assembling effective RAG systems and demonstrate the utility of evolutionary search for full-pipeline optimization. 

**Abstract (ZH)**: RAGSmith：一种基于进化搜索的模块化框架，用于端到端的RAG架构搜索 

---
# AI Literacy in UAE Libraries: Assessing Competencies, Training Needs, and Ethical Considerations for the Digital Age 

**Title (ZH)**: 阿联酋图书馆的AI素养评估：数字时代的能力要求、培训需求及伦理考量 

**Authors**: Zafar Imam Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01353)  

**Abstract**: The study explores the current state of artificial intelligence (AI) literacy levels among library professionals employing a quantitative approach consisting of 92 surveys of LIS professionals in the United Arab Emirates (UAE). Findings of the study revealed the presence of strong cognitive competencies, while there were gaps observed in behavioral and normative competencies, especially related to AI biases, AI-powered learning, and ethical considerations. There was a disconnect observed between the perceived importance of AI skills and the effectiveness of the current training programs. 

**Abstract (ZH)**: 本研究采用定量方法，通过对阿拉伯联合酋长国（UAE）92名图书馆与信息科学专业人员进行调查，探讨了图书馆专业人员的人工智能素养现状。研究发现，图书馆专业人员在认知能力方面表现出较强的实力，但在行为和规范能力方面存在差距，尤其是在人工智能偏见、人工智能赋能学习和伦理考虑方面。研究还发现，专业人员感知到的人工智能技能重要性与当前培训项目的有效性之间存在脱节。 

---
# The Future of Generative AI in Software Engineering: A Vision from Industry and Academia in the European GENIUS Project 

**Title (ZH)**: 欧洲GENIUS项目视角下的生成式AI在未来软件工程中的前景：来自行业与学术界的愿景 

**Authors**: Robin Gröpler, Steffen Klepke, Jack Johns, Andreas Dreschinski, Klaus Schmid, Benedikt Dornauer, Eray Tüzün, Joost Noppen, Mohammad Reza Mousavi, Yongjian Tang, Johannes Viehmann, Selin Şirin Aslangül, Beum Seuk Lee, Adam Ziolkowski, Eric Zie  

**Link**: [PDF](https://arxiv.org/pdf/2511.01348)  

**Abstract**: Generative AI (GenAI) has recently emerged as a groundbreaking force in Software Engineering, capable of generating code, suggesting fixes, and supporting quality assurance. While its use in coding tasks shows considerable promise, applying GenAI across the entire Software Development Life Cycle (SDLC) has not yet been fully explored. Critical uncertainties in areas such as reliability, accountability, security, and data privacy demand deeper investigation and coordinated action. The GENIUS project, comprising over 30 European industrial and academic partners, aims to address these challenges by advancing AI integration across all SDLC phases. It focuses on GenAI's potential, the development of innovative tools, and emerging research challenges, actively shaping the future of software engineering. This vision paper presents a shared perspective on the future of GenAI-based software engineering, grounded in cross-sector dialogue and experience within the GENIUS consortium, supported by an exploratory literature review. The paper explores four central elements: (1) a structured overview of current challenges in GenAI adoption across the SDLC; (2) a forward-looking vision outlining key technological and methodological advances expected over the next five years; (3) anticipated shifts in the roles and required skill sets of software professionals; and (4) the contribution of GENIUS in realizing this transformation through practical tools and industrial validation. By aligning technical innovation with business relevance, this paper aims to inform both research agendas and industrial strategies, providing a foundation for reliable, scalable, and industry-ready GenAI solutions for software engineering teams. 

**Abstract (ZH)**: Generative AI (GenAI)在软件工程中的前景：GENIUS项目探索全过程软件开发生命周期中的机遇与挑战 

---
# Beyond Permissions: Investigating Mobile Personalization with Simulated Personas 

**Title (ZH)**: 超越权限：基于模拟人像探究移动个性化服务 

**Authors**: Ibrahim Khalilov, Chaoran Chen, Ziang Xiao, Tianshi Li, Toby Jia-Jun Li, Yaxing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01336)  

**Abstract**: Mobile applications increasingly rely on sensor data to infer user context and deliver personalized experiences. Yet the mechanisms behind this personalization remain opaque to users and researchers alike. This paper presents a sandbox system that uses sensor spoofing and persona simulation to audit and visualize how mobile apps respond to inferred behaviors. Rather than treating spoofing as adversarial, we demonstrate its use as a tool for behavioral transparency and user empowerment. Our system injects multi-sensor profiles - generated from structured, lifestyle-based personas - into Android devices in real time, enabling users to observe app responses to contexts such as high activity, location shifts, or time-of-day changes. With automated screenshot capture and GPT-4 Vision-based UI summarization, our pipeline helps document subtle personalization cues. Preliminary findings show measurable app adaptations across fitness, e-commerce, and everyday service apps such as weather and navigation. We offer this toolkit as a foundation for privacy-enhancing technologies and user-facing transparency interventions. 

**Abstract (ZH)**: 移动应用日益依赖传感器数据推断用户上下文并提供个性化体验，但其个性化机制对用户和研究者来说仍然不透明。本文提出一个沙盒系统，通过传感器模拟和 persona 模拟来审计和可视化移动应用对推断行为的响应。我们不仅将模拟视为敌对行为，还展示了其作为一种行为透明性和用户赋权工具的应用。该系统实时将基于生活方式的人格特征生成的多传感器配置文件注入 Android 设备，使用户能够观察应用对高强度活动、位置变化或时间变化等上下文的响应。借助自动屏幕截图捕获和基于 GPT-4 Vision 的 UI 总结，我们的流水线有助于记录微妙的个性化线索。初步研究结果表明，健身、电子商务和日常生活服务（如天气和导航）应用的可衡量适应性改进。我们提供这一工具包作为增强隐私技术和面向用户的透明干预的基础。 

---
# AI for Requirements Engineering: Industry adoption and Practitioner perspectives 

**Title (ZH)**: AI在需求工程中的应用：行业采纳与实践者视角 

**Authors**: Lekshmi Murali Rani, Richard Berntsson Svensson, Robert Feldt  

**Link**: [PDF](https://arxiv.org/pdf/2511.01324)  

**Abstract**: The integration of AI for Requirements Engineering (RE) presents significant benefits but also poses real this http URL RE is fundamental to software engineering, limited research has examined AI adoption in this http URL surveyed 55 software practitioners to map AI usage across four RE phases:Elicitation, Analysis, Specification, and Validation, and four approaches for decision making: human only decisions, AI validation, Human AI Collaboration (HAIC), and full AI this http URL also shared their perceptions, challenges, and opportunities when applying AI for RE this http URL data show that 58.2% of respondents already use AI in RE, and 69.1% view its impact as positive or very this http URL dominates practice, accounting for 54.4% of all RE techniques, while full AI automation remains minimal at 5.4%.Passive AI validation (4.4 to 6.2%) lags even further behind, indicating that practitioners value AI's active support over passive this http URL findings suggest that AI is most effective when positioned as a collaborative partner rather than a replacement for human this http URL also highlights the need for RE specific HAIC frameworks along with robust and responsible AI governance as AI adoption in RE grows. 

**Abstract (ZH)**: AI在需求工程中的集成：优势与挑战 

---
# Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models 

**Title (ZH)**: 不扰动图像，扰动模型：通过抗个性化扩散模型迈向稳健的隐私保护 

**Authors**: Tae-Young Lee, Juwon Seo, Jong Hwan Ko, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.01307)  

**Abstract**: Recent advances in diffusion models have enabled high-quality synthesis of specific subjects, such as identities or objects. This capability, while unlocking new possibilities in content creation, also introduces significant privacy risks, as personalization techniques can be misused by malicious users to generate unauthorized content. Although several studies have attempted to counter this by generating adversarially perturbed samples designed to disrupt personalization, they rely on unrealistic assumptions and become ineffective in the presence of even a few clean images or under simple image transformations. To address these challenges, we shift the protection target from the images to the diffusion model itself to hinder the personalization of specific subjects, through our novel framework called Anti-Personalized Diffusion Models (APDM). We first provide a theoretical analysis demonstrating that a naive approach of existing loss functions to diffusion models is inherently incapable of ensuring convergence for robust anti-personalization. Motivated by this finding, we introduce Direct Protective Optimization (DPO), a novel loss function that effectively disrupts subject personalization in the target model without compromising generative quality. Moreover, we propose a new dual-path optimization strategy, coined Learning to Protect (L2P). By alternating between personalization and protection paths, L2P simulates future personalization trajectories and adaptively reinforces protection at each step. Experimental results demonstrate that our framework outperforms existing methods, achieving state-of-the-art performance in preventing unauthorized personalization. The code is available at this https URL. 

**Abstract (ZH)**: Recent Advances in Anti-Personalized Diffusion Models 

---
# DeepSpecs: Expert-Level Questions Answering in 5G 

**Title (ZH)**: DeepSpecs: 5G领域的专家级别问题回答 

**Authors**: Aman Ganapathy Manvattira, Yifei Xu, Ziyue Dang, Songwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01305)  

**Abstract**: 5G technology enables mobile Internet access for billions of users. Answering expert-level questions about 5G specifications requires navigating thousands of pages of cross-referenced standards that evolve across releases. Existing retrieval-augmented generation (RAG) frameworks, including telecom-specific approaches, rely on semantic similarity and cannot reliably resolve cross-references or reason about specification evolution. We present DeepSpecs, a RAG system enhanced by structural and temporal reasoning via three metadata-rich databases: SpecDB (clause-aligned specification text), ChangeDB (line-level version diffs), and TDocDB (standardization meeting documents). DeepSpecs explicitly resolves cross-references by recursively retrieving referenced clauses through metadata lookup, and traces specification evolution by mining changes and linking them to Change Requests that document design rationale. We curate two 5G QA datasets: 573 expert-annotated real-world questions from practitioner forums and educational resources, and 350 evolution-focused questions derived from approved Change Requests. Across multiple LLM backends, DeepSpecs outperforms base models and state-of-the-art telecom RAG systems; ablations confirm that explicit cross-reference resolution and evolution-aware retrieval substantially improve answer quality, underscoring the value of modeling the structural and temporal properties of 5G standards. 

**Abstract (ZH)**: 5G技术使 billions of 用户能够接入移动互联网。回答关于5G规范的高级问题需要导航数千页跨参考的标准，这些标准在不同版本中不断演进。现有的检索增强生成（RAG）框架，包括电信特定的方法，依赖于语义相似性，无法可靠地解决跨参考问题或推理规范演进。我们提出DeepSpecs，一种通过三个元数据丰富的数据库增强的RAG系统：SpecDB（条款对齐的规范文本）、ChangeDB（行级版本差异）和TDocDB（标准化会议文档）。DeepSpecs通过递归检索通过元数据查找引用的条款来明确解决跨参考问题，并通过挖掘变化并将它们链接到记录设计理性的变更请求，跟踪规范演进。我们 curate 了两个5G QA数据集：357个由实践者论坛和教育资源标注的真实世界问题，以及350个专注于演进的问题，源自已批准的变更请求。在多个LLM后端中，DeepSpecs优于基础模型和最先进的电信RAG系统；消融实验确认，明确的跨参考解析和演进感知检索显著提高了答案质量，突显了建模5G标准的结构和时间属性的价值。 

---
# LSHFed: Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping 

**Title (ZH)**: LSHFed：基于局部敏感哈希梯度映射的健壮且通信高效的联邦学习 

**Authors**: Guanjie Cheng, Mengzhen Yang, Xinkui Zhao, Shuyi Yu, Tianyu Du, Yangyang Wu, Mengying Zhu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01296)  

**Abstract**: Federated learning (FL) enables collaborative model training across distributed nodes without exposing raw data, but its decentralized nature makes it vulnerable in trust-deficient environments. Inference attacks may recover sensitive information from gradient updates, while poisoning attacks can degrade model performance or induce malicious behaviors. Existing defenses often suffer from high communication and computation costs, or limited detection precision. To address these issues, we propose LSHFed, a robust and communication-efficient FL framework that simultaneously enhances aggregation robustness and privacy preservation. At its core, LSHFed incorporates LSHGM, a novel gradient verification mechanism that projects high-dimensional gradients into compact binary representations via multi-hyperplane locally-sensitive hashing. This enables accurate detection and filtering of malicious gradients using only their irreversible hash forms, thus mitigating privacy leakage risks and substantially reducing transmission overhead. Extensive experiments demonstrate that LSHFed maintains high model performance even when up to 50% of participants are collusive adversaries while achieving up to a 1000x reduction in gradient verification communication compared to full-gradient methods. 

**Abstract (ZH)**: LSHFed：一种同时增强聚合鲁棒性和隐私保护的联邦学习框架 

---
# Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions 

**Title (ZH)**: 基础模型在医学图像分析中的适应策略、挑战与未来方向 

**Authors**: Karma Phuntsho, Abdullah, Kyungmi Lee, Ickjai Lee, Euijoon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2511.01284)  

**Abstract**: Foundation models (FMs) have emerged as a transformative paradigm in medical image analysis, offering the potential to provide generalizable, task-agnostic solutions across a wide range of clinical tasks and imaging modalities. Their capacity to learn transferable representations from large-scale data has the potential to address the limitations of conventional task-specific models. However, adaptation of FMs to real-world clinical practice remains constrained by key challenges, including domain shifts, limited availability of high-quality annotated data, substantial computational demands, and strict privacy requirements. This review presents a comprehensive assessment of strategies for adapting FMs to the specific demands of medical imaging. We examine approaches such as supervised fine-tuning, domain-specific pretraining, parameter-efficient fine-tuning, self-supervised learning, hybrid methods, and multimodal or cross-modal frameworks. For each, we evaluate reported performance gains, clinical applicability, and limitations, while identifying trade-offs and unresolved challenges that prior reviews have often overlooked. Beyond these established techniques, we also highlight emerging directions aimed at addressing current gaps. These include continual learning to enable dynamic deployment, federated and privacy-preserving approaches to safeguard sensitive data, hybrid self-supervised learning to enhance data efficiency, data-centric pipelines that combine synthetic generation with human-in-the-loop validation, and systematic benchmarking to assess robust generalization under real-world clinical variability. By outlining these strategies and associated research gaps, this review provides a roadmap for developing adaptive, trustworthy, and clinically integrated FMs capable of meeting the demands of real-world medical imaging. 

**Abstract (ZH)**: 基础模型（FMs）在医学图像分析中 emerged 为一项变革性范式，提供了一种在广泛临床任务和成像模ality 中提供泛化、任务无关解决方案的潜力。它们从大规模数据中学习到的可迁移表示有能力解决传统任务特定模型的限制。然而，将 FMs 融入实际临床实践中仍受到关键挑战的约束，包括领域转移、高质量标注数据的有限可用性、巨大的计算需求以及严格的隐私要求。本文综述了适应医学成像特定需求的策略。我们探讨了诸如监督微调、领域特定预训练、参数高效微调、自我监督学习、混合方法和多模态或跨模态框架等方法。对于每种方法，我们评估了报告的性能增益、临床适用性和限制，同时还指出了先前综述往往忽视的权衡和未解决的挑战。除了这些成熟的技术外，我们还强调了旨在解决当前空白的新兴方向。这些包括持续学习以实现动态部署、分布式和隐私保护方法以保护敏感数据、混合自我监督学习以提高数据效率、以合成生成与人工在环验证相结合的数据为中心的管道以及系统基准测试以评估在现实世界临床变异性下的鲁棒泛化能力。通过概述这些策略及其相关的研究空白，本文为开发适应性强、值得信赖且临床集成的基础模型提供了路线图，这些模型能够满足医学成像的实际需求。 

---
# Adversarial Spatio-Temporal Attention Networks for Epileptic Seizure Forecasting 

**Title (ZH)**: 基于对抗时空注意力的癫痫发作预测网络 

**Authors**: Zan Li, Kyongmin Yeo, Wesley Gifford, Lara Marcuse, Madeline Fields, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2511.01275)  

**Abstract**: Forecasting epileptic seizures from multivariate EEG signals represents a critical challenge in healthcare time series prediction, requiring high sensitivity, low false alarm rates, and subject-specific adaptability. We present STAN, an Adversarial Spatio-Temporal Attention Network that jointly models spatial brain connectivity and temporal neural dynamics through cascaded attention blocks with alternating spatial and temporal modules. Unlike existing approaches that assume fixed preictal durations or separately process spatial and temporal features, STAN captures bidirectional dependencies between spatial and temporal patterns through a unified cascaded architecture. Adversarial training with gradient penalty enables robust discrimination between interictal and preictal states learned from clearly defined 15-minute preictal windows. Continuous 90-minute pre-seizure monitoring reveals that the learned spatio-temporal attention patterns enable early detection: reliable alarms trigger at subject-specific times (typically 15-45 minutes before onset), reflecting the model's capacity to capture subtle preictal dynamics without requiring individualized training. Experiments on two benchmark EEG datasets (CHB-MIT scalp: 8 subjects, 46 events; MSSM intracranial: 4 subjects, 14 events) demonstrate state-of-the-art performance: 96.6% sensitivity with 0.011 false detections per hour and 94.2% sensitivity with 0.063 false detections per hour, respectively, while maintaining computational efficiency (2.3M parameters, 45 ms latency, 180 MB memory) for real-time edge deployment. Beyond epilepsy, the proposed framework provides a general paradigm for spatio-temporal forecasting in healthcare and other time series domains where individual heterogeneity and interpretability are crucial. 

**Abstract (ZH)**: 基于多变量EEG信号的癫痫发作预测代表了医疗健康时间序列预测中的一个关键挑战，要求高灵敏度、低误报率和个体特定的适应性。我们提出STAN（Adversarial Spatio-Temporal Attention Network），通过交替的空间和时间模块组成的级联注意力块，联合建模脑的空间连接性和时间神经动力学。不同于现有方法假设固定的前驱期长度或分别处理空间和时间特征，STAN通过统一的级联架构捕获空间和时间模式的双向依赖性。基于明确定义的15分钟前驱期窗口学习的间歇期和前驱期状态之间的鲁棒区分通过对抗训练和梯度惩罚实现。连续90分钟的预发作监测表明，学习到的空间-时间注意力模式能够实现早期检测：可靠的警报在个体特定的时间触发（通常在发作前15-45分钟），体现了模型捕获微妙的前驱期动力学的能力，无需个体化训练。实验在两个基准EEG数据集（CHB-MIT头皮：8名患者，46次事件；MSSM颅内：4名患者，14次事件）上展示了最先进的性能：分别达到96.6%和94.2%的灵敏度，每小时误报率分别为0.011和0.063，同时保持计算效率（2.3百万参数，45毫秒延迟，180MB内存），适用于实时边缘部署。除了癫痫外，所提出的框架为医疗健康和其他时间序列领域中个体异质性和可解释性的关键要求提供了通用的空间-时间预测范式。 

---
# Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play 

**Title (ZH)**: Speech-DRAME: 人类对齐框架在语音角色扮演基准中的应用 

**Authors**: Jiatong Shi, Jionghao Han, Yichen Lu, Santiago Pascual, Pengfei Wu, Chenye Cui, Shinji Watanabe, Chao Weng, Cong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01261)  

**Abstract**: Role-play has become a key testbed for generative models, expanding from text-only dialogue to multimodal interaction. Extending role-play to speech captures prosody, emotion, and delivery, but also poses new evaluation challenges. Current pipelines often use audio large language models (ALLMs) as zero-shot judges, which miss paralinguistic cues, collapse multiple aspects into coarse scores, and rely on synthetic speech references that fail to reflect real-world roles. We present Speech-DRAME, a unified framework that contributes at three levels: (i) Speech-DRAME-EvalBench, an evaluation benchmark with bilingual human-annotated data and protocols for training and testing speech evaluation models (SEMs), (ii) DRAME-Eval, a fine-tuned evaluation model, which substantially outperforms zero-shot and few-shot ALLMs, and (iii) Speech-DRAME-RoleBench, a speech role-play benchmark that leverages DRAME-Eval as an automatic judge to compare speech foundation models (SFMs). Speech-DRAME distinguishes between two complementary evaluation strategies: Archetype Evaluation, a top-down approach measuring adherence to broad role archetypes, and Realism Evaluation, a bottom-up approach grounded in real human speech that emphasizes nuanced role quality. Compared to zero-shot ALLM judges, DRAME-Eval achieves stronger agreement with human ratings (Pearson correlation from 0.480 to 0.629 in archetypes, and 0.390 to 0.625 in realism). By integrating transparent benchmark resources, modeling approaches, and system-level evaluation, Speech-DRAME provides the first comprehensive, reproducible foundation for assessing spoken role-play. 

**Abstract (ZH)**: 角色扮演已成为生成模型的关键测试床，从仅文本对话扩展至多模态交互。将角色扮演扩展到语音捕捉了语调、情感和表达方式，但也带来了新的评估挑战。当前管道通常使用音音频大型语言模型（ALLMs）作为零样本评判者，这会忽略副语言线索，将多个方面合并为粗略评分，并依赖于无法反映真实角色的合成语音参考。我们提出Speech-DRAME，一个统一框架，从三个层面贡献：（i）Speech-DRAME-EvalBench，一个包含双语人工标注数据和训练与测试语音评估模型（SEMs）协议的评估基准；（ii）DRAME-Eval，一个微调的评估模型，显著优于零样本和少样本ALLMs；（iii）Speech-DRAME-RoleBench，一个利用DRAME-Eval作为自动评判者的语音角色扮演基准，用于比较语音基础模型（SFMs）。Speech-DRAME 区分了两种互补的评估策略：原型评估，一种自上而下的方法，衡量对广泛角色原型的遵循程度，和现实性评估，一种自下而上的方法，基于真实人类语音，强调角色质量的细腻之处。与零样本ALLM评判者相比，DRAME-Eval 与人类评分的一致性更强（在原型中的皮尔逊相关系数从0.480提升至0.629，在现实性中的皮尔逊相关系数从0.390提升至0.625）。通过整合透明基准资源、模型方法和系统级评估，Speech-DRAME 提供了评估口语角色扮演的第一个全面且可复制的基础。 

---
# Quantum Deep Learning Still Needs a Quantum Leap 

**Title (ZH)**: 量子深度学习仍需量子飞跃 

**Authors**: Hans Gundlach, Hrvoje Kukina, Jayson Lynch, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2511.01253)  

**Abstract**: Quantum computing technology is advancing rapidly. Yet, even accounting for these trends, a quantum leap would be needed for quantum computers to mean- ingfully impact deep learning over the coming decade or two. We arrive at this conclusion based on a first-of-its-kind survey of quantum algorithms and how they match potential deep learning applications. This survey reveals three important areas where quantum computing could potentially accelerate deep learning, each of which faces a challenging roadblock to realizing its potential. First, quantum algorithms for matrix multiplication and other algorithms central to deep learning offer small theoretical improvements in the number of operations needed, but this advantage is overwhelmed on practical problem sizes by how slowly quantum computers do each operation. Second, some promising quantum algorithms depend on practical Quantum Random Access Memory (QRAM), which is underdeveloped. Finally, there are quantum algorithms that offer large theoretical advantages, but which are only applicable to special cases, limiting their practical benefits. In each of these areas, we support our arguments using quantitative forecasts of quantum advantage that build on the work by Choi et al. [2023] as well as new research on limitations and quantum hardware trends. Our analysis outlines the current scope of quantum deep learning and points to research directions that could lead to greater practical advances in the field. 

**Abstract (ZH)**: 量子计算技术正在迅速发展。然而，即使考虑到这些趋势，量子计算机在未来一 two 十年对深度学习产生有意义的影响仍需巨大的突破。我们基于一项开创性的量子算法调查及其与潜在深度学习应用的匹配，得出了这一结论。这项调查揭示了量子计算可能加速深度学习的三个重要领域，每个领域都面临着实现其潜力的巨大挑战。首先，用于矩阵乘法和其他深度学习核心算法的量子算法在降低操作数量上提供了微小的理论改进，但在实际问题规模下，量子计算机执行每个操作的速度低使得这一优势被抹平。其次，一些有前途的量子算法依赖于尚未充分开发的量子随机存取存储器（QRAM）。最后，有些量子算法在理论上具有显著的优势，但仅适用于特殊案例，限制了它们的实际益处。在这些每个领域，我们通过基于 Choi 等人 [2023] 的工作以及对限制和量子硬件趋势的新研究进行的定量预测来支持我们的论点。我们的分析界定了当前量子深度学习的范围，并指出了可能导致该领域取得更大实际进展的研究方向。 

---
# Influence-aware Causal Autoencoder Network for Node Importance Ranking in Complex Networks 

**Title (ZH)**: 基于影响意识的因果自编码网络：复杂网络节点重要性排序 

**Authors**: Jiahui Gao, Kuang Zhou, Yuchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01228)  

**Abstract**: Node importance ranking is a fundamental problem in graph data analysis. Existing approaches typically rely on node features derived from either traditional centrality measures or advanced graph representation learning methods, which depend directly on the target network's topology. However, this reliance on structural information raises privacy concerns and often leads to poor generalization across different networks. In this work, we address a key question: Can we design a node importance ranking model trained exclusively on synthetic networks that is effectively appliable to real-world networks, eliminating the need to rely on the topology of target networks and improving both practicality and generalizability? We answer this question affirmatively by proposing the Influence-aware Causal Autoencoder Network (ICAN), a novel framework that leverages causal representation learning to get robust, invariant node embeddings for cross-network ranking tasks. Firstly, ICAN introduces an influence-aware causal representation learning module within an autoencoder architecture to extract node embeddings that are causally related to node importance. Moreover, we introduce a causal ranking loss and design a unified optimization framework that jointly optimizes the reconstruction and ranking objectives, enabling mutual reinforcement between node representation learning and ranking optimization. This design allows ICAN, trained on synthetic networks, to generalize effectively across diverse real-world graphs. Extensive experiments on multiple benchmark datasets demonstrate that ICAN consistently outperforms state-of-the-art baselines in terms of both ranking accuracy and generalization capability. 

**Abstract (ZH)**: 基于合成网络的因果自编码器网络在节点重要性排序中的应用 

---
# An Interdisciplinary and Cross-Task Review on Missing Data Imputation 

**Title (ZH)**: 跨学科与跨任务的缺失数据插补综述 

**Authors**: Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01196)  

**Abstract**: Missing data is a fundamental challenge in data science, significantly hindering analysis and decision-making across a wide range of disciplines, including healthcare, bioinformatics, social science, e-commerce, and industrial monitoring. Despite decades of research and numerous imputation methods, the literature remains fragmented across fields, creating a critical need for a comprehensive synthesis that connects statistical foundations with modern machine learning advances. This work systematically reviews core concepts-including missingness mechanisms, single versus multiple imputation, and different imputation goals-and examines problem characteristics across various domains. It provides a thorough categorization of imputation methods, spanning classical techniques (e.g., regression, the EM algorithm) to modern approaches like low-rank and high-rank matrix completion, deep learning models (autoencoders, GANs, diffusion models, graph neural networks), and large language models. Special attention is given to methods for complex data types, such as tensors, time series, streaming data, graph-structured data, categorical data, and multimodal data. Beyond methodology, we investigate the crucial integration of imputation with downstream tasks like classification, clustering, and anomaly detection, examining both sequential pipelines and joint optimization frameworks. The review also assesses theoretical guarantees, benchmarking resources, and evaluation metrics. Finally, we identify critical challenges and future directions, emphasizing model selection and hyperparameter optimization, the growing importance of privacy-preserving imputation via federated learning, and the pursuit of generalizable models that can adapt across domains and data types, thereby outlining a roadmap for future research. 

**Abstract (ZH)**: 缺失数据是数据科学中的一个基本挑战，显著阻碍了在卫生保健、生物信息学、社会科学、电子商务和工业监控等广泛的学科领域中进行分析和决策。尽管有数十年的研究和众多的插补方法，文献仍然在各个领域分散，因此迫切需要一个综合性的综述来将统计基础与现代机器学习进步联系起来。本文系统地回顾了核心概念，包括缺失机制、单次插补与多次插补以及不同的插补目标，并考察了不同领域的问题特征。它提供了插补方法的全面分类，涵盖了从经典技术（例如，回归、EM算法）到现代方法（如低秩和高秩矩阵完成、深度学习模型（自编码器、GAN、扩散模型、图神经网络）和大语言模型）的方法。特别注意了复杂数据类型的方法，例如张量、时间序列、流数据、图结构数据、分类数据和多模态数据。除了方法研究之外，我们还探讨了插补与下游任务（如分类、聚类和异常检测）的整合，包括顺序流水线和联合优化框架。综述还评估了理论保证、基准资源和评估指标。最后，我们识别了关键挑战和未来方向，强调模型选择和超参数优化，隐私保护插补在联邦学习中的日益重要性，以及寻求可以适用于不同领域和数据类型的可泛化模型，从而为未来研究指明了方向。 

---
# Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning 

**Title (ZH)**: 自我和谐：在测试时强化学习中学习协调自我监督与自我博弈 

**Authors**: Ru Wang, Wei Huang, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01191)  

**Abstract**: Test-time reinforcement learning (TTRL) offers a label-free paradigm for adapting models using only synthetic signals at inference, but its success hinges on constructing reliable learning signals. Standard approaches such as majority voting often collapse to spurious yet popular answers. We introduce Self-Harmony, a framework built on a simple intuition: the correct answer should remain stable across both an original question and its paraphrase. Self-Harmony operationalizes this by employing a single model in two complementary roles: a Solver to produce answers and a Reframer to rephrase the input. Based on this, we further propose a pseudo-label method: instead of majority voting, it aggregates answer frequencies across these original and reframed views using the harmonic mean. This is a process that naturally selects for solutions stable under reframing, thereby avoiding the common trap of favoring view-dependent, spurious answers. Crucially, this requires no human supervision or auxiliary models. Across diverse reasoning benchmarks, Self-Harmony achieves state-of-the-art results at the label-free test-time setting, ranking first in 28 of 30 settings across multiple methods. Beyond accuracy, it demonstrates unprecedented robustness, with zero training failures in all experiments, underscoring its stability and reliability. 

**Abstract (ZH)**: 自洽性框架：无标签测试时强化学习的新范式 

---
# Adapt under Attack and Domain Shift: Unified Adversarial Meta-Learning and Domain Adaptation for Robust Automatic Modulation Classification 

**Title (ZH)**: 在攻击下适应和领域转移：统一的对抗元学习与领域适应方法以提高自动调制分类的鲁棒性 

**Authors**: Ali Owfi, Amirmohammad Bamdad, Tolunay Seyfi, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2511.01172)  

**Abstract**: Deep learning has emerged as a leading approach for Automatic Modulation Classification (AMC), demonstrating superior performance over traditional methods. However, vulnerability to adversarial attacks and susceptibility to data distribution shifts hinder their practical deployment in real-world, dynamic environments. To address these threats, we propose a novel, unified framework that integrates meta-learning with domain adaptation, making AMC systems resistant to both adversarial attacks and environmental changes. Our framework utilizes a two-phase strategy. First, in an offline phase, we employ a meta-learning approach to train the model on clean and adversarially perturbed samples from a single source domain. This method enables the model to generalize its defense, making it resistant to a combination of previously unseen attacks. Subsequently, in the online phase, we apply domain adaptation to align the model's features with a new target domain, allowing it to adapt without requiring substantial labeled data. As a result, our framework achieves a significant improvement in modulation classification accuracy against these combined threats, offering a critical solution to the deployment and operational challenges of modern AMC systems. 

**Abstract (ZH)**: 深度学习已成为自动调制分类（AMC）的领先方法，展现出优于传统方法的性能。然而，对抗攻击的易受性和数据分布偏移的敏感性阻碍了其在现实动态环境中的实际部署。为解决这些威胁，我们提出了一种结合元学习与领域适应的新型统一框架，使AMC系统对对抗攻击和环境变化具有抵抗力。该框架采用两阶段策略。首先，在离线阶段，我们使用元学习方法在单一源域的干净样本和对抗扰动样本上训练模型，使模型能够泛化其防御，从而对新型未见攻击具有抵抗力。随后，在在线阶段，我们应用领域适应以使模型的特征与新目标域对齐，使其能够在无需大量标注数据的情况下进行适应。因此，该框架在面对这些综合威胁时显著提高了调制分类准确率，为现代AMC系统的部署和运营挑战提供了关键解决方案。 

---
# A High-Throughput Spiking Neural Network Processor Enabling Synaptic Delay Emulation 

**Title (ZH)**: 一种支持突触延时仿真的高吞吐量神经网络处理器 

**Authors**: Faquan Chen, Qingyang Tian, Ziren Wu, Rendong Ying, Fei Wen, Peilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01158)  

**Abstract**: Synaptic delay has attracted significant attention in neural network dynamics for integrating and processing complex spatiotemporal information. This paper introduces a high-throughput Spiking Neural Network (SNN) processor that supports synaptic delay-based emulation for edge applications. The processor leverages a multicore pipelined architecture with parallel compute engines, capable of real-time processing of the computational load associated with synaptic delays. We develop a SoC prototype of the proposed processor on PYNQ Z2 FPGA platform and evaluate its performance using the Spiking Heidelberg Digits (SHD) benchmark for low-power keyword spotting tasks. The processor achieves 93.4% accuracy in deployment and an average throughput of 104 samples/sec at a typical operating frequency of 125 MHz and 282 mW power consumption. 

**Abstract (ZH)**: Synaptic Delay 在神经网络动力学中用于整合和处理复杂的空间-时间信息引起了广泛关注。本文介绍了一种支持基于突触延时的模拟的高通量脉冲神经网络（SNN）处理器，适用于边缘应用。该处理器采用多核流水线架构，配备并行计算引擎，能够实时处理与突触延时相关的计算负载。我们在PYNQ Z2 FPGA平台上开发了所提出的处理器的SoC原型，并使用Spiking Heidelberg Digits（SHD）基准测试其在低功耗关键词识别任务中的性能。处理器在部署中的准确率为93.4%，在 typical 125 MHz 工作频率下的平均吞吐量为104样本/秒，功耗为282 mW。 

---
# SliceVision-F2I: A Synthetic Feature-to-Image Dataset for Visual Pattern Representation on Network Slices 

**Title (ZH)**: SliceVision-F2I：用于网络切片中视觉模式表示的合成特征到图像数据集 

**Authors**: Md. Abid Hasan Rafi, Mst. Fatematuj Johora, Pankaj Bhowmik  

**Link**: [PDF](https://arxiv.org/pdf/2511.01087)  

**Abstract**: The emergence of 5G and 6G networks has established network slicing as a significant part of future service-oriented architectures, demanding refined identification methods supported by robust datasets. The article presents SliceVision-F2I, a dataset of synthetic samples for studying feature visualization in network slicing for next-generation networking systems. The dataset transforms multivariate Key Performance Indicator (KPI) vectors into visual representations through four distinct encoding methods: physically inspired mappings, Perlin noise, neural wallpapering, and fractal branching. For each encoding method, 30,000 samples are generated, each comprising a raw KPI vector and a corresponding RGB image at low-resolution pixels. The dataset simulates realistic and noisy network conditions to reflect operational uncertainties and measurement imperfections. SliceVision-F2I is suitable for tasks involving visual learning, network state classification, anomaly detection, and benchmarking of image-based machine learning techniques applied to network data. The dataset is publicly available and can be reused in various research contexts, including multivariate time series analysis, synthetic data generation, and feature-to-image transformations. 

**Abstract (ZH)**: 5G和6G网络的出现已将网络切片确立为未来服务导向架构中的重要组成部分，需要由稳健的数据集支持的精细识别方法。本文介绍了SliceVision-F2I，一个用于研究网络切片中下一代网络系统特征可视化的人工合成样本数据集。该数据集通过四种不同的编码方法将多变量关键性能指标（KPI）向量转换为视觉表示：物理启发映射、Perlin噪声、神经壁纸和分形分支。每种编码方法生成30,000个样本，每个样本包含一个原始KPI向量和一个相应的低分辨率RGB图像。该数据集模拟了实际和嘈杂的网络条件，以反映操作不确定性及测量不完善性。SliceVision-F2I适用于涉及视觉学习、网络状态分类、异常检测及基于图像的机器学习技术在网络数据中应用的基准测试的任务。该数据集公开可用，并可在包括多变量时间序列分析、合成数据生成和特征到图像转换在内的多种研究背景下重用。 

---
# HAFixAgent: History-Aware Automated Program Repair Agent 

**Title (ZH)**: HAFixAgent: 历史意识的自动化程序修复代理 

**Authors**: Yu Shi, Hao Li, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01047)  

**Abstract**: Automated program repair (APR) has recently shifted toward large language models and agent-based systems, yet most systems rely on local snapshot context, overlooking repository history. Prior work shows that repository history helps repair single-line bugs, since the last commit touching the buggy line is often the bug-introducing one. In this paper, we investigate whether repository history can also improve agentic APR systems at scale, especially for complex multi-hunk bugs. We present HAFixAgent, a History-Aware Bug-Fixing Agent that injects blame-derived repository heuristics into its repair loop. A preliminary study of all 854 real-world bugs from Defects4J motivates our design, showing that bug-relevant history is both widely available and highly concentrated. Empirical comparison of HAFixAgent with two state-of-the-art baselines shows: (1) Effectiveness: HAFixAgent significantly improves over the agent-based baseline (by 212.3%) and the multi-hunk baseline (by 29.9%). (2) Efficiency: history does not significantly increase agent steps and keeps token costs comparable, with notably lower median costs for complex multi-file-multi-hunk bugs. (3) Practicality: combining different historical heuristics repairs more bugs, offering a clear cost-benefit trade-off. HAFixAgent offers a practical recipe for history-aware agentic APR: ground the agent in version control history, prioritize diff-based historical context, and integrate complementary heuristics when needed. 

**Abstract (ZH)**: 面向历史的代理自动化程序修复（Repository History-Aware Automated Program Repair with HAFixAgent） 

---
# Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer 

**Title (ZH)**: 种子诱导的独特性在变压器模型中的表现：子空间对齐支配潜移学习 

**Authors**: Ayşe Selin Okatan, Mustafa İlhan Akbaş, Laxima Niure Kandel, Berker Peköz  

**Link**: [PDF](https://arxiv.org/pdf/2511.01023)  

**Abstract**: We analyze subliminal transfer in Transformer models, where a teacher embeds hidden traits that can be linearly decoded by a student without degrading main-task performance. Prior work often attributes transferability to global representational similarity, typically quantified with Centered Kernel Alignment (CKA). Using synthetic corpora with disentangled public and private labels, we distill students under matched and independent random initializations. We find that transfer strength hinges on alignment within a trait-discriminative subspace: same-seed students inherit this alignment and show higher leakage {\tau \approx} 0.24, whereas different-seed students--despite global CKA > 0.9--exhibit substantially reduced excess accuracy {\tau \approx} 0.12 - 0.13. We formalize this with subspace-level CKA diagnostic and residualized probes, showing that leakage tracks alignment within the trait-discriminative subspace rather than global representational similarity. Security controls (projection penalty, adversarial reversal, right-for-the-wrong-reasons regularization) reduce leakage in same-base models without impairing public-task fidelity. These results establish seed-induced uniqueness as a resilience property and argue for subspace-aware diagnostics for secure multi-model deployments. 

**Abstract (ZH)**: 子空间内隐转移在Transformer模型中的分析：基于特征辨别子空间的转移强度依赖于定向匹配 

---
# Keys in the Weights: Transformer Authentication Using Model-Bound Latent Representations 

**Title (ZH)**: 权重中的密钥：基于模型约束潜在表示的变压器身份验证 

**Authors**: Ayşe S. Okatan, Mustafa İlhan Akbaş, Laxima Niure Kandel, Berker Peköz  

**Link**: [PDF](https://arxiv.org/pdf/2511.00973)  

**Abstract**: We introduce Model-Bound Latent Exchange (MoBLE), a decoder-binding property in Transformer autoencoders formalized as Zero-Shot Decoder Non-Transferability (ZSDN). In identity tasks using iso-architectural models trained on identical data but differing in seeds, self-decoding achieves more than 0.91 exact match and 0.98 token accuracy, while zero-shot cross-decoding collapses to chance without exact matches. This separation arises without injected secrets or adversarial training, and is corroborated by weight-space distances and attention-divergence diagnostics. We interpret ZSDN as model binding, a latent-based authentication and access-control mechanism, even when the architecture and training recipe are public: encoder's hidden state representation deterministically reveals the plaintext, yet only the correctly keyed decoder reproduces it in zero-shot. We formally define ZSDN, a decoder-binding advantage metric, and outline deployment considerations for secure artificial intelligence (AI) pipelines. Finally, we discuss learnability risks (e.g., adapter alignment) and outline mitigations. MoBLE offers a lightweight, accelerator-friendly approach to secure AI deployment in safety-critical domains, including aviation and cyber-physical systems. 

**Abstract (ZH)**: MoBLE：一种形式化的零样本解码器不可转移性机制 

---
# Using Synthetic Data to estimate the True Error is theoretically and practically doable 

**Title (ZH)**: 使用合成数据估计真正错误在理论上和实践中是可行的 

**Authors**: Hai Hoang Thanh, Duy-Tung Nguyen, Hung The Tran, Khoat Than  

**Link**: [PDF](https://arxiv.org/pdf/2511.00964)  

**Abstract**: Accurately evaluating model performance is crucial for deploying machine learning systems in real-world applications. Traditional methods often require a sufficiently large labeled test set to ensure a reliable evaluation. However, in many contexts, a large labeled dataset is costly and labor-intensive. Therefore, we sometimes have to do evaluation by a few labeled samples, which is theoretically challenging. Recent advances in generative models offer a promising alternative by enabling the synthesis of high-quality data. In this work, we make a systematic investigation about the use of synthetic data to estimate the test error of a trained model under limited labeled data conditions. To this end, we develop novel generalization bounds that take synthetic data into account. Those bounds suggest novel ways to optimize synthetic samples for evaluation and theoretically reveal the significant role of the generator's quality. Inspired by those bounds, we propose a theoretically grounded method to generate optimized synthetic data for model evaluation. Experimental results on simulation and tabular datasets demonstrate that, compared to existing baselines, our method achieves accurate and more reliable estimates of the test error. 

**Abstract (ZH)**: 准确评估模型性能对于在实际应用中部署机器学习系统至关重要。传统的方法通常需要一个足够大的标记测试集以确保评估的可靠性。然而，在许多情况下，收集大量标记数据既昂贵又劳动密集。因此，我们有时必须使用少量标记样本进行评估，这是理论上具有挑战性的。最近生成模型的进步为此提供了一种有希望的替代方案，通过生成高质量的数据。在本文中，我们对在标记数据有限的情况下使用合成数据估算训练模型的测试误差进行了系统的调查。为此，我们开发了新的泛化边界，考虑了合成数据。这些边界提出了优化合成样本以进行评估的新方法，并从理论上揭示了生成器质量的重要作用。受到这些边界的启发，我们提出了一种基于理论的方法来生成用于模型评估的优化合成数据。实验结果表明，与现有基线相比，我们的方法能够更准确和可靠地估算测试误差。 

---
# The Hidden Power of Normalization: Exponential Capacity Control in Deep Neural Networks 

**Title (ZH)**: 归一化的隐含力量：深层神经网络中的指数容量控制 

**Authors**: Khoat Than  

**Link**: [PDF](https://arxiv.org/pdf/2511.00958)  

**Abstract**: Normalization methods are fundamental components of modern deep neural networks (DNNs). Empirically, they are known to stabilize optimization dynamics and improve generalization. However, the underlying theoretical mechanism by which normalization contributes to both optimization and generalization remains largely unexplained, especially when using many normalization layers in a DNN architecture.
In this work, we develop a theoretical framework that elucidates the role of normalization through the lens of capacity control. We prove that an unnormalized DNN can exhibit exponentially large Lipschitz constants with respect to either its parameters or inputs, implying excessive functional capacity and potential overfitting. Such bad DNNs are uncountably many. In contrast, the insertion of normalization layers provably can reduce the Lipschitz constant at an exponential rate in the number of normalization operations. This exponential reduction yields two fundamental consequences: (1) it smooths the loss landscape at an exponential rate, facilitating faster and more stable optimization; and (2) it constrains the effective capacity of the network, thereby enhancing generalization guarantees on unseen data. Our results thus offer a principled explanation for the empirical success of normalization methods in deep learning. 

**Abstract (ZH)**: Normalization方法是现代深度神经网络（DNNs）的基本组件。实证研究表明，它们可以稳定优化动态并提高泛化能力。然而，归一化如何通过控制容量的方式同时促进优化和泛化的确切理论机制仍然解释不足，尤其是在DNN架构中使用大量归一化层的情况下。
在本文中，我们开发了一种理论框架，通过容量控制的视角阐明了归一化的作用。我们证明了未归一化的DNN可能在参数或输入方面表现出指数级大的Lipschitz常数，暗示了过大的功能容量和潜在的过拟合。这样的不良DNN是不可数的。相比之下，插入归一化层可以证明以指数级速率减少Lipschitz常数。这种指数级减少产生了两个基本后果：（1）它以指数级速率平滑损失景观，促进更快更稳定的优化；（2）它限制了网络的有效容量，从而增强未见数据上的泛化保证。因此，我们的结果为归一化方法在深度学习中的 empirical 成功提供了基本原则性的解释。 

---
# Dynamic Logic of Trust-Based Beliefs 

**Title (ZH)**: 基于信任的信念动态逻辑 

**Authors**: Junli Jiang, Pavel Naumov, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00899)  

**Abstract**: Traditionally, an agent's beliefs would come from what the agent can see, hear, or sense. In the modern world, beliefs are often based on the data available to the agents. In this work, we investigate a dynamic logic of such beliefs that incorporates public announcements of data. The main technical contribution is a sound and complete axiomatisation of the interplay between data-informed beliefs and data announcement modalities. We also describe a non-trivial polynomial model checking algorithm for this logical system. 

**Abstract (ZH)**: 传统上，一个代理的信念来自于它所能看见、听到或感觉到的内容。在现代社会，信念通常是基于代理所拥有的数据。本工作中，我们探讨了一种包含数据公告的动态逻辑，以研究数据驱动信念与数据公告模态之间的互动。我们的主要技术贡献是对这种逻辑系统的公理化进行了声学和完备性的证明。我们还描述了一个非平凡的多项式模型检验算法用于此逻辑系统。 

---
# Android Malware Detection: A Machine Leaning Approach 

**Title (ZH)**: Android恶意软件检测：一种机器学习方法 

**Authors**: Hasan Abdulla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00894)  

**Abstract**: This study examines machine learning techniques like Decision Trees, Support Vector Machines, Logistic Regression, Neural Networks, and ensemble methods to detect Android malware. The study evaluates these models on a dataset of Android applications and analyzes their accuracy, efficiency, and real-world applicability. Key findings show that ensemble methods demonstrate superior performance, but there are trade-offs between model interpretability, efficiency, and accuracy. Given its increasing threat, the insights guide future research and practical use of ML to combat Android malware. 

**Abstract (ZH)**: 本研究探讨了决策树、支持向量机、逻辑回归、神经网络以及 ensemble 方法等机器学习技术在检测 Android 恶意软件中的应用，并在 Android 应用程序数据集上评估了这些模型的准确性、效率及其实际适用性。研究发现，ensemble 方法表现 Superior，但存在模型可解释性、效率和准确性之间的权衡。鉴于 Android 恶意软件威胁的不断增加，研究结果为未来通过机器学习对抗 Android 恶意软件的研究和实践提供了指导。 

---
# Fast Stochastic Greedy Algorithm for $k$-Submodular Cover Problem 

**Title (ZH)**: 快速随机贪婪算法求解$k$-次模覆盖问题 

**Authors**: Hue T. Nguyen, Tan D. Tran, Nguyen Long Giang, Canh V. Pham  

**Link**: [PDF](https://arxiv.org/pdf/2511.00869)  

**Abstract**: We study the $k$-Submodular Cover ($kSC$) problem, a natural generalization of the classical Submodular Cover problem that arises in artificial intelligence and combinatorial optimization tasks such as influence maximization, resource allocation, and sensor placement. Existing algorithms for $\kSC$ often provide weak approximation guarantees or incur prohibitively high query complexity. To overcome these limitations, we propose a \textit{Fast Stochastic Greedy} algorithm that achieves strong bicriteria approximation while substantially lowering query complexity compared to state-of-the-art methods. Our approach dramatically reduces the number of function evaluations, making it highly scalable and practical for large-scale real-world AI applications where efficiency is essential. 

**Abstract (ZH)**: 我们研究$k$-子模覆盖($kSC$)问题，这是经典子模覆盖问题在人工智能和组合优化任务如影响力最大化、资源分配和传感器放置中的自然推广。现有的$kSC$算法往往提供较差的近似保证或引发难以接受的高查询复杂度。为克服这些限制，我们提出了一种快速随机贪婪算法，该算法实现了强双准则近似，并显着降低了查询复杂度，相比现有最佳方法更具优势。我们的方法大大减少了函数评估的次数，使其对于需要高效性的大规模实际AI应用具有高度的可扩展性和实用性。 

---
# MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models 

**Title (ZH)**: MULTI-Bench：评估口语对话模型情感智能能力的多轮交互基准 

**Authors**: Yayue Deng, Guoqiang Hu, Haiyang Sun, Xiangyu Zhang, Haoyang Zhang, Fei Tian, Xuerui Yang, Gang Yu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2511.00850)  

**Abstract**: Spoken Dialogue Models (SDMs) have advanced rapidly, yet their ability to sustain genuinely interactive multi-turn conversations remains underexplored, as most benchmarks focus on single-turn exchanges. We introduce Multi-Bench, the first benchmark explicitly designed to evaluate SDMs in multi-turn interactive dialogue with an emphasis on emotional intelligence. Multi-Bench employs a hierarchical structure with a basic track for emotion understanding and reasoning and an advanced track for emotion support and application. It comprises five carefully designed tasks and about 3.2K samples, ranging from emotion recognition to complex reasoning and interactive dialogue, supported by a reproducible evaluation framework. We evaluate six representative SDMs on eight subsets of Multi-Bench. Results show that while current SDMs achieve good performance on basic understanding tasks, they still have room for improvement in advanced multi-turn interactive dialogue and reasoning-related tasks, particularly in emotion awareness and application. 

**Abstract (ZH)**: spoken dialogue models (sdms)在多轮互动对话中的情感智能评估：multi-bench 

---
# OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks 

**Title (ZH)**: OmniBrainBench: 跨多阶段临床任务的综合多模态脑成像分析基准 

**Authors**: Zhihao Peng, Cheng Wang, Shengyuan Liu, Zhiying Liang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00846)  

**Abstract**: Brain imaging analysis is vital for diagnosing and treating brain disorders, and multimodal large language models (MLLMs) are increasingly assisting in that analysis. However, current brain-oriented visual question-answering (VQA) benchmarks either cover a few imaging modalities or are limited to coarse-grained pathological descriptions, hindering a comprehensive assessment of MLLMs throughout the full clinical continuum. To address these, we introduce OmniBrainBench, the first comprehensive multimodal VQA benchmark specifically designed to assess the multimodal comprehension capabilities of MLLMs in brain imaging this http URL consists of 15 distinct brain imaging modalities collected from 30 verified medical sources, yielding 9,527 validated VQA pairs and 31,706 images. It simulates clinical workflows and encompasses 15 multi-stage clinical tasks rigorously validated by a professional radiologist. Evaluation of 24 state-of-the-art models, including open-source, medical, and proprietary MLLMs, highlights the substantial challenges posed by OmniBrainBench. Our experiments reveal: (1) proprietary MLLMs (e.g., GPT-5) beat open-source and medical models but lag physicians; (2) medical MLLMs vary widely in performance; (3) open-source MLLMs trail overall but excel in specific tasks; (4) MLLMs underperform sharply in complex preoperative tasks, revealing a visual-to-clinical reasoning gap. OmniBrainBench sets a new standard for evaluating and advancing MLLMs in brain imaging analysis, highlighting gaps compared to expert clinical reasoning. We release it at benchmark \& code. 

**Abstract (ZH)**: 全面的脑成像多模态视觉问答基准 OmniBrainBench：评估和推进脑成像分析中多模态大语言模型的标准 

---
# CodeClash: Benchmarking Goal-Oriented Software Engineering 

**Title (ZH)**: CodeClash: 目标导向的软件工程benchmarking 

**Authors**: John Yang, Kilian Lieret, Joyce Yang, Carlos E. Jimenez, Ofir Press, Ludwig Schmidt, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00839)  

**Abstract**: Current benchmarks for coding evaluate language models (LMs) on concrete, well-specified tasks such as fixing specific bugs or writing targeted tests. However, human programmers do not spend all day incessantly addressing isolated tasks. Instead, real-world software development is grounded in the pursuit of high-level goals, like improving user retention or reducing costs. Evaluating whether LMs can also iteratively develop code to better accomplish open-ended objectives without any explicit guidance remains an open challenge. To address this, we introduce CodeClash, a benchmark where LMs compete in multi-round tournaments to build the best codebase for achieving a competitive objective. Each round proceeds in two phases: agents edit their code, then their codebases compete head-to-head in a code arena that determines winners based on objectives like score maximization, resource acquisition, or survival. Whether it's writing notes, scrutinizing documentation, analyzing competition logs, or creating test suites, models must decide for themselves how to improve their codebases both absolutely and against their opponents. We run 1680 tournaments (25,200 rounds total) to evaluate 8 LMs across 6 arenas. Our results reveal that while models exhibit diverse development styles, they share fundamental limitations in strategic reasoning. Models also struggle with long-term codebase maintenance, as repositories become progressively messy and redundant. These limitations are stark: top models lose every round against expert human programmers. We open-source CodeClash to advance the study of autonomous, goal-oriented code development. 

**Abstract (ZH)**: CodeClash：面向实现开放目标的模型竞争基准 

---
# Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack 

**Title (ZH)**: 通过局部乱序和基于样本的攻击增强视觉-语言预训练模型的对抗迁移性 

**Authors**: Xin Liu, Aoyang Zhou, Aoyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00831)  

**Abstract**: Visual-Language Pre-training (VLP) models have achieved significant performance across various downstream tasks. However, they remain vulnerable to adversarial examples. While prior efforts focus on improving the adversarial transferability of multimodal adversarial examples through cross-modal interactions, these approaches suffer from overfitting issues, due to a lack of input diversity by relying excessively on information from adversarial examples in one modality when crafting attacks in another. To address this issue, we draw inspiration from strategies in some adversarial training methods and propose a novel attack called Local Shuffle and Sample-based Attack (LSSA). LSSA randomly shuffles one of the local image blocks, thus expanding the original image-text pairs, generating adversarial images, and sampling around them. Then, it utilizes both the original and sampled images to generate the adversarial texts. Extensive experiments on multiple models and datasets demonstrate that LSSA significantly enhances the transferability of multimodal adversarial examples across diverse VLP models and downstream tasks. Moreover, LSSA outperforms other advanced attacks on Large Vision-Language Models. 

**Abstract (ZH)**: 视觉语言预训练模型（VLP）已经在多种下游任务中取得了显著性能，但仍然容易受到对抗样本的影响。为了应对这一问题，我们借鉴了一些对抗训练方法中的策略，提出了一种新的攻击方法——局部打乱和采样攻击（LSSA）。实验结果表明，LSSA显著增强了跨多种VLP模型和下游任务的多模态对抗样本的转移性，并且在大型视觉语言模型上优于其他先进的攻击方法。 

---
# Towards Ultra-Low Latency: Binarized Neural Network Architectures for In-Vehicle Network Intrusion Detection 

**Title (ZH)**: 面向极低延迟：车载网络入侵检测的二值神经网络架构 

**Authors**: Huiyao Dong, Igor Kotenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.00828)  

**Abstract**: The Control Area Network (CAN) protocol is essential for in-vehicle communication, facilitating high-speed data exchange among Electronic Control Units (ECUs). However, its inherent design lacks robust security features, rendering vehicles susceptible to cyberattacks. While recent research has investigated machine learning and deep learning techniques to enhance network security, their practical applicability remains uncertain. This paper presents a lightweight intrusion detection technique based on Binarized Neural Networks (BNNs), which utilizes payload data, message IDs, and CAN message frequencies for effective intrusion detection. Additionally, we develop hybrid binary encoding techniques to integrate non-binary features, such as message IDs and frequencies. The proposed method, namely the BNN framework specifically optimized for in-vehicle intrusion detection combined with hybrid binary quantization techniques for non-payload attributes, demonstrates efficacy in both anomaly detection and multi-class network traffic classification. The system is well-suited for deployment on micro-controllers and Gateway ECUs, aligning with the real-time requirements of CAN bus safety applications. 

**Abstract (ZH)**: 基于二值神经网络的轻量级车载入侵检测技术研究 

---
# Attention Saturation and Gradient Suppression at Inflection Layers: Diagnosing and Mitigating Bottlenecks in Transformer Adaptation 

**Title (ZH)**: 注意力饱和与渐变抑制在拐点层：诊断和缓解Transformer适配瓶颈 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2511.00797)  

**Abstract**: Pre-trained Transformers often exhibit over-confidence in source patterns and difficulty in forming new target-domain patterns during fine-tuning. We formalize the mechanism of output saturation leading to gradient suppression through standard cross-entropy and softmax analysis, showing that gradient suppression at inflection layers confines adaptation to high-level recombination of existing features while preventing low-level reconstruction. We introduce a set of layer-wise diagnostic metrics -- attention entropy (saturation proxy), activation gradient norm, parameter gradient norm, and Delta-CKA under a shared PCA basis -- to identify inflection layers characterized by both low attention entropy and steep gradient decay. Building on these findings, we propose a diagnose-first, inject-light fine-tuning strategy: selectively inserting LoRA adapters at inflection layers to restore suppressed backward signals with minimal parameter overhead. Experiments on BERT-base transfer from SST-2 to Rotten Tomatoes under under-trained and over-trained source regimes reveal that over-trained initialization benefits from inflection-layer LoRA injection, while under-trained initialization suffers performance degradation. When base features are strong, unblocking inflection layers facilitates high-level compositional adaptation; when base features are weak, full-pathway unblocking is required for low-level reconstruction, as supported by joint analysis of layer-wise activation gradients and Delta-CKA dynamics. 

**Abstract (ZH)**: 预训练变换器經常在微调過程中表現出對源模式的過度自信及形成新的目標域模式 difficulties，我們通過標準交叉熵和Softmax分析形式化了輸出飽和導致梯度壓制的機制，顯示出在拐點層梯度壓制限制了高水平現有特徵的重組，同時阻止了低水平重建。我們引入了一組分層診斷指標——注意力熵（飽和代理）、激活梯度范数、参数梯度范数以及共享PCA基底下的Delta-CKA，以識別同時表現低注意力熵和陡峭梯度衰減的拐點層。基於這些發現，我們提出了先診斷後，輕量級微调策略：在拐點层选择性插入LoRA适配器，以最小的参数开销恢复被抑制的反向信号。在从SST-2到烂番茄数据集的BERT-base迁移任务中，我們发现过训练初始化受益于拐点层LoRA注入，而欠训练初始化则遭受性能下降。当基础特征较强时，解锁拐点层促进高层次组合适应；当基础特征较弱时，需要全程路径解锁以实现低层次重建，这得到了分层激活梯度和Delta-CKA动态的联合分析的支持。 

---
# FedOnco-Bench: A Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation with Synthetic CT Data 

**Title (ZH)**: FedOnco-Bench：一种基于合成CT数据的隐私意识联邦肿瘤分割可再现基准 

**Authors**: Viswa Chaitanya Marella, Suhasnadh Reddy Veluru, Sai Teja Erukude  

**Link**: [PDF](https://arxiv.org/pdf/2511.00795)  

**Abstract**: Federated Learning (FL) allows multiple institutions to cooperatively train machine learning models while retaining sensitive data at the source, which has great utility in privacy-sensitive environments. However, FL systems remain vulnerable to membership-inference attacks and data heterogeneity. This paper presents FedOnco-Bench, a reproducible benchmark for privacy-aware FL using synthetic oncologic CT scans with tumor annotations. It evaluates segmentation performance and privacy leakage across FL methods: FedAvg, FedProx, FedBN, and FedAvg with DP-SGD. Results show a distinct trade-off between privacy and utility: FedAvg is high performance (Dice around 0.85) with more privacy leakage (attack AUC about 0.72), while DP-SGD provides a higher level of privacy (AUC around 0.25) at the cost of accuracy (Dice about 0.79). FedProx and FedBN offer balanced performance under heterogeneous data, especially with non-identical distributed client data. FedOnco-Bench serves as a standardized, open-source platform for benchmarking and developing privacy-preserving FL methods for medical image segmentation. 

**Abstract (ZH)**: 联邦学习(FedOnco-Bench): 一种基于合成肿瘤CT扫描的隐私意识联邦学习可重现基准 

---
# Fast PINN Eigensolvers via Biconvex Reformulation 

**Title (ZH)**: 快速PINN特征值求解器通过双凸重写 

**Authors**: Akshay Sai Banderwaar, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.00792)  

**Abstract**: Eigenvalue problems have a distinctive forward-inverse structure and are fundamental to characterizing a system's thermal response, stability, and natural modes. Physics-Informed Neural Networks (PINNs) offer a mesh-free alternative for solving such problems but are often orders of magnitude slower than classical numerical schemes. In this paper, we introduce a reformulated PINN approach that casts the search for eigenpairs as a biconvex optimization problem, enabling fast and provably convergent alternating convex search (ACS) over eigenvalues and eigenfunctions using analytically optimal updates. Numerical experiments show that PINN-ACS attains high accuracy with convergence speeds up to 500$\times$ faster than gradient-based PINN training. We release our codes at this https URL. 

**Abstract (ZH)**: Eigen值得问题具有独特的前向-逆向结构，是表征系统热响应、稳定性和自然模式的基础。基于物理的信息神经网络（PINNs）提供了一种无网格的求解方法，但通常比经典数值方案慢几个数量级。本文提出了一种重新公式化的PINN方法，将对偶凸优化问题用于搜索特征对，实现了特征值和特征函数的交替凸搜索，并使用分析最优更新保证了快速且可证明收敛。数值实验表明，PINN-ACS 达到了高精度，其收敛速度比基于梯度的PINN训练快500倍。我们已将代码发布在 <https://>。 

---
# Quantifying truth and authenticity in AI-assisted candidate evaluation: A multi-domain pilot analysis 

**Title (ZH)**: 量化AI辅助候选人评估中的真实性和 authenticity: 多领域试点分析 

**Authors**: Eldred Lee, Nicholas Worley, Koshu Takatsuji  

**Link**: [PDF](https://arxiv.org/pdf/2511.00774)  

**Abstract**: This paper presents a retrospective analysis of anonymized candidate-evaluation data collected during pilot hiring campaigns conducted through AlteraSF, an AI-native resume-verification platform. The system evaluates resume claims, generates context-sensitive verification questions, and measures performance along quantitative axes of factual validity and job fit, complemented by qualitative integrity detection. Across six job families and 1,700 applications, the platform achieved a 90-95% reduction in screening time and detected measurable linguistic patterns consistent with AI-assisted or copied responses. The analysis demonstrates that candidate truthfulness can be assessed not only through factual accuracy but also through patterns of linguistic authenticity. The results suggest that a multi-dimensional verification framework can improve both hiring efficiency and trust in AI-mediated evaluation systems. 

**Abstract (ZH)**: 本文通过对通过AlteraSF进行的试点招聘活动中收集的匿名候选人评估数据进行回顾性分析，展示了AI原生简历验证平台在评估简历主张、生成上下文相关验证问题以及衡量绩效方面的能力，同时还包括定量的事实准确性和岗位匹配度评估，以及定性的诚信检测。在六类职位和1,700份申请中，该平台实现了筛查时间90-95%的减少，并检测到了与AI辅助或抄袭响应一致的可衡量的语言模式。分析表明，候选人诚信不仅可以通过事实准确性还可以通过语言真实性的模式来评估。结果表明，多维度验证框架可以提高招聘效率并增强对AI驱动评估系统的信任。 

---
# EP-HDC: Hyperdimensional Computing with Encrypted Parameters for High-Throughput Privacy-Preserving Inference 

**Title (ZH)**: EP-HDC：加密参数下的高维度计算以实现高吞吐量的隐私保护推理 

**Authors**: Jaewoo Park, Chenghao Quan, Jongeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.00737)  

**Abstract**: While homomorphic encryption (HE) provides strong privacy protection, its high computational cost has restricted its application to simple tasks. Recently, hyperdimensional computing (HDC) applied to HE has shown promising performance for privacy-preserving machine learning (PPML). However, when applied to more realistic scenarios such as batch inference, the HDC-based HE has still very high compute time as well as high encryption and data transmission overheads. To address this problem, we propose HDC with encrypted parameters (EP-HDC), which is a novel PPML approach featuring client-side HE, i.e., inference is performed on a client using a homomorphically encrypted model. Our EP-HDC can effectively mitigate the encryption and data transmission overhead, as well as providing high scalability with many clients while providing strong protection for user data and model parameters. In addition to application examples for our client-side PPML, we also present design space exploration involving quantization, architecture, and HE-related parameters. Our experimental results using the BFV scheme and the Face/Emotion datasets demonstrate that our method can improve throughput and latency of batch inference by orders of magnitude over previous PPML methods (36.52~1068x and 6.45~733x, respectively) with less than 1% accuracy degradation. 

**Abstract (ZH)**: 基于加密参数的高维计算 homomorphic 加密（EP-HDC）：面向批量推理的隐私保护机器学习新方法 

---
# FeNN-DMA: A RISC-V SoC for SNN acceleration 

**Title (ZH)**: FeNN-DMA：一种用于SNN加速的RISC-V系统级芯片 

**Authors**: Zainab Aizaz, James C. Knight, Thomas Nowotny  

**Link**: [PDF](https://arxiv.org/pdf/2511.00732)  

**Abstract**: Spiking Neural Networks (SNNs) are a promising, energy-efficient alternative to standard Artificial Neural Networks (ANNs) and are particularly well-suited to spatio-temporal tasks such as keyword spotting and video classification. However, SNNs have a much lower arithmetic intensity than ANNs and are therefore not well-matched to standard accelerators like GPUs and TPUs. Field Programmable Gate Arrays(FPGAs) are designed for such memory-bound workloads and here we develop a novel, fully-programmable RISC-V-based system-on-chip (FeNN-DMA), tailored to simulating SNNs on modern UltraScale+ FPGAs. We show that FeNN-DMA has comparable resource usage and energy requirements to state-of-the-art fixed-function SNN accelerators, yet it is capable of simulating much larger and more complex models. Using this functionality, we demonstrate state-of-the-art classification accuracy on the Spiking Heidelberg Digits and Neuromorphic MNIST tasks. 

**Abstract (ZH)**: 基于可编程RISC-V的FPGA系统（FeNN-DMA）：用于现代UltraScale+ FPGA上模拟脉冲神经网络 

---
# TRISKELION-1: Unified Descriptive-Predictive-Generative AI 

**Title (ZH)**: TRISKELION-1：统一的描述-预测-生成AI 

**Authors**: Nardeep Kumar, Arun Kanwar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00711)  

**Abstract**: TRISKELION-1 is a unified descriptive-predictive-generative architecture that integrates statistical, mechanistic, and generative reasoning within a single encoder-decoder framework. The model demonstrates how descriptive representation learning, predictive inference, and generative synthesis can be jointly optimized using variational objectives. Experiments on MNIST validate that descriptive reconstruction, predictive classification, and generative sampling can coexist stably within one model. The framework provides a blueprint toward universal intelligence architectures that connect interpretability, accuracy, and creativity. 

**Abstract (ZH)**: TRISKELION-1是一种统一的描述-预测-生成架构，结合了统计、机理和生成推理于一体，并在单个编码器-解码器框架中进行了集成。该模型展示了如何使用变分目标 jointly 优化描述性表示学习、预测性推理和生成性合成。MNIST上的实验证明，描述性重构、预测性分类和生成性采样可以在一个模型中稳定共存。该框架为连接可解释性、准确性和创造性的一般智能架构提供了蓝图。 

---
# Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Control 

**Title (ZH)**: 基于元数据对齐的3D MRI表示方法及其在对比度理解和质量控制中的应用 

**Authors**: Mehmet Yigit Avci, Pedro Borges, Virginia Fernandez, Paul Wright, Mehmet Yigitsoy, Sebastien Ourselin, Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2511.00681)  

**Abstract**: Magnetic Resonance Imaging suffers from substantial data heterogeneity and the absence of standardized contrast labels across scanners, protocols, and institutions, which severely limits large-scale automated analysis. A unified representation of MRI contrast would enable a wide range of downstream utilities, from automatic sequence recognition to harmonization and quality control, without relying on manual annotations. To this end, we introduce MR-CLIP, a metadata-guided framework that learns MRI contrast representations by aligning volumetric images with their DICOM acquisition parameters. The resulting embeddings shows distinct clusters of MRI sequences and outperform supervised 3D baselines under data scarcity in few-shot sequence classification. Moreover, MR-CLIP enables unsupervised data quality control by identifying corrupted or inconsistent metadata through image-metadata embedding distances. By transforming routinely available acquisition metadata into a supervisory signal, MR-CLIP provides a scalable foundation for label-efficient MRI analysis across diverse clinical datasets. 

**Abstract (ZH)**: 磁共振成像受数据异质性和标准化对比标签缺乏的严重影响，这严重限制了大型自动化分析的应用。一种统一的磁共振成像对比表示将能够广泛支持从自动序列识别到规范化和质量控制的各种下游应用，无需依赖手动注释。为此，我们提出了MR-CLIP，这是一种由元数据指导的框架，通过将体容图像与其DICOM采集参数对齐来学习MRI对比表示。生成的嵌入显示出MRI序列的明显聚类，在数据稀缺的小样本序列分类中优于监督的3D基线。此外，MR-CLIP还通过图像-元数据嵌入距离识别损坏或不一致的元数据，以实现无监督的数据质量控制。通过将常规可用的采集元数据转换为监督信号，MR-CLIP为各种临床数据集的高效标签MRI分析提供了可扩展的基础。 

---
# Isotropic Curvature Model for Understanding Deep Learning Optimization: Is Gradient Orthogonalization Optimal? 

**Title (ZH)**: 用于理解深度学习优化的各向同性曲率模型：梯度正交化是最优的吗？ 

**Authors**: Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00674)  

**Abstract**: In this paper, we introduce a model for analyzing deep learning optimization over a single iteration by leveraging the matrix structure of the weights. We derive the model by assuming isotropy of curvature, including the second-order Hessian and higher-order terms, of the loss function across all perturbation directions; hence, we call it the isotropic curvature model. This model is a convex optimization program amenable to analysis, which allows us to understand how an update on the weights in the form of a matrix relates to the change in the total loss function. As an application, we use the isotropic curvature model to analyze the recently introduced Muon optimizer and other matrix-gradient methods for training language models. First, we show that under a general growth condition on the curvature, the optimal update matrix is obtained by making the spectrum of the original gradient matrix more homogeneous -- that is, making its singular values closer in ratio -- which in particular improves the conditioning of the update matrix. Next, we show that the orthogonalized gradient becomes optimal for the isotropic curvature model when the curvature exhibits a phase transition in growth. Taken together, these results suggest that the gradient orthogonalization employed in Muon and other related methods is directionally correct but may not be strictly optimal. Finally, we discuss future research on how to leverage the isotropic curvature model for designing new optimization methods for training deep learning and language models. 

**Abstract (ZH)**: 本文通过利用权重的矩阵结构引入了一种分析单迭代深度学习优化的方法，并假设损失函数在所有扰动方向上曲率的各向同性，包括二阶海森矩阵和高阶项，因此称之为各向同性曲率模型。该模型是一个便于分析的凸优化程序，使我们能够理解矩阵形式的权重更新如何与总损失函数的变化相关。作为一种应用，我们使用各向同性曲率模型来分析最近提出的Muon优化器及其他矩阵梯度方法用于训练语言模型的情况。首先，我们证明在曲率一般增长条件下，最优更新矩阵通过使原始梯度矩阵的谱更均匀——即使其奇异值的比例更接近——从而在某种程度上改善了更新矩阵的条件。其次，当曲率表现出增长相变时，正交化梯度成为各向同性曲率模型下的最优解。综上所述，这些结果表明，Muon及其他相关方法中采用的梯度正交化在方向上是正确的，但可能不是严格最优的。最后，我们讨论了如何利用各向同性曲率模型设计新的深度学习和语言模型优化方法的未来研究方向。 

---
# Lessons Learned from the Use of Generative AI in Engineering and Quality Assurance of a WEB System for Healthcare 

**Title (ZH)**: 基于用于医疗保健的WEB系统工程与质量保证中生成式AI应用的教训总结 

**Authors**: Guilherme H. Travassos, Sabrina Rocha, Rodrigo Feitosa, Felipe Assis, Patricia Goncalves, Andre Gheventer, Larissa Galeno, Arthur Sasse, Julio Cesar Guimaraes, Carlos Brito, Joao Pedro Wieland  

**Link**: [PDF](https://arxiv.org/pdf/2511.00658)  

**Abstract**: The advances and availability of technologies involving Generative Artificial Intelligence (AI) are evolving clearly and explicitly, driving immediate changes in various work activities. Software Engineering (SE) is no exception and stands to benefit from these new technologies, enhancing productivity and quality in its software development processes. However, although the use of Generative AI in SE practices is still in its early stages, considering the lack of conclusive results from ongoing research and the limited technological maturity, we have chosen to incorporate these technologies in the development of a web-based software system to be used in clinical trials by a thoracic diseases research group at our university. For this reason, we decided to share this experience report documenting our development team's learning journey in using Generative AI during the software development process. Project management, requirements specification, design, development, and quality assurance activities form the scope of observation. Although we do not yet have definitive technological evidence to evolve our development process significantly, the results obtained and the suggestions shared here represent valuable insights for software organizations seeking to innovate their development practices to achieve software quality with generative AI. 

**Abstract (ZH)**: 生成型人工智能技术的进步及其在软件工程中的应用：一项基于临床试验的网页软件系统开发经验报告 

---
# More Than A Shortcut: A Hyperbolic Approach To Early-Exit Networks 

**Title (ZH)**: 不只是一个捷径：双曲方法在早期退出网络中的应用 

**Authors**: Swapnil Bhosale, Cosmin Frateanu, Camilla Clark, Arnoldas Jasonas, Chris Mitchell, Xiatian Zhu, Vamsi Krishna Ithapu, Giacomo Ferroni, Cagdas Bilen, Sanjeel Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2511.00641)  

**Abstract**: Deploying accurate event detection on resource-constrained devices is challenged by the trade-off between performance and computational cost. While Early-Exit (EE) networks offer a solution through adaptive computation, they often fail to enforce a coherent hierarchical structure, limiting the reliability of their early predictions. To address this, we propose Hyperbolic Early-Exit networks (HypEE), a novel framework that learns EE representations in the hyperbolic space. Our core contribution is a hierarchical training objective with a novel entailment loss, which enforces a partial-ordering constraint to ensure that deeper network layers geometrically refine the representations of shallower ones. Experiments on multiple audio event detection tasks and backbone architectures show that HypEE significantly outperforms standard Euclidean EE baselines, especially at the earliest, most computationally-critical exits. The learned geometry also provides a principled measure of uncertainty, enabling a novel triggering mechanism that makes the overall system both more efficient and more accurate than a conventional EE and standard backbone models without early-exits. 

**Abstract (ZH)**: 在资源受限设备上部署精确的事件检测面临性能和计算成本之间的权衡挑战。Hyperbolic Early-Exit 网络 (HypEE): 一种学习双曲空间中Early-Exit表示的新框架 

---
# Node Preservation and its Effect on Crossover in Cartesian Genetic Programming 

**Title (ZH)**: 节点保留及其对笛卡尔遗传编程交叉操作的影响 

**Authors**: Mark Kocherovsky, Illya Bakurov, Wolfgang Banzhaf  

**Link**: [PDF](https://arxiv.org/pdf/2511.00634)  

**Abstract**: While crossover is a critical and often indispensable component in other forms of Genetic Programming, such as Linear- and Tree-based, it has consistently been claimed that it deteriorates search performance in CGP. As a result, a mutation-alone $(1+\lambda)$ evolutionary strategy has become the canonical approach for CGP. Although several operators have been developed that demonstrate an increased performance over the canonical method, a general solution to the problem is still lacking. In this paper, we compare basic crossover methods, namely one-point and uniform, to variants in which nodes are ``preserved,'' including the subgraph crossover developed by Roman Kalkreuth, the difference being that when ``node preservation'' is active, crossover is not allowed to break apart instructions. We also compare a node mutation operator to the traditional point mutation; the former simply replaces an entire node with a new one. We find that node preservation in both mutation and crossover improves search using symbolic regression benchmark problems, moving the field towards a general solution to CGP crossover. 

**Abstract (ZH)**: 而在其他形式的遗传编程，如线性和树形基于的遗传编程中，交叉操作是一个关键且 Often indispensable 组件，然而在计算型遗传编程中，交叉操作一直被认为会损害搜索性能。因此，仅使用突变的 $(1+\lambda)$ 进化策略成为了计算型遗传编程的经典方法。尽管已经开发出了一些优于经典方法的操作符，但针对该问题的通用解决方案仍然缺乏。在本文中，我们将基础的交叉方法，如单点交叉和均匀交叉，与包括由罗马·卡尔克鲁斯开发的子图交叉在内的“节点保留”变种进行比较，其中“节点保留”的区别在于当“节点保留”激活时，交叉不会被允许打断指令。我们还将节点突变操作符与传统的位点突变进行比较，前者简单地用一个新的节点替换整个节点。我们发现，在突变和交叉中实施“节点保留”都能在符号回归基准问题上改善搜索，从而朝着计算型遗传编程交叉的通用解决方案迈进。 

---
# EPARA: Parallelizing Categorized AI Inference in Edge Clouds 

**Title (ZH)**: EPARA: 边缘云中按类别并行化AI推断 

**Authors**: Yubo Wang, Yubo Cui, Tuo Shi, Danyang Li, Wenxin Li, Lide Suo, Tao Wang, Xin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.00603)  

**Abstract**: With the increasing adoption of AI applications such as large language models and computer vision AI, the computational demands on AI inference systems are continuously rising, making the enhancement of task processing capacity using existing hardware a primary objective in edge clouds. We propose EPARA, an end-to-end AI parallel inference framework in edge, aimed at enhancing the edge AI serving capability. Our key idea is to categorize tasks based on their sensitivity to latency/frequency and requirement for GPU resources, thereby achieving both request-level and service-level task-resource allocation. EPARA consists of three core components: 1) a task-categorized parallelism allocator that decides the parallel mode of each task, 2) a distributed request handler that performs the calculation for the specific request, and 3) a state-aware scheduler that periodically updates service placement in edge clouds. We implement a EPARA prototype and conduct a case study on the EPARA operation for LLMs and segmentation tasks. Evaluation through testbed experiments involving edge servers, embedded devices, and microcomputers shows that EPARA achieves up to 2.1$\times$ higher goodput in production workloads compared to prior frameworks, while adapting to various edge AI inference tasks. 

**Abstract (ZH)**: 面向边缘云的端到端AI并行推理框架E_PARA：提升边缘AI服务能力 

---
# FTT-GRU: A Hybrid Fast Temporal Transformer with GRU for Remaining Useful Life Prediction 

**Title (ZH)**: FTT-GRU: 一种结合快速时域变换器与GRU的剩余寿命预测模型 

**Authors**: Varun Teja Chirukiri, Udaya Bhasker Cheerala, Sandeep Kanta, Abdul Karim, Praveen Damacharla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00564)  

**Abstract**: Accurate prediction of the remaining useful life (RUL) of industrial machinery is essential for reducing downtime and optimizing maintenance schedules. Existing approaches, such as long short-term memory (LSTM) networks and convolutional neural networks (CNNs), often struggle to model both global temporal dependencies and fine-grained degradation trends in multivariate sensor data. We propose a hybrid model, FTT-GRU, which combines a Fast Temporal Transformer (FTT) -- a lightweight Transformer variant using linearized attention via fast Fourier transform (FFT) -- with a gated recurrent unit (GRU) layer for sequential modeling. To the best of our knowledge, this is the first application of an FTT with a GRU for RUL prediction on NASA CMAPSS, enabling simultaneous capture of global and local degradation patterns in a compact architecture. On CMAPSS FD001, FTT-GRU attains RMSE 30.76, MAE 18.97, and $R^2=0.45$, with 1.12 ms CPU latency at batch=1. Relative to the best published deep baseline (TCN--Attention), it improves RMSE by 1.16\% and MAE by 4.00\%. Training curves averaged over $k=3$ runs show smooth convergence with narrow 95\% confidence bands, and ablations (GRU-only, FTT-only) support the contribution of both components. These results demonstrate that a compact Transformer-RNN hybrid delivers accurate and efficient RUL predictions on CMAPSS, making it suitable for real-time industrial prognostics. 

**Abstract (ZH)**: 准确预测工业机械的剩余使用寿命（RUL）对于减少停机时间和优化维护计划至关重要。现有的方法，如长短期记忆（LSTM）网络和卷积神经网络（CNNs），往往难以 modeling 多变量传感器数据中的全局时间依赖性和细微的退化趋势。我们提出了一种名为 FTT-GRU 的混合模型，它将 Fast Temporal Transformer（FTT）——一种使用快速傅立叶变换（FFT）进行线性化注意力的轻量级 Transformer 变体——与门控循环单元（GRU）层结合使用，以进行序列建模。据我们所知，这是首次将 FTT 与 GRU 结合用于 NASA CMAPSS 的 RUL 预测，从而使全局和局部退化模式在同一紧凑架构中同时被捕获。在 CMAPSS FD001 上，FTT-GRU 获得了 RMSE 30.76、MAE 18.97 和 $R^2=0.45$，批处理大小为 1 时 CPU 延迟为 1.12 ms。与最佳已发布的深度基线（TCN--Attention）相比，它的 RMSE 改进了 1.16%，MAE 改进了 4.00%。平均 k=3 次训练曲线显示平滑收敛，并具有狭窄的 95% 置信区间，消融实验（仅 GRU、仅 FTT）支持两者的贡献。这些结果表明，紧凑的 Transformer-RNN 混合模型在 CMAPSS 上提供了准确且高效的 RUL 预测，使其适合用于实时工业预测性维护。 

---
# Temporal Fusion Transformer for Multi-Horizon Probabilistic Forecasting of Weekly Retail Sales 

**Title (ZH)**: 用于weekly零售销售多 horizon 概率预测的Temporal Fusion Transformer 

**Authors**: Santhi Bharath Punati, Sandeep Kanta, Udaya Bhasker Cheerala, Madhusudan G Lanjewar, Praveen Damacharla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00552)  

**Abstract**: Accurate multi-horizon retail forecasts are critical for inventory and promotions. We present a novel study of weekly Walmart sales (45 stores, 2010--2012) using a Temporal Fusion Transformer (TFT) that fuses static store identifiers with time-varying exogenous signals (holidays, CPI, fuel price, temperature). The pipeline produces 1--5-week-ahead probabilistic forecasts via Quantile Loss, yielding calibrated 90\% prediction intervals and interpretability through variable-selection networks, static enrichment, and temporal attention. On a fixed 2012 hold-out dataset, TFT achieves an RMSE of \$57.9k USD per store-week and an $R^2$ of 0.9875. Across a 5-fold chronological cross-validation, the averages are RMSE = \$64.6k USD and $R^2$ = 0.9844, outperforming the XGB, CNN, LSTM, and CNN-LSTM baseline models. These results demonstrate practical value for inventory planning and holiday-period optimization, while maintaining model transparency. 

**Abstract (ZH)**: 准确的多时间尺度零售预测对于库存管理和促销活动至关重要：基于Temporal Fusion Transformer的Walmart周销售预测研究 

---
# Air Pollution Forecasting in Bucharest 

**Title (ZH)**: 布加勒斯特的空气污染预报 

**Authors**: Dragoş-Andrei Şerban, Răzvan-Alexandru Smădu, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00532)  

**Abstract**: Air pollution, especially the particulate matter 2.5 (PM2.5), has become a growing concern in recent years, primarily in urban areas. Being exposed to air pollution is linked to developing numerous health problems, like the aggravation of respiratory diseases, cardiovascular disorders, lung function impairment, and even cancer or early death. Forecasting future levels of PM2.5 has become increasingly important over the past few years, as it can provide early warnings and help prevent diseases. This paper aims to design, fine-tune, test, and evaluate machine learning models for predicting future levels of PM2.5 over various time horizons. Our primary objective is to assess and compare the performance of multiple models, ranging from linear regression algorithms and ensemble-based methods to deep learning models, such as advanced recurrent neural networks and transformers, as well as large language models, on this forecasting task. 

**Abstract (ZH)**: 空气污染，尤其是细颗粒物PM2.5，近年来在城市地区已成为日益增长的关注点。暴露于空气污染与多种健康问题相关，包括呼吸道疾病加重、心血管疾病、肺功能损害，甚至癌症或早死。预测未来PM2.5水平在过去几年变得越来越重要，因为它可以提供早期预警并帮助预防疾病。本文旨在设计、微调、测试和评估用于预测不同时间范围PM2.5水平的机器学习模型。我们的主要目标是评估和比较从线性回归算法、集成方法到深度学习模型（如高级循环神经网络和变压器）以及大型语言模型在该预测任务上的性能。 

---
# Reasoning Planning for Language Models 

**Title (ZH)**: 语言模型的推理规划 

**Authors**: Bao Nguyen, Hieu Trung Nguyen, Ruifeng She, Xiaojin Fu, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00521)  

**Abstract**: Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at this https URL. 

**Abstract (ZH)**: 选择合适的推理方法仍然是语言模型生成中的一个关键挑战。现有的方法通常生成多个候选回复，并使用聚合策略选择输出答案，往往会假设更多的候选答案能获得更高的准确性。我们通过严格的理论分析重新审视这一假设，推导出在固定生成分布和候选规模下标准聚合方法的准确性界。基于这些见解，我们提出了EPIC（对比学习下的ensemble规划框架），用于学习一个既能捕捉模型推理能力又能反映查询-方法兼容性的共享表示空间。EPIC将我们的概率界作为驱动效用优化中的正则化项，平衡准确性与计算成本。在多样化的数学推理任务上的实验显示，EPIC一致地选择最优的推理方法，同时提高准确性并减少计算开销。可供代码见[this https URL]。 

---
# A Multimodal Dataset for Indoor Radio Mapping with 3D Point Clouds and RSSI 

**Title (ZH)**: 基于3D点云和RSSI的室内无线地图 multimodal 数据集 

**Authors**: Ljupcho Milosheski, Kuon Akiyama, Blaž Bertalanič, Jernej Hribar, Ryoichi Shinkuma  

**Link**: [PDF](https://arxiv.org/pdf/2511.00494)  

**Abstract**: The growing number of smart devices supporting bandwidth-intensive and latency-sensitive applications, such as real-time video analytics, smart sensing, and Extended Reality (XR), necessitates reliable wireless connectivity in indoor environments. Therein, accurate estimation of Radio Environment Maps (REMs) enables adaptive wireless network planning and optimization of Access Point (AP) placement. However, generating realistic REMs remains challenging due to the complexity of indoor spaces. To overcome this challenge, this paper introduces a multimodal dataset that integrates high-resolution 3D LiDAR scans with Wi-Fi Received Signal Strength Indicator (RSSI) measurements collected under 20 distinct AP configurations in a multi-room indoor environment. The dataset captures two measurement scenarios: the first without human presence in the environment, and the second with human presence. Thus, the presented dataset supports the study of dynamic environmental effects on wireless signal propagation. This resource is designed to facilitate research in data-driven wireless modeling, particularly in the context of emerging high-frequency standards such as IEEE 802.11be (Wi-Fi 7), and aims to advance the development of robust, high-capacity indoor communication systems. 

**Abstract (ZH)**: 基于多模态数据集的无线环境图估计：推动室内高频频段通信系统的发展 

---
# Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation 

**Title (ZH)**: 探究标签偏差和表征来源对年龄相关医学分割差异的影响 

**Authors**: Aditya Parikh, Sneha Das, Aasa Feragen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00477)  

**Abstract**: Algorithmic bias in medical imaging can perpetuate health disparities, yet its causes remain poorly understood in segmentation tasks. While fairness has been extensively studied in classification, segmentation remains underexplored despite its clinical importance. In breast cancer segmentation, models exhibit significant performance disparities against younger patients, commonly attributed to physiological differences in breast density. We audit the MAMA-MIA dataset, establishing a quantitative baseline of age-related bias in its automated labels, and reveal a critical Biased Ruler effect where systematically flawed labels for validation misrepresent a model's actual bias. However, whether this bias originates from lower-quality annotations (label bias) or from fundamentally more challenging image characteristics remains unclear. Through controlled experiments, we systematically refute hypotheses that the bias stems from label quality sensitivity or quantitative case difficulty imbalance. Balancing training data by difficulty fails to mitigate the disparity, revealing that younger patient cases are intrinsically harder to learn. We provide direct evidence that systemic bias is learned and amplified when training on biased, machine-generated labels, a critical finding for automated annotation pipelines. This work introduces a systematic framework for diagnosing algorithmic bias in medical segmentation and demonstrates that achieving fairness requires addressing qualitative distributional differences rather than merely balancing case counts. 

**Abstract (ZH)**: 医学影像中的算法偏见可以加剧健康不平等，但在分割任务中的成因仍知之甚少。尽管分类中的公平性已得到广泛研究，但分割任务仍因临床重要性不足而未被充分探索。在乳腺癌分割中，模型在年轻患者中表现出显著的性能差异，通常归因于乳腺密度的生理差异。我们审计了MAMA-MIA数据集，建立了与年龄相关的偏见的量化基线，并揭示了一个关键的“偏差尺子”效应，其中系统性错误的标签在验证过程中歪曲了模型的实际偏见。然而，这种偏见是源自较低质量的注释（标签偏见）还是源自更为基本的图像特征挑战性仍不清楚。通过控制实验，我们系统地反驳了偏见源于质量敏感性注释或案例难度量化的假设。通过难度平衡训练数据无法缓解这种差异，揭示了年轻患者案例本身更具学习难度。我们提供了直接证据，表明系统性偏见是在使用偏差的机器生成标签进行训练时被学习和放大的，这是一个关于自动化注释管道的关键发现。本文引入了一套系统框架来诊断医学分割中的算法偏见，并证明实现公平性需要解决定性的分布差异而不是仅仅平衡案例数量。 

---
# Longitudinal Vestibular Schwannoma Dataset with Consensus-based Human-in-the-loop Annotations 

**Title (ZH)**: 基于共识的人工在环Longitudinal前庭神经鞘瘤数据集 

**Authors**: Navodini Wijethilake, Marina Ivory, Oscar MacCormac, Siddhant Kumar, Aaron Kujawa, Lorena Garcia-Foncillas Macias, Rebecca Burger, Amanda Hitchings, Suki Thomson, Sinan Barazi, Eleni Maratos, Rupert Obholzer, Dan Jiang, Fiona McClenaghan, Kazumi Chia, Omar Al-Salihi, Nick Thomas, Steve Connor, Tom Vercauteren, Jonathan Shapey  

**Link**: [PDF](https://arxiv.org/pdf/2511.00472)  

**Abstract**: Accurate segmentation of vestibular schwannoma (VS) on Magnetic Resonance Imaging (MRI) is essential for patient management but often requires time-intensive manual annotations by experts. While recent advances in deep learning (DL) have facilitated automated segmentation, challenges remain in achieving robust performance across diverse datasets and complex clinical cases. We present an annotated dataset stemming from a bootstrapped DL-based framework for iterative segmentation and quality refinement of VS in MRI. We combine data from multiple centres and rely on expert consensus for trustworthiness of the annotations. We show that our approach enables effective and resource-efficient generalisation of automated segmentation models to a target data distribution. The framework achieved a significant improvement in segmentation accuracy with a Dice Similarity Coefficient (DSC) increase from 0.9125 to 0.9670 on our target internal validation dataset, while maintaining stable performance on representative external datasets. Expert evaluation on 143 scans further highlighted areas for model refinement, revealing nuanced cases where segmentation required expert intervention. The proposed approach is estimated to enhance efficiency by approximately 37.4% compared to the conventional manual annotation process. Overall, our human-in-the-loop model training approach achieved high segmentation accuracy, highlighting its potential as a clinically adaptable and generalisable strategy for automated VS segmentation in diverse clinical settings. The dataset includes 190 patients, with tumour annotations available for 534 longitudinal contrast-enhanced T1-weighted (T1CE) scans from 184 patients, and non-annotated T2-weighted scans from 6 patients. This dataset is publicly accessible on The Cancer Imaging Archive (TCIA) (this https URL). 

**Abstract (ZH)**: 基于深度学习的迭代分割和质量精炼框架在磁共振成像中精准分割前庭 Schwannoma 的标注数据集 

---
# Why Federated Optimization Fails to Achieve Perfect Fitting? A Theoretical Perspective on Client-Side Optima 

**Title (ZH)**: 为什么联邦优化无法实现完美拟合？从客户端最优解的理论视角探讨 

**Authors**: Zhongxiang Lei, Qi Yang, Ping Qiu, Gang Zhang, Yuanchi Ma, Jinyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00469)  

**Abstract**: Federated optimization is a constrained form of distributed optimization that enables training a global model without directly sharing client data. Although existing algorithms can guarantee convergence in theory and often achieve stable training in practice, the reasons behind performance degradation under data heterogeneity remain unclear. To address this gap, the main contribution of this paper is to provide a theoretical perspective that explains why such degradation occurs. We introduce the assumption that heterogeneous client data lead to distinct local optima, and show that this assumption implies two key consequences: 1) the distance among clients' local optima raises the lower bound of the global objective, making perfect fitting of all client data impossible; and 2) in the final training stage, the global model oscillates within a region instead of converging to a single optimum, limiting its ability to fully fit the data. These results provide a principled explanation for performance degradation in non-iid settings, which we further validate through experiments across multiple tasks and neural network architectures. The framework used in this paper is open-sourced at: this https URL. 

**Abstract (ZH)**: 联邦优化是一种受限形式的分布式优化，能够在不直接共享客户端数据的情况下训练全局模型。尽管现有算法在理论上可以保证收敛，并且在实践中通常能够实现稳定的训练，但在数据异构性下性能下降的原因仍然不明确。为了填补这一缺口，本文的主要贡献是提供了一个理论视角，以解释为何会出现这种性能下降。我们提出了异构客户端数据会导致不同的局部最优解的假设，并展示了这一假设意味着两个关键后果：1）客户端局部最优解之间的距离提高了全局目标函数的下界，使得所有客户端数据的完美拟合变得不可能；2）在最终训练阶段，全局模型在某一区域内振荡而无法收敛到单一最优解，限制了其完全拟合数据的能力。这些结果为非iid设置下的性能下降提供了原则上解释，并通过多个任务和神经网络架构上的实验进一步验证了这一结论。本文使用的框架已开源于：this https URL。 

---
# LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026 

**Title (ZH)**: LIR：ECIR 2026第一届Late Interaction和多向量检索研讨会 

**Authors**: Benjamin Clavié, Xianming Li, Antoine Chaffin, Omar Khattab, Tom Aarsen, Manuel Faysse, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00444)  

**Abstract**: Late interaction retrieval methods, pioneered by ColBERT, have emerged as a powerful alternative to single-vector neural IR. By leveraging fine-grained, token-level representations, they have been demonstrated to deliver strong generalisation and robustness, particularly in out-of-domain settings. They have recently been shown to be particularly well-suited for novel use cases, such as reasoning-based or cross-modality retrieval. At the same time, these models pose significant challenges of efficiency, usability, and integrations into fully fledged systems; as well as the natural difficulties encountered while researching novel application domains. Recent years have seen rapid advances across many of these areas, but research efforts remain fragmented across communities and frequently exclude practitioners. The purpose of this workshop is to create an environment where all aspects of late interaction can be discussed, with a focus on early research explorations, real-world outcomes, and negative or puzzling results to be freely shared and discussed. The aim of LIR is to provide a highly-interactive environment for researchers from various backgrounds and practitioners to freely discuss their experience, fostering further collaboration. 

**Abstract (ZH)**: Late 交互检索方法：ColBERT引领的迟态交互检索方法已成为单向量神经IR的强大替代方案。通过利用细粒度的token级别表示，它们已被证明在泛化能力和鲁棒性方面表现出色，尤其是在领域外设置中。最近的研究表明，它们特别适合新型应用场景，如基于推理或跨模态检索。与此同时，这些模型带来了效率、易用性和与完整系统的集成等方面的重大挑战；此外，探索新型应用领域时还面临着自然的困难。近年来，这些领域的研究取得了快速进展，但研究努力仍然分散在不同的社区中，且经常排除实际应用人员。这次研讨会的目标是创建一个平台，让来自不同背景的研究人员和实践者能够自由讨论各个方面，重点关注早期研究探索、实际应用结果以及可以自由分享和讨论的反常或困惑的结果。迟态交互检索（LIR）旨在为来自不同领域的研究人员和实践者提供一个高互动性的环境，促进进一步的合作探索。 

---
# Region-Aware Reconstruction Strategy for Pre-training fMRI Foundation Model 

**Title (ZH)**: 基于区域感知的重建策略预训练fMRI基础模型 

**Authors**: Ruthwik Reddy Doodipala, Pankaj Pandey, Carolina Torres Rojas, Manob Jyoti Saikia, Ranganatha Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2511.00443)  

**Abstract**: The emergence of foundation models in neuroimaging is driven by the increasing availability of large-scale and heterogeneous brain imaging datasets. Recent advances in self-supervised learning, particularly reconstruction-based objectives, have demonstrated strong potential for pretraining models that generalize effectively across diverse downstream functional MRI (fMRI) tasks. In this study, we explore region-aware reconstruction strategies for a foundation model in resting-state fMRI, moving beyond approaches that rely on random region masking. Specifically, we introduce an ROI-guided masking strategy using the Automated Anatomical Labelling Atlas (AAL3), applied directly to full 4D fMRI volumes to selectively mask semantically coherent brain regions during self-supervised pretraining. Using the ADHD-200 dataset comprising 973 subjects with resting-state fMRI scans, we show that our method achieves a 4.23% improvement in classification accuracy for distinguishing healthy controls from individuals diagnosed with ADHD, compared to conventional random masking. Region-level attribution analysis reveals that brain volumes within the limbic region and cerebellum contribute most significantly to reconstruction fidelity and model representation. Our results demonstrate that masking anatomical regions during model pretraining not only enhances interpretability but also yields more robust and discriminative representations. In future work, we plan to extend this approach by evaluating it on additional neuroimaging datasets, and developing new loss functions explicitly derived from region-aware reconstruction objectives. These directions aim to further improve the robustness and interpretability of foundation models for functional neuroimaging. 

**Abstract (ZH)**: 基础模型在神经影像学中的出现受到大规模和异质性脑影像数据可用性的推动。最近在自监督学习方面的进展，尤其是基于重建的目标，证明了预训练能够在多种下游功能磁共振成像(fMRI)任务中实现有效的泛化能力。在本研究中，我们探讨了在静息态fMRI中使用区域感知重建策略的基础模型，超越了依赖随机区域掩蔽的方法。具体地，我们引入了一种基于ROI的掩蔽策略，使用Automated Anatomical Labeling Atlas（AAL3），直接应用于完整的4D fMRI体数据，在自监督预训练过程中选择性地掩蔽语义一致的脑区。使用包含973例静息态fMRI扫描的ADHD-200数据集，我们表明，我们的方法在区分健康对照组和 ADHD 病例的分类准确性上比传统的随机掩蔽提升了4.23%。区域级别的归因分析表明，边缘区和小脑的脑体积对重建保真度和模型表示的贡献最大。我们的结果表明，在模型预训练期间掩蔽解剖学区域不仅提高了可解释性，还产生了更稳健和区分度更高的表示。在未来的工作中，我们将通过在其他神经影像学数据集上评估该方法，并开发从区域感知重建目标显式导出的新损失函数，来扩展这种方法。这些方向旨在进一步提高功能神经影像学领域基础模型的稳健性和可解释性。 

---
# LGCA: Enhancing Semantic Representation via Progressive Expansion 

**Title (ZH)**: LGCA：通过分阶扩张增强语义表示 

**Authors**: Thanh Hieu Cao, Trung Khang Tran, Gia Thinh Pham, Tuong Nghiem Diep, Thanh Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00419)  

**Abstract**: Recent advancements in large-scale pretraining in natural language processing have enabled pretrained vision-language models such as CLIP to effectively align images and text, significantly improving performance in zero-shot image classification tasks. Subsequent studies have further demonstrated that cropping images into smaller regions and using large language models to generate multiple descriptions for each caption can further enhance model performance. However, due to the inherent sensitivity of CLIP, random image crops can introduce misinformation and bias, as many images share similar features at small scales. To address this issue, we propose Localized-Globalized Cross-Alignment (LGCA), a framework that first captures the local features of an image and then repeatedly selects the most salient regions and expands them. The similarity score is designed to incorporate both the original and expanded images, enabling the model to capture both local and global features while minimizing misinformation. Additionally, we provide a theoretical analysis demonstrating that the time complexity of LGCA remains the same as that of the original model prior to the repeated expansion process, highlighting its efficiency and scalability. Extensive experiments demonstrate that our method substantially improves zero-shot performance across diverse datasets, outperforming state-of-the-art baselines. 

**Abstract (ZH)**: 大规模预训练在自然语言处理中的 recent 进展使得如 CLIP 等预训练视觉-语言模型能够有效对齐图像和文本，显著提高了零样本图像分类任务的性能。后续研究进一步表明，将图像裁剪成较小区域，并使用大规模语言模型为每个描述生成多个版本，可以进一步提升模型性能。然而，由于 CLIP 的固有敏感性，随机图像裁剪可能会引入 misinformation 和偏差，因为许多图像在小尺度上具有相似的特征。为了解决这一问题，我们提出了一种局部化-全局化交叉对齐（LGCA）框架，该框架首先捕获图像的局部特征，然后反复选择最具显著性的区域并进行扩展。相似度评分设计成同时考虑原始图像和扩展后的图像，从而使模型能够同时捕捉局部和全局特征并最大限度地减少 misinformation。此外，我们提供了理论分析，证明 LGCA 在重复扩展过程之前的计算复杂度与原模型相同，突显了其高效性和可扩展性。广泛的实验证明，我们的方法在多种数据集上显著提高了零样本性能，并优于现有最先进的基线。 

---
# Human-AI Programming Role Optimization: Developing a Personality-Driven Self-Determination Framework 

**Title (ZH)**: 人类-人工智能编程角色优化：开发一种以个性驱动的自主决定框架 

**Authors**: Marcel Valovy  

**Link**: [PDF](https://arxiv.org/pdf/2511.00417)  

**Abstract**: As artificial intelligence transforms software development, a critical question emerges: how can developers and AI systems collaborate most effectively? This dissertation optimizes human-AI programming roles through self-determination theory and personality psychology, introducing the Role Optimization Motivation Alignment (ROMA) framework.
Through Design Science Research spanning five cycles, this work establishes empirically-validated connections between personality traits, programming role preferences, and collaborative outcomes, engaging 200 experimental participants and 46 interview respondents.
Key findings demonstrate that personality-driven role optimization significantly enhances self-determination and team dynamics, yielding 23% average motivation increases among professionals and up to 65% among undergraduates. Five distinct personality archetypes emerge: The Explorer (high Openness/low Agreeableness), The Orchestrator (high Extraversion/Agreeableness), The Craftsperson (high Neuroticism/low Extraversion), The Architect (high Conscientiousness), and The Adapter (balanced profile). Each exhibits distinct preferences for programming roles (Co-Pilot, Co-Navigator, Agent), with assignment modes proving crucial for satisfaction.
The dissertation contributes: (1) an empirically-validated framework linking personality traits to role preferences and self-determination outcomes; (2) a taxonomy of AI collaboration modalities mapped to personality profiles while preserving human agency; and (3) an ISO/IEC 29110 extension enabling Very Small Entities to implement personality-driven role optimization within established standards.
Keywords: artificial intelligence, human-computer interaction, behavioral software engineering, self-determination theory, personality psychology, phenomenology, intrinsic motivation, pair programming, design science research, ISO/IEC 29110 

**Abstract (ZH)**: 人工智能变革软件开发过程中，一个关键问题浮出水面：开发者与AI系统如何实现最为有效的协作？本论文通过自我决定理论和人格心理学优化人机编程角色，引入角色优化动机对齐（ROMA）框架。
通过历时五轮的设计科学研究，本研究确立了人格特质、编程角色偏好与协作成效之间的实证联系，涉及200名实验参与者和46名访谈对象。
主要发现表明，基于人格特征的角色优化显著增强了自我决定及其团队动态，专业人员平均动机提升23%，本科生可达65%。五种独特的人格模式浮现：探路者（高开放性/低宜人性）、指挥家（高外向性/宜人性）、匠人（高神经质/低外向性）、建筑师（高尽责性）和适应者（平衡型）。每种模式显示出不同的编程角色偏好（协同程序员、协同导航员、代理），角色分配方式对满意度至关重要。
本论文贡献包括：(1) 一个基于人格特征与角色偏好和自我决定成果的实证框架；(2) 一种AI协作模式分类，映射到人格特征，同时保留人类自主权；(3) 一种ISO/IEC 29110扩展，使小型实体能够在现有标准中实施基于人格特征的角色优化。
关键词：人工智能、人机交互、行为软件工程、自我决定理论、人格心理学、现象学、内在动机、对等编程、设计科学研究、ISO/IEC 29110 

---
# PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks 

**Title (ZH)**: PADBen: 一种全面的基准，用于评估AI文本检测器对抗 paraphrase 攻击的能力 

**Authors**: Yiwei Zha, Rui Min, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2511.00416)  

**Abstract**: While AI-generated text (AIGT) detectors achieve over 90\% accuracy on direct LLM outputs, they fail catastrophically against iteratively-paraphrased content. We investigate why iteratively-paraphrased text -- itself AI-generated -- evades detection systems designed for AIGT identification. Through intrinsic mechanism analysis, we reveal that iterative paraphrasing creates an intermediate laundering region characterized by semantic displacement with preserved generation patterns, which brings up two attack categories: paraphrasing human-authored text (authorship obfuscation) and paraphrasing LLM-generated text (plagiarism evasion). To address these vulnerabilities, we introduce PADBen, the first benchmark systematically evaluating detector robustness against both paraphrase attack scenarios. PADBen comprises a five-type text taxonomy capturing the full trajectory from original content to deeply laundered text, and five progressive detection tasks across sentence-pair and single-sentence challenges. We evaluate 11 state-of-the-art detectors, revealing critical asymmetry: detectors successfully identify the plagiarism evasion problem but fail for the case of authorship obfuscation. Our findings demonstrate that current detection approaches cannot effectively handle the intermediate laundering region, necessitating fundamental advances in detection architectures beyond existing semantic and stylistic discrimination methods. For detailed code implementation, please see this https URL. 

**Abstract (ZH)**: AI生成文本检测中的迭代伪装文本逃避机制与PADBen基准研究 

---
# Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling 

**Title (ZH)**: 通过梯度引导采样实现探索与利用平衡以增强对抗迁移性 

**Authors**: Zenghao Niu, Weicheng Xie, Siyang Song, Zitong Yu, Feng Liu, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00411)  

**Abstract**: Adversarial attacks present a critical challenge to deep neural networks' robustness, particularly in transfer scenarios across different model architectures. However, the transferability of adversarial attacks faces a fundamental dilemma between Exploitation (maximizing attack potency) and Exploration (enhancing cross-model generalization). Traditional momentum-based methods over-prioritize Exploitation, i.e., higher loss maxima for attack potency but weakened generalization (narrow loss surface). Conversely, recent methods with inner-iteration sampling over-prioritize Exploration, i.e., flatter loss surfaces for cross-model generalization but weakened attack potency (suboptimal local maxima). To resolve this dilemma, we propose a simple yet effective Gradient-Guided Sampling (GGS), which harmonizes both objectives through guiding sampling along the gradient ascent direction to improve both sampling efficiency and stability. Specifically, based on MI-FGSM, GGS introduces inner-iteration random sampling and guides the sampling direction using the gradient from the previous inner-iteration (the sampling's magnitude is determined by a random distribution). This mechanism encourages adversarial examples to reside in balanced regions with both flatness for cross-model generalization and higher local maxima for strong attack potency. Comprehensive experiments across multiple DNN architectures and multimodal large language models (MLLMs) demonstrate the superiority of our method over state-of-the-art transfer attacks. Code is made available at this https URL. 

**Abstract (ZH)**: 对抗攻击对深度神经网络在不同模型架构间的鲁棒性构成了关键挑战，特别是迁移场景下的挑战。然而，对抗攻击的可迁移性在利用（最大化攻击效果）和探索（增强跨模型泛化）之间面临基本的矛盾。传统的基于动量的方法过分强调利用，即攻效果最优但泛化能力下降（损失面狭窄）。相反，近期的基于内迭代采样的方法过分强调探索，即损失面平坦但攻击效果减弱（局部最大值次优）。为解决这一矛盾，我们提出了一种简单有效的梯度引导采样（GGS），通过沿梯度上升方向引导采样来同时提升采样效率和稳定性。具体而言，基于MI-FGSM，GGS引入了内迭代随机采样，并使用上一次内迭代的梯度来引导采样方向（样本的大小由随机分布决定）。该机制促使对抗样本集中于既能促进跨模型泛化又能增强攻击效果的平衡区域。在多种DNN架构和多模态大规模语言模型（MLLMs）上的全面实验表明，我们的方法在最新的迁移攻击中具有优越性。代码可在以下链接获取：this https URL。 

---
# Quantum Machine Unlearning: Foundations, Mechanisms, and Taxonomy 

**Title (ZH)**: 量子机器遗忘：基础、机制及分类学 

**Authors**: Thanveer Shaik, Xiaohui Tao, Haoran Xie, Robert Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00406)  

**Abstract**: Quantum Machine Unlearning has emerged as a foundational challenge at the intersection of quantum information theory privacypreserving computation and trustworthy artificial intelligence This paper advances QMU by establishing a formal framework that unifies physical constraints algorithmic mechanisms and ethical governance within a verifiable paradigm We define forgetting as a contraction of distinguishability between pre and postunlearning models under completely positive trace-preserving dynamics grounding data removal in the physics of quantum irreversibility Building on this foundation we present a fiveaxis taxonomy spanning scope guarantees mechanisms system context and hardware realization linking theoretical constructs to implementable strategies Within this structure we incorporate influence and quantum Fisher information weighted updates parameter reinitialization and kernel alignment as practical mechanisms compatible with noisy intermediatescale quantum NISQ devices The framework extends naturally to federated and privacyaware settings via quantum differential privacy homomorphic encryption and verifiable delegation enabling scalable auditable deletion across distributed quantum systems Beyond technical design we outline a forwardlooking research roadmap emphasizing formal proofs of forgetting scalable and secure architectures postunlearning interpretability and ethically auditable governance Together these contributions elevate QMU from a conceptual notion to a rigorously defined and ethically aligned discipline bridging physical feasibility algorithmic verifiability and societal accountability in the emerging era of quantum intelligence. 

**Abstract (ZH)**: 量子机器去学习已成为量子信息理论、隐私保护计算和可信赖人工智能交汇处的一项基础性挑战。本文通过建立一个正式框架，将物理约束、算法机制和伦理治理统一在一个可验证的范式内，推动了量子机器去学习（QMU）的发展。我们定义遗忘为在完全正迹不变动力学下，学习前后的模型可区分性收缩，从而将数据删除基于量子不可逆性的物理原理。在这一基础上，我们提出了一个涵盖范围、保证、机制、系统上下文和硬件实现的五轴分类体系，将理论构架与可实施策略联系起来。在此结构中，我们整合了受量子差分隐私、同态加密和可验证委派影响的加权更新、参数重新初始化和核对齐等实用机制，这些机制与嘈杂的中间尺度量子（NISQ）设备兼容。该框架自然扩展到联邦和隐私意识设置中，通过量子差分隐私、同态加密和可验证委派，实现分布式量子系统中可扩展且可审计的数据删除。除了技术设计之外，我们还提出了一项前瞻性的研究 roadmap，强调遗忘的正式证明、可扩展和安全的架构、去学习后的解释性和伦理可审计的治理。这些贡献将QMU从一个概念性概念提升为一个严格定义且伦理对齐的学科，连接了物理可行性、算法验证性和社会问责制，在量子智能新兴时代架起了桥梁。 

---
# Emotion Detection in Speech Using Lightweight and Transformer-Based Models: A Comparative and Ablation Study 

**Title (ZH)**: 基于轻量级和Transformer模型的语音情感检测：比较与消融研究 

**Authors**: Lucky Onyekwelu-Udoka, Md Shafiqul Islam, Md Shahedul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00402)  

**Abstract**: Emotion recognition from speech plays a vital role in the development of empathetic human-computer interaction systems. This paper presents a comparative analysis of lightweight transformer-based models, DistilHuBERT and PaSST, by classifying six core emotions from the CREMA-D dataset. We benchmark their performance against a traditional CNN-LSTM baseline model using MFCC features. DistilHuBERT demonstrates superior accuracy (70.64%) and F1 score (70.36%) while maintaining an exceptionally small model size (0.02 MB), outperforming both PaSST and the baseline. Furthermore, we conducted an ablation study on three variants of the PaSST, Linear, MLP, and Attentive Pooling heads, to understand the effect of classification head architecture on model performance. Our results indicate that PaSST with an MLP head yields the best performance among its variants but still falls short of DistilHuBERT. Among the emotion classes, angry is consistently the most accurately detected, while disgust remains the most challenging. These findings suggest that lightweight transformers like DistilHuBERT offer a compelling solution for real-time speech emotion recognition on edge devices. The code is available at: this https URL. 

**Abstract (ZH)**: 语音情感识别对于发展共情的人机交互系统具有重要作用。本文通过分类CREMA-D数据集中的六种核心情感，对比分析了轻量级Transformer模型DistilHuBERT和PaSST的性能。我们使用MFCC特征将这两种模型与传统的CNN-LSTM基线模型进行了基准测试。结果表明，DistilHuBERT在保持极小模型大小（0.02 MB）的同时，展示了更高的准确率（70.64%）和F1分值（70.36%），在性能上优于PaSST和基线模型。此外，我们还对PaSST的三种变体——Linear、MLP和注意池化头部进行了消融研究，以了解分类头部架构对模型性能的影响。我们的结果表明，具有MLP头部的PaSST在其变体中表现最佳，但仍不及DistilHuBERT。在情感类别中，愤怒情感被一致准确地检测到，而厌恶则是最具挑战性的。这些发现表明，轻量级Transformer模型如DistilHuBERT为边缘设备上的实时语音情感识别提供了令人信服的解决方案。代码可用于此链接：this https URL。 

---
# Balancing Interpretability and Performance in Motor Imagery EEG Classification: A Comparative Study of ANFIS-FBCSP-PSO and EEGNet 

**Title (ZH)**: 在_motor想象_脑电图分类中的可解释性与性能平衡：ANFIS-FBCSP-PSO与EEGNet的比较研究 

**Authors**: Farjana Aktar, Mohd Ruhul Ameen, Akif Islam, Md Ekramul Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2511.00369)  

**Abstract**: Achieving both accurate and interpretable classification of motor imagery EEG remains a key challenge in brain computer interface (BCI) research. This paper compares a transparent fuzzy reasoning approach (ANFIS-FBCSP-PSO) with a deep learning benchmark (EEGNet) using the BCI Competition IV-2a dataset. The ANFIS pipeline combines filter bank common spatial pattern feature extraction with fuzzy IF-THEN rules optimized via particle swarm optimization, while EEGNet learns hierarchical spatial temporal representations directly from raw EEG data. In within-subject experiments, the fuzzy neural model performed better (68.58 percent +/- 13.76 percent accuracy, kappa = 58.04 percent +/- 18.43), while in cross-subject (LOSO) tests, the deep model exhibited stronger generalization (68.20 percent +/- 12.13 percent accuracy, kappa = 57.33 percent +/- 16.22). The study provides practical guidance for selecting MI-BCI systems according to design goals: interpretability or robustness across users. Future investigations into transformer based and hybrid neuro symbolic frameworks are expected to advance transparent EEG decoding. 

**Abstract (ZH)**: 实现准确且可解释的运动想象EEG分类仍然是脑机接口（BCI）研究中的关键挑战。本文使用BCI竞赛IV-2a数据集将透明模糊推理方法（ANFIS-FBCSP-PSO）与深度学习基准（EEGNet）进行比较。在单被试实验中，模糊神经模型表现更好（准确率68.58%±13.76%，κ=58.04%±18.43），而在跨被试（LOSO）测试中，深度模型展现出更强的泛化能力（准确率68.20%±12.13%，κ=57.33%±16.22）。该研究提供了根据设计目标选择MI-BCI系统的实用指导：可解释性或跨用户鲁棒性。未来对基于变换器和混合神经符号框架的研究有望推动透明EEG解码的发展。 

---
# MalDataGen: A Modular Framework for Synthetic Tabular Data Generation in Malware Detection 

**Title (ZH)**: MalDataGen: 一种用于恶意软件检测的模块化合成表格数据生成框架 

**Authors**: Kayua Oleques Paim, Angelo Gaspar Diniz Nogueira, Diego Kreutz, Weverton Cordeiro, Rodrigo Brandao Mansilha  

**Link**: [PDF](https://arxiv.org/pdf/2511.00361)  

**Abstract**: High-quality data scarcity hinders malware detection, limiting ML performance. We introduce MalDataGen, an open-source modular framework for generating high-fidelity synthetic tabular data using modular deep learning models (e.g., WGAN-GP, VQ-VAE). Evaluated via dual validation (TR-TS/TS-TR), seven classifiers, and utility metrics, MalDataGen outperforms benchmarks like SDV while preserving data utility. Its flexible design enables seamless integration into detection pipelines, offering a practical solution for cybersecurity applications. 

**Abstract (ZH)**: 高质量数据稀缺妨碍恶意软件检测，限制了机器学习性能。我们引入了MalDataGen，这是一个基于模块化深度学习模型（例如WGAN-GP、VQ-VAE）生成高保真合成表格数据的开源模块化框架。通过双重验证（TR-TS/TS-TR）、七种分类器和实用性指标评估，MalDataGen在保持数据实用性的基础上优于SDV等基准模型。其灵活的设计使其能够无缝集成到检测管道中，为网络安全应用提供了一种实用的解决方案。 

---
# Mind the Gap: Missing Cyber Threat Coverage in NIDS Datasets for the Energy Sector 

**Title (ZH)**: 注意缺口：能源部门NIDS数据集中缺失的网络威胁覆盖范围 

**Authors**: Adrita Rahman Tory, Khondokar Fida Hasan, Md Saifur Rahman, Nickolaos Koroniotis, Mohammad Ali Moni  

**Link**: [PDF](https://arxiv.org/pdf/2511.00360)  

**Abstract**: Network Intrusion Detection Systems (NIDS) developed us- ing publicly available datasets predominantly focus on enterprise environ- ments, raising concerns about their effectiveness for converged Informa- tion Technology (IT) and Operational Technology (OT) in energy infras- tructures. This study evaluates the representativeness of five widely used datasets: CIC-IDS2017, SWaT, WADI, Sherlock, and CIC-Modbus2023 against network-detectable MITRE ATT&CK techniques extracted from documented energy sector incidents. Using a structured five-step analyt- ical approach, this article successfully developed and performed a gap analysis that identified 94 network observable techniques from an initial pool of 274 ATT&CK techniques. Sherlock dataset exhibited the high- est mean coverage (0.56), followed closely by CIC-IDS2017 (0.55), while SWaT and WADI recorded the lowest scores (0.38). Combining CIC- IDS2017, Sherlock, and CIC-Modbus2023 achieved an aggregate coverage of 92%, highlighting their complementary strengths. The analysis identi- fies critical gaps, particularly in lateral movement and industrial protocol manipulation, providing a clear pathway for dataset enhancement and more robust NIDS evaluation in hybrid IT/OT energy environments. 

**Abstract (ZH)**: 利用公开数据集开发的网络入侵检测系统（NIDS）主要集中在企业环境中，这引起了对其在能源基础设施中结合的信息技术（IT）和操作技术（OT）环境中的有效性担忧。本文评估了五种广泛使用的数据集——CIC-IDS2017、SWaT、WADI、Sherlock和CIC-Modbus2023——在初始池中的274种MITRE ATT&CK技术中的表示性，这些技术是从记录的能源领域事件中提取的可网络检测的MITRE ATT&CK技术。本文采用结构化的五步分析方法，成功地开发并执行了空白分析，从初始池中识别出了94种网络可观察技术。Sherlock数据集的平均覆盖率最高（0.56），紧随其后的是CIC-IDS2017（0.55），而SWaT和WADI记录的最低分数（0.38）。结合使用CIC-IDS2017、Sherlock和CIC-Modbus2023实现了92%的整体覆盖率，突显了它们互补的优势。分析识别了关键空白，尤其是在横向移动和工业协议操控方面，为数据集的增强和混合IT/OT能源环境中更 robust的NIDS评估提供了清晰的路径。 

---
# Toward Unifying Group Fairness Evaluation from a Sparsity Perspective 

**Title (ZH)**: 从稀疏性视角统一群组公平性评估 

**Authors**: Zhecheng Sheng, Jiawei Zhang, Enmao Diao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00359)  

**Abstract**: Ensuring algorithmic fairness remains a significant challenge in machine learning, particularly as models are increasingly applied across diverse domains. While numerous fairness criteria exist, they often lack generalizability across different machine learning problems. This paper examines the connections and differences among various sparsity measures in promoting fairness and proposes a unified sparsity-based framework for evaluating algorithmic fairness. The framework aligns with existing fairness criteria and demonstrates broad applicability to a wide range of machine learning tasks. We demonstrate the effectiveness of the proposed framework as an evaluation metric through extensive experiments on a variety of datasets and bias mitigation methods. This work provides a novel perspective to algorithmic fairness by framing it through the lens of sparsity and social equity, offering potential for broader impact on fairness research and applications. 

**Abstract (ZH)**: 确保算法公平性仍然是机器学习中的一个重大挑战，尤其是随着模型在各个领域中的广泛应用。尽管存在多种公平性标准，但它们往往在不同的机器学习问题中缺乏普适性。本文探讨了各种稀疏性度量在促进公平性方面的联系与差异，并提出了一种基于稀疏性的统一框架，用于评估算法公平性。该框架与现有的公平性标准相一致，并表现出对多种机器学习任务的广泛适用性。我们通过在多种数据集和偏见缓解方法上进行广泛的实验，证明了所提框架作为评估指标的有效性。本文通过稀疏性和社会公平性的视角提供了算法公平性的一种新颖观点，为其在公平性研究和应用中的更广泛影响提供了潜在可能性。 

---
# MH-1M: A 1.34 Million-Sample Comprehensive Multi-Feature Android Malware Dataset for Machine Learning, Deep Learning, Large Language Models, and Threat Intelligence Research 

**Title (ZH)**: MH-1M：用于机器学习、深度学习、大规模语言模型和威胁情报研究的综合多特征Android恶意软件数据集，包含134万样本 

**Authors**: Hendrio Braganca, Diego Kreutz, Vanderson Rocha, Joner Assolin, and Eduardo Feitosa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00342)  

**Abstract**: We present MH-1M, one of the most comprehensive and up-to-date datasets for advanced Android malware research. The dataset comprises 1,340,515 applications, encompassing a wide range of features and extensive metadata. To ensure accurate malware classification, we employ the VirusTotal API, integrating multiple detection engines for comprehensive and reliable assessment. Our GitHub, Figshare, and Harvard Dataverse repositories provide open access to the processed dataset and its extensive supplementary metadata, totaling more than 400 GB of data and including the outputs of the feature extraction pipeline as well as the corresponding VirusTotal reports. Our findings underscore the MH-1M dataset's invaluable role in understanding the evolving landscape of malware. 

**Abstract (ZH)**: MH-1M：面向先进安卓恶意软件研究的最全面和最新的数据集 

---
# Towards Automated Petrography 

**Title (ZH)**: 面向自动岩相学的研究 

**Authors**: Isai Daniel Chacón, Paola Ruiz Puentes, Jillian Pearse, Pablo Arbeláez  

**Link**: [PDF](https://arxiv.org/pdf/2511.00328)  

**Abstract**: Petrography is a branch of geology that analyzes the mineralogical composition of rocks from microscopical thin section samples. It is essential for understanding rock properties across geology, archaeology, engineering, mineral exploration, and the oil industry. However, petrography is a labor-intensive task requiring experts to conduct detailed visual examinations of thin section samples through optical polarization microscopes, thus hampering scalability and highlighting the need for automated techniques. To address this challenge, we introduce the Large-scale Imaging and Thin section Optical-polarization Set (LITHOS), the largest and most diverse publicly available experimental framework for automated petrography. LITHOS includes 211,604 high-resolution RGB patches of polarized light and 105,802 expert-annotated grains across 25 mineral categories. Each annotation consists of the mineral class, spatial coordinates, and expert-defined major and minor axes represented as intersecting vector paths, capturing grain geometry and orientation. We evaluate multiple deep learning techniques for mineral classification in LITHOS and propose a dual-encoder transformer architecture that integrates both polarization modalities as a strong baseline for future reference. Our method consistently outperforms single-polarization models, demonstrating the value of polarization synergy in mineral classification. We have made the LITHOS Benchmark publicly available, comprising our dataset, code, and pretrained models, to foster reproducibility and further research in automated petrographic analysis. 

**Abstract (ZH)**: 岩石学是地质学的一个分支，通过显微薄片样本分析岩石的矿物组成。它对于理解地质学、考古学、工程学、矿产勘探以及石油工业中的岩石性质至关重要。然而，岩石学是一个劳动密集型任务，需要专家通过光学偏振显微镜进行详细的视觉检查，这限制了其可扩展性，并突显了需要自动技术的必要性。为解决这一挑战，我们推出了大规模成像和偏振光学显微镜数据集（LITHOS），这是目前规模最大、多样性最高的公开可用的自动岩石学实验框架。LITHOS 包含 211,604 个高分辨率 RGB 偏振光斑块和 105,802 个专家标注的颗粒，这些颗粒覆盖了 25 种矿物类别。每个标注包含矿物类别、空间坐标以及专家定义的主要轴和次轴，表示为相交的向量路径，捕捉颗粒的几何形状和方向。我们评估了多种深度学习技术在 LITHOS 中的矿物分类性能，并提出了一种双编码器变压器架构，该架构结合了偏振模态作为未来参考的强基准。我们的方法在矿物分类准确性上始终优于单偏振模型，证明了偏振数据协同作用的价值。我们已将 LITHOS 基准公开，包含我们的数据集、代码和预训练模型，以促进自动岩石学分析的可再现性和进一步研究。 

---
# POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation 

**Title (ZH)**: POSESTITCH-SLT：基于语言启发的姿势拼接端到端手语翻译 

**Authors**: Abhinav Joshi, Vaibhav Sharma, Sanjeet Singh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00270)  

**Abstract**: Sign language translation remains a challenging task due to the scarcity of large-scale, sentence-aligned datasets. Prior arts have focused on various feature extraction and architectural changes to support neural machine translation for sign languages. We propose POSESTITCH-SLT, a novel pre-training scheme that is inspired by linguistic-templates-based sentence generation technique. With translation comparison on two sign language datasets, How2Sign and iSign, we show that a simple transformer-based encoder-decoder architecture outperforms the prior art when considering template-generated sentence pairs in training. We achieve BLEU-4 score improvements from 1.97 to 4.56 on How2Sign and from 0.55 to 3.43 on iSign, surpassing prior state-of-the-art methods for pose-based gloss-free translation. The results demonstrate the effectiveness of template-driven synthetic supervision in low-resource sign language settings. 

**Abstract (ZH)**: 基于肢体模板的预训练方案在手语翻译中的应用 

---
# FedReplay: A Feature Replay Assisted Federated Transfer Learning Framework for Efficient and Privacy-Preserving Smart Agriculture 

**Title (ZH)**: FedReplay: 一种用于高效且隐私保护的智能农业联邦转移学习框架（基于特征重放辅助） 

**Authors**: Long Li, Jiajia Li, Dong Chen, Lina Pu, Haibo Yao, Yanbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00269)  

**Abstract**: Accurate classification plays a pivotal role in smart agriculture, enabling applications such as crop monitoring, fruit recognition, and pest detection. However, conventional centralized training often requires large-scale data collection, which raises privacy concerns, while standard federated learning struggles with non-independent and identically distributed (non-IID) data and incurs high communication costs. To address these challenges, we propose a federated learning framework that integrates a frozen Contrastive Language-Image Pre-training (CLIP) vision transformer (ViT) with a lightweight transformer classifier. By leveraging the strong feature extraction capability of the pre-trained CLIP ViT, the framework avoids training large-scale models from scratch and restricts federated updates to a compact classifier, thereby reducing transmission overhead significantly. Furthermore, to mitigate performance degradation caused by non-IID data distribution, a small subset (1%) of CLIP-extracted feature representations from all classes is shared across clients. These shared features are non-reversible to raw images, ensuring privacy preservation while aligning class representation across participants. Experimental results on agricultural classification tasks show that the proposed method achieve 86.6% accuracy, which is more than 4 times higher compared to baseline federated learning approaches. This demonstrates the effectiveness and efficiency of combining vision-language model features with federated learning for privacy-preserving and scalable agricultural intelligence. 

**Abstract (ZH)**: 准确分类在智能农业中起着关键作用，使作物监测、果实识别和害虫检测等应用成为可能。然而，传统的集中式训练通常需要大规模数据收集，这引发了隐私问题，而标准的联邦学习则难以处理非独立同分布（非-IID）数据，并且通信成本较高。为此，我们提出了一种结合冻结的预训练对比语言-图像变换器（CLIP）视觉变换器（ViT）和轻量级变换器分类器的联邦学习框架。通过利用预训练CLIP ViT的强大特征提取能力，该框架避免了从头训练大规模模型，并将联邦更新限制在紧凑的分类器上，从而显著减少了传输开销。此外，为了减轻由非-IID数据分布引起的表现下降，所有类别中提取的CLIP特征表示的小部分（1%）将在客户端之间共享。这些共享特征对原始图像不可逆，确保了隐私保护同时在参与者之间对齐类别表示。在农业分类任务上的实验结果表明，所提出的方法准确率为86.6%，比基线联邦学习方法高出4倍以上。这证明了将视觉-语言模型特征与联邦学习结合用于隐私保护和扩展的农业智能的有效性和效率。 

---
# IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval 

**Title (ZH)**: IL-PCSR: 前案和法律条文检索语料库 

**Authors**: Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00268)  

**Abstract**: Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance. 

**Abstract (ZH)**: 识别/检索与给定法律情况相关的法律法规和 precedents 是法律从业者常见的任务。研究人员到目前为止分别独立处理这两个任务，因而为每个任务开发了完全不同的数据集和模型；然而，这两个检索任务本质上是相关的，例如，相似的案例往往会引用相似的法律法规（由于类似的事实情况）。在本文中，我们填补了这一差距。我们提出了 IL-PCR（Indian Legal corpus for Prior Case and Statute Retrieval），这是一个独特的语料库，提供了同时开发两项任务（法律法规检索和 precedents 检索）模型的共同测试平台，这些模型可以利用两项任务之间的依赖性。我们在任务中广泛实验了包括词项模型、语义模型以及基于GNN的集成模型等多种基线模型。为进一步利用两项任务之间的依赖性，我们开发了一种基于LLM的重 ranking 方法，该方法性能最佳。 

---
# An Efficient and Generalizable Transfer Learning Method for Weather Condition Detection on Ground Terminals 

**Title (ZH)**: 一种高效且通用的气象条件检测传输学习方法 

**Authors**: Wenxuan Zhang, Peng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00211)  

**Abstract**: The increasing adoption of satellite Internet with low-Earth-orbit (LEO) satellites in mega-constellations allows ubiquitous connectivity to rural and remote areas. However, weather events have a significant impact on the performance and reliability of satellite Internet. Adverse weather events such as snow and rain can disturb the performance and operations of satellite Internet's essential ground terminal components, such as satellite antennas, significantly disrupting the space-ground link conditions between LEO satellites and ground stations. This challenge calls for not only region-based weather forecasts but also fine-grained detection capability on ground terminal components of fine-grained weather conditions. Such a capability can assist in fault diagnostics and mitigation for reliable satellite Internet, but its solutions are lacking, not to mention the effectiveness and generalization that are essential in real-world deployments. This paper discusses an efficient transfer learning (TL) method that can enable a ground component to locally detect representative weather-related conditions. The proposed method can detect snow, wet, and other conditions resulting from adverse and typical weather events and shows superior performance compared to the typical deep learning methods, such as YOLOv7, YOLOv9, Faster R-CNN, and R-YOLO. Our TL method also shows the advantage of being generalizable to various scenarios. 

**Abstract (ZH)**: 低 Earth 轨道卫星巨星座卫星互联网在农村和偏远地区的普及应用受到恶劣天气事件的显著影响：一种高效的迁移学习方法实现地端代表性天气条件的本地检测 

---
# Diffusion Models at the Drug Discovery Frontier: A Review on Generating Small Molecules versus Therapeutic Peptides 

**Title (ZH)**: 药物 discovery 前沿的扩散模型：生成小分子与治疗肽的综述 

**Authors**: Yiquan Wang, Yahui Ma, Yuhan Chang, Jiayao Yan, Jialin Zhang, Minnuo Cai, Kai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2511.00209)  

**Abstract**: Diffusion models have emerged as a leading framework in generative modeling, showing significant potential to accelerate and transform the traditionally slow and costly process of drug discovery. This review provides a systematic comparison of their application in designing two principal therapeutic modalities: small molecules and therapeutic peptides. We analyze how a unified framework of iterative denoising is adapted to the distinct molecular representations, chemical spaces, and design objectives of each modality. For small molecules, these models excel at structure-based design, generating novel, pocket-fitting ligands with desired physicochemical properties, yet face the critical hurdle of ensuring chemical synthesizability. Conversely, for therapeutic peptides, the focus shifts to generating functional sequences and designing de novo structures, where the primary challenges are achieving biological stability against proteolysis, ensuring proper folding, and minimizing immunogenicity. Despite these distinct challenges, both domains face shared hurdles: the need for more accurate scoring functions, the scarcity of high-quality experimental data, and the crucial requirement for experimental validation. We conclude that the full potential of diffusion models will be unlocked by bridging these modality-specific gaps and integrating them into automated, closed-loop Design-Build-Test-Learn (DBTL) platforms, thereby shifting the paradigm from chemical exploration to the targeted creation of novel therapeutics. 

**Abstract (ZH)**: 扩散模型已成为生成建模的领先框架，显示出显著的潜力，以加速和变革传统上缓慢和昂贵的药物发现过程。本文综述了它们在设计两种主要治疗模式：小分子和治疗性肽方面的应用。我们分析了统一的迭代去噪框架如何适应每种模式独特的分子表示、化学空间和设计目标。对于小分子，这些模型在基于结构的设计方面表现出色，生成具有所需物理化学性质的新型口袋配体，但面临确保化学可合成性的关键挑战。相反，对于治疗性肽，重点转向生成功能性序列和从头设计结构，其中主要挑战是实现生物稳定性以对抗酶解、确保正确的折叠并最大限度地减少免疫原性。尽管存在这些独特的挑战，两个领域都面临共同的障碍：需要更准确的评分函数、高质量实验数据的稀缺以及至关重要的实验验证需求。我们得出结论，扩散模型的全部潜力将通过弥合这些模式特定的差距并将其整合到自动化、闭环的“设计-构建-测试-学习”（DBTL）平台中而得以释放，从而将范式从化学探索转变为有针对性地创建新型治疗药物。 

---
# Generative Modeling Enables Molecular Structure Retrieval from Coulomb Explosion Imaging 

**Title (ZH)**: 生成建模使从库仑爆炸成像中检索分子结构成为可能 

**Authors**: Xiang Li, Till Jahnke, Rebecca Boll, Jiaqi Han, Minkai Xu, Michael Meyer, Maria Novella Piancastelli, Daniel Rolles, Artem Rudenko, Florian Trinter, Thomas J.A. Wolf, Jana B. Thayer, James P. Cryan, Stefano Ermon, Phay J. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2511.00179)  

**Abstract**: Capturing the structural changes that molecules undergo during chemical reactions in real space and time is a long-standing dream and an essential prerequisite for understanding and ultimately controlling femtochemistry. A key approach to tackle this challenging task is Coulomb explosion imaging, which benefited decisively from recently emerging high-repetition-rate X-ray free-electron laser sources. With this technique, information on the molecular structure is inferred from the momentum distributions of the ions produced by the rapid Coulomb explosion of molecules. Retrieving molecular structures from these distributions poses a highly non-linear inverse problem that remains unsolved for molecules consisting of more than a few atoms. Here, we address this challenge using a diffusion-based Transformer neural network. We show that the network reconstructs unknown molecular geometries from ion-momentum distributions with a mean absolute error below one Bohr radius, which is half the length of a typical chemical bond. 

**Abstract (ZH)**: 在实空间和时间中捕获分子在化学反应中经历的结构变化是长期以来的梦想，并且是理解最终控制飞秒化学的必要前提。克服这一挑战的关键方法是库仑爆炸成像，该方法受益于最近出现的高重复率X射线自由电子激光源。通过该技术，从分子的快速库仑爆炸产生的离子的动量分布中推断出分子结构信息。从这些分布中检索分子结构构成了一个高度非线性的逆问题，对于由更多于几个原子组成的分子，该问题尚未解决。我们使用基于扩散的Transformer神经网络来应对这一挑战。我们展示了该网络可以从离子动量分布中重建未知的分子几何结构，其平均绝对误差低于一个 Bohr 半径，即典型化学键长度的一半。 

---
# Feature Importance Guided Random Forest Learning with Simulated Annealing Based Hyperparameter Tuning 

**Title (ZH)**: 基于模拟退火基于超参数调优的特征重要性引导随机森林学习 

**Authors**: Kowshik Balasubramanian, Andre Williams, Ismail Butun  

**Link**: [PDF](https://arxiv.org/pdf/2511.00133)  

**Abstract**: This paper introduces a novel framework for enhancing Random Forest classifiers by integrating probabilistic feature sampling and hyperparameter tuning via Simulated Annealing. The proposed framework exhibits substantial advancements in predictive accuracy and generalization, adeptly tackling the multifaceted challenges of robust classification across diverse domains, including credit risk evaluation, anomaly detection in IoT ecosystems, early-stage medical diagnostics, and high-dimensional biological data analysis. To overcome the limitations of conventional Random Forests, we present an approach that places stronger emphasis on capturing the most relevant signals from data while enabling adaptive hyperparameter configuration. The model is guided towards features that contribute more meaningfully to classification and optimizing this with dynamic parameter tuning. The results demonstrate consistent accuracy improvements and meaningful insights into feature relevance, showcasing the efficacy of combining importance aware sampling and metaheuristic optimization. 

**Abstract (ZH)**: 本文提出了一种通过结合概率特征采样和使用模拟退火进行超参数调优来增强随机森林分类器的新框架。该提出的框架在预测准确性和泛化能力方面展示了显著的进步，有效地解决了跨多个领域（包括信用风险评估、物联网生态系统中的异常检测、医学早期诊断和高维生物数据分析）的稳健分类的复杂挑战。为了克服传统随机森林的局限性，我们提出了一种更强地关注从数据中捕捉最具相关性的信号、并允许自适应超参数配置的方法。该模型被引导关注对分类贡献更大的特征，并通过动态参数调整实现这一目标。结果表明，这种模型在准确性提升方面表现出一致的改进，并且能够揭示特征的相关性，展示了结合重要性感知采样和元启发式优化的有效性。 

---
# Casing Collar Identification using AlexNet-based Neural Networks for Depth Measurement in Oil and Gas Wells 

**Title (ZH)**: 基于AlexNet的神经网络在油气井深度测量中的套管 collar 识别 

**Authors**: Siyu Xiao, Xindi Zhao, Tianhao Mao, Yiwei Wang, Yuqiao Chen, Hongyun Zhang, Jian Wang, Junjie Wang, Shuang Liu, Tupei Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00129)  

**Abstract**: Accurate downhole depth measurement is essential for oil and gas well operations, directly influencing reservoir contact, production efficiency, and operational safety. Collar correlation using a casing collar locator (CCL) is fundamental for precise depth calibration. While neural network-based CCL signal recognition has achieved significant progress in collar identification, preprocessing methods for such applications remain underdeveloped. Moreover, the limited availability of real well data poses substantial challenges for training neural network models that require extensive datasets. This paper presents a system integrated into downhole tools for CCL signal acquisition to facilitate dataset construction. We propose comprehensive preprocessing methods for data augmentation and evaluate their effectiveness using our AlexNet-based neural network models. Through systematic experimentation across various configuration combinations, we analyze the contribution of each augmentation method. Results demonstrate that standardization, label distribution smoothing (LDS), and random cropping are fundamental requirements for model training, while label smoothing regularization (LSR), time scaling, and multiple sampling significantly enhance model generalization capability. The F1 scores of our two benchmark models trained with the proposed augmentation methods maximumly improve from 0.937 and 0.952 to 1.0 and 1.0, respectively. Performance validation on real CCL waveforms confirms the effectiveness and practical applicability of our approach. This work addresses the gaps in data augmentation methodologies for training casing collar recognition models in CCL data-limited environments. 

**Abstract (ZH)**: 准确的井下深度测量对于油气井作业至关重要，直接关系到储层接触、生产效率和操作安全。使用套管接箍定位器（CCL）进行接箍相关是精确深度校准的基础。虽然基于神经网络的CCL信号识别在接箍识别方面取得了显著进展，但这类应用的数据预处理方法仍然匮乏。此外，真实井数据的有限可用性给需要大量数据集训练的神经网络模型带来了重大挑战。本文提出了一种集成在井下工具中的系统，用于采集CCL信号，以促进数据集的构建。我们提出了全面的数据增强方法，并使用基于AlexNet的神经网络模型评估其有效性。通过系统实验分析各种配置组合，我们分析了每种增强方法的贡献。结果显示，标准化、标签分布平滑（LDS）和随机裁剪是模型训练的基本要求，而标签平滑正则化（LSR）、时间缩放和多次采样显著增强了模型的泛化能力。我们提出的两种基准模型经过增强方法训练后的F1分数分别从0.937和0.952最大化提高到1.0和1.0。对真实CCL波形的性能验证确认了我们方法的有效性和实际可应用性。本工作填补了在CCL数据受限环境下训练套管接箍识别模型的数据增强方法缺口。 

---
# Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features 

**Title (ZH)**: 基于成对排序和元特征的轨迹预测动态模型选择 

**Authors**: Lu Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00126)  

**Abstract**: Recent deep trajectory predictors (e.g., Jiang et al., 2023; Zhou et al., 2022) have achieved strong average accuracy but remain unreliable in complex long-tail driving scenarios. These limitations reveal the weakness of the prevailing "one-model-fits-all" paradigm, particularly in safety-critical urban contexts where simpler physics-based models can occasionally outperform advanced networks (Kalman, 1960). To bridge this gap, we propose a dynamic multi-expert gating framework that adaptively selects the most reliable trajectory predictor among a physics-informed LSTM, a Transformer, and a fine-tuned GameFormer on a per-sample basis.
Our method leverages internal model signals (meta-features) such as stability and uncertainty (Gal and Ghahramani, 2016), which we demonstrate to be substantially more informative than geometric scene descriptors. To the best of our knowledge, this is the first work to formulate trajectory expert selection as a pairwise-ranking problem over internal model signals (Burges et al., 2005), directly optimizing decision quality without requiring post-hoc calibration.
Evaluated on the nuPlan-mini dataset (Caesar et al., 2021) with 1,287 samples, our LLM-enhanced tri-expert gate achieves a Final Displacement Error (FDE) of 2.567 m, representing a 9.5 percent reduction over GameFormer (2.835 m), and realizes 57.8 percent of the oracle performance bound. In open-loop simulations, after trajectory horizon alignment, the same configuration reduces FDE on left-turn scenarios by approximately 10 percent, demonstrating consistent improvements across both offline validation and open-loop evaluation. These results indicate that adaptive hybrid systems enhance trajectory reliability in safety-critical autonomous driving, providing a practical pathway beyond static single-model paradigms. 

**Abstract (ZH)**: Recent深轨迹预测器（如Jiang等，2023；Zhou等，2022）在平均准确性上取得了显著效果，但在复杂长尾驾驶场景中仍然不可靠。这些限制揭示了现有“一模型适配所有”范式的弱点，特别是在安全性至关重要的城市环境中，基于物理的简单模型有时会优于先进的网络（Kalman，1960）。为弥合这一差距，我们提出了一种动态多专家门控框架，该框架在每个样本基础上适应性地选择最可靠的轨迹预测器，包括一个物理信息LSTM、一个Transformer和一个微调的GameFormer。

我们的方法利用了内部模型信号（元特征），如稳定性和不确定性（Gal和Ghahramani，2016），我们证明这些信号比几何场景描述符更具信息量。据我们所知，这是首次将轨迹专家选择公式化为内部模型信号的成对排名问题（Burges等，2005），直接优化决策质量，无需后续校准。

在nuPlan-mini数据集（Caesar等，2021）的1,287个样本上评估，我们的增强三专家门控机制实现了2.567米的最终位移误差（FDE），相比GameFormer（2.835米）减少了9.5%，达到了最优性能的57.8%。在开环仿真中，对齐轨迹时间范围后，相同的配置在左转场景中减少FDE约10%，表明在离线验证和开环评估中都实现了持续改进。这些结果表明，自适应混合系统能够增强安全性关键的自主驾驶中的轨迹可靠性，提供了一种超越静态单模型范式的实用途径。 

---
# Cross-fluctuation phase transitions reveal sampling dynamics in diffusion models 

**Title (ZH)**: 跨波动相转换揭示扩散模型中的采样动态 

**Authors**: Sai Niranjan Ramachandran, Manish Krishan Lal, Suvrit Sra  

**Link**: [PDF](https://arxiv.org/pdf/2511.00124)  

**Abstract**: We analyse how the sampling dynamics of distributions evolve in score-based diffusion models using cross-fluctuations, a centered-moment statistic from statistical physics. Specifically, we show that starting from an unbiased isotropic normal distribution, samples undergo sharp, discrete transitions, eventually forming distinct events of a desired distribution while progressively revealing finer structure. As this process is reversible, these transitions also occur in reverse, where intermediate states progressively merge, tracing a path back to the initial distribution. We demonstrate that these transitions can be detected as discontinuities in $n^{\text{th}}$-order cross-fluctuations. For variance-preserving SDEs, we derive a closed-form for these cross-fluctuations that is efficiently computable for the reverse trajectory. We find that detecting these transitions directly boosts sampling efficiency, accelerates class-conditional and rare-class generation, and improves two zero-shot tasks--image classification and style transfer--without expensive grid search or retraining. We also show that this viewpoint unifies classical coupling and mixing from finite Markov chains with continuous dynamics while extending to stochastic SDEs and non Markovian samplers. Our framework therefore bridges discrete Markov chain theory, phase analysis, and modern generative modeling. 

**Abstract (ZH)**: 我们使用统计物理中的中心矩统计量交叉波动来分析分数基于扩散模型中分布的采样动力学演化。具体而言，我们展示了从无偏等向正态分布开始，样本经历锐利的、离散的转换，最终形成所需分布的明显事件，同时逐步揭示更精细的结构。由于这一过程可逆，这些转换也会在反向进行，其中中间状态逐步合并，勾勒出一条回溯到初始分布的路径。我们证明了这些转换可以通过检测 nth 阶交叉波动中的不连续性来识别。对于方差保持的 SDE，我们推导出了这些交叉波动的闭式解，该解可以高效地计算反向轨迹。我们发现直接检测这些转换可以提升采样效率，加速条件类和稀类生成，并改进两种零样本任务—图像分类和风格转换，无需昂贵的网格搜索或重新训练。我们还展示了这种视角将经典的有限马尔可夫链耦合和混合统一并与连续动态相结合，并扩展到随机 SDE 和非马尔可夫采样器。因此，我们的框架将离散马尔可夫链理论、相分析和现代生成模型结合在了一起。 

---
# DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads 

**Title (ZH)**: DCcluster-Opt: 地理分布式数据中거工作负载的动态多目标优化基准测试 

**Authors**: Antonio Guillen-Perez, Avisek Naug, Vineet Gundecha, Sahand Ghorbanpour, Ricardo Luna Gutierrez, Ashwin Ramesh Babu, Munther Salim, Shubhanker Banerjee, Eoin H. Oude Essink, Damien Fay, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00117)  

**Abstract**: The increasing energy demands and carbon footprint of large-scale AI require intelligent workload management in globally distributed data centers. Yet progress is limited by the absence of benchmarks that realistically capture the interplay of time-varying environmental factors (grid carbon intensity, electricity prices, weather), detailed data center physics (CPUs, GPUs, memory, HVAC energy), and geo-distributed network dynamics (latency and transmission costs). To bridge this gap, we present DCcluster-Opt: an open-source, high-fidelity simulation benchmark for sustainable, geo-temporal task scheduling. DCcluster-Opt combines curated real-world datasets, including AI workload traces, grid carbon intensity, electricity markets, weather across 20 global regions, cloud transmission costs, and empirical network delay parameters with physics-informed models of data center operations, enabling rigorous and reproducible research in sustainable computing. It presents a challenging scheduling problem where a top-level coordinating agent must dynamically reassign or defer tasks that arrive with resource and service-level agreement requirements across a configurable cluster of data centers to optimize multiple objectives. The environment also models advanced components such as heat recovery. A modular reward system enables an explicit study of trade-offs among carbon emissions, energy costs, service level agreements, and water use. It provides a Gymnasium API with baseline controllers, including reinforcement learning and rule-based strategies, to support reproducible ML research and a fair comparison of diverse algorithms. By offering a realistic, configurable, and accessible testbed, DCcluster-Opt accelerates the development and validation of next-generation sustainable computing solutions for geo-distributed data centers. 

**Abstract (ZH)**: 全球分布数据中心可持续时空任务调度的开放源高保真模拟基准：DCcluster-Opt 

---
# Wayfinding through the AI wilderness: Mapping rhetorics of ChatGPT prompt writing on X (formerly Twitter) to promote critical AI literacies 

**Title (ZH)**: 通过AI荒野导航：将ChatGPT提示写作在X平台（原Twitter）上的修辞映射以促进批判性AI素养 

**Authors**: Anuj Gupta, Ann Shivers-McNair  

**Link**: [PDF](https://arxiv.org/pdf/2511.00106)  

**Abstract**: In this paper, we demonstrate how studying the rhetorics of ChatGPT prompt writing on social media can promote critical AI literacies. Prompt writing is the process of writing instructions for generative AI tools like ChatGPT to elicit desired outputs and there has been an upsurge of conversations about it on social media. To study this rhetorical activity, we build on four overlapping traditions of digital writing research in computers and composition that inform how we frame literacies, how we study social media rhetorics, how we engage iteratively and reflexively with methodologies and technologies, and how we blend computational methods with qualitative methods. Drawing on these four traditions, our paper shows our iterative research process through which we gathered and analyzed a dataset of 32,000 posts (formerly known as tweets) from X (formerly Twitter) about prompt writing posted between November 2022 to May 2023. We present five themes about these emerging AI literacy practices: (1) areas of communication impacted by prompt writing, (2) micro-literacy resources shared for prompt writing, (3) market rhetoric shaping prompt writing, (4) rhetorical characteristics of prompts, and (5) definitions of prompt writing. In discussing these themes and our methodologies, we highlight takeaways for digital writing teachers and researchers who are teaching and analyzing critical AI literacies. 

**Abstract (ZH)**: 在本文中，我们展示了如何研究ChatGPT提示写作在社交媒体中的修辞方式可以促进批判性人工智能素养。提示写作是为生成型人工智能工具如ChatGPT编写指令以获取所需输出的过程，社交媒体上关于这一主题的讨论日益增多。为了研究这一修辞活动，我们借鉴了计算机与写作领域中四种相互重叠的数字写作研究传统，这些传统指导我们如何界定素养，如何研究社交媒体修辞，如何迭代反思地与方法论和技术互动，以及如何将计算方法与定性方法相结合。基于这四种传统，我们的论文展示了我们从2022年11月至2023年5月收集并分析了来自X（原Twitter）平台上32,000个关于提示写作的帖子（原知会）的过程。我们提出了五个关于这些新兴人工智能实践的主题：（1）提示写作影响的沟通领域，（2）共享的微素养资源以用于提示写作，（3）塑造提示写作的市场修辞，（4）提示的修辞特征，以及（5）提示写作的定义。在讨论这些主题和方法论时，我们强调了数字写作教师和研究人员在教授和分析批判性人工智能素养方面的收获。 

---
# Artificial Intelligence in Elementary STEM Education: A Systematic Review of Current Applications and Future Challenges 

**Title (ZH)**: 人工智能在基础STEM教育中的应用：现有应用与未来挑战的系统评价 

**Authors**: Majid Memari, Krista Ruggles  

**Link**: [PDF](https://arxiv.org/pdf/2511.00105)  

**Abstract**: Artificial intelligence (AI) is transforming elementary STEM education, yet evidence remains fragmented. This systematic review synthesizes 258 studies (2020-2025) examining AI applications across eight categories: intelligent tutoring systems (45% of studies), learning analytics (18%), automated assessment (12%), computer vision (8%), educational robotics (7%), multimodal sensing (6%), AI-enhanced extended reality (XR) (4%), and adaptive content generation. The analysis shows that most studies focus on upper elementary grades (65%) and mathematics (38%), with limited cross-disciplinary STEM integration (15%). While conversational AI demonstrates moderate effectiveness (d = 0.45-0.70 where reported), only 34% of studies include standardized effect sizes. Eight major gaps limit real-world impact: fragmented ecosystems, developmental inappropriateness, infrastructure barriers, lack of privacy frameworks, weak STEM integration, equity disparities, teacher marginalization, and narrow assessment scopes. Geographic distribution is also uneven, with 90% of studies originating from North America, East Asia, and Europe. Future directions call for interoperable architectures that support authentic STEM integration, grade-appropriate design, privacy-preserving analytics, and teacher-centered implementations that enhance rather than replace human expertise. 

**Abstract (ZH)**: 人工智能（AI）正在transforming小学STEM教育，但证据依然支离破碎。本系统综述综合了2020-2025年间258项研究，这些研究探讨了AI在八个类别中的应用：智能辅导系统（45%的研究）、学习分析（18%）、自动化评估（12%）、计算机视觉（8%）、教育机器人（7%）、多模态感知（6%）、AI增强扩展现实（XR）（4%）和适应性内容生成。分析表明，大多数研究集中在高年级小学（65%）和数学（38%）上，跨学科STEM整合不足（15%）。虽然对话式AI显示出中等效果（当报告时d值为0.45-0.70），但只有34%的研究包括标准化效果大小。八个主要缺口限制了实际影响：碎片化的生态系统、发展不适宜、基础设施障碍、缺乏隐私框架、薄弱的STEM整合、公平差异、教师边缘化以及狭窄的评估范围。地域分布也不均匀，90%的研究源自北美、东亚和欧洲。未来方向需要支持真实STEM整合的可互操作架构、适合年级的设计、保护隐私的分析以及以教师为中心的实施方案，这些方案增强而非替代人类专长。 

---
# Automated Discovery of Conservation Laws via Hybrid Neural ODE-Transformers 

**Title (ZH)**: 自动发现 conservation laws 的混合神经 ODE-变换器方法 

**Authors**: Vivan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00102)  

**Abstract**: The discovery of conservation laws is a cornerstone of scientific progress. However, identifying these invariants from observational data remains a significant challenge. We propose a hybrid framework to automate the discovery of conserved quantities from noisy trajectory data. Our approach integrates three components: (1) a Neural Ordinary Differential Equation (Neural ODE) that learns a continuous model of the system's dynamics, (2) a Transformer that generates symbolic candidate invariants conditioned on the learned vector field, and (3) a symbolic-numeric verifier that provides a strong numerical certificate for the validity of these candidates. We test our framework on canonical physical systems and show that it significantly outperforms baselines that operate directly on trajectory data. This work demonstrates the robustness of a decoupled learn-then-search approach for discovering mathematical principles from imperfect data. 

**Abstract (ZH)**: 从噪声轨迹数据中自动化发现守恒律的一种混合框架 

---
# GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation 

**Title (ZH)**: GraphKeeper：通过知识解缠和保存实现图领域增量学习 

**Authors**: Zihao Guo, Qingyun Sun, Ziwei Zhang, Haonan Yuan, Huiping Zhuang, Xingcheng Fu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00097)  

**Abstract**: Graph incremental learning (GIL), which continuously updates graph models by sequential knowledge acquisition, has garnered significant interest recently. However, existing GIL approaches focus on task-incremental and class-incremental scenarios within a single domain. Graph domain-incremental learning (Domain-IL), aiming at updating models across multiple graph domains, has become critical with the development of graph foundation models (GFMs), but remains unexplored in the literature. In this paper, we propose Graph Domain-Incremental Learning via Knowledge Dientanglement and Preservation (GraphKeeper), to address catastrophic forgetting in Domain-IL scenario from the perspectives of embedding shifts and decision boundary deviations. Specifically, to prevent embedding shifts and confusion across incremental graph domains, we first propose the domain-specific parameter-efficient fine-tuning together with intra- and inter-domain disentanglement objectives. Consequently, to maintain a stable decision boundary, we introduce deviation-free knowledge preservation to continuously fit incremental domains. Additionally, for graphs with unobservable domains, we perform domain-aware distribution discrimination to obtain precise embeddings. Extensive experiments demonstrate the proposed GraphKeeper achieves state-of-the-art results with 6.5%~16.6% improvement over the runner-up with negligible forgetting. Moreover, we show GraphKeeper can be seamlessly integrated with various representative GFMs, highlighting its broad applicative potential. 

**Abstract (ZH)**: Graph域增量学习通过知识解纠缠与保存（GraphKeeper） 

---
# MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning 

**Title (ZH)**: MaGNet：一种基于时空因果和全局关系学习的Mamba双超图网络股票预测模型 

**Authors**: Peilin Tan, Chuanqi Shi, Dian Tu, Liang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.00085)  

**Abstract**: Stock trend prediction is crucial for profitable trading strategies and portfolio management yet remains challenging due to market volatility, complex temporal dynamics and multifaceted inter-stock relationships. Existing methods struggle to effectively capture temporal dependencies and dynamic inter-stock interactions, often neglecting cross-sectional market influences, relying on static correlations, employing uniform treatments of nodes and edges, and conflating diverse relationships. This work introduces MaGNet, a novel Mamba dual-hyperGraph Network for stock prediction, integrating three key innovations: (1) a MAGE block, which leverages bidirectional Mamba with adaptive gating mechanisms for contextual temporal modeling and integrates a sparse Mixture-of-Experts layer to enable dynamic adaptation to diverse market conditions, alongside multi-head attention for capturing global dependencies; (2) Feature-wise and Stock-wise 2D Spatiotemporal Attention modules enable precise fusion of multivariate features and cross-stock dependencies, effectively enhancing informativeness while preserving intrinsic data structures, bridging temporal modeling with relational reasoning; and (3) a dual hypergraph framework consisting of the Temporal-Causal Hypergraph (TCH) that captures fine-grained causal dependencies with temporal constraints, and Global Probabilistic Hypergraph (GPH) that models market-wide patterns through soft hyperedge assignments and Jensen-Shannon Divergence weighting mechanism, jointly disentangling localized temporal influences from instantaneous global structures for multi-scale relational learning. Extensive experiments on six major stock indices demonstrate MaGNet outperforms state-of-the-art methods in both superior predictive performance and exceptional investment returns with robust risk management capabilities. Codes available at: this https URL. 

**Abstract (ZH)**: 股市趋势预测对于盈利性交易策略和资产配置至关重要，但由于市场波动性、复杂的时序动态和多方面的股票间关系，这一任务仍然具有挑战性。现有方法难以有效地捕捉时序依赖性和动态的股票间互动，往往忽视横截面市场影响，依赖静态相关性，对节点和边采用统一处理，并混淆了不同的关系。本工作提出了MaGNet，这是一种用于股票预测的新颖Mamba双超图网络，综合了三项创新：（1）MAGE块，利用双向Mamba与自适应门控机制进行上下文时序建模，并集成一个稀疏Mixture-of-Experts层以实现对不同市场条件的动态适应，同时采用多头注意机制捕捉全局依赖；（2）特征维度和股票维度的二维时空注意模块能够精确融合多变量特征和跨股票依赖关系，有效增强信息量同时保留原始数据结构，将时序建模与关系推理相结合；（3）一种双超图框架，包括时序因果超图（TCH），捕捉具有时间约束的细粒度因果依赖关系，和全局概率超图（GPH），通过软超边分配和Jensen-Shannon散度加权机制建模市场范围内的模式，共同分离局部时间影响与瞬时全局结构，以实现多尺度关系学习。在六个主要股票指数上的广泛实验展示了MaGNet在卓越预测性能和异常投资回报率方面的优势，同时具备稳健的风险管理能力。代码地址：这个链接。 

---
# Application of predictive machine learning in pen & paper RPG game design 

**Title (ZH)**: 预测机器学习在笔纸角色扮演游戏设计中的应用 

**Authors**: Jolanta Śliwa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00084)  

**Abstract**: In recent years, the pen and paper RPG market has experienced significant growth. As a result, companies are increasingly exploring the integration of AI technologies to enhance player experience and gain a competitive edge.
One of the key challenges faced by publishers is designing new opponents and estimating their challenge level. Currently, there are no automated methods for determining a monster's level; the only approaches used are based on manual testing and expert evaluation. Although these manual methods can provide reasonably accurate estimates, they are time-consuming and resource-intensive.
Level prediction can be approached using ordinal regression techniques. This thesis presents an overview and evaluation of state-of-the-art methods for this task. It also details the construction of a dedicated dataset for level estimation. Furthermore, a human-inspired model was developed to serve as a benchmark, allowing comparison between machine learning algorithms and the approach typically employed by pen and paper RPG publishers. In addition, a specialized evaluation procedure, grounded in domain knowledge, was designed to assess model performance and facilitate meaningful comparisons. 

**Abstract (ZH)**: 近年来，纸上角色扮演（RPG）市场经历了显著增长。因此，公司越来越多地探索AI技术的集成，以提升玩家体验并获得竞争优势。
其中，出版商面临的一个关键挑战是设计新的对手并估计其难度等级。目前，尚无自动化方法来确定怪兽的等级；现有的方法仅基于手动测试和专家评估。尽管这些手动方法可以提供相对准确的估计，但却是耗时且资源密集型的。
等级预测可以使用序数回归技术来实现。本文综述并评估了该任务的先进方法，并详细描述了一个专门用于等级估计的数据集构建。此外，还开发了一个基于人类启发的方法作为基准，以比较机器学习算法与纸上角色扮演（RPG）出版商通常使用的方法。同时，还设计了一种基于领域知识的专业评估程序，以评估模型性能并促进有意义的比较。 

---
# Fixed-point graph convolutional networks against adversarial attacks 

**Title (ZH)**: 固定点图卷积网络对抗 adversarial 攻击 

**Authors**: Shakib Khan, A. Ben Hamza, Amr Youssef  

**Link**: [PDF](https://arxiv.org/pdf/2511.00083)  

**Abstract**: Adversarial attacks present a significant risk to the integrity and performance of graph neural networks, particularly in tasks where graph structure and node features are vulnerable to manipulation. In this paper, we present a novel model, called fixed-point iterative graph convolutional network (Fix-GCN), which achieves robustness against adversarial perturbations by effectively capturing higher-order node neighborhood information in the graph without additional memory or computational complexity. Specifically, we introduce a versatile spectral modulation filter and derive the feature propagation rule of our model using fixed-point iteration. Unlike traditional defense mechanisms that rely on additional design elements to counteract attacks, the proposed graph filter provides a flexible-pass filtering approach, allowing it to selectively attenuate high-frequency components while preserving low-frequency structural information in the graph signal. By iteratively updating node representations, our model offers a flexible and efficient framework for preserving essential graph information while mitigating the impact of adversarial manipulation. We demonstrate the effectiveness of the proposed model through extensive experiments on various benchmark graph datasets, showcasing its resilience against adversarial attacks. 

**Abstract (ZH)**: 对抗攻击对图神经网络的完整性和性能构成重大风险，尤其是在图结构和节点特征易受操纵的任务中。本文提出了一种名为固定点迭代图卷积网络（Fix-GCN）的新型模型，该模型通过有效地捕获图中的高级节点邻域信息来抵御对抗性扰动，而无需增加额外的内存或计算复杂度。具体而言，我们引入了灵活的频谱调制滤波器，并使用固定点迭代推导了模型的特征传播规则。与传统的依赖于额外设计元素来对抗攻击的防御机制不同，所提出的图滤波器提供了一种灵活的通过过滤方法，能够选择性地衰减高频分量同时保留图信号中的低频结构性信息。通过迭代更新节点表示，该模型提供了一种灵活且高效的框架，可在减轻对抗性操纵影响的同时保留关键的图信息。我们通过在各种基准图数据集上的大量实验展示了所提模型的有效性，证明了其对对抗攻击的鲁棒性。 

---
# RailEstate: An Interactive System for Metro Linked Property Trends 

**Title (ZH)**: RailEstate：一种基于地铁线路的房产趋势交互系统 

**Authors**: Chen-Wei Chang, Yu-Chieh Cheng, Yun-En Tsai, Fanglan Chen, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00078)  

**Abstract**: Access to metro systems plays a critical role in shaping urban housing markets by enhancing neighborhood accessibility and driving property demand. We present RailEstate, a novel web based system that integrates spatial analytics, natural language interfaces, and interactive forecasting to analyze how proximity to metro stations influences residential property prices in the Washington metropolitan area. Unlike static mapping tools or generic listing platforms, RailEstate combines 25 years of historical housing data with transit infrastructure to support low latency geospatial queries, time series visualizations, and predictive modeling. Users can interactively explore ZIP code level price patterns, investigate long term trends, and forecast future housing values around any metro station. A key innovation is our natural language chatbot, which translates plain-English questions e.g., What is the highest price in Falls Church in the year 2000? into executable SQL over a spatial database. This unified and interactive platform empowers urban planners, investors, and residents to derive actionable insights from metro linked housing data without requiring technical expertise. 

**Abstract (ZH)**: 地铁系统接入对城市住房市场产生关键影响，通过提升邻里可达性和推动房地产需求。我们推出了RailEstate，一个结合空间分析、自然语言界面和互动预测的新一代网页系统，用于分析地铁站 proximity 对华盛顿大都市区住宅房产价格的影响。不同于静态地图工具或通用房源平台，RailEstate 结合了 25 年的历史住房数据和交通基础设施，支持低延迟地理空间查询、时间序列可视化和预测建模。用户可以交互式地探索 ZIP 码级别的价格模式，调查长期趋势，并预测任何地铁站周围的未来住房价值。一项关键创新是我们的自然语言聊天机器人，它可以将简单的英文问题，如 2000 年 Falls Church 的最高价格是多少？转化为对时空数据库的可执行 SQL 查询。这一统一且互动的平台使城市规划者、投资者和居民能够从与地铁相关的住房数据中提取 actionable 瞳识，无需具备技术专长。 

---
# Latent Domain Prompt Learning for Vision-Language Models 

**Title (ZH)**: 潜类别领域提示学习用于 vision-language 模型 

**Authors**: Zhixing Li, Arsham Gholamzadeh Khoee, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00067)  

**Abstract**: The objective of domain generalization (DG) is to enable models to be robust against domain shift. DG is crucial for deploying vision-language models (VLMs) in real-world applications, yet most existing methods rely on domain labels that may not be available and often ambiguous. We instead study the DG setting where models must generalize well without access to explicit domain labels. Our key idea is to represent an unseen target domain as a combination of latent domains automatically discovered from training data, enabling the model to adaptively transfer knowledge across domains. To realize this, we perform latent domain clustering on image features and fuse domain-specific text features based on the similarity between the input image and each latent domain. Experiments on four benchmarks show that this strategy yields consistent gains over VLM-based baselines and provides new insights into improving robustness under domain shift. 

**Abstract (ZH)**: 域适应（Domain Generalization）的目的是使模型能够抵御域移变的影响。域适应对于在实际应用场景中部署视觉-语言模型（VLMs）至关重要，但现有大多数方法依赖于可能不可用且经常模糊的域标签。相反，我们研究的是模型在无法访问明确域标签的情况下仍能良好泛化的域适应设置。我们的核心思想是将未见过的目标域表示为从训练数据中自动发现的潜在域的组合，从而使模型能够自适应地在域之间转移知识。为了实现这一点，我们在图像特征上进行潜在域聚类，并根据输入图像与每个潜在域之间的相似性融合特定域的文本特征。在四个基准上的实验表明，这种策略在基于VLM的基线方法上提供了持续的改进，并为提高在域移变下的鲁棒性提供了新的见解。 

---
# Automatically Finding Rule-Based Neurons in OthelloGPT 

**Title (ZH)**: 自动发现基于规则的神经元在OthelloGPT中 

**Authors**: Aditya Singh, Zihang Wen, Srujananjali Medicherla, Adam Karvonen, Can Rager  

**Link**: [PDF](https://arxiv.org/pdf/2511.00059)  

**Abstract**: OthelloGPT, a transformer trained to predict valid moves in Othello, provides an ideal testbed for interpretability research. The model is complex enough to exhibit rich computational patterns, yet grounded in rule-based game logic that enables meaningful reverse-engineering. We present an automated approach based on decision trees to identify and interpret MLP neurons that encode rule-based game logic. Our method trains regression decision trees to map board states to neuron activations, then extracts decision paths where neurons are highly active to convert them into human-readable logical forms. These descriptions reveal highly interpretable patterns; for instance, neurons that specifically detect when diagonal moves become legal. Our findings suggest that roughly half of the neurons in layer 5 can be accurately described by compact, rule-based decision trees ($R^2 > 0.7$ for 913 of 2,048 neurons), while the remainder likely participate in more distributed or non-rule-based computations. We verify the causal relevance of patterns identified by our decision trees through targeted interventions. For a specific square, for specific game patterns, we ablate neurons corresponding to those patterns and find an approximately 5-10 fold stronger degradation in the model's ability to predict legal moves along those patterns compared to control patterns. To facilitate future work, we provide a Python tool that maps rule-based game behaviors to their implementing neurons, serving as a resource for researchers to test whether their interpretability methods recover meaningful computational structures. 

**Abstract (ZH)**: OthelloGPT：一种用于预测奥赛罗棋有效走法的变换器模型，提供了一种理想的解释性研究测试平台。基于决策树的自动化方法识别并解释编码规则性游戏逻辑的MLP神经元。方法通过训练回归决策树将棋盘状态映射到神经元激活，并提取高度活跃的决策路径，将其转换为可读的逻辑形式。这些描述揭示了高度可解释的模式，例如专门检测对角走法何时合法的神经元。研究发现，大约一半的第5层神经元可以由紧凑的规则性决策树准确描述（2048个神经元中有913个的$R^2 > 0.7$），而其余的神经元可能参与更分布式或非规则性的计算。我们通过有针对性的干预验证了通过决策树识别的模式的因果相关性。提供了用于映射规则性游戏行为及其实现神经元的Python工具，作为研究人员测试其解释性方法是否恢复有意义的计算结构的资源。 

---
# Exploring Federated Learning for Thermal Urban Feature Segmentation -- A Comparison of Centralized and Decentralized Approaches 

**Title (ZH)**: 探索联邦学习在热城市特征分割中的应用——集中式与去中心化方法的比较 

**Authors**: Leonhard Duda, Khadijeh Alibabaei, Elena Vollmer, Leon Klug, Valentin Kozlov, Lisana Berberi, Mishal Benz, Rebekka Volk, Juan Pedro Gutiérrez Hermosillo Muriedas, Markus Götz, Judith Sáínz-Pardo Díaz, Álvaro López García, Frank Schultmann, Achim Streit  

**Link**: [PDF](https://arxiv.org/pdf/2511.00055)  

**Abstract**: Federated Learning (FL) is an approach for training a shared Machine Learning (ML) model with distributed training data and multiple participants. FL allows bypassing limitations of the traditional Centralized Machine Learning CL if data cannot be shared or stored centrally due to privacy or technical restrictions -- the participants train the model locally with their training data and do not need to share it among the other participants. This paper investigates the practical implementation and effectiveness of FL in a real-world scenario, specifically focusing on unmanned aerial vehicle (UAV)-based thermal images for common thermal feature detection in urban environments. The distributed nature of the data arises naturally and makes it suitable for FL applications, as images captured in two German cities are available. This application presents unique challenges due to non-identical distribution and feature characteristics of data captured at both locations. The study makes several key contributions by evaluating FL algorithms in real deployment scenarios rather than simulation. We compare several FL approaches with a centralized learning baseline across key performance metrics such as model accuracy, training time, communication overhead, and energy usage. This paper also explores various FL workflows, comparing client-controlled workflows and server-controlled workflows. The findings of this work serve as a valuable reference for understanding the practical application and limitations of the FL methods in segmentation tasks in UAV-based imaging. 

**Abstract (ZH)**: 联邦学习（FL）是一种在分布式训练数据和多个参与者之间训练共享机器学习（ML）模型的方法。FL允许在由于隐私或技术限制无法共享或集中存储数据的情况下绕过传统集中式机器学习（CL）的局限性——参与者可以在本地使用其训练数据训练模型，无需与其他参与者分享数据。本文在真实世界场景中探讨了FL的实用实现和有效性，特别是针对基于无人机（UAV）的热成像在城市环境中进行常见热特征检测的应用。由于数据的分布式特性自然产生，使得该应用非常适合FL应用，因为在两个德国城市收集的图像数据可供使用。这一应用因两个采集地点捕获的数据分布和特征特性不一致而具有独特的挑战性。研究通过在真实部署场景中评估FL算法而非仿真来做出多项关键贡献。我们根据模型准确度、训练时间、通信开销和能耗等关键性能指标，比较了几种FL方法和集中式学习基准。此外，本文还探讨了各种FL工作流，比较了客户端控制的工作流和服务器控制的工作流。本研究的发现为理解在无人机成像中分割任务中的FL方法的实际应用及其局限性提供了宝贵的参考。 

---
# SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation 

**Title (ZH)**: SpatialTraceGen: 高保真轨迹以提高VLM空间推理提炼效率 

**Authors**: Gio Huh, Dhruv Sheth, Rayhan Zirvi, Frank Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00054)  

**Abstract**: While Vision-Language Models (VLMs) excel in many areas, they struggle with complex spatial reasoning, which requires problem decomposition and strategic tool use. Fine-tuning smaller, more deployable models offers an efficient path to strong performance, but this is hampered by a major bottleneck: the absence of high-quality, step-by-step reasoning data. To address this data-efficiency gap, we introduce SpatialTraceGen, a framework to distill the reasoning processes of a large teacher model into a high-quality dataset of multi-hop, multi-tool reasoning traces. A key innovation is our automated Verifier, which scalably ensures the fidelity of each reasoning step, providing a cost-effective alternative to manual human annotation. On the CLEVR-Humans benchmark, this verifier-guided process improves the average quality score of traces by 17\% while reducing quality variance by over 40\%. SpatialTraceGen delivers a dataset of expert traces, providing the structured, step-by-step examples of tool use necessary for effective fine-tuning and sample-efficient offline reinforcement learning. 

**Abstract (ZH)**: 视觉-语言模型在视觉-语言推理中的空间 reasoning 数据生成框架：SpatialTraceGen 

---
# Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models 

**Title (ZH)**: 二次直接预测训练多步时间序列预测模型 

**Authors**: Hao Wang, Licheng Pan, Yuan Lu, Zhichao Chen, Tianqiao Liu, Shuting He, Zhixuan Chu, Qingsong Wen, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00053)  

**Abstract**: The design of training objective is central to training time-series forecasting models. Existing training objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which we found leading to the following two issues: (1) overlook the label autocorrelation effect among future steps, leading to biased training objective; (2) fail to set heterogeneous task weights for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted training objective, addressing both of the issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the most preferable weights of the forecasting tasks with varying future steps. To achieve this, we propose a Quadratic Direct Forecast (QDF) learning algorithm, which trains the forecast model using the adaptively updated quadratic-form weighting matrix. Experiments show that our QDF effectively improves performance of various forecast models, achieving state-of-the-art results. Code is available at this https URL. 

**Abstract (ZH)**: 时间序列预测模型训练目标的设计是训练的核心。现有的训练目标，如均方误差，通常将每个未来步骤视为独立且等权重的任务，这会导致以下两个问题：（1）忽视未来步骤之间的标签自相关效应，导致训练目标偏差；（2）无法为不同的预测任务设置异质任务权重，限制了预测性能。为了解决这些问题，我们提出了一种新颖的二次形式加权训练目标，同时解决了这两个问题。具体而言，加权矩阵的非对角元素考虑了标签自相关效应，而非均匀的对角线元素则期望匹配不同未来步骤的预测任务的最适权重。为此，我们提出了二次直接预测（QDF）学习算法，该算法使用自适应更新的二次形式加权矩阵来训练预测模型。实验表明，我们的QDF显著提高了各种预测模型的性能，达到了最先进的效果。代码可在以下链接获取：this https URL。 

---
# Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT 

**Title (ZH)**: 校准与旋转：PEFT 中权重调整的统一框架 

**Authors**: Da Chang, Peng Xue, Yu Li, Yongxiang Liu, Pengxiang Xu, Shixun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00051)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods are crucial for adapting large pre-trained models. Among these, LoRA is considered a foundational approach. Building on this, the influential DoRA method enhances performance by decomposing weight updates into magnitude and direction. However, its underlying mechanism remains unclear, and it introduces significant computational overhead. In this work, we first identify that DoRA's success stems from its capacity to increase the singular value entropy of the weight update matrix, which promotes a more uniform update distribution akin to full fine-tuning. We then reformulate DoRA into a mathematically equivalent and more efficient matrix form, revealing it as a learnable weight conditioning method. Based on this insight, we propose a unified framework for designing advanced PEFT methods by exploring two orthogonal dimensions: the architectural placement and the transformation type of the conditioning matrix. Within this framework, we introduce two novel methods: (1) \textbf{Pre-Diag}, which applies a diagonal conditioning matrix before the LoRA update to efficiently calibrate the pre-trained weights, thereby enhancing performance while reducing training time; and (2) \textbf{S}kewed \textbf{O}rthogonal \textbf{R}otation \textbf{A}daptation (\textbf{SORA}), which employs a parameter-efficient orthogonal rotation to perform a more powerful, norm-preserving transformation of the feature space. Extensive experiments on natural language understanding and generation tasks demonstrate that our proposed methods achieve superior performance and efficiency compared to both LoRA and DoRA. The code is available at this https URL. 

**Abstract (ZH)**: 基于参数高效微调的参数高效细调（PEFT）方法对于适配大规模预训练模型至关重要。其中，LoRA 被认为是一种基础方法。在此基础上，有影响力的DoRA方法通过将权重更新分解为幅度和方向来提升性能，但其工作机制尚不清晰，并引入了显著的计算开销。在本工作中，我们首先发现DoRA的成功在于其能够增加权重更新矩阵的奇异值熵，从而促进更均匀的更新分布，类似于全面微调。然后，我们将DoRA重新形式化为一个等效且更高效的矩阵形式，揭示其为可学习的权重调整方法。基于此见解，我们提出了一种综合框架，通过探索可调整微调方法设计的两个正交维度——结构调整和条件矩阵变换类型来设计先进的PEFT方法。在这一框架内，我们提出了两种新的方法：(1) \textbf{Pre-Diag}，在LoRA更新之前应用一个对角线条件矩阵，以高效校准预训练权重，从而提高性能并减少训练时间；(2) \textbf{S}kewed \textbf{O}rthogonal \textbf{R}otation \textbf{A}daptation (\textbf{SORA})，采用参数高效的正交旋转，对特征空间进行更强大且范数保持的变换。在自然语言理解和生成任务上的广泛实验表明，我们提出的方法在性能和效率方面优于LoRA和DoRA。代码可在以下链接获得：this https URL。 

---
# Adaptive Spatio-Temporal Graphs with Self-Supervised Pretraining for Multi-Horizon Weather Forecasting 

**Title (ZH)**: 自监督预训练的自适应时空图用于多 horizons 天气预报 

**Authors**: Yao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00049)  

**Abstract**: Accurate and robust weather forecasting remains a fundamental challenge due to the inherent spatio-temporal complexity of atmospheric systems. In this paper, we propose a novel self-supervised learning framework that leverages spatio-temporal structures to improve multi-variable weather prediction. The model integrates a graph neural network (GNN) for spatial reasoning, a self-supervised pretraining scheme for representation learning, and a spatio-temporal adaptation mechanism to enhance generalization across varying forecasting horizons. Extensive experiments on both ERA5 and MERRA-2 reanalysis datasets demonstrate that our approach achieves superior performance compared to traditional numerical weather prediction (NWP) models and recent deep learning methods. Quantitative evaluations and visual analyses in Beijing and Shanghai confirm the model's capability to capture fine-grained meteorological patterns. The proposed framework provides a scalable and label-efficient solution for future data-driven weather forecasting systems. 

**Abstract (ZH)**: 准确且稳健的天气预报依然是一个基本挑战，由于大气系统的固有时空复杂性所致。本文提出了一种新颖的自我监督学习框架，利用时空结构以提高多变量天气预测的准确性。该模型整合了图神经网络（GNN）进行空间推理、自我监督预训练方案进行表示学习，以及时空适应机制以增强不同预报时效的泛化能力。在ERA5和MERRA-2再分析数据集上的广泛实验表明，本方法在传统数值天气预报（NWP）模型和近期深度学习方法中表现出更优性能。在北京市和上海市的定量评估和可视化分析中，证实了该模型捕捉细粒度气象模式的能力。提出的框架为未来数据驱动的天气预报系统提供了可扩展且标签高效的解决方案。 

---
# DynBERG: Dynamic BERT-based Graph neural network for financial fraud detection 

**Title (ZH)**: DynBERG: 基于动态BERT的图神经网络在金融欺诈检测中的应用 

**Authors**: Omkar Kulkarni, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2511.00047)  

**Abstract**: Financial fraud detection is critical for maintaining the integrity of financial systems, particularly in decentralised environments such as cryptocurrency networks. Although Graph Convolutional Networks (GCNs) are widely used for financial fraud detection, graph Transformer models such as Graph-BERT are gaining prominence due to their Transformer-based architecture, which mitigates issues such as over-smoothing. Graph-BERT is designed for static graphs and primarily evaluated on citation networks with undirected edges. However, financial transaction networks are inherently dynamic, with evolving structures and directed edges representing the flow of money. To address these challenges, we introduce DynBERG, a novel architecture that integrates Graph-BERT with a Gated Recurrent Unit (GRU) layer to capture temporal evolution over multiple time steps. Additionally, we modify the underlying algorithm to support directed edges, making DynBERG well-suited for dynamic financial transaction analysis. We evaluate our model on the Elliptic dataset, which includes Bitcoin transactions, including all transactions during a major cryptocurrency market event, the Dark Market Shutdown. By assessing DynBERG's resilience before and after this event, we analyse its ability to adapt to significant market shifts that impact transaction behaviours. Our model is benchmarked against state-of-the-art dynamic graph classification approaches, such as EvolveGCN and GCN, demonstrating superior performance, outperforming EvolveGCN before the market shutdown and surpassing GCN after the event. Additionally, an ablation study highlights the critical role of incorporating a time-series deep learning component, showcasing the effectiveness of GRU in modelling the temporal dynamics of financial transactions. 

**Abstract (ZH)**: 动态金融交易欺诈检测中基于图Transformer的新型架构DynBERG研究 

---
# Semi-Supervised Preference Optimization with Limited Feedback 

**Title (ZH)**: 半监督偏好优化在有限反馈情况下 

**Authors**: Seonggyun Lee, Sungjun Lim, Seojin Park, Soeun Cheon, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.00040)  

**Abstract**: The field of preference optimization has made outstanding contributions to the alignment of language models with human preferences. Despite these advancements, recent methods still rely heavily on substantial paired (labeled) feedback data, leading to substantial resource expenditures. To address these challenges, we study the problem of Semi-Supervised Preference Optimization (SSPO) in which the idea is to learn from both a small number of pairwise preference labels and a large pool of unpaired samples simultaneously. Our key theoretical contribution proves the existence of an optimal reward threshold capable of separating winning and losing responses with high probability, which enables a principled pseudo-labeling of unpaired data. By leveraging these pseudo-labels, SSPO effectively distills latent preferences from large-scale unpaired data, thus maintaining human alignment while drastically reducing acquisition costs. Extensive experiments across datasets validate this remarkable data efficiency; for instance, SSPO trained with Llama3-8B-Instruct on just 1% of UltraFeedback consistently surpasses strong baselines trained on 10% of UltraFeedback. 

**Abstract (ZH)**: 半监督偏好优化（SSPO） 

---
# From Uniform to Adaptive: General Skip-Block Mechanisms for Efficient PDE Neural Operators 

**Title (ZH)**: 从统一到自适应：高效偏微分方程神经算子的通用跳过块机制 

**Authors**: Lei Liu, Zhongyi Yu, Hong Wang, Huanshuo Dong, Haiyang Xin, Hongwei Zhao, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00032)  

**Abstract**: In recent years, Neural Operators(NO) have gradually emerged as a popular approach for solving Partial Differential Equations (PDEs). However, their application to large-scale engineering tasks suffers from significant computational overhead. And the fact that current models impose a uniform computational cost while physical fields exhibit vastly different complexities constitutes a fundamental mismatch, which is the root of this inefficiency. For instance, in turbulence flows, intricate vortex regions require deeper network processing compared to stable flows. To address this, we introduce a framework: Skip-Block Routing (SBR), a general framework designed for Transformer-based neural operators, capable of being integrated into their multi-layer architectures. First, SBR uses a routing mechanism to learn the complexity and ranking of tokens, which is then applied during inference. Then, in later layers, it decides how many tokens are passed forward based on this ranking. This way, the model focuses more processing capacity on the tokens that are more complex. Experiments demonstrate that SBR is a general framework that seamlessly integrates into various neural operators. Our method reduces computational cost by approximately 50% in terms of Floating Point Operations (FLOPs), while still delivering up to 2x faster inference without sacrificing accuracy. 

**Abstract (ZH)**: 基于Transformer的神经运算器中的一种Skip-Block Routing框架：提高偏微分方程求解效率的同时减少计算成本 

---
# Mutual Information guided Visual Contrastive Learning 

**Title (ZH)**: 互信息引导的视觉对比学习 

**Authors**: Hanyang Chen, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00028)  

**Abstract**: Representation learning methods utilizing the InfoNCE loss have demonstrated considerable capacity in reducing human annotation effort by training invariant neural feature extractors. Although different variants of the training objective adhere to the information maximization principle between the data and learned features, data selection and augmentation still rely on human hypotheses or engineering, which may be suboptimal. For instance, data augmentation in contrastive learning primarily focuses on color jittering, aiming to emulate real-world illumination changes. In this work, we investigate the potential of selecting training data based on their mutual information computed from real-world distributions, which, in principle, should endow the learned features with better generalization when applied in open environments. Specifically, we consider patches attached to scenes that exhibit high mutual information under natural perturbations, such as color changes and motion, as positive samples for learning with contrastive loss. We evaluate the proposed mutual-information-informed data augmentation method on several benchmarks across multiple state-of-the-art representation learning frameworks, demonstrating its effectiveness and establishing it as a promising direction for future research. 

**Abstract (ZH)**: 基于现实世界互信息选择训练数据的方法在对比损失下的 Representation 学习中具有潜在优势：一种新的数据增强方向 

---
# Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All? 

**Title (ZH)**: 立场论文：如果人工智能的创新系统性地违背了基本权利，那还是创新吗？ 

**Authors**: Josu Eguiluz Castañeira, Axel Brando, Migle Laukyte, Marc Serra-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2511.00027)  

**Abstract**: Artificial intelligence (AI) now permeates critical infrastructures and decision-making systems where failures produce social, economic, and democratic harm. This position paper challenges the entrenched belief that regulation and innovation are opposites. As evidenced by analogies from aviation, pharmaceuticals, and welfare systems and recent cases of synthetic misinformation, bias and unaccountable decision-making, the absence of well-designed regulation has already created immeasurable damage. Regulation, when thoughtful and adaptive, is not a brake on innovation--it is its foundation. The present position paper examines the EU AI Act as a model of risk-based, responsibility-driven regulation that addresses the Collingridge Dilemma: acting early enough to prevent harm, yet flexibly enough to sustain innovation. Its adaptive mechanisms--regulatory sandboxes, small and medium enterprises (SMEs) support, real-world testing, fundamental rights impact assessment (FRIA) -- demonstrate how regulation can accelerate responsibly, rather than delay, technological progress. The position paper summarises how governance tools transform perceived burdens into tangible advantages: legal certainty, consumer trust, and ethical competitiveness. Ultimately, the paper reframes progress: innovation and regulation advance together. By embedding transparency, impact assessments, accountability, and AI literacy into design and deployment, the EU framework defines what responsible innovation truly means--technological ambition disciplined by democratic values and fundamental rights. 

**Abstract (ZH)**: 人工智能（AI）现在渗透到关键基础设施和决策系统中，失败会导致社会、经济和民主层面的损害。本文挑战了监管与创新水火不容的既有认知。正如航空、制药和福利系统中的先例以及合成错误信息、偏见和不可问责决策的最新案例所示，缺乏周到设计的监管已经造成了无法估量的损害。当监管富有远见且具有适应性时，它不是创新的阻碍，而是其基础。本文检视欧盟AI法案作为基于风险、以负责任的态度驱动的监管模型，它解决了考林дей尔难题：在早于风险发生时采取行动以防止损害，同时保持足够的灵活性以支持创新。其适应性机制——监管沙盒、中小企业（SMEs）支持、实际测试、基本权利影响评估（FRIA）——展示了监管是如何能够促进负责任的技术进步，而不是拖延。本文总结了治理工具如何将感知中的负担转化为实际的优势：法律法规的确定性、消费者信任以及道德竞争力。最终，本文重新定义了进步：创新与监管同步发展。通过将透明性、影响评估、问责制和AI素养融入设计和部署中，欧盟框架定义了负责任创新的真正含义——技术雄心与民主价值和基本权利相结合。 

---
# Sorting by Strip Swaps is NP-Hard 

**Title (ZH)**: 条形交换排序是NP难问题 

**Authors**: Swapnoneel Roy, Asai Asaithambi, Debajyoti Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2511.00015)  

**Abstract**: We show that \emph{Sorting by Strip Swaps} (SbSS) is NP-hard by a polynomial reduction of \emph{Block Sorting}. The key idea is a local gadget, a \emph{cage}, that replaces every decreasing adjacency $(a_i,a_{i+1})$ by a guarded triple $a_i,m_i,a_{i+1}$ enclosed by guards $L_i,U_i$, so the only decreasing adjacencies are the two inside the cage. Small \emph{hinge} gadgets couple adjacent cages that share an element and enforce that a strip swap that removes exactly two adjacencies corresponds bijectively to a block move that removes exactly one decreasing adjacency in the source permutation. This yields a clean equivalence between exact SbSS schedules and perfect block schedules, establishing NP-hardness. 

**Abstract (ZH)**: 我们通过将Block Sorting问题转化为多项式时间归约来证明Strip Swaps排序问题是NP-hard的。关键思想是一种局部构件“笼子”，它用受保护的三元组$a_i,m_i,a_{i+1}$（由守卫$L_i,U_i$保护）来替换每一个递减邻接关系$(a_i,a_{i+1})$，使得仅在“笼子”内部存在递减邻接关系。小型铰链构件连接共享元素的相邻“笼子”，并确保一次去除两个邻接关系的条带交换与源排列中去除一个递减邻接关系的块移动之间存在一一对应关系。这建立了精确的Strip Swaps调度与完美块调度之间的清洁等价性，从而证明了NP-hard性。 

---
# VRScout: Towards Real-Time, Autonomous Testing of Virtual Reality Games 

**Title (ZH)**: VRScout: 向实时自主测试虚拟现实游戏方向迈进 

**Authors**: Yurun Wu, Yousong Sun, Burkhard Wunsche, Jia Wang, Elliott Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00002)  

**Abstract**: Virtual Reality (VR) has rapidly become a mainstream platform for gaming and interactive experiences, yet ensuring the quality, safety, and appropriateness of VR content remains a pressing challenge. Traditional human-based quality assurance is labor-intensive and cannot scale with the industry's rapid growth. While automated testing has been applied to traditional 2D and 3D games, extending it to VR introduces unique difficulties due to high-dimensional sensory inputs and strict real-time performance requirements. We present VRScout, a deep learning-based agent capable of autonomously navigating VR environments and interacting with virtual objects in a human-like and real-time manner. VRScout learns from human demonstrations using an enhanced Action Chunking Transformer that predicts multi-step action sequences. This enables our agent to capture higher-level strategies and generalize across diverse environments. To balance responsiveness and precision, we introduce a dynamically adjustable sliding horizon that adapts the agent's temporal context at runtime. We evaluate VRScout on commercial VR titles and show that it achieves expert-level performance with only limited training data, while maintaining real-time inference at 60 FPS on consumer-grade hardware. These results position VRScout as a practical and scalable framework for automated VR game testing, with direct applications in both quality assurance and safety auditing. 

**Abstract (ZH)**: 基于深度学习的VR内容自动导航与测试框架：VRScout 

---
# A Two Level Neural Approach Combining Off-Chip Prediction with Adaptive Prefetch Filtering 

**Title (ZH)**: 结合_off-芯片预测与自适应预取过滤的两级神经方法 

**Authors**: Alexandre Valentin Jamet, Georgios Vavouliotis, Daniel A. Jiménez, Lluc Alvarez, Marc Casas  

**Link**: [PDF](https://arxiv.org/pdf/2403.15181)  

**Abstract**: To alleviate the performance and energy overheads of contemporary applications with large data footprints, we propose the Two Level Perceptron (TLP) predictor, a neural mechanism that effectively combines predicting whether an access will be off-chip with adaptive prefetch filtering at the first-level data cache (L1D). TLP is composed of two connected microarchitectural perceptron predictors, named First Level Predictor (FLP) and Second Level Predictor (SLP). FLP performs accurate off-chip prediction by using several program features based on virtual addresses and a novel selective delay component. The novelty of SLP relies on leveraging off-chip prediction to drive L1D prefetch filtering by using physical addresses and the FLP prediction as features. TLP constitutes the first hardware proposal targeting both off-chip prediction and prefetch filtering using a multi-level perceptron hardware approach. TLP only requires 7KB of storage. To demonstrate the benefits of TLP we compare its performance with state-of-the-art approaches using off-chip prediction and prefetch filtering on a wide range of single-core and multi-core workloads. Our experiments show that TLP reduces the average DRAM transactions by 30.7% and 17.7%, as compared to a baseline using state-of-the-art cache prefetchers but no off-chip prediction mechanism, across the single-core and multi-core workloads, respectively, while recent work significantly increases DRAM transactions. As a result, TLP achieves geometric mean performance speedups of 6.2% and 11.8% across single-core and multi-core workloads, respectively. In addition, our evaluation demonstrates that TLP is effective independently of the L1D prefetching logic. 

**Abstract (ZH)**: 基于两级感知机的预测过滤器（TLP）：缓解大型数据足迹应用的性能和能效开销 

---
