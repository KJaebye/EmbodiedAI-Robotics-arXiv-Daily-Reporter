# Enhancing System Self-Awareness and Trust of AI: A Case Study in Trajectory Prediction and Planning 

**Title (ZH)**: 增强系统自我意识和对AI的信任：轨迹预测与规划案例研究 

**Authors**: Lars Ullrich, Zurab Mujirishvili, Knut Graichen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18421)  

**Abstract**: In the trajectory planning of automated driving, data-driven statistical artificial intelligence (AI) methods are increasingly established for predicting the emergent behavior of other road users. While these methods achieve exceptional performance in defined datasets, they usually rely on the independent and identically distributed (i.i.d.) assumption and thus tend to be vulnerable to distribution shifts that occur in the real world. In addition, these methods lack explainability due to their black box nature, which poses further challenges in terms of the approval process and social trustworthiness. Therefore, in order to use the capabilities of data-driven statistical AI methods in a reliable and trustworthy manner, the concept of TrustMHE is introduced and investigated in this paper. TrustMHE represents a complementary approach, independent of the underlying AI systems, that combines AI-driven out-of-distribution detection with control-driven moving horizon estimation (MHE) to enable not only detection and monitoring, but also intervention. The effectiveness of the proposed TrustMHE is evaluated and proven in three simulation scenarios. 

**Abstract (ZH)**: 在自动驾驶的轨迹规划中，基于数据驱动的统计人工智能方法逐渐建立起来以预测其他道路用户的行为。尽管这些方法在定义的数据集上表现出色，但通常依赖独立同分布（i.i.d.）假设，因此在现实世界中容易受到分布变化的影响。此外，由于其黑盒性质，这些方法缺乏可解释性，这在审批过程和社交信任方面提出了进一步的挑战。因此，为了可靠且可信地利用数据驱动的统计人工智能方法的能力，本文引介入一种名为TrustMHE的概念，并对其进行研究。TrustMHE代表一种独立于底层AI系统的补充方法，结合了基于AI的异常检测和基于控制的移动窗口估计（MHE），不仅实现检测和监控，还实现了干预。提出的TrustMHE在三个仿真场景中进行了评估并得到验证。 

---
# Range-based 6-DoF Monte Carlo SLAM with Gradient-guided Particle Filter on GPU 

**Title (ZH)**: 基于范围测量的6-DoF蒙特卡洛SLAM与梯度引导粒子滤波在GPU上的实现 

**Authors**: Takumi Nakao, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Hisashi Date  

**Link**: [PDF](https://arxiv.org/pdf/2504.18056)  

**Abstract**: This paper presents range-based 6-DoF Monte Carlo SLAM with a gradient-guided particle update strategy. While non-parametric state estimation methods, such as particle filters, are robust in situations with high ambiguity, they are known to be unsuitable for high-dimensional problems due to the curse of dimensionality. To address this issue, we propose a particle update strategy that improves the sampling efficiency by using the gradient information of the likelihood function to guide particles toward its mode. Additionally, we introduce a keyframe-based map representation that represents the global map as a set of past frames (i.e., keyframes) to mitigate memory consumption. The keyframe poses for each particle are corrected using a simple loop closure method to maintain trajectory consistency. The combination of gradient information and keyframe-based map representation significantly enhances sampling efficiency and reduces memory usage compared to traditional RBPF approaches. To process a large number of particles (e.g., 100,000 particles) in real-time, the proposed framework is designed to fully exploit GPU parallel processing. Experimental results demonstrate that the proposed method exhibits extreme robustness to state ambiguity and can even deal with kidnapping situations, such as when the sensor moves to different floors via an elevator, with minimal heuristics. 

**Abstract (ZH)**: 基于范围的6自由度蒙特卡洛SLAM结合梯度导向粒子更新策略 

---
# Sky-Drive: A Distributed Multi-Agent Simulation Platform for Socially-Aware and Human-AI Collaborative Future Transportation 

**Title (ZH)**: Sky-Drive：一种面向社会意识和人机协同未来交通的分布式多Agent仿真平台 

**Authors**: Zilin Huang, Zihao Sheng, Zhengyang Wan, Yansong Qu, Yuhao Luo, Boyue Wang, Pei Li, Yen-Jung Chen, Jiancong Chen, Keke Long, Jiayi Meng, Yue Leng, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18010)  

**Abstract**: Recent advances in autonomous system simulation platforms have significantly enhanced the safe and scalable testing of driving policies. However, existing simulators do not yet fully meet the needs of future transportation research, particularly in modeling socially-aware driving agents and enabling effective human-AI collaboration. This paper introduces Sky-Drive, a novel distributed multi-agent simulation platform that addresses these limitations through four key innovations: (a) a distributed architecture for synchronized simulation across multiple terminals; (b) a multi-modal human-in-the-loop framework integrating diverse sensors to collect rich behavioral data; (c) a human-AI collaboration mechanism supporting continuous and adaptive knowledge exchange; and (d) a digital twin (DT) framework for constructing high-fidelity virtual replicas of real-world transportation environments. Sky-Drive supports diverse applications such as autonomous vehicle (AV)-vulnerable road user (VRU) interaction modeling, human-in-the-loop training, socially-aware reinforcement learning, personalized driving policy, and customized scenario generation. Future extensions will incorporate foundation models for context-aware decision support and hardware-in-the-loop (HIL) testing for real-world validation. By bridging scenario generation, data collection, algorithm training, and hardware integration, Sky-Drive has the potential to become a foundational platform for the next generation of socially-aware and human-centered autonomous transportation research. The demo video and code are available at:this https URL 

**Abstract (ZH)**: 最近在自主系统仿真平台方面的进展显著提高了驾驶策略的安全和可扩展测试。然而，现有的仿真器尚未完全满足未来交通研究的需求，特别是在建模社会感知驾驶代理和促进有效的人类-人工智能协作方面。本文介绍了Sky-Drive，这是一种新型的分布式多智能体仿真平台，通过四项关键创新解决了这些局限性：(a) 多终端同步仿真分布架构；(b) 多模态人机环框架，集成多种传感器以收集丰富的行为数据；(c) 支持持续和自适应知识交流的人机协作机制；(d) 虚拟孪生框架，用于构建高度仿真的现实世界交通环境的虚拟副本。Sky-Drive 支持多种应用，包括无人驾驶车辆（AV）与脆弱道路使用者（VRU）的交互建模、人机环训练、社会感知强化学习、个性化驾驶策略以及定制化场景生成。未来扩展将 Incorporate 基于上下文的决策支持基础模型和硬件在环（HIL）测试以进行现实世界的验证。通过链接场景生成、数据收集、算法训练和硬件集成，Sky-Drive 有望成为下一代社会感知和以人为本的自主交通研究的基础平台。演示视频和代码可在以下链接获取：this https URL。 

---
# Plug-and-Play Physics-informed Learning using Uncertainty Quantified Port-Hamiltonian Models 

**Title (ZH)**: 使用不确定性量化哈密尔顿模型的即插即用物理约束学习 

**Authors**: Kaiyuan Tan, Peilun Li, Jun Wang, Thomas Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2504.17966)  

**Abstract**: The ability to predict trajectories of surrounding agents and obstacles is a crucial component in many robotic applications. Data-driven approaches are commonly adopted for state prediction in scenarios where the underlying dynamics are unknown. However, the performance, reliability, and uncertainty of data-driven predictors become compromised when encountering out-of-distribution observations relative to the training data. In this paper, we introduce a Plug-and-Play Physics-Informed Machine Learning (PnP-PIML) framework to address this challenge. Our method employs conformal prediction to identify outlier dynamics and, in that case, switches from a nominal predictor to a physics-consistent model, namely distributed Port-Hamiltonian systems (dPHS). We leverage Gaussian processes to model the energy function of the dPHS, enabling not only the learning of system dynamics but also the quantification of predictive uncertainty through its Bayesian nature. In this way, the proposed framework produces reliable physics-informed predictions even for the out-of-distribution scenarios. 

**Abstract (ZH)**: 基于物理信息的插件式机器学习框架（PnP-PIML）：处理分布外观测的可靠轨迹预测 

---
# Generalization Capability for Imitation Learning 

**Title (ZH)**: 模仿学习的泛化能力 

**Authors**: Yixiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18538)  

**Abstract**: Imitation learning holds the promise of equipping robots with versatile skills by learning from expert demonstrations. However, policies trained on finite datasets often struggle to generalize beyond the training distribution. In this work, we present a unified perspective on the generalization capability of imitation learning, grounded in both information theorey and data distribution property. We first show that the generalization gap can be upper bounded by (i) the conditional information bottleneck on intermediate representations and (ii) the mutual information between the model parameters and the training dataset. This characterization provides theoretical guidance for designing effective training strategies in imitation learning, particularly in determining whether to freeze, fine-tune, or train large pretrained encoders (e.g., vision-language models or vision foundation models) from scratch to achieve better generalization. Furthermore, we demonstrate that high conditional entropy from input to output induces a flatter likelihood landscape, thereby reducing the upper bound on the generalization gap. In addition, it shortens the stochastic gradient descent (SGD) escape time from sharp local minima, which may increase the likelihood of reaching global optima under fixed optimization budgets. These insights explain why imitation learning often exhibits limited generalization and underscore the importance of not only scaling the diversity of input data but also enriching the variability of output labels conditioned on the same input. 

**Abstract (ZH)**: 模仿学习在信息理论和数据分布性质的基础上提供了一种统一的观点，以装备机器人具备多样化技能。然而，基于有限数据集训练的策略往往难以在训练分布之外进行泛化。在这项工作中，我们展示了模仿学习泛化能力的一种统一视角，该视角基于信息理论和数据分布的特性。我们首先证明了泛化差距可以通过(i) 中间表示的条件信息瓶颈和(ii) 模型参数与训练数据集之间的互信息进行上界估计。这种表征为设计有效的模仿学习训练策略提供了理论指导，尤其是在确定是否冻结、微调或从零开始训练大型预训练编码器（例如视觉语言模型或视觉基础模型）以实现更好的泛化效果时。此外，我们还证明了从输入到输出的高条件熵会导致更平坦的似然景观，从而降低泛化差距的上界。此外，这缩短了从尖峰局部最小值逃离随机梯度下降（SGD）的时间，从而可能在固定优化预算下增加达到全局最优的可能性。这些见解解释了为什么模仿学习常常表现出有限的泛化能力，并强调了不仅扩展输入数据多样性而且在相同输入条件下丰富输出标签变化性的重要性。 

---
# A Taylor Series Approach to Correction of Input Errors in Gaussian Process Regression 

**Title (ZH)**: 高斯过程回归中输入误差矫正的泰勒级数方法 

**Authors**: Muzaffar Qureshi, Tochukwu Elijah Ogri, Zachary I. Bell, Wanjiku A. Makumi, Rushikesh Kamalapurkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.18463)  

**Abstract**: Gaussian Processes (GPs) are widely recognized as powerful non-parametric models for regression and classification. Traditional GP frameworks predominantly operate under the assumption that the inputs are either accurately known or subject to zero-mean noise. However, several real-world applications such as mobile sensors have imperfect localization, leading to inputs with biased errors. These biases can typically be estimated through measurements collected over time using, for example, Kalman filters. To avoid recomputation of the entire GP model when better estimates of the inputs used in the training data become available, we introduce a technique for updating a trained GP model to incorporate updated estimates of the inputs. By leveraging the differentiability of the mean and covariance functions derived from the squared exponential kernel, a second-order correction algorithm is developed to update the trained GP models. Precomputed Jacobians and Hessians of kernels enable real-time refinement of the mean and covariance predictions. The efficacy of the developed approach is demonstrated using two simulation studies, with error analyses revealing improvements in both predictive accuracy and uncertainty quantification. 

**Abstract (ZH)**: 高斯过程（GPs）被认为是回归和分类的强非参数模型。传统的GP框架主要假设输入要么准确已知，要么受到零均值噪声的影响。然而，如移动传感器等实际应用场景可能具有不完美的定位，导致输入带有有偏误差。这些偏差可以通过时间序列测量，例如使用卡尔曼滤波器进行估计。为了避免在获得更好的输入估计值时重新计算整个GP模型，我们提出了一种更新训练好的GP模型以融入更新的输入估计值的技术。通过利用来自指数平方核的均值和协方差函数的可微性，开发了一种二次校正算法来更新训练好的GP模型。预计算的核的雅可比行列式和海森矩阵使对均值和协方差预测的实时优化成为可能。通过两个模拟研究验证了该方法的有效性，误差分析显示预测准确性和不确定性量化均有提升。 

---
# Offline Learning of Controllable Diverse Behaviors 

**Title (ZH)**: 离线学习可控多样化行为 

**Authors**: Mathieu Petitbois, Rémy Portelas, Sylvain Lamprier, Ludovic Denoyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18160)  

**Abstract**: Imitation Learning (IL) techniques aim to replicate human behaviors in specific tasks. While IL has gained prominence due to its effectiveness and efficiency, traditional methods often focus on datasets collected from experts to produce a single efficient policy. Recently, extensions have been proposed to handle datasets of diverse behaviors by mainly focusing on learning transition-level diverse policies or on performing entropy maximization at the trajectory level. While these methods may lead to diverse behaviors, they may not be sufficient to reproduce the actual diversity of demonstrations or to allow controlled trajectory generation. To overcome these drawbacks, we propose a different method based on two key features: a) Temporal Consistency that ensures consistent behaviors across entire episodes and not just at the transition level as well as b) Controllability obtained by constructing a latent space of behaviors that allows users to selectively activate specific behaviors based on their requirements. We compare our approach to state-of-the-art methods over a diverse set of tasks and environments. Project page: this https URL 

**Abstract (ZH)**: 模仿学习（IL）技术旨在在一个特定任务中复制人类行为。尽管由于其有效性与高效性，模仿学习已获得广泛关注，但传统方法通常侧重于从专家收集的数据集以生成单一高效的策略。最近，已提出扩展方法来处理包含多样行为的数据集，这些方法主要关注学习过渡层面的多样策略或在轨迹层面进行熵最大化。虽然这些方法可能会导致多样行为，但它们可能不足以再现演示的实际多样性或允许受控轨迹生成。为克服这些缺点，我们提出了一种基于两个关键特征的不同方法：a) 时间一致性，确保整个 episodes 的行为一致性，而不仅仅是过渡层面的一致性；b) 可控性，通过构建行为的潜在空间，允许用户根据其需求选择性地激活特定行为。我们在多种任务和环境中将我们的方法与最先进的方法进行了比较。项目页面: this https URL 

---
# CaRL: Learning Scalable Planning Policies with Simple Rewards 

**Title (ZH)**: CaRL: 通过简单的奖励学习可扩展的规划策略 

**Authors**: Bernhard Jaeger, Daniel Dauner, Jens Beißwenger, Simon Gerstenecker, Kashyap Chitta, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.17838)  

**Abstract**: We investigate reinforcement learning (RL) for privileged planning in autonomous driving. State-of-the-art approaches for this task are rule-based, but these methods do not scale to the long tail. RL, on the other hand, is scalable and does not suffer from compounding errors like imitation learning. Contemporary RL approaches for driving use complex shaped rewards that sum multiple individual rewards, \eg~progress, position, or orientation rewards. We show that PPO fails to optimize a popular version of these rewards when the mini-batch size is increased, which limits the scalability of these approaches. Instead, we propose a new reward design based primarily on optimizing a single intuitive reward term: route completion. Infractions are penalized by terminating the episode or multiplicatively reducing route completion. We find that PPO scales well with higher mini-batch sizes when trained with our simple reward, even improving performance. Training with large mini-batch sizes enables efficient scaling via distributed data parallelism. We scale PPO to 300M samples in CARLA and 500M samples in nuPlan with a single 8-GPU node. The resulting model achieves 64 DS on the CARLA longest6 v2 benchmark, outperforming other RL methods with more complex rewards by a large margin. Requiring only minimal adaptations from its use in CARLA, the same method is the best learning-based approach on nuPlan. It scores 91.3 in non-reactive and 90.6 in reactive traffic on the Val14 benchmark while being an order of magnitude faster than prior work. 

**Abstract (ZH)**: 我们探讨了强化学习在自主驾驶中特权规划中的应用。最新的方法基于规则，但这些方法无法扩展至长尾场景。相比之下，强化学习是可扩展的，并且不会遭受模仿学习中累积错误的问题。用于驾驶的现代强化学习方法使用复杂形状的奖励，这些奖励由多个个体奖励项之和构成，例如进度、位置或方向奖励。我们发现，当mini-batch大小增加时，PPO无法优化这些奖励的一种流行版本，这限制了这些方法的可扩展性。相反，我们提出了一种新的奖励设计，主要是基于优化单一直观的奖励项：路线完成。违规行为通过终止episode或乘性地减少路线完成度予以惩罚。我们发现，在使用我们简单奖励进行训练时，PPO在较高的mini-batch大小下表现出良好的可扩展性，甚至提高了性能。使用大mini-batch大小进行训练能够通过分布式数据并行实现高效的扩展。我们使用单个8-GPU节点将PPO扩展至CARLA中的3亿样本和nuPlan中的5亿样本。所得到的模型在CARLA的longest6 v2基准测试中实现了64 DS，显著优于使用更复杂奖励的其他RL方法。仅需对其在CARLA中的应用进行少量适应，该方法在nuPlan中也是最优的基于学习的方法。它在Val14基准测试中的非反应性交通得分为91.3，在反应性交通得分为90.6，比之前的工作快了几个数量级。 

---
# Adapting Probabilistic Risk Assessment for AI 

**Title (ZH)**: 适配人工智能的概率风险评估 

**Authors**: Anna Katariina Wisakanto, Joe Rogero, Avyay M. Casheekar, Richard Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2504.18536)  

**Abstract**: Modern general-purpose artificial intelligence (AI) systems present an urgent risk management challenge, as their rapidly evolving capabilities and potential for catastrophic harm outpace our ability to reliably assess their risks. Current methods often rely on selective testing and undocumented assumptions about risk priorities, frequently failing to make a serious attempt at assessing the set of pathways through which Al systems pose direct or indirect risks to society and the biosphere. This paper introduces the probabilistic risk assessment (PRA) for AI framework, adapting established PRA techniques from high-reliability industries (e.g., nuclear power, aerospace) for the new challenges of advanced AI. The framework guides assessors in identifying potential risks, estimating likelihood and severity, and explicitly documenting evidence, underlying assumptions, and analyses at appropriate granularities. The framework's implementation tool synthesizes the results into a risk report card with aggregated risk estimates from all assessed risks. This systematic approach integrates three advances: (1) Aspect-oriented hazard analysis provides systematic hazard coverage guided by a first-principles taxonomy of AI system aspects (e.g. capabilities, domain knowledge, affordances); (2) Risk pathway modeling analyzes causal chains from system aspects to societal impacts using bidirectional analysis and incorporating prospective techniques; and (3) Uncertainty management employs scenario decomposition, reference scales, and explicit tracing protocols to structure credible projections with novelty or limited data. Additionally, the framework harmonizes diverse assessment methods by integrating evidence into comparable, quantified absolute risk estimates for critical decisions. We have implemented this as a workbook tool for AI developers, evaluators, and regulators, available on the project website. 

**Abstract (ZH)**: 现代通用人工智能的probabilistic风险评估框架：适应先进人工智能的新挑战 

---
# Scaling Laws For Scalable Oversight 

**Title (ZH)**: 可扩展监督的标度定律 

**Authors**: Joshua Engels, David D. Baek, Subhash Kantamneni, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2504.18530)  

**Abstract**: Scalable oversight, the process by which weaker AI systems supervise stronger ones, has been proposed as a key strategy to control future superintelligent systems. However, it is still unclear how scalable oversight itself scales. To address this gap, we propose a framework that quantifies the probability of successful oversight as a function of the capabilities of the overseer and the system being overseen. Specifically, our framework models oversight as a game between capability-mismatched players; the players have oversight-specific and deception-specific Elo scores that are a piecewise-linear function of their general intelligence, with two plateaus corresponding to task incompetence and task saturation. We validate our framework with a modified version of the game Nim and then apply it to four oversight games: "Mafia", "Debate", "Backdoor Code" and "Wargames". For each game, we find scaling laws that approximate how domain performance depends on general AI system capability (using Chatbot Arena Elo as a proxy for general capability). We then build on our findings in a theoretical study of Nested Scalable Oversight (NSO), a process in which trusted models oversee untrusted stronger models, which then become the trusted models in the next step. We identify conditions under which NSO succeeds and derive numerically (and in some cases analytically) the optimal number of oversight levels to maximize the probability of oversight success. In our numerical examples, the NSO success rate is below 52% when overseeing systems that are 400 Elo points stronger than the baseline overseer, and it declines further for overseeing even stronger systems. 

**Abstract (ZH)**: 可扩展的监督：一种量化监督成功概率的框架及其在多层次信任模型中的应用 

---
# Pseudo-Boolean Proof Logging for Optimal Classical Planning 

**Title (ZH)**: 伪布尔证明记录在最优经典规划中的应用 

**Authors**: Simon Dold, Malte Helmert, Jakob Nordström, Gabriele Röger, Tanja Schindler  

**Link**: [PDF](https://arxiv.org/pdf/2504.18443)  

**Abstract**: We introduce lower-bound certificates for classical planning tasks, which can be used to prove the unsolvability of a task or the optimality of a plan in a way that can be verified by an independent third party. We describe a general framework for generating lower-bound certificates based on pseudo-Boolean constraints, which is agnostic to the planning algorithm used.
As a case study, we show how to modify the $A^{*}$ algorithm to produce proofs of optimality with modest overhead, using pattern database heuristics and $h^\textit{max}$ as concrete examples. The same proof logging approach works for any heuristic whose inferences can be efficiently expressed as reasoning over pseudo-Boolean constraints. 

**Abstract (ZH)**: 我们引入了用于经典规划任务的下界证书，可以通过独立第三方验证这些证书来证明任务的不可解或计划的最优性。我们描述了一个基于伪布尔约束生成下界证书的通用框架，该框架与使用的规划算法无关。作为案例研究，我们展示了如何通过修改$A^{*}$算法并结合模式数据库启发式和$h^\textit{max}$启发式，以适度的开销产生最优性证明。相同的方法适用于任何可以高效表达为伪布尔约束推理的启发式算法。 

---
# Combating the Bucket Effect:Multi-Knowledge Alignment for Medication Recommendation 

**Title (ZH)**: 对抗桶效应：多知识对齐的药品推荐方法 

**Authors**: Xiang Li, Haixu Ma, Guanyong Wu, Shi Mu, Chen Li, Shunpan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18096)  

**Abstract**: Medication recommendation is crucial in healthcare, offering effective treatments based on patient's electronic health records (EHR). Previous studies show that integrating more medication-related knowledge improves medication representation accuracy. However, not all medications encompass multiple types of knowledge data simultaneously. For instance, some medications provide only textual descriptions without structured data. This imbalance in data availability limits the performance of existing models, a challenge we term the "bucket effect" in medication recommendation. Our data analysis uncovers the severity of the "bucket effect" in medication recommendation. To fill this gap, we introduce a cross-modal medication encoder capable of seamlessly aligning data from different modalities and propose a medication recommendation framework to integrate Multiple types of Knowledge, named MKMed. Specifically, we first pre-train a cross-modal encoder with contrastive learning on five knowledge modalities, aligning them into a unified space. Then, we combine the multi-knowledge medication representations with patient records for recommendations. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that MKMed mitigates the "bucket effect" in data, and significantly outperforms state-of-the-art baselines in recommendation accuracy and safety. 

**Abstract (ZH)**: 基于电子健康记录的药品推荐至关重要：整合多类型知识的跨模态药品编码器（MKMed） 

---
# Differential Privacy-Driven Framework for Enhancing Heart Disease Prediction 

**Title (ZH)**: 差分隐私驱动的心脏疾病预测增强框架 

**Authors**: Yazan Otoum, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.18007)  

**Abstract**: With the rapid digitalization of healthcare systems, there has been a substantial increase in the generation and sharing of private health data. Safeguarding patient information is essential for maintaining consumer trust and ensuring compliance with legal data protection regulations. Machine learning is critical in healthcare, supporting personalized treatment, early disease detection, predictive analytics, image interpretation, drug discovery, efficient operations, and patient monitoring. It enhances decision-making, accelerates research, reduces errors, and improves patient outcomes. In this paper, we utilize machine learning methodologies, including differential privacy and federated learning, to develop privacy-preserving models that enable healthcare stakeholders to extract insights without compromising individual privacy. Differential privacy introduces noise to data to guarantee statistical privacy, while federated learning enables collaborative model training across decentralized datasets. We explore applying these technologies to Heart Disease Data, demonstrating how they preserve privacy while delivering valuable insights and comprehensive analysis. Our results show that using a federated learning model with differential privacy achieved a test accuracy of 85%, ensuring patient data remained secure and private throughout the process. 

**Abstract (ZH)**: 随着 healthcare 系统的快速数字化，私人健康数据的生成和共享显著增加。保护患者信息对于维护消费者信任并确保遵守数据保护法规至关重要。机器学习在医疗保健中至关重要，支持个性化的治疗、疾病的早期检测、预测分析、图像解释、药物发现、高效运营和患者监测。它能增强决策、加速研究、减少错误并改善患者结果。在本文中，我们利用机器学习方法，包括差分隐私和联邦学习，开发了保护隐私的模型，从而使医疗保健利益相关者能够在不泄露个人隐私的情况下提取有用信息。差分隐私通过对数据添加噪声来保证统计隐私，而联邦学习则允许多个分散的数据集共同训练模型。我们探讨了将这些技术应用于心脏病数据的可行性，展示了它们如何在保护隐私的同时提供有价值的洞察和全面的分析。结果表明，使用联邦学习模型结合差分隐私实现了85%的测试准确率，确保整个过程中患者数据的安全性和隐私性。 

---
# DeSIA: Attribute Inference Attacks Against Limited Fixed Aggregate Statistics 

**Title (ZH)**: DeSIA：针对有限固定聚合统计的属性推断攻击 

**Authors**: Yifeng Mao, Bozhidar Stevanoski, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2504.18497)  

**Abstract**: Empirical inference attacks are a popular approach for evaluating the privacy risk of data release mechanisms in practice. While an active attack literature exists to evaluate machine learning models or synthetic data release, we currently lack comparable methods for fixed aggregate statistics, in particular when only a limited number of statistics are released. We here propose an inference attack framework against fixed aggregate statistics and an attribute inference attack called DeSIA. We instantiate DeSIA against the U.S. Census PPMF dataset and show it to strongly outperform reconstruction-based attacks. In particular, we show DeSIA to be highly effective at identifying vulnerable users, achieving a true positive rate of 0.14 at a false positive rate of $10^{-3}$. We then show DeSIA to perform well against users whose attributes cannot be verified and when varying the number of aggregate statistics and level of noise addition. We also perform an extensive ablation study of DeSIA and show how DeSIA can be successfully adapted to the membership inference task. Overall, our results show that aggregation alone is not sufficient to protect privacy, even when a relatively small number of aggregates are being released, and emphasize the need for formal privacy mechanisms and testing before aggregate statistics are released. 

**Abstract (ZH)**: 经验推理攻击是评估数据发布机制实际隐私风险的一种流行方法。虽然存在针对机器学习模型或合成数据发布的主动攻击文献，但在仅发布有限数量统计信息的情况下，我们目前缺乏可比的方法，尤其是在固定聚合统计方面。我们在此提出一种针对固定聚合统计的推理攻击框架以及一种属性推理攻击，称为DeSIA。我们将DeSIA实例化应用于美国人口普查PPMF数据集，并证明它在重建攻击中表现更为优异。特别是，我们展示DeSIA在识别易受攻击用户方面极其有效，实现了在假阳性率为$10^{-3}$时的真实阳性率为0.14。随后，我们展示DeSIA在用户属性无法验证以及聚合统计数量和噪声添加水平变化时的表现良好。我们还进行了DeSIA的广泛消融研究，并展示了如何成功地将DeSIA适应到成员推理任务。总体而言，我们的研究结果表明，仅聚合本身不足以保护隐私，即使发布的聚合数量相对较少，强调了在发布聚合统计之前需要正式的隐私机制和测试。 

---
# Fast-Slow Thinking for Large Vision-Language Model Reasoning 

**Title (ZH)**: 快速-缓慢思考在大规模视觉-语言模型推理中的应用 

**Authors**: Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18458)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have revealed an \textit{overthinking} phenomenon, where models generate verbose reasoning across all tasks regardless of questions. To address this issue, we present \textbf{FAST}, a novel \textbf{Fa}st-\textbf{S}low \textbf{T}hinking framework that dynamically adapts reasoning depth based on question characteristics. Through empirical analysis, we establish the feasibility of fast-slow thinking in LVLMs by investigating how response length and data distribution affect performance. We develop FAST-GRPO with three components: model-based metrics for question characterization, an adaptive thinking reward mechanism, and difficulty-aware KL regularization. Experiments across seven reasoning benchmarks demonstrate that FAST achieves state-of-the-art accuracy with over 10\% relative improvement compared to the base model, while reducing token usage by 32.7-67.3\% compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy. 

**Abstract (ZH)**: 最近的大规模视觉-语言模型（LVLMs）研究揭示了过度推理现象，即模型在所有任务中不分情境地产生冗长的推理过程。为了解决这一问题，我们提出了FAST（快速-缓慢思考）框架，这是一种新的动态适应推理深度的框架，基于问题特征调整推理深度。通过实证分析，我们研究了响应长度和数据分布如何影响性能，以证明在LVLMs中实施快速-缓慢思考的可行性。我们开发了FAST-GRPO，它包括基于模型的问题特征度量、自适应思考奖励机制和难度感知的KL正则化。在七个推理基准测试中的实验表明，FAST 在相对于基线模型的相对准确性提高了10%以上的同时，相比之前缓慢思考的方法减少了32.7%-67.3%的_token_使用量，有效地平衡了推理长度和准确性。 

---
# Enhancing Pre-Trained Model-Based Class-Incremental Learning through Neural Collapse 

**Title (ZH)**: 基于神经坍缩提升预训练模型驱动的类别增量学习 

**Authors**: Kun He, Zijian Song, Shuoxi Zhang, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2504.18437)  

**Abstract**: Class-Incremental Learning (CIL) is a critical capability for real-world applications, enabling learning systems to adapt to new tasks while retaining knowledge from previous ones. Recent advancements in pre-trained models (PTMs) have significantly advanced the field of CIL, demonstrating superior performance over traditional methods. However, understanding how features evolve and are distributed across incremental tasks remains an open challenge. In this paper, we propose a novel approach to modeling feature evolution in PTM-based CIL through the lens of neural collapse (NC), a striking phenomenon observed in the final phase of training, which leads to a well-separated, equiangular feature space. We explore the connection between NC and CIL effectiveness, showing that aligning feature distributions with the NC geometry enhances the ability to capture the dynamic behavior of continual learning. Based on this insight, we introduce Neural Collapse-inspired Pre-Trained Model-based CIL (NCPTM-CIL), a method that dynamically adjusts the feature space to conform to the elegant NC structure, thereby enhancing the continual learning process. Extensive experiments demonstrate that NCPTM-CIL outperforms state-of-the-art methods across four benchmark datasets. Notably, when initialized with ViT-B/16-IN1K, NCPTM-CIL surpasses the runner-up method by 6.73% on VTAB, 1.25% on CIFAR-100, and 2.5% on OmniBenchmark. 

**Abstract (ZH)**: 基于神经衰减的预训练模型增量学习（NCPTM-CIL） 

---
# Paradigm shift on Coding Productivity Using GenAI 

**Title (ZH)**: 使用生成式人工智能促进编码生产力范式转移 

**Authors**: Liang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18404)  

**Abstract**: Generative AI (GenAI) applications are transforming software engineering by enabling automated code co-creation. However, empirical evidence on GenAI's productivity effects in industrial settings remains limited. This paper investigates the adoption of GenAI coding assistants (e.g., Codeium, Amazon Q) within telecommunications and FinTech domains. Through surveys and interviews with industrial domain-experts, we identify primary productivity-influencing factors, including task complexity, coding skills, domain knowledge, and GenAI integration. Our findings indicate that GenAI tools enhance productivity in routine coding tasks (e.g., refactoring and Javadoc generation) but face challenges in complex, domain-specific activities due to limited context-awareness of codebases and insufficient support for customized design rules. We highlight new paradigms for coding transfer, emphasizing iterative prompt refinement, immersive development environment, and automated code evaluation as essential for effective GenAI usage. 

**Abstract (ZH)**: Generative AI编码助手在电信和金融科技领域的采用及其生产力影响：基于工业环境的实证研究 

---
# Spatial Reasoner: A 3D Inference Pipeline for XR Applications 

**Title (ZH)**: 三维推理管道：XR应用中的空间 reasoning 

**Authors**: Steven Häsler, Philipp Ackermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.18380)  

**Abstract**: Modern extended reality XR systems provide rich analysis of image data and fusion of sensor input and demand AR/VR applications that can reason about 3D scenes in a semantic manner. We present a spatial reasoning framework that bridges geometric facts with symbolic predicates and relations to handle key tasks such as determining how 3D objects are arranged among each other ('on', 'behind', 'near', etc.). Its foundation relies on oriented 3D bounding box representations, enhanced by a comprehensive set of spatial predicates, ranging from topology and connectivity to directionality and orientation, expressed in a formalism related to natural language. The derived predicates form a spatial knowledge graph and, in combination with a pipeline-based inference model, enable spatial queries and dynamic rule evaluation. Implementations for client- and server-side processing demonstrate the framework's capability to efficiently translate geometric data into actionable knowledge, ensuring scalable and technology-independent spatial reasoning in complex 3D environments. The Spatial Reasoner framework is fostering the creation of spatial ontologies, and seamlessly integrates with and therefore enriches machine learning, natural language processing, and rule systems in XR applications. 

**Abstract (ZH)**: 现代扩展现实XR系统提供了丰富的图像数据分析和传感器输入融合，需要能够以语义方式推理3D场景的AR/VR应用。我们提出了一种空间推理框架，将几何事实与符号谓词和关系相结合，处理诸如确定3D对象彼此如何排列（'在...上'、'在...后面'、'在...附近'等）的关键任务。该框架的基础依赖于定向的3D边界框表示，并通过一系列空间谓词得到了增强，这些谓词涵盖了拓扑、连接性、方向性和方向信息，并使用与自然语言相关的形式化表示。得出的谓词形成了空间知识图谱，并结合基于流水线的推理模型，实现了空间查询和动态规则评估。客户端和服务器端的实现证明了该框架能够高效地将几何数据转换为可操作的知识，确保在复杂的3D环境中实现可扩展且技术独立的空间推理。空间推理框架促进了空间本体的创建，并无缝集成和丰富了XR应用中的机器学习、自然语言处理和规则系统。 

---
# Testing Individual Fairness in Graph Neural Networks 

**Title (ZH)**: 测试图神经网络的个体公平性 

**Authors**: Roya Nasiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.18353)  

**Abstract**: The biases in artificial intelligence (AI) models can lead to automated decision-making processes that discriminate against groups and/or individuals based on sensitive properties such as gender and race. While there are many studies on diagnosing and mitigating biases in various AI models, there is little research on individual fairness in Graph Neural Networks (GNNs). Unlike traditional models, which treat data features independently and overlook their inter-relationships, GNNs are designed to capture graph-based structure where nodes are interconnected. This relational approach enables GNNs to model complex dependencies, but it also means that biases can propagate through these connections, complicating the detection and mitigation of individual fairness violations. This PhD project aims to develop a testing framework to assess and ensure individual fairness in GNNs. It first systematically reviews the literature on individual fairness, categorizing existing approaches to define, measure, test, and mitigate model biases, creating a taxonomy of individual fairness. Next, the project will develop a framework for testing and ensuring fairness in GNNs by adapting and extending current fairness testing and mitigation techniques. The framework will be evaluated through industrial case studies, focusing on graph-based large language models. 

**Abstract (ZH)**: 人工intelligence模型中的偏差可能导致基于性别、种族等敏感属性对群体和个人进行歧视性的自动决策过程。虽然已有许多研究专注于诊断和减轻各种AI模型中的偏差，但对于图神经网络（GNNs）中的个体公平性研究却甚少。与传统模型独立处理数据特征并忽略其相互关系不同，GNNs旨在捕捉节点间相连的图结构。这种关系方法使得GNNs能够建模复杂的依赖关系，但也意味着偏见可以通过这些连接传播，增加了检测和减轻个体公平性侵犯的复杂性。本博士项目旨在开发一个测试框架以评估和确保GNNs中的个体公平性。项目首先系统地回顾个体公平性的文献，分类现有的方法以定义、衡量、测试和减轻模型偏差，创建个体公平性的分类框架。之后，项目将开发一个用于测试和确保GNNs公平性的框架，通过适应和扩展当前公平性测试和缓解技术来实现。该框架将通过基于图的大语言模型的工业案例研究进行评估。 

---
# TSCL:Multi-party loss Balancing scheme for deep learning Image steganography based on Curriculum learning 

**Title (ZH)**: TSCL：基于 Curriculum 学习的多 party 损失均衡的深度学习图像隐写术 

**Authors**: Fengchun Liu. Tong Zhang, Chunying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18348)  

**Abstract**: For deep learning-based image steganography frameworks, in order to ensure the invisibility and recoverability of the information embedding, the loss function usually contains several losses such as embedding loss, recovery loss and steganalysis loss. In previous research works, fixed loss weights are usually chosen for training optimization, and this setting is not linked to the importance of the steganography task itself and the training process. In this paper, we propose a Two-stage Curriculum Learning loss scheduler (TSCL) for balancing multinomial losses in deep learning image steganography algorithms. TSCL consists of two phases: a priori curriculum control and loss dynamics control. The first phase firstly focuses the model on learning the information embedding of the original image by controlling the loss weights in the multi-party adversarial training; secondly, it makes the model shift its learning focus to improving the decoding accuracy; and finally, it makes the model learn to generate a steganographic image that is resistant to steganalysis. In the second stage, the learning speed of each training task is evaluated by calculating the loss drop of the before and after iteration rounds to balance the learning of each task. Experimental results on three large public datasets, ALASKA2, VOC2012 and ImageNet, show that the proposed TSCL strategy improves the quality of steganography, decoding accuracy and security. 

**Abstract (ZH)**: 基于深度学习的图像隐写分析Loss调度的两阶段 Curriculum学习方法 

---
# PHEATPRUNER: Interpretable Data-centric Feature Selection for Multivariate Time Series Classification through Persistent Homology 

**Title (ZH)**: PHEATPRUNER: 基于持久同调的可解释多变量时间序列分类数据中心特征选择 

**Authors**: Anh-Duy Pham, Olivier Basole Kashongwe, Martin Atzmueller, Tim Römer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18329)  

**Abstract**: Balancing performance and interpretability in multivariate time series classification is a significant challenge due to data complexity and high dimensionality. This paper introduces PHeatPruner, a method integrating persistent homology and sheaf theory to address these challenges. Persistent homology facilitates the pruning of up to 45% of the applied variables while maintaining or enhancing the accuracy of models such as Random Forest, CatBoost, XGBoost, and LightGBM, all without depending on posterior probabilities or supervised optimization algorithms. Concurrently, sheaf theory contributes explanatory vectors that provide deeper insights into the data's structural nuances. The approach was validated using the UEA Archive and a mastitis detection dataset for dairy cows. The results demonstrate that PHeatPruner effectively preserves model accuracy. Furthermore, our results highlight PHeatPruner's key features, i.e. simplifying complex data and offering actionable insights without increasing processing time or complexity. This method bridges the gap between complexity reduction and interpretability, suggesting promising applications in various fields. 

**Abstract (ZH)**: 在多变量时间序列分类中平衡性能和可解释性是一项由于数据复杂性和高维度而面临的重大挑战。本文介绍了一种结合持久同调和层理论的PHeatPruner方法，以应对这些挑战。 

---
# Towards Adaptive Software Agents for Debugging 

**Title (ZH)**: 面向自适应软件代理的调试研究 

**Authors**: Yacine Majdoub, Eya Ben Charrada, Haifa Touati  

**Link**: [PDF](https://arxiv.org/pdf/2504.18316)  

**Abstract**: Using multiple agents was found to improve the debugging capabilities of Large Language Models. However, increasing the number of LLM-agents has several drawbacks such as increasing the running costs and rising the risk for the agents to lose focus. In this work, we propose an adaptive agentic design, where the number of agents and their roles are determined dynamically based on the characteristics of the task to be achieved. In this design, the agents roles are not predefined, but are generated after analyzing the problem to be solved. Our initial evaluation shows that, with the adaptive design, the number of agents that are generated depends on the complexity of the buggy code. In fact, for simple code with mere syntax issues, the problem was usually fixed using one agent only. However, for more complex problems, we noticed the creation of a higher number of agents. Regarding the effectiveness of the fix, we noticed an average improvement of 11% compared to the one-shot prompting. Given these promising results, we outline future research directions to improve our design for adaptive software agents that can autonomously plan and conduct their software goals. 

**Abstract (ZH)**: 使用多个代理被发现能够提高大型语言模型的调试能力。然而，增加LLM代理的数量也带来了一些缺点，如增加运行成本和增加代理失去焦点的风险。在此工作中，我们提出了一个自适应代理设计，其中代理的数量及其角色根据要实现的任务特性动态确定。在此设计中，代理的角色并非预先定义，而是在分析待解决问题后生成。初步评估显示，随着自适应设计的应用，生成的代理数量取决于存在错误代码的复杂性。实际上，对于仅有语法问题的简单代码，通常只需一个代理即可解决问题。然而，对于更复杂的问题，我们观察到代理的数量增加。关于修复的有效性，我们发现与单次提示相比，修复效果平均提高了11%。鉴于这些有前景的结果，我们概述了未来的研究方向，以改进适应性软件代理的设计，使其能够自主规划和实现其软件目标。 

---
# Artificial Intelligence health advice accuracy varies across languages and contexts 

**Title (ZH)**: 人工智能健康建议的准确性在不同语言和背景下有所差异 

**Authors**: Prashant Garg, Thiemo Fetzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18310)  

**Abstract**: Using basic health statements authorized by UK and EU registers and 9,100 journalist-vetted public-health assertions on topics such as abortion, COVID-19 and politics from sources ranging from peer-reviewed journals and government advisories to social media and news across the political spectrum, we benchmark six leading large language models from in 21 languages, finding that, despite high accuracy on English-centric textbook claims, performance falls in multiple non-European languages and fluctuates by topic and source, highlighting the urgency of comprehensive multilingual, domain-aware validation before deploying AI in global health communication. 

**Abstract (ZH)**: 使用英国和欧盟注册机构授权的基本健康声明以及9100条经记者和专家审核的公共健康主张，涵盖堕胎、COVID-19和政治等话题，来源包括同行评审期刊、政府建议、社交媒体和政治光谱内的新闻，我们对21种语言的六种领先大型语言模型进行了基准测试，发现尽管在以英语为中心的教科书主张方面表现准确，但在多种非欧洲语言中的表现却下降，并且在不同话题和来源之间波动，这强调了在全球健康沟通中部署AI之前进行全面多语言、领域意识验证的紧迫性。 

---
# Enhancing Long-Term Re-Identification Robustness Using Synthetic Data: A Comparative Analysis 

**Title (ZH)**: 使用合成数据增强长期重识别鲁棒性：一种比较分析 

**Authors**: Christian Pionzewski, Rebecca Rademacher, Jérôme Rutinowski, Antonia Ponikarov, Stephan Matzke, Tim Chilla, Pia Schreynemackers, Alice Kirchheim  

**Link**: [PDF](https://arxiv.org/pdf/2504.18286)  

**Abstract**: This contribution explores the impact of synthetic training data usage and the prediction of material wear and aging in the context of re-identification. Different experimental setups and gallery set expanding strategies are tested, analyzing their impact on performance over time for aging re-identification subjects. Using a continuously updating gallery, we were able to increase our mean Rank-1 accuracy by 24%, as material aging was taken into account step by step. In addition, using models trained with 10% artificial training data, Rank-1 accuracy could be increased by up to 13%, in comparison to a model trained on only real-world data, significantly boosting generalized performance on hold-out data. Finally, this work introduces a novel, open-source re-identification dataset, pallet-block-2696. This dataset contains 2,696 images of Euro pallets, taken over a period of 4 months. During this time, natural aging processes occurred and some of the pallets were damaged during their usage. These wear and tear processes significantly changed the appearance of the pallets, providing a dataset that can be used to generate synthetically aged pallets or other wooden materials. 

**Abstract (ZH)**: 合成训练数据使用对重识别中材料磨损和老化预测的影响研究：一种不断更新的画廊及其在重识别中老化对象性能的影响分析，并介绍新型开源重识别数据集pallet-block-2696 

---
# Neural operators struggle to learn complex PDEs in pedestrian mobility: Hughes model case study 

**Title (ZH)**: 神经算子在行人流动性中难以学习复杂偏微分方程：豪 ug 斯模型案例研究 

**Authors**: Prajwal Chauhan, Salah Eddine Choutri, Mohamed Ghattassi, Nader Masmoudi, Saif Eddin Jabari  

**Link**: [PDF](https://arxiv.org/pdf/2504.18267)  

**Abstract**: This paper investigates the limitations of neural operators in learning solutions for a Hughes model, a first-order hyperbolic conservation law system for crowd dynamics. The model couples a Fokker-Planck equation representing pedestrian density with a Hamilton-Jacobi-type (eikonal) equation. This Hughes model belongs to the class of nonlinear hyperbolic systems that often exhibit complex solution structures, including shocks and discontinuities. In this study, we assess the performance of three state-of-the-art neural operators (Fourier Neural Operator, Wavelet Neural Operator, and Multiwavelet Neural Operator) in various challenging scenarios. Specifically, we consider (1) discontinuous and Gaussian initial conditions and (2) diverse boundary conditions, while also examining the impact of different numerical schemes.
Our results show that these neural operators perform well in easy scenarios with fewer discontinuities in the initial condition, yet they struggle in complex scenarios with multiple initial discontinuities and dynamic boundary conditions, even when trained specifically on such complex samples. The predicted solutions often appear smoother, resulting in a reduction in total variation and a loss of important physical features. This smoothing behavior is similar to issues discussed by Daganzo (1995), where models that introduce artificial diffusion were shown to miss essential features such as shock waves in hyperbolic systems. These results suggest that current neural operator architectures may introduce unintended regularization effects that limit their ability to capture transport dynamics governed by discontinuities. They also raise concerns about generalizing these methods to traffic applications where shock preservation is essential. 

**Abstract (ZH)**: 本文探讨了神经运算符在学习Hughes模型解中的局限性，Hughes模型是用于 crowd dynamics 的一个一阶双曲守恒律系统，该模型将表示行人密度的Fokker-Planck方程与Hamilton-Jacobi类型（eikonal）方程耦合。本文评估了四种先进神经运算符（Fourier神经运算符、小波神经运算符和多重小波神经运算符）在各种具有挑战性的场景中的性能，具体考虑了具有不连续和高斯初始条件以及多样化的边界条件的情形，并考察了不同数值方案的影响。 

---
# Time and Frequency Domain-based Anomaly Detection in Smart Meter Data for Distribution Network Studies 

**Title (ZH)**: 基于时间和频率域的智能电表数据异常检测在配电网研究中的应用 

**Authors**: Petar Labura, Tomislav Antic, Tomislav Capuder  

**Link**: [PDF](https://arxiv.org/pdf/2504.18231)  

**Abstract**: The widespread integration of new technologies in low-voltage distribution networks on the consumer side creates the need for distribution system operators to perform advanced real-time calculations to estimate network conditions. In recent years, data-driven models based on machine learning and big data analysis have emerged for calculation purposes, leveraging the information available in large datasets obtained from smart meters and other advanced measurement infrastructure. However, existing data-driven algorithms do not take into account the quality of data collected from smart meters. They lack built-in anomaly detection mechanisms and fail to differentiate anomalies based on whether the value or context of anomalous data instances deviates from the norm. This paper focuses on methods for detecting and mitigating the impact of anomalies on the consumption of active and reactive power datasets. It proposes an anomaly detection framework based on the Isolation Forest machine learning algorithm and Fast Fourier Transform filtering that works in both the time and frequency domain and is unaffected by point anomalies or contextual anomalies of the power consumption data. The importance of integrating anomaly detection methods is demonstrated in the analysis important for distribution networks with a high share of smart meters. 

**Abstract (ZH)**: 低电压配电网络中消费侧新技術的广泛集成促使配电网运营商需要进行高级实时计算以估计网络状况。近年来，基于机器学习和大数据分析的数据驱动模型在计算中崭露头角，利用智能电表和其他先进测量基础设施获得的大数据集中的信息。然而，现有的数据驱动算法并未考虑到从智能电表收集的数据质量。它们缺乏内置的异常检测机制，无法根据异常数据实例的价值或上下文是否偏离常态来区分异常。本文专注于检测和减轻异常对有功和无功功率数据消耗影响的方法，提出了一种基于孤立森林机器学习算法和快速傅里叶变换滤波的异常检测框架，该框架在时频域均有效，不受电能消耗数据点异常或上下文异常的影响。在高智能电表渗透率的配电网分析中，集成异常检测方法的重要性得到体现。 

---
# Learning to fuse: dynamic integration of multi-source data for accurate battery lifespan prediction 

**Title (ZH)**: 基于学习的融合：多源数据的动态集成以实现精确的电池寿命预测 

**Authors**: He Shanxuan, Lin Zuhong, Yu Bolun, Gao Xu, Long Biao, Yao Jingjing  

**Link**: [PDF](https://arxiv.org/pdf/2504.18230)  

**Abstract**: Accurate prediction of lithium-ion battery lifespan is vital for ensuring operational reliability and reducing maintenance costs in applications like electric vehicles and smart grids. This study presents a hybrid learning framework for precise battery lifespan prediction, integrating dynamic multi-source data fusion with a stacked ensemble (SE) modeling approach. By leveraging heterogeneous datasets from the National Aeronautics and Space Administration (NASA), Center for Advanced Life Cycle Engineering (CALCE), MIT-Stanford-Toyota Research Institute (TRC), and nickel cobalt aluminum (NCA) chemistries, an entropy-based dynamic weighting mechanism mitigates variability across heterogeneous datasets. The SE model combines Ridge regression, long short-term memory (LSTM) networks, and eXtreme Gradient Boosting (XGBoost), effectively capturing temporal dependencies and nonlinear degradation patterns. It achieves a mean absolute error (MAE) of 0.0058, root mean square error (RMSE) of 0.0092, and coefficient of determination (R2) of 0.9839, outperforming established baseline models with a 46.2% improvement in R2 and an 83.2% reduction in RMSE. Shapley additive explanations (SHAP) analysis identifies differential discharge capacity (Qdlin) and temperature of measurement (Temp_m) as critical aging indicators. This scalable, interpretable framework enhances battery health management, supporting optimized maintenance and safety across diverse energy storage systems, thereby contributing to improved battery health management in energy storage systems. 

**Abstract (ZH)**: 基于多重数据融合与级联ensembles模型的锂离子电池寿命精准预测方法 

---
# EDU-NER-2025: Named Entity Recognition in Urdu Educational Texts using XLM-RoBERTa with X (formerly Twitter) 

**Title (ZH)**: EDU-NER-2025: Urdu教育文本中的命名实体识别使用XLM-RoBERTa with X（ formerly Twitter） 

**Authors**: Fida Ullah, Muhammad Ahmad, Muhammad Tayyab Zamir, Muhammad Arif, Grigori sidorov, Edgardo Manuel Felipe Riverón, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2504.18142)  

**Abstract**: Named Entity Recognition (NER) plays a pivotal role in various Natural Language Processing (NLP) tasks by identifying and classifying named entities (NEs) from unstructured data into predefined categories such as person, organization, location, date, and time. While extensive research exists for high-resource languages and general domains, NER in Urdu particularly within domain-specific contexts like education remains significantly underexplored. This is Due to lack of annotated datasets for educational content which limits the ability of existing models to accurately identify entities such as academic roles, course names, and institutional terms, underscoring the urgent need for targeted resources in this domain. To the best of our knowledge, no dataset exists in the domain of the Urdu language for this purpose. To achieve this objective this study makes three key contributions. Firstly, we created a manually annotated dataset in the education domain, named EDU-NER-2025, which contains 13 unique most important entities related to education domain. Second, we describe our annotation process and guidelines in detail and discuss the challenges of labelling EDU-NER-2025 dataset. Third, we addressed and analyzed key linguistic challenges, such as morphological complexity and ambiguity, which are prevalent in formal Urdu texts. 

**Abstract (ZH)**: 命名实体识别(NER)在各种自然语言处理(NLP)任务中扮演着关键角色，通过从无结构数据中识别并分类命名实体(NEs)到预定义的类别，如人名、组织、地名、日期和时间。虽然高资源语言和通用领域的研究广泛存在，但特别是在教育等特定领域的乌尔都语命名实体识别仍然显著未被探索。这主要是由于缺乏教育内容的标注数据集，限制了现有模型准确识别学术角色、课程名称和机构术语的能力，突显了在该领域迫切需要针对性资源。据我们所知，目前乌尔都语领域中没有用于这一目的的数据集。为了实现这一目标，本研究作出三项关键贡献。首先，我们创建了一个针对教育领域的手动标注数据集，命名为EDU-NER-2025，包含13个与教育领域密切相关的重要实体。其次，我们详细描述了我们的标注过程和指南，并讨论了标注EDU-NER-2025数据集时遇到的挑战。第三，我们解决了并分析了诸如形态复杂性和歧义性等关键语言挑战，这些挑战在正式乌尔都语文本中普遍存在。 

---
# Learning from Less: SINDy Surrogates in RL 

**Title (ZH)**: 从少量数据中学习：SINDy代理在强化学习中的应用 

**Authors**: Aniket Dixit, Muhammad Ibrahim Khan, Faizan Ahmed, James Brusey  

**Link**: [PDF](https://arxiv.org/pdf/2504.18113)  

**Abstract**: This paper introduces an approach for developing surrogate environments in reinforcement learning (RL) using the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm. We demonstrate the effectiveness of our approach through extensive experiments in OpenAI Gym environments, particularly Mountain Car and Lunar Lander. Our results show that SINDy-based surrogate models can accurately capture the underlying dynamics of these environments while reducing computational costs by 20-35%. With only 75 interactions for Mountain Car and 1000 for Lunar Lander, we achieve state-wise correlations exceeding 0.997, with mean squared errors as low as 3.11e-06 for Mountain Car velocity and 1.42e-06 for LunarLander position. RL agents trained in these surrogate environments require fewer total steps (65,075 vs. 100,000 for Mountain Car and 801,000 vs. 1,000,000 for Lunar Lander) while achieving comparable performance to those trained in the original environments, exhibiting similar convergence patterns and final performance metrics. This work contributes to the field of model-based RL by providing an efficient method for generating accurate, interpretable surrogate environments. 

**Abstract (ZH)**: 基于Sparse Identification of Nonlinear Dynamics的强化学习代理环境 surrogate 环境开发方法及其有效性探索 

---
# Efficient GNN Training Through Structure-Aware Randomized Mini-Batching 

**Title (ZH)**: 结构感知随机 minibatch 训练高效 GNN 

**Authors**: Vignesh Balaji, Christos Kozyrakis, Gal Chechik, Haggai Maron  

**Link**: [PDF](https://arxiv.org/pdf/2504.18082)  

**Abstract**: Graph Neural Networks (GNNs) enable learning on realworld graphs and mini-batch training has emerged as the de facto standard for training GNNs because it can scale to very large graphs and improve convergence. Current mini-batch construction policies largely ignore efficiency considerations of GNN training. Specifically, existing mini-batching techniques employ randomization schemes to improve accuracy and convergence. However, these randomization schemes are often agnostic to the structural properties of the graph (for eg. community structure), resulting in highly irregular memory access patterns during GNN training that make suboptimal use of on-chip GPU caches. On the other hand, while deterministic mini-batching based solely on graph structure delivers fast runtime performance, the lack of randomness compromises both the final model accuracy and training convergence speed. In this paper, we present Community-structure-aware Randomized Mini-batching (COMM-RAND), a novel methodology that bridges the gap between the above extremes. COMM-RAND allows practitioners to explore the space between pure randomness and pure graph structural awareness during mini-batch construction, leading to significantly more efficient GNN training with similar accuracy. We evaluated COMM-RAND across four popular graph learning benchmarks. COMM-RAND cuts down GNN training time by up to 2.76x (1.8x on average) while achieving an accuracy that is within 1.79% points (0.42% on average) compared to popular random mini-batching approaches. 

**Abstract (ZH)**: 面向社区结构的随机化mini-batch训练方法（COMM-RAND） 

---
# Privacy-Preserving Personalized Federated Learning for Distributed Photovoltaic Disaggregation under Statistical Heterogeneity 

**Title (ZH)**: 统计异构性下具有隐私保护的个性化联邦学习在分布式光伏负荷辨识中的应用 

**Authors**: Xiaolu Chen, Chenghao Huang, Yanru Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18078)  

**Abstract**: The rapid expansion of distributed photovoltaic (PV) installations worldwide, many being behind-the-meter systems, has significantly challenged energy management and grid operations, as unobservable PV generation further complicates the supply-demand balance. Therefore, estimating this generation from net load, known as PV disaggregation, is critical. Given privacy concerns and the need for large training datasets, federated learning becomes a promising approach, but statistical heterogeneity, arising from geographical and behavioral variations among prosumers, poses new challenges to PV disaggregation. To overcome these challenges, a privacy-preserving distributed PV disaggregation framework is proposed using Personalized Federated Learning (PFL). The proposed method employs a two-level framework that combines local and global modeling. At the local level, a transformer-based PV disaggregation model is designed to generate solar irradiance embeddings for representing local PV conditions. A novel adaptive local aggregation mechanism is adopted to mitigate the impact of statistical heterogeneity on the local model, extracting a portion of global information that benefits the local model. At the global level, a central server aggregates information uploaded from multiple data centers, preserving privacy while enabling cross-center knowledge sharing. Experiments on real-world data demonstrate the effectiveness of this proposed framework, showing improved accuracy and robustness compared to benchmark methods. 

**Abstract (ZH)**: 全球分布式光伏安装的快速扩张，尤其是背对计量系统的安装，显著挑战了能源管理和电网运营，不可观测的光伏发电进一步 complicates 供需平衡。因此，从净负荷估算这一发电过程，即光伏解聚集，至关重要。鉴于隐私问题和需要大规模训练数据集，联邦学习成为一个有希望的方法，但地理和行为差异导致的数据统计异质性给光伏解聚集带来了新挑战。为了克服这些挑战，提出了一种基于个性化联邦学习（PFL）的隐私保护分布式光伏解聚集框架。该方法采用两层框架结合局部和全局建模。在局部层面，设计了一个基于变压器的光伏解聚集模型来生成太阳能辐照度嵌入表示局部光伏条件。采用了新颖的自适应局部聚合机制来减轻统计异质性对局部模型的影响，提取有助于局部模型的全局信息部分。在全局层面，中央服务器从多个数据中心收集信息，同时保护隐私并促进跨中心的知识共享。实验结果表明，与基准方法相比，该提出的框架在准确性和鲁棒性方面取得了改进。 

---
# Exploring Personality-Aware Interactions in Salesperson Dialogue Agents 

**Title (ZH)**: 探索销售代理对话中的人格意识交互 

**Authors**: Sijia Cheng, Wen-Yu Chang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18058)  

**Abstract**: The integration of dialogue agents into the sales domain requires a deep understanding of how these systems interact with users possessing diverse personas. This study explores the influence of user personas, defined using the Myers-Briggs Type Indicator (MBTI), on the interaction quality and performance of sales-oriented dialogue agents. Through large-scale testing and analysis, we assess the pre-trained agent's effectiveness, adaptability, and personalization capabilities across a wide range of MBTI-defined user types. Our findings reveal significant patterns in interaction dynamics, task completion rates, and dialogue naturalness, underscoring the future potential for dialogue agents to refine their strategies to better align with varying personality traits. This work not only provides actionable insights for building more adaptive and user-centric conversational systems in the sales domain but also contributes broadly to the field by releasing persona-defined user simulators. These simulators, unconstrained by domain, offer valuable tools for future research and demonstrate the potential for scaling personalized dialogue systems across diverse applications. 

**Abstract (ZH)**: 将对话代理整合到销售领域需要深刻理解这些系统与具有多样化人设的用户交互的方式。本研究探讨了使用迈尔斯-布里格斯类型指标（MBTI）定义的用户人设对销售导向对话代理的交互质量和性能的影响。通过大规模测试和分析，我们评估了预训练代理在各种MBTI定义的用户类型中的有效性、适应性和个性化能力。研究发现揭示了交互动力学、任务完成率和对话自然度的显著模式，强调了对话代理在未来如何根据不同个性特征优化其策略的潜力。本研究不仅为构建更加适应用户需求的销售领域对话系统提供了可操作的见解，还通过发布人设定义的用户模拟器为该领域未来的研究提供了广泛贡献，这些模拟器不受领域限制，为未来研究提供了有价值的工具，并展示了跨多种应用场景规模化个性化对话系统的能力。 

---
# AI Ethics and Social Norms: Exploring ChatGPT's Capabilities From What to How 

**Title (ZH)**: AI伦理与社会规范：探究ChatGPT能力的从“是什么”到“怎么做” 

**Authors**: Omid Veisi, Sasan Bahrami, Roman Englert, Claudia Müller  

**Link**: [PDF](https://arxiv.org/pdf/2504.18044)  

**Abstract**: Using LLMs in healthcare, Computer-Supported Cooperative Work, and Social Computing requires the examination of ethical and social norms to ensure safe incorporation into human life. We conducted a mixed-method study, including an online survey with 111 participants and an interview study with 38 experts, to investigate the AI ethics and social norms in ChatGPT as everyday life tools. This study aims to evaluate whether ChatGPT in an empirical context operates following ethics and social norms, which is critical for understanding actions in industrial and academic research and achieving machine ethics. The findings of this study provide initial insights into six important aspects of AI ethics, including bias, trustworthiness, security, toxicology, social norms, and ethical data. Significant obstacles related to transparency and bias in unsupervised data collection methods are identified as ChatGPT's ethical concerns. 

**Abstract (ZH)**: 在医疗保健、计算机支持的协作工作和社交计算中使用大语言模型需要审视伦理和社会规范以确保安全地融入人类生活。我们进行了一项混合方法研究，包括一项有111名参与者在线调查和一项有38名专家的访谈研究，以调查ChatGPT作为日常生活工具中的AI伦理和社会规范。本研究旨在评估在实际情境中ChatGPT是否遵循伦理和社会规范，这对于理解和实现机器伦理至关重要。本研究的发现提供了关于AI伦理的六个重要方面的初步见解，包括偏见、可信度、安全性、毒性、社会规范和伦理数据。识别出与无监督数据收集方法相关的透明度和偏见的重大障碍是ChatGPT的伦理关切。 

---
# Addressing Concept Mislabeling in Concept Bottleneck Models Through Preference Optimization 

**Title (ZH)**: 通过偏好优化解决概念瓶颈模型中的概念误标问题 

**Authors**: Emiliano Penaloza, Tianyue H. Zhan, Laurent Charlin, Mateo Espinosa Zarlenga  

**Link**: [PDF](https://arxiv.org/pdf/2504.18026)  

**Abstract**: Concept Bottleneck Models (CBMs) propose to enhance the trustworthiness of AI systems by constraining their decisions on a set of human understandable concepts. However, CBMs typically assume that datasets contains accurate concept labels an assumption often violated in practice, which we show can significantly degrade performance (by 25% in some cases). To address this, we introduce the Concept Preference Optimization (CPO) objective, a new loss function based on Direct Preference Optimization, which effectively mitigates the negative impact of concept mislabeling on CBM performance. We provide an analysis on some key properties of the CPO objective showing it directly optimizes for the concept's posterior distribution, and contrast it against Binary Cross Entropy (BCE) where we show CPO is inherently less sensitive to concept noise. We empirically confirm our analysis finding that CPO consistently outperforms BCE in three real world datasets with and without added label noise. 

**Abstract (ZH)**: 概念偏好优化（CPO）目标：提升概念瓶颈模型（CBM）的稳健性以应对概念标记错误问题 

---
# Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts 

**Title (ZH)**: 避免泄露污染：分布变化下的概念干预 

**Authors**: Mateo Espinosa Zarlenga, Gabriele Dominici, Pietro Barbiero, Zohreh Shams, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2504.17921)  

**Abstract**: In this paper, we investigate how concept-based models (CMs) respond to out-of-distribution (OOD) inputs. CMs are interpretable neural architectures that first predict a set of high-level concepts (e.g., stripes, black) and then predict a task label from those concepts. In particular, we study the impact of concept interventions (i.e., operations where a human expert corrects a CM's mispredicted concepts at test time) on CMs' task predictions when inputs are OOD. Our analysis reveals a weakness in current state-of-the-art CMs, which we term leakage poisoning, that prevents them from properly improving their accuracy when intervened on for OOD inputs. To address this, we introduce MixCEM, a new CM that learns to dynamically exploit leaked information missing from its concepts only when this information is in-distribution. Our results across tasks with and without complete sets of concept annotations demonstrate that MixCEMs outperform strong baselines by significantly improving their accuracy for both in-distribution and OOD samples in the presence and absence of concept interventions. 

**Abstract (ZH)**: 本文研究了基于概念的模型（CMs）对分布外（OOD）输入的响应。CMs 是可解释的神经架构，首先预测一组高层概念（例如：条纹、黑色），然后从这些概念中预测任务标签。特别地，我们研究了在测试时人为专家修正CM错误预测的概念对CM的任务预测的影响，特别是在输入为OOD时。我们的分析揭示了当前最先进的CMs的一个弱点，我们称之为泄露毒害，这阻碍了它们在人为干预时提高针对OOD输入的准确性。为了解决这个问题，我们提出了MixCEM，这是一种新的CM，能够仅在该信息为分布内时动态地利用其概念中缺失的泄露信息。我们的跨任务实验结果表明，在有和没有完整概念注解的情况下，MixCEMs在概念干预存在和不存在的情况下，显著提高了其对分布内和OOD样本的准确性，从而在基准模型中表现出色。 

---
# Token Sequence Compression for Efficient Multimodal Computing 

**Title (ZH)**: 高效的多模态计算中的令牌序列压缩 

**Authors**: Yasmine Omri, Parth Shroff, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2504.17892)  

**Abstract**: The exponential growth of Large Multimodal Models (LMMs) has driven advancements in cross-modal reasoning but at significant computational costs. In this work, we focus on visual language models. We highlight the redundancy and inefficiency in current vision encoders, and seek to construct an adaptive compression method for multimodal data. In this work, we characterize a panoply of visual token selection and merging approaches through both benchmarking and qualitative analysis. In particular, we demonstrate that simple cluster-level token aggregation outperforms prior state-of-the-art works in token selection and merging, including merging at the vision encoder level and attention-based approaches. We underline the redundancy in current vision encoders, and shed light on several puzzling trends regarding principles of visual token selection through cross-modal attention visualizations. This work is a first effort towards more effective encoding and processing of high-dimensional data, and paves the way for more scalable and sustainable multimodal systems. 

**Abstract (ZH)**: 大型多模态模型的指数增长推动了跨模态推理的进步，但伴随着巨大的计算成本。在此工作中，我们关注视觉语言模型。我们指出现有视觉编码器中的冗余和低效性，并寻求构建一种适应性压缩方法以处理多模态数据。在此工作中，我们通过基准测试和定性分析，表征了多种视觉标记选择和合并方法。特别是，我们证明了简单的聚类级别标记聚合在标记选择和合并方面优于之前的最佳方法，包括视编码器级别的合并和基于注意力的方法。我们指出现有视觉编码器中的冗余问题，并通过跨模态注意可视化揭示了几种关于视觉标记选择原理的令人困惑的趋势。此工作是更有效地编码和处理高维数据的首次尝试，并为更可扩展和可持续的多模态系统铺平了道路。 

---
# Crypto-ncRNA: Non-coding RNA (ncRNA) Based Encryption Algorithm 

**Title (ZH)**: Crypto-ncRNA：非编码RNA（ncRNA）基于的加密算法 

**Authors**: Xu Wang, Yiquan Wang, Tin-yeh Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17878)  

**Abstract**: In the looming post-quantum era, traditional cryptographic systems are increasingly vulnerable to quantum computing attacks that can compromise their mathematical foundations. To address this critical challenge, we propose crypto-ncRNA-a bio-convergent cryptographic framework that leverages the dynamic folding properties of non-coding RNA (ncRNA) to generate high-entropy, quantum-resistant keys and produce unpredictable ciphertexts. The framework employs a novel, multi-stage process: encoding plaintext into RNA sequences, predicting and manipulating RNA secondary structures using advanced algorithms, and deriving cryptographic keys through the intrinsic physical unclonability of RNA molecules. Experimental evaluations indicate that, although crypto-ncRNA's encryption speed is marginally lower than that of AES, it significantly outperforms RSA in terms of efficiency and scalability while achieving a 100% pass rate on the NIST SP 800-22 randomness tests. These results demonstrate that crypto-ncRNA offers a promising and robust approach for securing digital infrastructures against the evolving threats posed by quantum computing. 

**Abstract (ZH)**: 在即将到来的后量子时代，传统密码系统日益 Vulnerable 于量子计算攻击，这些攻击可以破坏其数学基础。为应对这一关键挑战，我们提出了一种基于非编码 RNA (ncRNA) 动态折叠性质的加密框架——crypto-ncRNA，利用 RNA 分子的固有物理不可克隆性生成高熵、量子抗性的密钥并产生不可预测的密文。该框架采用一种新颖的多阶段过程：将明文编码为 RNA 序列、使用高级算法预测和操控 RNA 的二级结构以及通过 RNA 分子的固有物理不可克隆性导出密码学密钥。实验评估表明，尽管 crypto-ncRNA 的加密速度略低于 AES，但在效率和扩展性方面，它显著优于 RSA，并且在 NIST SP 800-22 随机性测试中通过率达到了 100%。这些结果展示了 crypto-ncRNA 作为一种抵御量子计算所带来的演进威胁的有前景且稳健的保护数字基础设施的方法。 

---
# Evolution Meets Diffusion: Efficient Neural Architecture Generation 

**Title (ZH)**: 进化相遇扩散：高效神经架构生成 

**Authors**: Bingye Zhou, Caiyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17827)  

**Abstract**: Neural Architecture Search (NAS) has gained widespread attention for its transformative potential in deep learning model design. However, the vast and complex search space of NAS leads to significant computational and time costs. Neural Architecture Generation (NAG) addresses this by reframing NAS as a generation problem, enabling the precise generation of optimal architectures for specific tasks. Despite its promise, mainstream methods like diffusion models face limitations in global search capabilities and are still hindered by high computational and time demands. To overcome these challenges, we propose Evolutionary Diffusion-based Neural Architecture Generation (EDNAG), a novel approach that achieves efficient and training-free architecture generation. EDNAG leverages evolutionary algorithms to simulate the denoising process in diffusion models, using fitness to guide the transition from random Gaussian distributions to optimal architecture distributions. This approach combines the strengths of evolutionary strategies and diffusion models, enabling rapid and effective architecture generation. Extensive experiments demonstrate that EDNAG achieves state-of-the-art (SOTA) performance in architecture optimization, with an improvement in accuracy of up to 10.45%. Furthermore, it eliminates the need for time-consuming training and boosts inference speed by an average of 50 times, showcasing its exceptional efficiency and effectiveness. 

**Abstract (ZH)**: 基于进化扩散的神经架构生成（Evolutionary Diffusion-based Neural Architecture Generation）：高效无训练的架构生成 

---
# The Cloud Weaving Model for AI development 

**Title (ZH)**: AI发展中的云编织模型 

**Authors**: Darcy Kim, Aida Kalender, Sennay Ghebreab, Giovanni Sileno  

**Link**: [PDF](https://arxiv.org/pdf/2504.17823)  

**Abstract**: While analysing challenges in pilot projects developing AI with marginalized communities, we found it difficult to express them within commonly used paradigms. We therefore constructed an alternative conceptual framework to ground AI development in the social fabric -- the Cloud Weaving Model -- inspired (amongst others) by indigenous knowledge, motifs from nature, and Eastern traditions. This paper introduces and elaborates on the fundamental elements of the model (clouds, spiders, threads, spiderwebs, and weather) and their interpretation in an AI context. The framework is then applied to comprehend patterns observed in co-creation pilots approaching marginalized communities, highlighting neglected yet relevant dimensions for responsible AI development. 

**Abstract (ZH)**: 基于云织模型：在边缘化社区中开发人工智能所面临的挑战及其替代概念框架 

---
# Fuzzy Logic -- Based Scheduling System for Part-Time Workforce 

**Title (ZH)**: 基于模糊逻辑的兼职人员调度系统 

**Authors**: Tri Nguyen, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17805)  

**Abstract**: This paper explores the application of genetic fuzzy systems to efficiently generate schedules for a team of part-time student workers at a university. Given the preferred number of working hours and availability of employees, our model generates feasible solutions considering various factors, such as maximum weekly hours, required number of workers on duty, and the preferred number of working hours. The algorithm is trained and tested with availability data collected from students at the University of Cincinnati. The results demonstrate the algorithm's efficiency in producing schedules that meet operational criteria and its robustness in understaffed conditions. 

**Abstract (ZH)**: 本研究探讨了遗传模糊系统在高效生成大学兼职学生员工团队工作时间表中的应用。根据员工的偏好工作时数和可用性，我们的模型在考虑诸如每周最大工时、所需的值班员工数量以及偏好工作时数等因素的情况下生成可行的解决方案。该算法使用俄亥俄州辛辛那提大学学生收集的可用性数据进行训练和测试。研究结果展示了该算法在满足运营标准方面和 understaffed 条件下的鲁棒性。 

---
# Subfunction Structure Matters: A New Perspective on Local Optima Networks 

**Title (ZH)**: 子功能结构很重要：局部最优网络的一个新视角 

**Authors**: S. L. Thomson, M. W. Przewozniczek  

**Link**: [PDF](https://arxiv.org/pdf/2504.17799)  

**Abstract**: Local optima networks (LONs) capture fitness landscape information. They are typically constructed in a black-box manner; information about the problem structure is not utilised. This also applies to the analysis of LONs: knowledge about the problem, such as interaction between variables, is not considered. We challenge this status-quo with an alternative approach: we consider how LON analysis can be improved by incorporating subfunction-based information - this can either be known a-priori or learned during search. To this end, LONs are constructed for several benchmark pseudo-boolean problems using three approaches: firstly, the standard algorithm; a second algorithm which uses deterministic grey-box crossover; and a third algorithm which selects perturbations based on learned information about variable interactions. Metrics related to subfunction changes in a LON are proposed and compared with metrics from previous literature which capture other aspects of a LON. Incorporating problem structure in LON construction and analysing it can bring enriched insight into optimisation dynamics. Such information may be crucial to understanding the difficulty of solving a given problem with state-of-the-art linkage learning optimisers. In light of the results, we suggest incorporation of problem structure as an alternative paradigm in landscape analysis for problems with known or suspected subfunction structure. 

**Abstract (ZH)**: 局部最优网络（LONs）捕获了适应度景观信息。它们通常以黑盒方式构建；问题结构的相关信息未被利用。这也适用于LON分析：关于问题的知识，例如变量间的交互作用，未被考虑。我们提出了一个替代方法来挑战这一现状：我们探讨如何通过结合基于子函数的信息来改进LON分析——这些信息可以先验地已知或者在搜索过程中学习到。为此，我们使用三种方法为几种基准伪布尔问题构建LON：首先，标准算法；其次，使用确定性灰盒 crossover 的算法；最后，基于关于变量交互信息的学习来进行扰动选择的算法。提出了与LON中的子函数变化相关的度量，并将其与捕捉LON其他方面的度量进行比较。在LON构建和分析中融入问题结构可以提供更丰富的优化动力学见解。这些信息对于理解利用当前最先进的连接学习优化器解决给定问题的难度可能是至关重要的。基于结果，我们建议将问题结构的融入作为景观分析中的一种替代范式，用于具有已知或疑似子函数结构的问题。 

---
# My Precious Crash Data: Barriers and Opportunities in Encouraging Autonomous Driving Companies to Share Safety-Critical Data 

**Title (ZH)**: 我宝贵的安全数据：鼓励自动驾驶公司共享关键安全数据的障碍与机遇 

**Authors**: Hauke Sandhaus, Angel Hsing-Chi Hwang, Wendy Ju, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17792)  

**Abstract**: Safety-critical data, such as crash and near-crash records, are crucial to improving autonomous vehicle (AV) design and development. Sharing such data across AV companies, academic researchers, regulators, and the public can help make all AVs safer. However, AV companies rarely share safety-critical data externally. This paper aims to pinpoint why AV companies are reluctant to share safety-critical data, with an eye on how these barriers can inform new approaches to promote sharing. We interviewed twelve AV company employees who actively work with such data in their day-to-day work. Findings suggest two key, previously unknown barriers to data sharing: (1) Datasets inherently embed salient knowledge that is key to improving AV safety and are resource-intensive. Therefore, data sharing, even within a company, is fraught with politics. (2) Interviewees believed AV safety knowledge is private knowledge that brings competitive edges to their companies, rather than public knowledge for social good. We discuss the implications of these findings for incentivizing and enabling safety-critical AV data sharing, specifically, implications for new approaches to (1) debating and stratifying public and private AV safety knowledge, (2) innovating data tools and data sharing pipelines that enable easier sharing of public AV safety data and knowledge; (3) offsetting costs of curating safety-critical data and incentivizing data sharing. 

**Abstract (ZH)**: 安全关键数据，例如碰撞和接近碰撞记录，对于提升自动驾驶车辆（AV）的设计与开发至关重要。跨AV公司、学术研究人员、监管机构和公众共享此类数据有助于提高所有AV的安全性。然而，AV公司很少对外共享安全关键数据。本文旨在探究AV公司不愿共享安全关键数据的原因，并分析这些障碍如何有助于提出新的促进数据共享的方法。我们对十二名在日常工作中积极处理此类数据的AV公司员工进行了访谈。研究发现存在两个新的关键障碍：（1）数据集本身嵌入了对提升AV安全性至关重要的核心知识，并且资源密集，因此即使在公司内部，共享数据也充满了政治因素。（2）受访者认为AV安全知识是公司的私有知识，为公司带来竞争优势，而不是社会公共知识。我们讨论了这些发现对激励和促进安全关键AV数据共享的含义，具体包括（1）关于争论和划分公共与私营AV安全知识的新方法，（2）创新数据工具和数据共享管道以使公共AV安全数据和知识更容易共享，（3）抵消收集安全关键数据的成本并激励数据共享的含义。 

---
