# Towards more efficient quantitative safety validation of residual risk for assisted and automated driving 

**Title (ZH)**: 面向辅助和自动驾驶残余风险更高效的定量安全验证 

**Authors**: Daniel Betschinske, Malte Schrimpf, Steven Peters, Kamil Klonecki, Jan Peter Karch, Moritz Lippert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10363)  

**Abstract**: The safety validation of Advanced Driver Assistance Systems (ADAS) and Automated Driving Systems (ADS) increasingly demands efficient and reliable methods to quantify residual risk while adhering to international standards such as ISO 21448. Traditionally, Field Operational Testing (FOT) has been pivotal for macroscopic safety validation of automotive driving functions up to SAE automation level 2. However, state-of-the-art derivations for empirical safety demonstrations using FOT often result in impractical testing efforts, particularly at higher automation levels. Even at lower automation levels, this limitation - coupled with the substantial costs associated with FOT - motivates the exploration of approaches to enhance the efficiency of FOT-based macroscopic safety validation. Therefore, this publication systematically identifies and evaluates state-of-the-art Reduction Approaches (RAs) for FOT, including novel methods reported in the literature. Based on an analysis of ISO 21448, two models are derived: a generic model capturing the argumentation components of the standard, and a base model, exemplarily applied to Automatic Emergency Braking (AEB) systems, establishing a baseline for the real-world driving requirement for a Quantitative Safety Validation of Residual Risk (QSVRR). Subsequently, the RAs are assessed using four criteria: quantifiability, threats to validity, missing links, and black box compatibility, highlighting potential benefits, inherent limitations, and identifying key areas for further research. Our evaluation reveals that, while several approaches offer potential, none are free from missing links or other substantial shortcomings. Moreover, no identified alternative can fully replace FOT, reflecting its crucial role in the safety validation of ADAS and ADS. 

**Abstract (ZH)**: 高级驾驶辅助系统（ADAS）和自动驾驶系统（ADS）的安全验证越来越多地需要符合国际标准（如ISO 21448）的高效可靠方法来量化剩余风险。传统的场操作测试（FOT）在宏观层面验证自动驾驶功能（至SAE自动化水平2）方面一直至关重要。然而，用于FOT的先进经验性安全演示推导往往导致不切实际的测试努力，特别是在更高自动化水平时。即使在较低自动化水平，FOT的这一局限性以及其高昂的成本促使探索提高FOT宏层面安全验证效率的方法。因此，本文系统地识别和评估了FOT的最新减小型方案（RAs），包括文献中报道的新方法。基于对ISO 21448的分析，提出了两个模型：一个泛化模型捕捉标准的论点组件，以及一个基础模型，用于例示自动紧急制动（AEB）系统的实车安全性要求，建立量化剩余风险（QSVRR）的实际驾驶需求基准。随后，使用四个标准（可量化性、有效性的威胁、缺失链接和黑盒兼容性）评估RAs，强调潜在益处、固有的局限性，并确定需要进一步研究的关键领域。我们的评估表明，尽管一些方法具有潜力，但没有一个方法是完美的，更没有一种方法能够完全取代FOT在ADAS和ADS安全性验证中的关键作用。 

---
# Multi-Timescale Dynamics Model Bayesian Optimization for Plasma Stabilization in Tokamaks 

**Title (ZH)**: 多时间尺度动力学模型贝叶斯优化方法在托卡马克中的等离子体稳定化 

**Authors**: Rohit Sonker, Alexandre Capone, Andrew Rothstein, Hiro Josep Farre Kaga, Egemen Kolemen, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.10287)  

**Abstract**: Machine learning algorithms often struggle to control complex real-world systems. In the case of nuclear fusion, these challenges are exacerbated, as the dynamics are notoriously complex, data is poor, hardware is subject to failures, and experiments often affect dynamics beyond the experiment's duration. Existing tools like reinforcement learning, supervised learning, and Bayesian optimization address some of these challenges but fail to provide a comprehensive solution. To overcome these limitations, we present a multi-scale Bayesian optimization approach that integrates a high-frequency data-driven dynamics model with a low-frequency Gaussian process. By updating the Gaussian process between experiments, the method rapidly adapts to new data, refining the predictions of the less reliable dynamical model. We validate our approach by controlling tearing instabilities in the DIII-D nuclear fusion plant. Offline testing on historical data shows that our method significantly outperforms several baselines. Results on live experiments on the DIII-D tokamak, conducted under high-performance plasma scenarios prone to instabilities, shows a 50% success rate, marking a 117% improvement over historical outcomes. 

**Abstract (ZH)**: 机器学习算法在控制复杂现实系统时常常面临挑战。在核聚变领域，这些挑战更为严峻，因为动力学极其复杂、数据质量较差、硬件易故障，且实验往往会影响实验持续时间之外的动力学。现有的工具如强化学习、监督学习和贝叶斯优化部分解决了这些问题，但未能提供全面的解决方案。为克服这些限制，我们提出了一种多尺度贝叶斯优化方法，该方法将高频数据驱动的动力学模型与低频高斯过程结合。通过在实验之间更新高斯过程，该方法能够迅速适应新数据，改进可靠性较低的动力学模型的预测。我们通过控制DIII-D核聚变装置中的剥离不稳定性来验证我们的方法。离线历史数据测试表明，我们的方法显著优于几种基线方法。在DIII-D托卡马克进行的实时实验中，在高度易发生不稳定的高性能等离子体场景下，成功率达到50%，比历史结果提高了117%。 

---
# Learning Safe Control via On-the-Fly Bandit Exploration 

**Title (ZH)**: 基于即时bandit探索的.safe控制学习 

**Authors**: Alexandre Capone, Ryan Cosner, Aaaron Ames, Sandra Hirche  

**Link**: [PDF](https://arxiv.org/pdf/2506.10279)  

**Abstract**: Control tasks with safety requirements under high levels of model uncertainty are increasingly common. Machine learning techniques are frequently used to address such tasks, typically by leveraging model error bounds to specify robust constraint-based safety filters. However, if the learned model uncertainty is very high, the corresponding filters are potentially invalid, meaning no control input satisfies the constraints imposed by the safety filter. While most works address this issue by assuming some form of safe backup controller, ours tackles it by collecting additional data on the fly using a Gaussian process bandit-type algorithm. We combine a control barrier function with a learned model to specify a robust certificate that ensures safety if feasible. Whenever infeasibility occurs, we leverage the control barrier function to guide exploration, ensuring the collected data contributes toward the closed-loop system safety. By combining a safety filter with exploration in this manner, our method provably achieves safety in a setting that allows for a zero-mean prior dynamics model, without requiring a backup controller. To the best of our knowledge, it is the first safe learning-based control method that achieves this. 

**Abstract (ZH)**: 在高模型不确定性下执行具有安全要求的任务正变得越来越常见。机器学习技术通常被用来处理这类任务，通常通过利用模型误差边界来指定鲁棒的基于约束的安全过滤器。然而，如果学习到的模型不确定性非常高，相应的过滤器可能是无效的，这意味着没有控制输入能够满足安全过滤器施加的约束。大多数工作通过假设某种形式的安全备用控制器来解决这个问题，而我们则通过使用高斯过程宝瓶式的算法收集额外数据来处理这一问题。我们将控制屏障函数与学习到的模型相结合，以指定一个鲁棒证书，该证书确保如果可行，能确保安全性。每当不可行性发生时，我们利用控制屏障函数来引导探索，确保收集的数据有助于闭环系统的安全性。通过以这种方式结合安全过滤器和探索性，我们的方法在允许零均值先验动力学模型的环境中能够证明实现安全性，而无需备用控制器。据我们所知，这是第一个能够在不需要备用控制器的情况下实现这一目标的安全学习控制方法。 

---
# Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models 

**Title (ZH)**: 基于高斯混合 copula 模型的场景参数联合概率估计算法 

**Authors**: Christian Reichenbächer, Philipp Rank, Jochen Hipp, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10098)  

**Abstract**: This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependencies. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from scenarios defined in United Nations Regulation No. 157. Our evaluation across 18 million scenario instances demonstrates that Gaussian Mixture Copula Models provide a better fit to the data in terms of both likelihood and Sinkhorn distance. These results suggest that Gaussian Mixture Copula Models are a compelling foundation for future scenario-based validation frameworks. 

**Abstract (ZH)**: 基于高斯混合 copula 模型的自动驾驶场景统计建模及其安全验证应用 

---
# Provable Sim-to-Real Transfer via Offline Domain Randomization 

**Title (ZH)**: 可验证的离线领域随机化实现从模拟到现实的迁移 

**Authors**: Arnaud Fickinger, Abderrahim Bendahi, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2506.10133)  

**Abstract**: Reinforcement-learning agents often struggle when deployed from simulation to the real-world. A dominant strategy for reducing the sim-to-real gap is domain randomization (DR) which trains the policy across many simulators produced by sampling dynamics parameters, but standard DR ignores offline data already available from the real system. We study offline domain randomization (ODR), which first fits a distribution over simulator parameters to an offline dataset. While a growing body of empirical work reports substantial gains with algorithms such as DROPO, the theoretical foundations of ODR remain largely unexplored. In this work, we (i) formalize ODR as a maximum-likelihood estimation over a parametric simulator family, (ii) prove consistency of this estimator under mild regularity and identifiability conditions, showing it converges to the true dynamics as the dataset grows, (iii) derive gap bounds demonstrating ODRs sim-to-real error is up to an O(M) factor tighter than uniform DR in the finite-simulator case (and analogous gains in the continuous setting), and (iv) introduce E-DROPO, a new version of DROPO which adds an entropy bonus to prevent variance collapse, yielding broader randomization and more robust zero-shot transfer in practice. 

**Abstract (ZH)**: Offline Domain Randomizationagt;往往在从模拟环境部署到真实世界时难以适应。我们研究了Offline Domain Randomization（ODR），首先根据离线数据集拟合模拟器参数的分布。尽管有关ODR的实证研究显示了显著的改进，但其理论基础仍相对未被探索。在本文中，我们（i）将ODR形式化为参数化模拟器族的最大似然估计，（ii）在温和的正则性和可识别性条件下证明了该估计器的一致性，展示了随着数据集规模的增长，该估计器收敛于真实动力学，（iii）推导出间隙界，表明在有限模拟器情形下ODR的sim-to-real误差最多比均匀DR小O(M)倍（在连续情形下也有类似的优势），并（iv）引入了E-DROPO，这是一种改进的DROPO版本，增加了熵奖励以防止方差崩溃，实测中提供了更广泛的随机化和更强鲁棒性的零 shot 转移。 

---
# Spurious Rewards: Rethinking Training Signals in RLVR 

**Title (ZH)**: 虚假奖励：重新思考RLVR中的训练信号 

**Authors**: Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.10947)  

**Abstract**: We show that reinforcement learning with verifiable rewards (RLVR) can elicit strong mathematical reasoning in certain models even with spurious rewards that have little, no, or even negative correlation with the correct answer. For example, RLVR improves MATH-500 performance for Qwen2.5-Math-7B in absolute points by 21.4% (random reward), 13.8% (format reward), 24.1% (incorrect label), 26.0% (1-shot RL), and 27.1% (majority voting) -- nearly matching the 29.1% gained with ground truth rewards. However, the spurious rewards that work for Qwen often fail to yield gains with other model families like Llama3 or OLMo2. In particular, we find code reasoning -- thinking in code without actual code execution -- to be a distinctive Qwen2.5-Math behavior that becomes significantly more frequent after RLVR, from 65% to over 90%, even with spurious rewards. Overall, we hypothesize that, given the lack of useful reward signal, RLVR must somehow be surfacing useful reasoning representations learned during pretraining, although the exact mechanism remains a topic for future work. We suggest that future RLVR research should possibly be validated on diverse models rather than a single de facto choice, as we show that it is easy to get significant performance gains on Qwen models even with completely spurious reward signals. 

**Abstract (ZH)**: 我们展示了一种验证性奖励（RLVR）强化学习即使在与正确答案相关性很低、无相关或甚至负相关的虚假奖励下，仍能在某些模型中激发强烈的数学推理能力。强化学习与可验证奖励（RLVR）提高了Qwen2.5-Math-7B在MATH-500上的性能，分别在随机奖励、格式奖励、错误标签、单次RL和其他模型的多数投票中提高了21.4%、13.8%、24.1%、26.0%和27.1%——几乎与使用真实奖励信号获得的29.1%的改进持平。然而，适用于Qwen的虚假奖励往往不能为其他模型家族如Llama3或OLMo2带来提升。特别是，我们发现代码推理——在没有实际代码执行的情况下思考代码——是Qwen2.5-Math的独特行为，在RLVR后变得更为频繁，即使在虚假奖励下，这一比例也从65%上升到超过90%。总体而言，我们推测由于缺乏有用的奖励信号，RLVR必须以某种方式揭示预训练中学到的有用推理表示，但确切机制仍有待未来研究探讨。我们建议未来的RLVR研究应该在多种模型上进行验证，而不仅仅是在单一默认选择上，因为我们展示了即使在完全虚假的奖励信号下，也能在Qwen模型上获得显著性能提升。 

---
# Think before You Simulate: Symbolic Reasoning to Orchestrate Neural Computation for Counterfactual Question Answering 

**Title (ZH)**: 深思而后模拟：符号推理 orchestrating 神经计算以进行反事实问题回答 

**Authors**: Adam Ishay, Zhun Yang, Joohyung Lee, Ilgu Kang, Dongjae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10753)  

**Abstract**: Causal and temporal reasoning about video dynamics is a challenging problem. While neuro-symbolic models that combine symbolic reasoning with neural-based perception and prediction have shown promise, they exhibit limitations, especially in answering counterfactual questions. This paper introduces a method to enhance a neuro-symbolic model for counterfactual reasoning, leveraging symbolic reasoning about causal relations among events. We define the notion of a causal graph to represent such relations and use Answer Set Programming (ASP), a declarative logic programming method, to find how to coordinate perception and simulation modules. We validate the effectiveness of our approach on two benchmarks, CLEVRER and CRAFT. Our enhancement achieves state-of-the-art performance on the CLEVRER challenge, significantly outperforming existing models. In the case of the CRAFT benchmark, we leverage a large pre-trained language model, such as GPT-3.5 and GPT-4, as a proxy for a dynamics simulator. Our findings show that this method can further improve its performance on counterfactual questions by providing alternative prompts instructed by symbolic causal reasoning. 

**Abstract (ZH)**: 关于视频动态的因果和时间推理是一个具有挑战性的问题。虽然结合符号推理与基于神经网络的感知和预测的神经-符号模型显示出了潜力，但在回答反事实问题时表现出局限性。本文提出了一种增强神经-符号模型以进行反事实推理的方法，利用事件之间因果关系的符号推理。我们定义因果图来表示这些关系，并使用回答集编程（ASP），一种声明式逻辑编程方法，来找出感知和模拟模块的协调方式。我们在CLEVRER和CRAFT两个基准上验证了该方法的有效性。在CLEVRER挑战中，我们的增强方法达到了最先进的性能，显著优于现有模型。对于CRAFT基准，我们利用大型预训练语言模型，如GPT-3.5和GPT-4，作为动力学模拟器的代理。我们的研究结果表明，通过由符号因果推理提供的替代提示，这种方法可以在反事实问题上进一步提高其性能。 

---
# System ASPMT2SMT:Computing ASPMT Theories by SMT Solvers 

**Title (ZH)**: System ASPMT2SMT: 由SMT求解器计算ASPMT理论 

**Authors**: Michael Bartholomew, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.10708)  

**Abstract**: Answer Set Programming Modulo Theories (ASPMT) is an approach to combining answer set programming and satisfiability modulo theories based on the functional stable model semantics. It is shown that the tight fragment of ASPMT programs can be turned into SMT instances, thereby allowing SMT solvers to compute stable models of ASPMT programs. In this paper we present a compiler called {\sc aspsmt2smt}, which implements this translation. The system uses ASP grounder {\sc gringo} and SMT solver {\sc z3}. {\sc gringo} partially grounds input programs while leaving some variables to be processed by {\sc z3}. We demonstrate that the system can effectively handle real number computations for reasoning about continuous changes. 

**Abstract (ZH)**: Answer Set Programming Modulo Theories (基于理论的回答集编程)是一种结合回答集编程和理论饱和度的基础上的功能稳定模型语义的方法。证明了ASPMT的紧密片段可以转换为SMT实例，从而允许SMT求解器计算ASPMT程序的稳定模型。本文介绍了一个名为aspsmt2smt的编译器，实现了这一转换。该系统使用ASP填充器gringo和SMT求解器z3。gringo部分填充输入程序，保留一些变量供z3处理。我们展示了该系统能够有效处理实数计算，用于连续变化的推理。 

---
# Data Driven Diagnosis for Large Cyber-Physical-Systems with Minimal Prior Information 

**Title (ZH)**: 基于最少先验信息的数据驱动诊断在大型网络物理系统中的应用 

**Authors**: Henrik Sebastian Steude, Alexander Diedrich, Ingo Pill, Lukas Moddemann, Daniel Vranješ, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10613)  

**Abstract**: Diagnostic processes for complex cyber-physical systems often require extensive prior knowledge in the form of detailed system models or comprehensive training data. However, obtaining such information poses a significant challenge. To address this issue, we present a new diagnostic approach that operates with minimal prior knowledge, requiring only a basic understanding of subsystem relationships and data from nominal operations. Our method combines a neural network-based symptom generator, which employs subsystem-level anomaly detection, with a new graph diagnosis algorithm that leverages minimal causal relationship information between subsystems-information that is typically available in practice. Our experiments with fully controllable simulated datasets show that our method includes the true causal component in its diagnosis set for 82 p.c. of all cases while effectively reducing the search space in 73 p.c. of the scenarios. Additional tests on the real-world Secure Water Treatment dataset showcase the approach's potential for practical scenarios. Our results thus highlight our approach's potential for practical applications with large and complex cyber-physical systems where limited prior knowledge is available. 

**Abstract (ZH)**: 复杂 cyber-物理系统故障诊断过程往往需要大量的先验知识，形式上为详细的系统模型或完备的训练数据。然而，获取这些信息是一项重大挑战。为了应对这一问题，我们提出了一种新的诊断方法，该方法在极少先验知识的情况下运作，仅需对子系统关系有基本理解以及正常操作数据。该方法结合了一种基于神经网络的症状生成器，该生成器采用子系统级别的异常检测，以及一种新的图诊断算法，该算法利用子系统之间最小的因果关系信息——这种信息在实践中通常是可以获得的。我们的实验结果显示，在82%的情况下，我们的方法在诊断集中包含了真正的因果组件，并且在73%的场景中有效地减少了搜索空间。此外，对实际的Secure Water Treatment数据集的测试还展示了该方法在实际场景中的潜在应用。因此，我们的结果突显了在大量且复杂的 cyber-物理系统中，当先验知识有限时，该方法的潜在应用价值。 

---
# OIBench: Benchmarking Strong Reasoning Models with Olympiad in Informatics 

**Title (ZH)**: OIBench： Olympiad in Informatics 评估强大推理模型的基准测试 

**Authors**: Yaoming Zhu, Junxin Wang, Yiyang Li, Lin Qiu, ZongYu Wang, Jun Xu, Xuezhi Cao, Yuhuai Wei, Mingshi Wang, Xunliang Cai, Rong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.10481)  

**Abstract**: As models become increasingly sophisticated, conventional algorithm benchmarks are increasingly saturated, underscoring the need for more challenging benchmarks to guide future improvements in algorithmic reasoning. This paper introduces OIBench, a high-quality, private, and challenging olympiad-level informatics dataset comprising 250 carefully curated original problems. We detail the construction methodology of the benchmark, ensuring a comprehensive assessment across various programming paradigms and complexities, and we demonstrate its contamination-resistant properties via experiments. We propose Time/Space Completion Curves for finer-grained efficiency analysis and enable direct human-model comparisons through high-level participant evaluations. Our experiments reveal that while open-source models lag behind closed-source counterparts, current SOTA models already outperform most human participants in both correctness and efficiency, while still being suboptimal compared to the canonical solutions. By releasing OIBench as a fully open-source resource (this https URL), we hope this benchmark will contribute to advancing code reasoning capabilities for future LLMs. 

**Abstract (ZH)**: 随着模型日益复杂，传统算法基准逐渐饱和，强调了建立更具挑战性的基准以指导未来算法推理改进的必要性。本文介绍了OIBench，这是一个高质量、私有且具有奥林匹克级别信息学挑战性的数据集，包含250个精心筛选的原创问题。我们详细介绍了基准的构建方法，确保其在各类编程范式和复杂性方面的全面评估，并通过实验展示了其抵抗污染的特性。我们提出了时间/空间完成曲线用于更精细的效率分析，并通过高层次的人机比较使直接的人类模型对比成为可能。实验结果显示，开源模型落后于闭源模型，但当前的SOTA模型已经在正确性和效率上超越了大多数人类参与者，尽管仍然不及经典解决方案。通过将OIBench作为一个完全开源的资源发布（this https URL），我们希望此基准能促进未来LLM的代码推理能力的发展。 

---
# Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods 

**Title (ZH)**: 基于代理的方法多维度处理服务自动扩展比较 

**Authors**: Boris Sedlak, Alireza Furutanpey, Zihang Wang, Víctor Casamayor Pujol, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10420)  

**Abstract**: Edge computing breaks with traditional autoscaling due to strict resource constraints, thus, motivating more flexible scaling behaviors using multiple elasticity dimensions. This work introduces an agent-based autoscaling framework that dynamically adjusts both hardware resources and internal service configurations to maximize requirements fulfillment in constrained environments. We compare four types of scaling agents: Active Inference, Deep Q Network, Analysis of Structural Knowledge, and Deep Active Inference, using two real-world processing services running in parallel: YOLOv8 for visual recognition and OpenCV for QR code detection. Results show all agents achieve acceptable SLO performance with varying convergence patterns. While the Deep Q Network benefits from pre-training, the structural analysis converges quickly, and the deep active inference agent combines theoretical foundations with practical scalability advantages. Our findings provide evidence for the viability of multi-dimensional agent-based autoscaling for edge environments and encourage future work in this research direction. 

**Abstract (ZH)**: 边缘计算打破传统自动扩展模式，由于严格的资源约束，因此促进了多维度扩展行为的灵活性。本文介绍了一种基于代理的自动扩展框架，该框架能够动态调整硬件资源和内部服务配置，以在受限环境中最大化需求满足。我们使用两套并行运行的实际处理服务——YOLOv8用于视觉识别和OpenCV用于QR码检测——对比了四种类型的扩展代理：主动推理、深度Q网络、结构知识分析和深度主动推理。结果表明，所有代理都能达到可接受的服务水平目标（SLO）性能，但存在不同的收敛模式。深度Q网络从预训练中受益，结构分析迅速收敛，而深度主动推理代理结合了理论基础与实用扩展优势。本研究结果为多维度代理自动扩展在边缘环境中的可行性提供了证据，并鼓励未来在此研究方向上的工作。 

---
# NeuroPAL: Punctuated Anytime Learning with Neuroevolution for Macromanagement in Starcraft: Brood War 

**Title (ZH)**: NeuroPAL：《星际争霸：虫群之心》宏管理的间歇式持续学习神经进化的算法 

**Authors**: Jim O'Connor, Yeonghun Lee, Gary B Parker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10384)  

**Abstract**: StarCraft: Brood War remains a challenging benchmark for artificial intelligence research, particularly in the domain of macromanagement, where long-term strategic planning is required. Traditional approaches to StarCraft AI rely on rule-based systems or supervised deep learning, both of which face limitations in adaptability and computational efficiency. In this work, we introduce NeuroPAL, a neuroevolutionary framework that integrates Neuroevolution of Augmenting Topologies (NEAT) with Punctuated Anytime Learning (PAL) to improve the efficiency of evolutionary training. By alternating between frequent, low-fidelity training and periodic, high-fidelity evaluations, PAL enhances the sample efficiency of NEAT, enabling agents to discover effective strategies in fewer training iterations. We evaluate NeuroPAL in a fixed-map, single-race scenario in StarCraft: Brood War and compare its performance to standard NEAT-based training. Our results show that PAL significantly accelerates the learning process, allowing the agent to reach competitive levels of play in approximately half the training time required by NEAT alone. Additionally, the evolved agents exhibit emergent behaviors such as proxy barracks placement and defensive building optimization, strategies commonly used by expert human players. These findings suggest that structured evaluation mechanisms like PAL can enhance the scalability and effectiveness of neuroevolution in complex real-time strategy environments. 

**Abstract (ZH)**: StarCraft: Brood War依然是一个挑战性的人工智能基准，特别是在需要长期战略性规划的大局管理领域。传统的StarCraft AI方法依赖于基于规则的系统或监督深度学习，这两种方法在适应性和计算效率上都存在局限性。本工作中，我们引入了NeuroPAL，这是一种将Neuroevolution of Augmenting Topologies (NEAT)与Punctuated Anytime Learning (PAL)相结合的神经演化框架，以提高进化训练的效率。通过交替进行频繁的低保真度训练和定期的高保真度评估，PAL提升了NEAT的样本效率，使智能体能够在较少的训练迭代中发现有效的策略。我们在StarCraft: Brood War的固定地图、单种族场景中评估了NeuroPAL，并将其性能与基于标准NEAT的训练进行了比较。实验结果显示，PAL显著加速了学习过程，使代理能够在大约一半的训练时间内达到具有竞争力的水平。此外，进化出的代理还表现出诸如代理兵营布局和防御建筑优化等 emergent 行为，这些策略通常由专家人类玩家使用。这些发现表明，类似PAL的结构化评估机制可以增强在复杂实时战略环境中神经演化方法的扩展性和有效性。 

---
# A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon 

**Title (ZH)**: 跨多样团队策略的通用性基准：基于竞争宝可梦的benchmark 

**Authors**: Cameron Angliss, Jiaxun Cui, Jiaheng Hu, Arrasy Rahman, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2506.10326)  

**Abstract**: Developing AI agents that can robustly adapt to dramatically different strategic landscapes without retraining is a central challenge for multi-agent learning. Pokémon Video Game Championships (VGC) is a domain with an extraordinarily large space of possible team configurations of approximately $10^{139}$ - far larger than those of Dota or Starcraft. The highly discrete, combinatorial nature of team building in Pokémon VGC causes optimal strategies to shift dramatically depending on both the team being piloted and the opponent's team, making generalization uniquely challenging. To advance research on this problem, we introduce VGC-Bench: a benchmark that provides critical infrastructure, standardizes evaluation protocols, and supplies human-play datasets and a range of baselines - from large-language-model agents and behavior cloning to reinforcement learning and empirical game-theoretic methods such as self-play, fictitious play, and double oracle. In the restricted setting where an agent is trained and evaluated on a single-team configuration, our methods are able to win against a professional VGC competitor. We extensively evaluated all baseline methods over progressively larger team sets and find that even the best-performing algorithm in the single-team setting struggles at scaling up as team size grows. Thus, policy generalization across diverse team strategies remains an open challenge for the community. Our code is open sourced at this https URL. 

**Abstract (ZH)**: 开发能够在没有重新训练的情况下 robust 地适应大幅不同的战略景观的 AI 代理是多代理学习中的一个核心挑战。Pokémon 视频游戏锦标赛 (VGC) 是一个具有极为庞大可能队伍配置空间的领域，大约为 \(10^{139}\)，远超 Dota 或 Starcraft。在 Pokémon VGC 中，队伍构建的高度离散和组合性质使得最优策略随着被操控队伍和对手队伍的不同而大幅变化，这使得泛化变得尤为具有挑战性。为了推进这一问题的研究，我们引入了 VGC-Bench：一个基准，提供了关键基础设施、标准化评估协议，并提供了人类比赛的数据集以及从大规模语言模型代理和行为克隆到强化学习和经验博弈论方法（如自博弈、虚构玩和双oracle）的各种 baselines。在仅在单一队伍配置下进行训练和评估的受限设置中，我们的方法能够战胜职业 VGC 竞赛选手。我们在逐渐增大的队伍集合上广泛评估了所有 baselines 方法，并发现即使在单一队伍设置中表现最佳的算法在队伍规模扩大时也难以扩展。因此，跨多样队伍策略的策略泛化仍然是社区中的一个开放挑战。我们的代码已开源：this https URL。 

---
# The Alignment Trap: Complexity Barriers 

**Title (ZH)**: 对齐陷阱：复杂性障碍 

**Authors**: Jasper Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10304)  

**Abstract**: We establish fundamental computational complexity barriers to verifying AI safety as system capabilities scale. Our main results show that for AI systems with expressiveness EXP$(m)$ above a critical threshold $\tau$, safety verification requires exponential time and is coNP-complete. We formalize the Capability-Risk Scaling (CRS) dynamic, which demonstrates how increasing AI capability drives societal safety requirements toward perfection, creating an inescapable tension with verification complexity. Through four core theorems, we prove that (1) verification complexity grows exponentially with system expressiveness, (2) safe policies comprise at most a $2^{-2^m}$ fraction of the policy space, (3) no finite set of alignment techniques can provide universal coverage, and (4) robust safety properties form measure-zero sets for neural networks. These results characterize an "intractability gap" where practical safety requirements fall within the region of computational intractability. We conclude by presenting a strategic trilemma: AI development must either constrain system complexity to maintain verifiable safety, accept unverifiable risks while scaling capabilities, or develop fundamentally new safety paradigms beyond verification. Our work provides the first systematic complexity-theoretic analysis of AI alignment and establishes rigorous bounds that any safety approach must confront. A formal verification of the core theorems in Lean4 is currently in progress. 

**Abstract (ZH)**: 我们建立了随着AI系统能力增强而验证AI安全性基本计算复杂性障碍。我们的主要结果表明，对于表示能力EXP$(m)$超过临界阈值$\tau$的AI系统，安全性验证需要指数时间且是coNP完全问题。我们正式化了能力-风险扩展（CRS）动态，显示了增强的AI能力如何推动社会安全性要求向完美发展，创造了验证复杂性不可避免的紧张关系。通过四条核心定理，我们证明了：（1）验证复杂性随系统表示能力指数增长；（2）安全策略最多占策略空间的$2^{-2^m}$部分；（3）没有任何有限的对齐技术能够提供普遍覆盖；（4）对于神经网络，稳健的安全性质形成测度零集。这些结果刻画了“不可处理性差距”，其中实用的安全要求处于计算不可处理性的区域。最后，我们提出了一个战略三难困境：AI开发必须要么限制系统复杂性以维持可验证的安全性，要么在增强能力的同时接受不可验证的风险，要么开发超越验证的基本新的安全范式。我们的工作提供了对AI对齐的第一个系统性的复杂性理论分析，并建立了任何安全方法都必须面对的严格界线。目前，核心定理在Lean4中正在进行形式验证。 

---
# Towards Responsible AI: Advances in Safety, Fairness, and Accountability of Autonomous Systems 

**Title (ZH)**: 负责任的人工智能：自主系统在安全性、公平性和问责制方面的进展 

**Authors**: Filip Cano  

**Link**: [PDF](https://arxiv.org/pdf/2506.10192)  

**Abstract**: Ensuring responsible use of artificial intelligence (AI) has become imperative as autonomous systems increasingly influence critical societal domains. However, the concept of trustworthy AI remains broad and multi-faceted. This thesis advances knowledge in the safety, fairness, transparency, and accountability of AI systems. In safety, we extend classical deterministic shielding techniques to become resilient against delayed observations, enabling practical deployment in real-world conditions. We also implement both deterministic and probabilistic safety shields into simulated autonomous vehicles to prevent collisions with road users, validating the use of these techniques in realistic driving simulators. We introduce fairness shields, a novel post-processing approach to enforce group fairness in sequential decision-making settings over finite and periodic time horizons. By optimizing intervention costs while strictly ensuring fairness constraints, this method efficiently balances fairness with minimal interference. For transparency and accountability, we propose a formal framework for assessing intentional behaviour in probabilistic decision-making agents, introducing quantitative metrics of agency and intention quotient. We use these metrics to propose a retrospective analysis of intention, useful for determining responsibility when autonomous systems cause unintended harm. Finally, we unify these contributions through the ``reactive decision-making'' framework, providing a general formalization that consolidates previous approaches. Collectively, the advancements presented contribute practically to the realization of safer, fairer, and more accountable AI systems, laying the foundations for future research in trustworthy AI. 

**Abstract (ZH)**: 确保人工智能的负责任使用已成为必要，随着自主系统在关键社会领域中的影响不断增加。然而，可信赖人工智能的概念依然宽泛且多维度。本论文推进了在安全、公平、透明和问责方面对人工智能系统的知识。在安全性方面，我们将传统的确定性防护技术扩展为能够在延迟观察下保持韧性，从而在实际条件下实现实用部署。我们还将确定性和概率性安全防护应用到模拟的自主车辆中，以防止与道路使用者发生碰撞，验证了这些技术在现实驾驶模拟器中的使用。我们提出了公平防护，这是一种新颖的后处理方法，用于在有限和周期性的时间框架内的顺序决策环境中强制实施群体公平性。通过在严格确保公平约束的同时优化干预成本，该方法能够高效地在公平性和最小化干扰之间取得平衡。在透明度和问责方面，我们提出了一种正式框架来评估概率决策代理的意图行为，并引入了代理性和意图商的定量指标。利用这些指标，我们提出了关于意图的回顾性分析，这在确定自主系统导致意外损害时的责任问题上有用。最后，我们通过“反应性决策”框架统一这些贡献，提供了一个综合的正式化方法，将先前的方法整合在一起。总体而言，所提出的发展实用地促进了实现更安全、更公平和更负责任的人工智能系统，并为未来可信赖人工智能的研究奠定了基础。 

---
# Correlation vs causation in Alzheimer's disease: an interpretability-driven study 

**Title (ZH)**: 阿尔茨海默病中相关性与因果性的区别：一种基于可解释性的研究 

**Authors**: Hamzah Dabool, Raghad Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10179)  

**Abstract**: Understanding the distinction between causation and correlation is critical in Alzheimer's disease (AD) research, as it impacts diagnosis, treatment, and the identification of true disease drivers. This experiment investigates the relationships among clinical, cognitive, genetic, and biomarker features using a combination of correlation analysis, machine learning classification, and model interpretability techniques. Employing the XGBoost algorithm, we identified key features influencing AD classification, including cognitive scores and genetic risk factors. Correlation matrices revealed clusters of interrelated variables, while SHAP (SHapley Additive exPlanations) values provided detailed insights into feature contributions across disease stages. Our results highlight that strong correlations do not necessarily imply causation, emphasizing the need for careful interpretation of associative data. By integrating feature importance and interpretability with classical statistical analysis, this work lays groundwork for future causal inference studies aimed at uncovering true pathological mechanisms. Ultimately, distinguishing causal factors from correlated markers can lead to improved early diagnosis and targeted interventions for Alzheimer's disease. 

**Abstract (ZH)**: 理解因果关系与相关性的区别在阿尔茨海默病（AD）研究中至关重要，这影响着诊断、治疗以及真正疾病驱动因素的识别。本研究结合相关性分析、机器学习分类和模型可解释性技术，探讨临床、认知、遗传和生物标志物特征之间的关系。采用XGBoost算法，我们识别出影响AD分类的关键特征，包括认知评分和遗传风险因素。相关矩阵揭示了相互关联变量的集群，而SHAP值提供了对疾病各阶段特征贡献的详细见解。研究结果强调，强烈的相关性并不必然意味着因果关系，突出了对关联性数据谨慎解释的必要性。通过将特征重要性与解释性与经典统计分析相结合，本研究为未来旨在揭示真正病理机制的因果推理研究奠定了基础。最终，区分因果因素和相关标志物可以提高阿尔茨海默病的早期诊断并实现针对性的干预。 

---
# One Patient, Many Contexts: Scaling Medical AI Through Contextual Intelligence 

**Title (ZH)**: 一个患者，多种情境：通过情境智能扩大医疗AI的应用规模 

**Authors**: Michelle M. Li, Ben Y. Reis, Adam Rodman, Tianxi Cai, Noa Dagan, Ran D. Balicer, Joseph Loscalzo, Isaac S. Kohane, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2506.10157)  

**Abstract**: Medical foundation models, including language models trained on clinical notes, vision-language models on medical images, and multimodal models on electronic health records, can summarize clinical notes, answer medical questions, and assist in decision-making. Adapting these models to new populations, specialties, or settings typically requires fine-tuning, careful prompting, or retrieval from knowledge bases. This can be impractical, and limits their ability to interpret unfamiliar inputs and adjust to clinical situations not represented during training. As a result, models are prone to contextual errors, where predictions appear reasonable but fail to account for critical patient-specific or contextual information. These errors stem from a fundamental limitation that current models struggle with: dynamically adjusting their behavior across evolving contexts of medical care. In this Perspective, we outline a vision for context-switching in medical AI: models that dynamically adapt their reasoning without retraining to new specialties, populations, workflows, and clinical roles. We envision context-switching AI to diagnose, manage, and treat a wide range of diseases across specialties and regions, and expand access to medical care. 

**Abstract (ZH)**: 医疗情境切换的人工智能：模型无需重新训练即可动态适应新的专科、人群、工作流程和临床角色，以诊断、管理和治疗各种疾病并扩大医疗访问范围。 

---
# A Conjecture on a Fundamental Trade-Off between Certainty and Scope in Symbolic and Generative AI 

**Title (ZH)**: 关于符号性和生成性AI中确定性和范围基本权衡的一种推测 

**Authors**: Luciano Floridi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10130)  

**Abstract**: This article introduces a conjecture that formalises a fundamental trade-off between provable correctness and broad data-mapping capacity in Artificial Intelligence (AI) systems. When an AI system is engineered for deductively watertight guarantees (demonstrable certainty about the error-free nature of its outputs) -- as in classical symbolic AI -- its operational domain must be narrowly circumscribed and pre-structured. Conversely, a system that can input high-dimensional data to produce rich information outputs -- as in contemporary generative models -- necessarily relinquishes the possibility of zero-error performance, incurring an irreducible risk of errors or misclassification. By making this previously implicit trade-off explicit and open to rigorous verification, the conjecture significantly reframes both engineering ambitions and philosophical expectations for AI. After reviewing the historical motivations for this tension, the article states the conjecture in information-theoretic form and contextualises it within broader debates in epistemology, formal verification, and the philosophy of technology. It then offers an analysis of its implications and consequences, drawing on notions of underdetermination, prudent epistemic risk, and moral responsibility. The discussion clarifies how, if correct, the conjecture would help reshape evaluation standards, governance frameworks, and hybrid system design. The conclusion underscores the importance of eventually proving or refuting the inequality for the future of trustworthy AI. 

**Abstract (ZH)**: 本文介绍了一个公设，该公设正式化了人工智能（AI）系统中可验证正确性和广泛数据映射能力之间的基本权衡。当一个AI系统被设计为具有演绎严密的保证（对其输出无錯誤性质的可证明确定性）——如同传统的符号AI——其操作领域必须被严格限定和预结构化。相反，能够输入高维数据以生成丰富信息输出的系统——如同当代的生成模型——必然放弃了零错误性能的可能性，不可避免地承担了错误或误分类的风险。通过使这种先前隐含的权衡变得明确并且可以通过严格的验证进行审视，该公设显著重塑了工程目标和对AI的哲学期望。文章回顾了这种紧张关系的历史动机，以信息论形式表述该公设，并将其置于更广泛的认识论、形式验证和技术哲学辩论的背景下。然后，文章提供了对该结论的分析，借鉴了确定性不足、审慎的认知风险和道德责任的概念。讨论阐明了，如果该公设正确，它将如何有助于重塑评估标准、治理框架和混合系统设计。结论强调了最终证明或反驳不等式对于可信AI未来发展的重要性。 

---
# Rethinking Losses for Diffusion Bridge Samplers 

**Title (ZH)**: 重新思考扩散桥梁采样中的损失函数 

**Authors**: Sebastian Sanokowski, Lukas Gruber, Christoph Bartmann, Sepp Hochreiter, Sebastian Lehner  

**Link**: [PDF](https://arxiv.org/pdf/2506.10982)  

**Abstract**: Diffusion bridges are a promising class of deep-learning methods for sampling from unnormalized distributions. Recent works show that the Log Variance (LV) loss consistently outperforms the reverse Kullback-Leibler (rKL) loss when using the reparametrization trick to compute rKL-gradients. While the on-policy LV loss yields identical gradients to the rKL loss when combined with the log-derivative trick for diffusion samplers with non-learnable forward processes, this equivalence does not hold for diffusion bridges or when diffusion coefficients are learned. Based on this insight we argue that for diffusion bridges the LV loss does not represent an optimization objective that can be motivated like the rKL loss via the data processing inequality. Our analysis shows that employing the rKL loss with the log-derivative trick (rKL-LD) does not only avoid these conceptual problems but also consistently outperforms the LV loss. Experimental results with different types of diffusion bridges on challenging benchmarks show that samplers trained with the rKL-LD loss achieve better performance. From a practical perspective we find that rKL-LD requires significantly less hyperparameter optimization and yields more stable training behavior. 

**Abstract (ZH)**: 扩散桥梁是一类从未正规化分布中采样的有前途的深度学习方法。最近的研究表明，在使用重参数化技巧计算反Kullback-Leibler (rKL)梯度时，Log Variance (LV)损失一直优于rKL损失。当与非可学习前向过程结合使用日志导数技巧时，针对扩散采样的on-policy LV损失会提供与rKL损失相同的梯度，但这种等价性并不适用于扩散桥梁，或当扩散系数被学习时。基于这一洞察，我们认为对于扩散桥梁，LV损失并不是可以通过数据处理不等式进行动机说明的优化目标。我们的分析表明，通过日志导数技巧使用rKL损失 (rKL-LD) 不仅避免了这些概念性问题，而且在一致性上优于LV损失。不同类型扩散桥梁在具有挑战性的基准上的实验结果表明，使用rKL-LD损失训练的采样器表现出更好的性能。从实用的角度来看，我们发现使用rKL-LD损失需要显著较少的超参数优化，并且具有更稳定的训练行为。 

---
# Fine-Grained Perturbation Guidance via Attention Head Selection 

**Title (ZH)**: 基于注意力头选择的细粒度扰动引导 

**Authors**: Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Minjae Kim, Jaewon Min, Wooseok Jang, Saungwu Lee, Sayak Paul, Susung Hong, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10978)  

**Abstract**: Recent guidance methods in diffusion models steer reverse sampling by perturbing the model to construct an implicit weak model and guide generation away from it. Among these approaches, attention perturbation has demonstrated strong empirical performance in unconditional scenarios where classifier-free guidance is not applicable. However, existing attention perturbation methods lack principled approaches for determining where perturbations should be applied, particularly in Diffusion Transformer (DiT) architectures where quality-relevant computations are distributed across layers. In this paper, we investigate the granularity of attention perturbations, ranging from the layer level down to individual attention heads, and discover that specific heads govern distinct visual concepts such as structure, style, and texture quality. Building on this insight, we propose "HeadHunter", a systematic framework for iteratively selecting attention heads that align with user-centric objectives, enabling fine-grained control over generation quality and visual attributes. In addition, we introduce SoftPAG, which linearly interpolates each selected head's attention map toward an identity matrix, providing a continuous knob to tune perturbation strength and suppress artifacts. Our approach not only mitigates the oversmoothing issues of existing layer-level perturbation but also enables targeted manipulation of specific visual styles through compositional head selection. We validate our method on modern large-scale DiT-based text-to-image models including Stable Diffusion 3 and FLUX.1, demonstrating superior performance in both general quality enhancement and style-specific guidance. Our work provides the first head-level analysis of attention perturbation in diffusion models, uncovering interpretable specialization within attention layers and enabling practical design of effective perturbation strategies. 

**Abstract (ZH)**: Recent Guidance Methods in Diffusion Models通过扰动模型构造隐式弱模型并引导生成远离它，实现了逆向采样的调控。在这些方法中，注意力扰动在无条件场景中表现出强大的实证性能，尤其是在分类器自由指导不适用的情况下。然而，现有的注意力扰动方法缺乏确定扰动应应用于何处的原理性方法，特别是在质量相关的计算分布在各层中的扩散变换器(DiT)架构中。在本文中，我们考察了注意力扰动的粒度，从层级细化到单个注意力头，并发现特定的头控制着结构、样式和纹理质量等不同的视觉概念。基于这一洞察，我们提出了一种名为“HeadHunter”的系统框架，用于迭代选择与用户中心目标对齐的注意力头，从而实现对生成质量和视觉属性的细粒度控制。此外，我们引入了SoftPAG，它通过对每个选定头部的注意力图线性插值至单位矩阵，提供了一个连续的旋钮以调节扰动强度并抑制伪迹。我们的方法不仅缓解了现有层级扰动的过度平滑问题，还通过组成性头选择实现了对特定视觉样式的靶向操纵。我们在包括Stable Diffusion 3和FLUX.1的现代大规模基于DiT的文本到图像模型上验证了我们的方法，展示了在通用质量增强和样式特定指导方面的优越性能。我们的工作是首次对扩散模型中的注意力扰动进行头部级分析，揭示了注意力层内的可解释专业化，并为有效的扰动策略的设计提供了实用的设计指南。 

---
# Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning 

**Title (ZH)**: 原理性的方法将神经架构拓展至函数空间进行算子学习 

**Authors**: Julius Berner, Miguel Liu-Schiaffini, Jean Kossaifi, Valentin Duruisseaux, Boris Bonev, Kamyar Azizzadenesheli, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10973)  

**Abstract**: A wide range of scientific problems, such as those described by continuous-time dynamical systems and partial differential equations (PDEs), are naturally formulated on function spaces. While function spaces are typically infinite-dimensional, deep learning has predominantly advanced through applications in computer vision and natural language processing that focus on mappings between finite-dimensional spaces. Such fundamental disparities in the nature of the data have limited neural networks from achieving a comparable level of success in scientific applications as seen in other fields. Neural operators are a principled way to generalize neural networks to mappings between function spaces, offering a pathway to replicate deep learning's transformative impact on scientific problems. For instance, neural operators can learn solution operators for entire classes of PDEs, e.g., physical systems with different boundary conditions, coefficient functions, and geometries. A key factor in deep learning's success has been the careful engineering of neural architectures through extensive empirical testing. Translating these neural architectures into neural operators allows operator learning to enjoy these same empirical optimizations. However, prior neural operator architectures have often been introduced as standalone models, not directly derived as extensions of existing neural network architectures. In this paper, we identify and distill the key principles for constructing practical implementations of mappings between infinite-dimensional function spaces. Using these principles, we propose a recipe for converting several popular neural architectures into neural operators with minimal modifications. This paper aims to guide practitioners through this process and details the steps to make neural operators work in practice. Our code can be found at this https URL 

**Abstract (ZH)**: 基于函数空间映射的神经运算：从原理到实践 

---
# Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods 

**Title (ZH)**: 结构流形上的上下文学习理解：注意力与核方法的桥梁 

**Authors**: Zhaiming Shen, Alexander Hsu, Rongjie Lai, Wenjing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10959)  

**Abstract**: While in-context learning (ICL) has achieved remarkable success in natural language and vision domains, its theoretical understanding--particularly in the context of structured geometric data--remains unexplored. In this work, we initiate a theoretical study of ICL for regression of Hölder functions on manifolds. By establishing a novel connection between the attention mechanism and classical kernel methods, we derive generalization error bounds in terms of the prompt length and the number of training tasks. When a sufficient number of training tasks are observed, transformers give rise to the minimax regression rate of Hölder functions on manifolds, which scales exponentially with the intrinsic dimension of the manifold, rather than the ambient space dimension. Our result also characterizes how the generalization error scales with the number of training tasks, shedding light on the complexity of transformers as in-context algorithm learners. Our findings provide foundational insights into the role of geometry in ICL and novels tools to study ICL of nonlinear models. 

**Abstract (ZH)**: 关于流形上Hölder函数回归的上下文学习的理论研究 

---
# ReGuidance: A Simple Diffusion Wrapper for Boosting Sample Quality on Hard Inverse Problems 

**Title (ZH)**: ReGuidance: 一种简单的扩散包裹方法以在棘手的逆问题中提升样本质量 

**Authors**: Aayush Karan, Kulin Shah, Sitan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10955)  

**Abstract**: There has been a flurry of activity around using pretrained diffusion models as informed data priors for solving inverse problems, and more generally around steering these models using reward models. Training-free methods like diffusion posterior sampling (DPS) and its many variants have offered flexible heuristic algorithms for these tasks, but when the reward is not informative enough, e.g., in hard inverse problems with low signal-to-noise ratio, these techniques veer off the data manifold, failing to produce realistic outputs. In this work, we devise a simple wrapper, ReGuidance, for boosting both the sample realism and reward achieved by these methods. Given a candidate solution $\hat{x}$ produced by an algorithm of the user's choice, we propose inverting the solution by running the unconditional probability flow ODE in reverse starting from $\hat{x}$, and then using the resulting latent as an initialization for DPS. We evaluate our wrapper on hard inverse problems like large box in-painting and super-resolution with high upscaling. Whereas state-of-the-art baselines visibly fail, we find that applying our wrapper on top of these baselines significantly boosts sample quality and measurement consistency. We complement these findings with theory proving that on certain multimodal data distributions, ReGuidance simultaneously boosts the reward and brings the candidate solution closer to the data manifold. To our knowledge, this constitutes the first rigorous algorithmic guarantee for DPS. 

**Abstract (ZH)**: 基于奖励模型引导的预训练扩散模型在逆问题求解中的增强方法 

---
# SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks 

**Title (ZH)**: SWE-Factory: 您的自动化故障排除训练数据和评估基准工厂 

**Authors**: Lianghong Guo, Yanlin Wang, Caihua Li, Pengyu Yang, Jiachi Chen, Wei Tao, Yingtian Zou, Duyu Tang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10954)  

**Abstract**: Constructing large-scale datasets for the GitHub issue resolution task is crucial for both training and evaluating the software engineering capabilities of Large Language Models (LLMs). However, the traditional process for creating such benchmarks is notoriously challenging and labor-intensive, particularly in the stages of setting up evaluation environments, grading test outcomes, and validating task instances. In this paper, we propose SWE-Factory, an automated pipeline designed to address these challenges. To tackle these issues, our pipeline integrates three core automated components. First, we introduce SWE-Builder, a multi-agent system that automates evaluation environment construction, which employs four specialized agents that work in a collaborative, iterative loop and leverages an environment memory pool to enhance efficiency. Second, we introduce a standardized, exit-code-based grading method that eliminates the need for manually writing custom parsers. Finally, we automate the fail2pass validation process using these reliable exit code signals. Experiments on 671 issues across four programming languages show that our pipeline can effectively construct valid task instances; for example, with GPT-4.1-mini, our SWE-Builder constructs 269 valid instances at $0.045 per instance, while with Gemini-2.5-flash, it achieves comparable performance at the lowest cost of $0.024 per instance. We also demonstrate that our exit-code-based grading achieves 100% accuracy compared to manual inspection, and our automated fail2pass validation reaches a precision of 0.92 and a recall of 1.00. We hope our automated pipeline will accelerate the collection of large-scale, high-quality GitHub issue resolution datasets for both training and evaluation. Our code and datasets are released at this https URL. 

**Abstract (ZH)**: 构建大型数据集以解决GitHub问题对于训练和评估大型语言模型的软件工程能力至关重要。然而，传统基准创建过程既复杂又劳动密集，特别是在设置评估环境、评分测试结果和验证任务实例阶段。本文提出SWE-Factory，一种自动流水线，旨在解决这些问题。为了应对这些挑战，我们的流水线整合了三个核心自动组件。首先，我们引入SWE-Builder，一个多功能系统，自动构建评估环境，采用四个专门的代理在协作迭代循环中工作，并利用环境记忆池提高效率。其次，我们引入标准化的基于退出代码的评分方法，消除手动编写自定义解析器的需要。最后，我们使用可靠的退出代码信号自动实现fail2pass验证流程。在涵盖四种编程语言的671个问题的实验中，我们的流水线能够有效构建有效的任务实例；例如，使用GPT-4.1-mini时，SWE-Builder每实例成本0.045美元构建269个有效实例，使用Gemini-2.5-flash时，成本最低，每实例0.024美元。我们还证明基于退出代码的评分与人工检查相匹配，精确率为100%，自动化的fail2pass验证精度为0.92，召回率为1.00。我们希望我们的自动流水线能够加速收集用于训练和评估的大规模高质量GitHub问题解决数据集。我们的代码和数据集在此处公开。 

---
# Domain2Vec: Vectorizing Datasets to Find the Optimal Data Mixture without Training 

**Title (ZH)**: 域2Vec：将数据集向量化以找到最优数据混合 without 训练 

**Authors**: Mozhi Zhang, Howe Tissue, Lu Wang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10952)  

**Abstract**: We introduce~\textsc{Domain2Vec}, a novel approach that decomposes any dataset into a linear combination of several \emph{meta-domains}, a new concept designed to capture the key underlying features of datasets. \textsc{Domain2Vec} maintains a vocabulary of meta-domains and uses a classifier to decompose any given dataset into a domain vector that corresponds to a distribution over this vocabulary. These domain vectors enable the identification of the optimal data mixture for language model (LM) pretraining in a training-free manner under the \emph{\textbf{D}istribution \textbf{A}lignment \textbf{A}ssumption} (DA$^{2}$), which suggests that when the data distributions of the training set and the validation set are better aligned, a lower validation loss is achieved. Moreover, \textsc{Domain2vec} can be seamlessly integrated into previous works to model the relationship between domain vectors and LM performance, greatly enhancing the efficiency and scalability of previous methods. Extensive experiments demonstrate that \textsc{Domain2Vec} helps find the data mixture that enhances downstream task performance with minimal computational overhead. Specifically, \textsc{Domain2Vec} achieves the same validation loss on Pile-CC using only $51.5\%$ of the computation required when training on the original mixture of The Pile dataset. Under equivalent compute budget, \textsc{Domain2Vec} improves downstream performance by an average of $2.83\%$. 

**Abstract (ZH)**: 我们介绍了Domain2Vec，这是一种新颖的方法，将任意数据集分解为多个元领域（meta-domains）的线性组合，元领域是一种新设计的概念，用于捕捉数据集的关键底层特征。Domain2Vec维护一个元领域的词汇表，并使用分类器将任意给定的数据集分解为对应于该词汇表分布的领域向量。这些领域向量使得在分布对齐假设（DA²）下以无训练方式找到用于语言模型预训练的最优数据混合变得更加可能，分布对齐假设认为，当训练集和验证集的数据分布更加对齐时，会实现更低的验证损失。此外，Domain2Vec可以无缝集成到先前的工作中，用于建模领域向量与语言模型性能之间的关系，极大地提高了先前方法的效率与扩展性。大量实验表明，Domain2Vec在最小计算开销的情况下帮助找到增强下游任务性能的数据混合。具体而言，与在Pile原始混合数据集上进行训练相比，Domain2Vec仅使用51.5%的计算量就能在Pile-CC上达到相同的验证损失。在等效计算预算下，Domain2Vec将下游性能平均提高2.83%。 

---
# The Role of Generative AI in Facilitating Social Interactions: A Scoping Review 

**Title (ZH)**: 生成式人工智能在促进社会互动中的作用：一项范围性综述 

**Authors**: T. T. J. E. Arets, G. Perugia, M. Houben, W.A. IJsselsteijn  

**Link**: [PDF](https://arxiv.org/pdf/2506.10927)  

**Abstract**: Reduced social connectedness increasingly poses a threat to mental health, life expectancy, and general well-being. Generative AI (GAI) technologies, such as large language models (LLMs) and image generation tools, are increasingly integrated into applications aimed at enhancing human social experiences. Despite their growing presence, little is known about how these technologies influence social interactions. This scoping review investigates how GAI-based applications are currently designed to facilitate social interaction, what forms of social engagement they target, and which design and evaluation methodologies designers use to create and evaluate them. Through an analysis of 30 studies published since 2020, we identify key trends in application domains including storytelling, socio-emotional skills training, reminiscence, collaborative learning, music making, and general conversation. We highlight the role of participatory and co-design approaches in fostering both effective technology use and social engagement, while also examining socio-ethical concerns such as cultural bias and accessibility. This review underscores the potential of GAI to support dynamic and personalized interactions, but calls for greater attention to equitable design practices and inclusive evaluation strategies. 

**Abstract (ZH)**: reduced 社交联系的下降日益对心理健康、寿命和总体福祉构成威胁。生成型人工智能（GAI）技术，如大型语言模型（LLMs）和图像生成工具，正越来越多地被集成到旨在增强人类社交体验的应用中。尽管这些技术的影响力日益增强，但人们对它们如何影响社交互动知之甚少。本综述探讨了基于GAI的应用如何设计以促进社交互动、它们针对哪些形式的社交参与以及设计者采用哪些设计和评价方法来创建和评估这些应用。通过分析2020年以来发表的30项研究，我们识别了应用领域中的关键趋势，包括叙事、情感技能训练、回忆、协作学习、音乐创作和一般对话。 chúng我们强调参与式和共同设计方法在促进有效技术使用和社交参与中的作用，同时也探讨了文化偏见和可访问性等社会伦理问题。本综述强调了GAI在支持动态和个性化互动方面的潜力，但也呼吁对公平设计实践和包容性评估策略给予更多关注。 

---
# Agentic Semantic Control for Autonomous Wireless Space Networks: Extending Space-O-RAN with MCP-Driven Distributed Intelligence 

**Title (ZH)**: 自主无线太空网络中的代理语义控制：基于MCP驱动的分布式智能扩展Space-O-RAN 

**Authors**: Eduardo Baena, Paolo Testolina, Michele Polese, Sergi Aliaga, Andrew Benincasa, Dimitrios Koutsonikolas, Josep Jornet, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2506.10925)  

**Abstract**: Lunar surface operations impose stringent requirements on wireless communication systems, including autonomy, robustness to disruption, and the ability to adapt to environmental and mission-driven context. While Space-O-RAN provides a distributed orchestration model aligned with 3GPP standards, its decision logic is limited to static policies and lacks semantic integration. We propose a novel extension incorporating a semantic agentic layer enabled by the Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication protocols, allowing context-aware decision making across real-time, near-real-time, and non-real-time control layers. Distributed cognitive agents deployed in rovers, landers, and lunar base stations implement wireless-aware coordination strategies, including delay-adaptive reasoning and bandwidth-aware semantic compression, while interacting with multiple MCP servers to reason over telemetry, locomotion planning, and mission constraints. 

**Abstract (ZH)**: 月球表面操作对无线通信系统提出了严格要求，包括自主性、抗干扰能力和适应环境和任务驱动上下文的能力。尽管Space-O-RAN提供了与3GPP标准一致的分布式协同模型，但其决策逻辑仅限于静态策略且缺乏语义集成。我们提出了一种新的扩展，结合了由Model Context Protocol (MCP)和Agent-to-Agent (A2A)通信协议支持的语义代理层，允许跨实时、近实时和非实时控制层进行上下文感知决策。部署在月球车、着陆器和月球基站中的分布式认知代理实现了无线感知的协调策略，包括延迟自适应推理和带宽感知的语义压缩，同时与多个MCP服务器交互，以推理遥测、运动规划和任务约束。 

---
# AIR: Zero-shot Generative Model Adaptation with Iterative Refinement 

**Title (ZH)**: AIR：迭代精炼的零样本生成模型适应 

**Authors**: Guimeng Liu, Milad Abdollahzadeh, Ngai-Man Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.10895)  

**Abstract**: Zero-shot generative model adaptation (ZSGM) aims to adapt a pre-trained generator to a target domain using only text guidance and without any samples from the target domain. Central to recent ZSGM approaches are directional loss which use the text guidance in the form of aligning the image offset with text offset in the embedding space of a vision-language model like CLIP. This is similar to the analogical reasoning in NLP where the offset between one pair of words is used to identify a missing element in another pair by aligning the offset between these two pairs. However, a major limitation of existing ZSGM methods is that the learning objective assumes the complete alignment between image offset and text offset in the CLIP embedding space, resulting in quality degrade in generated images. Our work makes two main contributions. Inspired by the offset misalignment studies in NLP, as our first contribution, we perform an empirical study to analyze the misalignment between text offset and image offset in CLIP embedding space for various large publicly available datasets. Our important finding is that offset misalignment in CLIP embedding space is correlated with concept distance, i.e., close concepts have a less offset misalignment. To address the limitations of the current approaches, as our second contribution, we propose Adaptation with Iterative Refinement (AIR) which is the first ZSGM approach to focus on improving target domain image quality based on our new insight on offset this http URL, quantitative, and user study in 26 experiment setups consistently demonstrate the proposed AIR approach achieves SOTA performance. Additional experiments are in Supp. 

**Abstract (ZH)**: 零样本生成模型适应（ZSGM）旨在仅使用文本指导和支持目标领域无任何目标领域样本的情况下，适应预训练生成器到目标领域。近期ZSGM方法的核心在于方向性损失，它通过在类似于CLIP这类视觉-语言模型的嵌入空间中对齐图像偏移和文本偏移来利用文本指导。这类似于NLP中的类比推理，其中通过对齐一个词对之间的偏移来识别另一个词对中缺失的元素。然而，现有ZSGM方法的主要局限性在于学习目标假定了CLIP嵌入空间中图像偏移和文本偏移的完全对齐，导致生成的图像质量下降。我们的工作做出了两项主要贡献。受NLP中偏移不对齐研究的启发，作为我们第一项贡献，我们进行了实证研究，分析了各种大型公开数据集中CLIP嵌入空间中文本偏移和图像偏移之间的不对齐情况。我们的重要发现是，CLIP嵌入空间中的偏移不对齐与概念距离相关，即接近的概念具有较少的偏移不对齐。为了克服当前方法的局限性，作为我们第二项贡献，我们提出了一种迭代校准的适应方法（Adaptation with Iterative Refinement, AIR），这是首个基于我们对偏移的新洞察专注于提高目标领域图像质量的ZSGM方法。在26种实验设置下的定性、定量和用户研究结果一致表明，提出的AIR方法达到了当前最佳性能。附加实验详见附录。 

---
# A multi-scale loss formulation for learning a probabilistic model with proper score optimisation 

**Title (ZH)**: 多尺度损失函数 formulation 用于具有适当评分优化的概率模型学习 

**Authors**: Simon Lang, Martin Leutbecher, Pedro Maciel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10868)  

**Abstract**: We assess the impact of a multi-scale loss formulation for training probabilistic machine-learned weather forecasting models. The multi-scale loss is tested in AIFS-CRPS, a machine-learned weather forecasting model developed at the European Centre for Medium-Range Weather Forecasts (ECMWF). AIFS-CRPS is trained by directly optimising the almost fair continuous ranked probability score (afCRPS). The multi-scale loss better constrains small scale variability without negatively impacting forecast skill. This opens up promising directions for future work in scale-aware model training. 

**Abstract (ZH)**: 我们评估了多尺度损失函数对训练概率机器学习天气预报模型的影响。多尺度损失在欧洲中期天气预报中心（ECMWF）开发的AIFS-CRPS机器学习天气预报模型中进行了测试，通过直接优化近公允连续排名概率评分（afCRPS）进行训练。多尺度损失更好地约束了小尺度变异性，而不负面影响预报技巧。这为未来的尺度感知模型训练开辟了有 Promise 的方向。 

---
# Efficiency Robustness of Dynamic Deep Learning Systems 

**Title (ZH)**: 动态深度学习系统的效率稳健性 

**Authors**: Ravishka Rathnasuriya, Tingxi Li, Zexin Xu, Zihe Song, Mirazul Haque, Simin Chen, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10831)  

**Abstract**: Deep Learning Systems (DLSs) are increasingly deployed in real-time applications, including those in resourceconstrained environments such as mobile and IoT devices. To address efficiency challenges, Dynamic Deep Learning Systems (DDLSs) adapt inference computation based on input complexity, reducing overhead. While this dynamic behavior improves efficiency, such behavior introduces new attack surfaces. In particular, efficiency adversarial attacks exploit these dynamic mechanisms to degrade system performance. This paper systematically explores efficiency robustness of DDLSs, presenting the first comprehensive taxonomy of efficiency attacks. We categorize these attacks based on three dynamic behaviors: (i) attacks on dynamic computations per inference, (ii) attacks on dynamic inference iterations, and (iii) attacks on dynamic output production for downstream tasks. Through an in-depth evaluation, we analyze adversarial strategies that target DDLSs efficiency and identify key challenges in securing these systems. In addition, we investigate existing defense mechanisms, demonstrating their limitations against increasingly popular efficiency attacks and the necessity for novel mitigation strategies to secure future adaptive DDLSs. 

**Abstract (ZH)**: 深度学习系统（DLSs）越来越多地应用于实时应用，包括移动设备和物联网设备等资源受限环境中。为了应对效率挑战，动态深度学习系统（DDLSs）根据输入的复杂性调整推理计算，从而减少开销。虽然这种动态行为能够提高效率，但也引入了新的攻击表面。特别是效率对抗攻击利用这些动态机制以降低系统性能。本文系统地探讨了DDLSs的效率鲁棒性，提出了首个全面的效率攻击分类。我们根据三种动态行为对这些攻击进行分类：(i) 每次推理中的动态计算攻击，(ii) 动态推理迭代攻击，以及(iii) 动态输出生成攻击以供下游任务使用。通过深入评估，我们分析了针对DDLSs效率的目标攻击策略，并确定了这些系统安全性的关键挑战。此外，我们还调查了现有的防御机制，证明它们在应对日益流行的效率攻击方面的局限性，并强调未来需要新的缓解策略来确保动态适应性DDLSs的安全性。 

---
# ME: Trigger Element Combination Backdoor Attack on Copyright Infringement 

**Title (ZH)**: ME：版权侵权中的触发元素组合后门攻击 

**Authors**: Feiyu Yang, Siyuan Liang, Aishan Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10776)  

**Abstract**: The capability of generative diffusion models (DMs) like Stable Diffusion (SD) in replicating training data could be taken advantage of by attackers to launch the Copyright Infringement Attack, with duplicated poisoned image-text pairs. SilentBadDiffusion (SBD) is a method proposed recently, which shew outstanding performance in attacking SD in text-to-image tasks. However, the feasible data resources in this area are still limited, some of them are even constrained or prohibited due to the issues like copyright ownership or inappropriate contents; And not all of the images in current datasets are suitable for the proposed attacking methods; Besides, the state-of-the-art (SoTA) performance of SBD is far from ideal when few generated poisoning samples could be adopted for attacks. In this paper, we raised new datasets accessible for researching in attacks like SBD, and proposed Multi-Element (ME) attack method based on SBD by increasing the number of poisonous visual-text elements per poisoned sample to enhance the ability of attacking, while importing Discrete Cosine Transform (DCT) for the poisoned samples to maintain the stealthiness. The Copyright Infringement Rate (CIR) / First Attack Epoch (FAE) we got on the two new datasets were 16.78% / 39.50 and 51.20% / 23.60, respectively close to or even outperformed benchmark Pokemon and Mijourney datasets. In condition of low subsampling ratio (5%, 6 poisoned samples), MESI and DCT earned CIR / FAE of 0.23% / 84.00 and 12.73% / 65.50, both better than original SBD, which failed to attack at all. 

**Abstract (ZH)**: Generative扩散模型在版权侵权攻击中的应用与增强：SilentBadDiffusion及其Multi-Element攻击方法的研究 

---
# Stroke-based Cyclic Amplifier: Image Super-Resolution at Arbitrary Ultra-Large Scales 

**Title (ZH)**: 基于抽样周期的循环放大器：任意超大规模图像超分辨率 

**Authors**: Wenhao Guo, Peng Lu, Xujun Peng, Zhaoran Zhao, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10774)  

**Abstract**: Prior Arbitrary-Scale Image Super-Resolution (ASISR) methods often experience a significant performance decline when the upsampling factor exceeds the range covered by the training data, introducing substantial blurring. To address this issue, we propose a unified model, Stroke-based Cyclic Amplifier (SbCA), for ultra-large upsampling tasks. The key of SbCA is the stroke vector amplifier, which decomposes the image into a series of strokes represented as vector graphics for magnification. Then, the detail completion module also restores missing details, ensuring high-fidelity image reconstruction. Our cyclic strategy achieves ultra-large upsampling by iteratively refining details with this unified SbCA model, trained only once for all, while keeping sub-scales within the training range. Our approach effectively addresses the distribution drift issue and eliminates artifacts, noise and blurring, producing high-quality, high-resolution super-resolved images. Experimental validations on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing methods in ultra-large upsampling tasks (e.g. $\times100$), delivering visual quality far superior to state-of-the-art techniques. 

**Abstract (ZH)**: 基于笔画循环放大器的超大规模图像超分辨率方法 

---
# Learning Chaotic Dynamics with Neuromorphic Network Dynamics 

**Title (ZH)**: 学习混沌动态的神经形态网络动力学 

**Authors**: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic  

**Link**: [PDF](https://arxiv.org/pdf/2506.10773)  

**Abstract**: This study investigates how dynamical systems may be learned and modelled with a neuromorphic network which is itself a dynamical system. The neuromorphic network used in this study is based on a complex electrical circuit comprised of memristive elements that produce neuro-synaptic nonlinear responses to input electrical signals. To determine how computation may be performed using the physics of the underlying system, the neuromorphic network was simulated and evaluated on autonomous prediction of a multivariate chaotic time series, implemented with a reservoir computing framework. Through manipulating only input electrodes and voltages, optimal nonlinear dynamical responses were found when input voltages maximise the number of memristive components whose internal dynamics explore the entire dynamical range of the memristor model. Increasing the network coverage with the input electrodes was found to suppress other nonlinear responses that are less conducive to learning. These results provide valuable insights into how a practical neuromorphic network device can be optimised for learning complex dynamical systems using only external control parameters. 

**Abstract (ZH)**: 本研究探讨如何使用自身为动力学系统的神经形态网络来学习和建模动力学系统。该研究中使用的神经形态网络基于由忆阻元件组成的复杂数字电路，这些元件对输入电信号产生神经突触非线性响应。通过模拟和在动力学范围广的忆阻器模型中探究内部动力学的复杂数值时间序列的自主预测性能评估，研究了如何利用底层系统的物理特性来执行计算。通过仅调节输入电极和电压，研究发现在最大化探索忆阻器模型整个动力学范围的忆阻器组件数量时，可以实现最优的非线性动力学响应。增加输入电极覆盖范围被发现会抑制其他不利于学习的非线性响应。这些结果为仅通过外部控制参数优化实用神经形态网络设备以学习复杂动力学系统提供了有价值的见解。 

---
# TED-LaST: Towards Robust Backdoor Defense Against Adaptive Attacks 

**Title (ZH)**: TED-LaST: 面向适应性攻击的鲁棒后门防御方法 

**Authors**: Xiaoxing Mo, Yuxuan Cheng, Nan Sun, Leo Yu Zhang, Wei Luo, Shang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10722)  

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to backdoor attacks, where attackers implant hidden triggers during training to maliciously control model behavior. Topological Evolution Dynamics (TED) has recently emerged as a powerful tool for detecting backdoor attacks in DNNs. However, TED can be vulnerable to backdoor attacks that adaptively distort topological representation distributions across network layers. To address this limitation, we propose TED-LaST (Topological Evolution Dynamics against Laundry, Slow release, and Target mapping attack strategies), a novel defense strategy that enhances TED's robustness against adaptive attacks. TED-LaST introduces two key innovations: label-supervised dynamics tracking and adaptive layer emphasis. These enhancements enable the identification of stealthy threats that evade traditional TED-based defenses, even in cases of inseparability in topological space and subtle topological perturbations. We review and classify data poisoning tricks in state-of-the-art adaptive attacks and propose enhanced adaptive attack with target mapping, which can dynamically shift malicious tasks and fully leverage the stealthiness that adaptive attacks possess. Our comprehensive experiments on multiple datasets (CIFAR-10, GTSRB, and ImageNet100) and model architectures (ResNet20, ResNet101) show that TED-LaST effectively counteracts sophisticated backdoors like Adap-Blend, Adapt-Patch, and the proposed enhanced adaptive attack. TED-LaST sets a new benchmark for robust backdoor detection, substantially enhancing DNN security against evolving threats. 

**Abstract (ZH)**: 基于拓扑演化动力学的防洗牌缓释放标 targeted 防护策略：应对动态后门攻击 

---
# Deep Learning-based Multi Project InP Wafer Simulation for Unsupervised Surface Defect Detection 

**Title (ZH)**: 基于深度学习的多项目InP晶圆仿真用于无监督表面缺陷检测 

**Authors**: Emílio Dolgener Cantú, Rolf Klemens Wittmann, Oliver Abdeen, Patrick Wagner, Wojciech Samek, Moritz Baier, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10713)  

**Abstract**: Quality management in semiconductor manufacturing often relies on template matching with known golden standards. For Indium-Phosphide (InP) multi-project wafer manufacturing, low production scale and high design variability lead to such golden standards being typically unavailable. Defect detection, in turn, is manual and labor-intensive. This work addresses this challenge by proposing a methodology to generate a synthetic golden standard using Deep Neural Networks, trained to simulate photo-realistic InP wafer images from CAD data. We evaluate various training objectives and assess the quality of the simulated images on both synthetic data and InP wafer photographs. Our deep-learning-based method outperforms a baseline decision-tree-based approach, enabling the use of a 'simulated golden die' from CAD plans in any user-defined region of a wafer for more efficient defect detection. We apply our method to a template matching procedure, to demonstrate its practical utility in surface defect detection. 

**Abstract (ZH)**: 半导体制造中的质量管理通常依赖于与已知黄金标准进行模板匹配。对于InP多项目晶圆制造，由于低生产规模和高设计变异性，通常缺乏黄金标准。因此，缺陷检测需要手工进行且劳动密集型。本工作通过提出一种使用深度神经网络生成合成黄金标准的方法来应对这一挑战，该网络训练以从CAD数据中模拟出光现实的InP晶圆图像。我们评估了各种训练目标，并在合成数据和InP晶圆照片上评估模拟图像的质量。基于深度学习的方法优于基准的决策树方法，使得在晶圆的任何用户定义区域内使用“从CAD计划生成的模拟黄金晶圆”进行缺陷检测更加高效。我们应用该方法到模板匹配过程，以展示其在表面缺陷检测中的实际应用价值。 

---
# Continual Hyperbolic Learning of Instances and Classes 

**Title (ZH)**: 持续双曲学习实例和类別 

**Authors**: Melika Ayoughi, Mina Ghadimi Atigh, Mohammad Mahdi Derakhshani, Cees G. M. Snoek, Pascal Mettes, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2506.10710)  

**Abstract**: Continual learning has traditionally focused on classifying either instances or classes, but real-world applications, such as robotics and self-driving cars, require models to handle both simultaneously. To mirror real-life scenarios, we introduce the task of continual learning of instances and classes, at the same time. This task challenges models to adapt to multiple levels of granularity over time, which requires balancing fine-grained instance recognition with coarse-grained class generalization. In this paper, we identify that classes and instances naturally form a hierarchical structure. To model these hierarchical relationships, we propose HyperCLIC, a continual learning algorithm that leverages hyperbolic space, which is uniquely suited for hierarchical data due to its ability to represent tree-like structures with low distortion and compact embeddings. Our framework incorporates hyperbolic classification and distillation objectives, enabling the continual embedding of hierarchical relations. To evaluate performance across multiple granularities, we introduce continual hierarchical metrics. We validate our approach on EgoObjects, the only dataset that captures the complexity of hierarchical object recognition in dynamic real-world environments. Empirical results show that HyperCLIC operates effectively at multiple granularities with improved hierarchical generalization. 

**Abstract (ZH)**: 持续学习传统上侧重于分类实例或类，但机器人技术和自动驾驶汽车等实际应用要求模型同时处理实例和类。为模拟现实场景，我们引入了实例和类的同时持续学习任务。此任务要求模型随时间适应不同的粒度层次，需要在细粒度实例识别与粗粒度类泛化之间取得平衡。在本文中，我们发现类和实例自然形成了一个层次结构。为了建模这些层次关系，我们提出了一种名为HyperCLIC的持续学习算法，该算法利用双曲空间，因其能够以低失真的方式表示树状结构并提供紧凑的嵌入而特别适合层次数据。我们的框架结合了双曲分类和蒸馏目标，能够持续嵌入层次关系。为了评估多粒度下的性能，我们引入了持续层次度量。我们通过EgoObjects数据集验证了该方法，这是唯一一个捕捉动态现实环境中超级对象识别复杂性的数据集。经验结果表明，HyperCLIC在多个粒度层次上有效运行，并具有改进的层次泛化能力。 

---
# Formalising Software Requirements using Large Language Models 

**Title (ZH)**: 使用大型语言模型形式化软件需求 

**Authors**: Arshad Beg, Diarmuid O'Donoghue, Rosemary Monahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10704)  

**Abstract**: This paper is a brief introduction to our recently initiated project named VERIFAI: Traceability and verification of natural language requirements. The project addresses the challenges in the traceability and verification of formal specifications through providing support for the automatic generation of the formal specifications and the traceability of the requirements from the initial software design stage through the systems implementation and verification. Approaches explored in this project include Natural Language Processing, use of ontologies to describe the software system domain, reuse of existing software artefacts from similar systems (i.e. through similarity based reuse) and large language models to identify and declare the specifications as well as use of artificial intelligence to guide the process. 

**Abstract (ZH)**: 本论文是对最近启动的VERIFAI项目的一个简要介绍：自然语言需求的可追溯性和验证。该项目通过提供自动生成形式化规范和支持从初始软件设计阶段到系统实现和验证的可追溯性来解决形式化规范的可追溯性和验证挑战。项目中探索的方法包括自然语言处理、使用Ontology描述软件系统领域、基于相似性重用现有软件 artefacts以及使用大规模语言模型标识和声明规范，并利用人工智能指导这一过程。 

---
# Saturation Self-Organizing Map 

**Title (ZH)**: 饱和自组织地图 

**Authors**: Igor Urbanik, Paweł Gajewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.10680)  

**Abstract**: Continual learning poses a fundamental challenge for neural systems, which often suffer from catastrophic forgetting when exposed to sequential tasks. Self-Organizing Maps (SOMs), despite their interpretability and efficiency, are not immune to this issue. In this paper, we introduce Saturation Self-Organizing Maps (SatSOM)-an extension of SOMs designed to improve knowledge retention in continual learning scenarios. SatSOM incorporates a novel saturation mechanism that gradually reduces the learning rate and neighborhood radius of neurons as they accumulate information. This effectively freezes well-trained neurons and redirects learning to underutilized areas of the map. 

**Abstract (ZH)**: 持续学习对神经系统构成了根本性的挑战，当神经网络暴露于顺序任务时常会出现灾难性遗忘。自组织映射(SOM)尽管具备可解释性和高效性，但也不免受到这一问题的影响。本文引入了饱和自组织映射(SatSOM)——一种为持续学习场景设计、旨在提高知识保留能力的SOM扩展模型。SatSOM集成了一种新颖的饱和机制，该机制逐渐降低累积信息的神经元的学习率和邻域半径，从而有效冻结训练良好的神经元，并将学习过程转向映射中利用率较低的区域。 

---
# Contrastive Matrix Completion with Denoising and Augmented Graph Views for Robust Recommendation 

**Title (ZH)**: 降噪与增强图视图相结合的对比矩阵补全稳健推荐 

**Authors**: Narges Nemati, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10658)  

**Abstract**: Matrix completion is a widely adopted framework in recommender systems, as predicting the missing entries in the user-item rating matrix enables a comprehensive understanding of user preferences. However, current graph neural network (GNN)-based approaches are highly sensitive to noisy or irrelevant edges--due to their inherent message-passing mechanisms--and are prone to overfitting, which limits their generalizability. To overcome these challenges, we propose a novel method called Matrix Completion using Contrastive Learning (MCCL). Our approach begins by extracting local neighborhood subgraphs for each interaction and subsequently generates two distinct graph representations. The first representation emphasizes denoising by integrating GNN layers with an attention mechanism, while the second is obtained via a graph variational autoencoder that aligns the feature distribution with a standard prior. A mutual learning loss function is employed during training to gradually harmonize these representations, enabling the model to capture common patterns and significantly enhance its generalizability. Extensive experiments on several real-world datasets demonstrate that our approach not only improves the numerical accuracy of the predicted scores--achieving up to a 0.8% improvement in RMSE--but also produces superior rankings with improvements of up to 36% in ranking metrics. 

**Abstract (ZH)**: 矩阵补全是一种广泛采用的推荐系统框架，通过预测用户-项评分矩阵中的缺失条目，可以全面理解用户偏好。然而，当前基于图神经网络（GNN）的方法对其固有的消息传递机制极为敏感，容易受到噪声或无关边的影响，并且容易过拟合，这限制了它们的泛化能力。为克服这些挑战，我们提出了一种新颖的方法，称为对比学习下的矩阵补全（MCCL）。该方法首先为每种交互提取局部邻域子图，然后生成两种不同的图表示。第一种表示通过结合GNN层和注意机制强调去噪，第二种表示通过图变分自编码器获得，该编码器将特征分布与标准先验对齐。在训练过程中采用互学习损失函数逐步协调这两种表示，使模型能够捕捉共同模式，显著增强其泛化能力。在多个真实世界数据集上的 extensive 实验表明，我们的方法不仅提高了预测分数的数值准确性——在均方根误差（RMSE）上提高了多达 0.8%——而且在排名指标上也表现出更优的效果，排名改进幅度高达 36%。 

---
# Deep Learning-Based Digitization of Overlapping ECG Images with Open-Source Python Code 

**Title (ZH)**: 基于深度学习的重叠心电图图像数字化开源Python代码实现 

**Authors**: Reza Karbasi, Masoud Rahimi, Abdol-Hossein Vahabie, Hadi Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10617)  

**Abstract**: This paper addresses the persistent challenge of accurately digitizing paper-based electrocardiogram (ECG) recordings, with a particular focus on robustly handling single leads compromised by signal overlaps-a common yet under-addressed issue in existing methodologies. We propose a two-stage pipeline designed to overcome this limitation. The first stage employs a U-Net based segmentation network, trained on a dataset enriched with overlapping signals and fortified with custom data augmentations, to accurately isolate the primary ECG trace. The subsequent stage converts this refined binary mask into a time-series signal using established digitization techniques, enhanced by an adaptive grid detection module for improved versatility across different ECG formats and scales. Our experimental results demonstrate the efficacy of our approach. The U-Net architecture achieves an IoU of 0.87 for the fine-grained segmentation task. Crucially, our proposed digitization method yields superior performance compared to a well-established baseline technique across both non-overlapping and challenging overlapping ECG samples. For non-overlapping signals, our method achieved a Mean Squared Error (MSE) of 0.0010 and a Pearson Correlation Coefficient (rho) of 0.9644, compared to 0.0015 and 0.9366, respectively, for the baseline. On samples with signal overlap, our method achieved an MSE of 0.0029 and a rho of 0.9641, significantly improving upon the baseline's 0.0178 and 0.8676. This work demonstrates an effective strategy to significantly enhance digitization accuracy, especially in the presence of signal overlaps, thereby laying a strong foundation for the reliable conversion of analog ECG records into analyzable digital data for contemporary research and clinical applications. The implementation is publicly available at this GitHub repository: this https URL. 

**Abstract (ZH)**: 本文解决了一直存在的准确数字化基于纸张的心电图（ECG）记录的持续挑战，特别关注在现有方法中常被忽视但又十分常见的信号重叠导致单通道数据受损问题。我们提出了一种两阶段的流水线设计来克服这一限制。第一阶段采用一种基于U-Net的分割网络，该网络通过增强带有重叠信号的数据集并结合自定义数据增强进行训练，以准确地分离主要的ECG轨迹。第二阶段采用现有的数字化技术将这个精炼的二进制掩模转化为时间序列信号，并通过自适应网格检测模块提高不同ECG格式和规模下的灵活性。我们的实验结果证明了该方法的有效性。U-Net架构在细粒度分割任务中达到了0.87的IoU。至关重要的是，我们提出的数据数字化方法在非重叠信号和具有挑战性的重叠信号样本上均优于现有的基准技术。对于非重叠信号，该方法的均方误差（MSE）为0.0010，皮尔森相关系数（rho）为0.9644，而基准技术分别为0.0015和0.9366。在具有信号重叠的样本上，该方法的MSE为0.0029，rho为0.9641，显著优于基准技术的0.0178和0.8676。本文展示了在信号重叠存在的情况下显著提高数字化准确性的有效策略，为将模拟ECG记录可靠地转换为可分析的数字数据以供当前研究和临床应用奠定了坚实的基础。该实现已公开发布在 GitHub 仓库中：this https URL。 

---
# TexTailor: Customized Text-aligned Texturing via Effective Resampling 

**Title (ZH)**: TexTailor: 通过有效的重采样实现自定义文本对齐纹理化 

**Authors**: Suin Lee, Dae-Shik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10612)  

**Abstract**: We present TexTailor, a novel method for generating consistent object textures from textual descriptions. Existing text-to-texture synthesis approaches utilize depth-aware diffusion models to progressively generate images and synthesize textures across predefined multiple viewpoints. However, these approaches lead to a gradual shift in texture properties across viewpoints due to (1) insufficient integration of previously synthesized textures at each viewpoint during the diffusion process and (2) the autoregressive nature of the texture synthesis process. Moreover, the predefined selection of camera positions, which does not account for the object's geometry, limits the effective use of texture information synthesized from different viewpoints, ultimately degrading overall texture consistency. In TexTailor, we address these issues by (1) applying a resampling scheme that repeatedly integrates information from previously synthesized textures within the diffusion process, and (2) fine-tuning a depth-aware diffusion model on these resampled textures. During this process, we observed that using only a few training images restricts the model's original ability to generate high-fidelity images aligned with the conditioning, and therefore propose an performance preservation loss to mitigate this issue. Additionally, we improve the synthesis of view-consistent textures by adaptively adjusting camera positions based on the object's geometry. Experiments on a subset of the Objaverse dataset and the ShapeNet car dataset demonstrate that TexTailor outperforms state-of-the-art methods in synthesizing view-consistent textures. The source code for TexTailor is available at this https URL 

**Abstract (ZH)**: TexTailor：一种从文本描述生成一致对象纹理的新方法 

---
# Size-adaptive Hypothesis Testing for Fairness 

**Title (ZH)**: 自适应样本大小公平性假设检验 

**Authors**: Antonio Ferrara, Francesco Cozzi, Alan Perotti, André Panisson, Francesco Bonchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10586)  

**Abstract**: Determining whether an algorithmic decision-making system discriminates against a specific demographic typically involves comparing a single point estimate of a fairness metric against a predefined threshold. This practice is statistically brittle: it ignores sampling error and treats small demographic subgroups the same as large ones. The problem intensifies in intersectional analyses, where multiple sensitive attributes are considered jointly, giving rise to a larger number of smaller groups. As these groups become more granular, the data representing them becomes too sparse for reliable estimation, and fairness metrics yield excessively wide confidence intervals, precluding meaningful conclusions about potential unfair treatments.
In this paper, we introduce a unified, size-adaptive, hypothesis-testing framework that turns fairness assessment into an evidence-based statistical decision. Our contribution is twofold. (i) For sufficiently large subgroups, we prove a Central-Limit result for the statistical parity difference, leading to analytic confidence intervals and a Wald test whose type-I (false positive) error is guaranteed at level $\alpha$. (ii) For the long tail of small intersectional groups, we derive a fully Bayesian Dirichlet-multinomial estimator; Monte-Carlo credible intervals are calibrated for any sample size and naturally converge to Wald intervals as more data becomes available. We validate our approach empirically on benchmark datasets, demonstrating how our tests provide interpretable, statistically rigorous decisions under varying degrees of data availability and intersectionality. 

**Abstract (ZH)**: 确定算法决策系统是否针对特定 demographic 进行歧视通常涉及将公平性指标的单点估计与预定义的阈值进行比较。这一做法在统计上是脆弱的：它忽视了抽样误差，并将小的 demographic 子群体与大的子群体同等对待。在考虑多个敏感属性的交叉分析中，这一问题更加严重，产生了更多的小群体。随着这些群体变得越来越细分，代表它们的数据变得过于稀疏，无法进行可靠的估计，公平性指标导出了过宽的置信区间，阻碍了对潜在不公平待遇的有意义结论。本文介绍了一个统一体积自适应假设检验框架，将其公平性评估转化为基于证据的统计决策。我们的贡献包括：（i）对于足够大的子群体，我们证明了统计均等差的中心极限定理，从而得到分析置信区间和类型-I错误（虚假阳性）保证在水平$\alpha$的沃尔德检验；（ii）对于小的交叉子群体的长尾部分，我们推导出一个完全贝叶斯狄利克雷-多项式估计器；蒙特卡洛可信区间根据样本大小进行了校准，并随着可用数据的增加自然收敛到沃尔德区间。我们在基准数据集上进行了实证验证，展示了我们的方法如何在不同数据可用性和交叉性程度下提供可解释且统计上严谨的决策。 

---
# Balancing Tails when Comparing Distributions: Comprehensive Equity Index (CEI) with Application to Bias Evaluation in Operational Face Biometrics 

**Title (ZH)**: 在比较分布时平衡尾部：综合公平指数（CEI）及其在操作面部生物识别偏差评估中的应用 

**Authors**: Imanol Solano, Julian Fierrez, Aythami Morales, Alejandro Peña, Ruben Tolosana, Francisco Zamora-Martinez, Javier San Agustin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10564)  

**Abstract**: Demographic bias in high-performance face recognition (FR) systems often eludes detection by existing metrics, especially with respect to subtle disparities in the tails of the score distribution. We introduce the Comprehensive Equity Index (CEI), a novel metric designed to address this limitation. CEI uniquely analyzes genuine and impostor score distributions separately, enabling a configurable focus on tail probabilities while also considering overall distribution shapes. Our extensive experiments (evaluating state-of-the-art FR systems, intentionally biased models, and diverse datasets) confirm CEI's superior ability to detect nuanced biases where previous methods fall short. Furthermore, we present CEI^A, an automated version of the metric that enhances objectivity and simplifies practical application. CEI provides a robust and sensitive tool for operational FR fairness assessment. The proposed methods have been developed particularly for bias evaluation in face biometrics but, in general, they are applicable for comparing statistical distributions in any problem where one is interested in analyzing the distribution tails. 

**Abstract (ZH)**: 全面公平指数（CEI）在高 PERFORMANCE 人脸鉴别系统中的性别偏差检测 

---
# Starting Positions Matter: A Study on Better Weight Initialization for Neural Network Quantization 

**Title (ZH)**: 初始权重值的选择 matters：关于神经网络量化更优权重初始化方法的研究 

**Authors**: Stone Yun, Alexander Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.10463)  

**Abstract**: Deep neural network (DNN) quantization for fast, efficient inference has been an important tool in limiting the cost of machine learning (ML) model inference. Quantization-specific model development techniques such as regularization, quantization-aware training, and quantization-robustness penalties have served to greatly boost the accuracy and robustness of modern DNNs. However, very little exploration has been done on improving the initial conditions of DNN training for quantization. Just as random weight initialization has been shown to significantly impact test accuracy of floating point models, it would make sense that different weight initialization methods impact quantization robustness of trained models. We present an extensive study examining the effects of different weight initializations on a variety of CNN building blocks commonly used in efficient CNNs. This analysis reveals that even with varying CNN architectures, the choice of random weight initializer can significantly affect final quantization robustness. Next, we explore a new method for quantization-robust CNN initialization -- using Graph Hypernetworks (GHN) to predict parameters of quantized DNNs. Besides showing that GHN-predicted parameters are quantization-robust after regular float32 pretraining (of the GHN), we find that finetuning GHNs to predict parameters for quantized graphs (which we call GHN-QAT) can further improve quantized accuracy of CNNs. Notably, GHN-QAT shows significant accuracy improvements for even 4-bit quantization and better-than-random accuracy for 2-bits. To the best of our knowledge, this is the first in-depth study on quantization-aware DNN weight initialization. GHN-QAT offers a novel approach to quantized DNN model design. Future investigations, such as using GHN-QAT-initialized parameters for quantization-aware training, can further streamline the DNN quantization process. 

**Abstract (ZH)**: Deep神经网络（DNN）量化以实现快速、高效的推理一直是限制机器学习（ML）模型推理成本的重要工具。针对量化特异性的模型开发技术，如正则化、量化感知训练和量化鲁棒性惩罚，显著提升了现代DNN的准确性和鲁棒性。然而，在改进DNN训练的初始条件以适应量化方面，研究工作还很少。正如随机权重初始化已被证明对浮点模型的测试准确率有显著影响一样，不同权重初始化方法对训练模型的量化鲁棒性也理应有显著影响。我们进行了一项广泛的分析，研究了不同权重初始化方法对在高效CNN中常用的多种CNN构建块的影响。这一分析揭示了即使在不同的CNN架构下，随机权重初始化的选择也会显著影响最终的量化鲁棒性。随后，我们探索了一种新的量化鲁棒CNN初始化方法——使用Graph超网络（GHN）来预测量化DNN的参数。除了证明GHN预测的参数在常规float32预训练后具有量化鲁棒性之外，我们还发现，微调GHN以预测量化图的参数（我们称之为GHN-QAT）可以进一步提高CNN的量化准确性。值得注意的是，GHN-QAT甚至在4位量化时显示出显著的准确性提升，并在2位量化时优于随机准确性。据我们所知，这是关于量化感知DNN权重初始化的首个详细研究。GHN-QAT为量化DNN模型设计提供了一种新颖的方法。未来的进一步研究，如使用GHN-QAT初始化参数进行量化感知训练，可以进一步简化DNN的量化过程。 

---
# Equitable Mechanism Design for Facility Location 

**Title (ZH)**: 公平设施定位机制设计 

**Authors**: Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10460)  

**Abstract**: We consider strategy proof mechanisms for facility location which maximize equitability between agents. As is common in the literature, we measure equitability with the Gini index. We first prove a simple but fundamental impossibility result that no strategy proof mechanism can bound the approximation ratio of the optimal Gini index of utilities for one or more facilities. We propose instead computing approximation ratios of the complemented Gini index of utilities, and consider how well both deterministic and randomized mechanisms approximate this. In addition, as Nash welfare is often put forwards as an equitable compromise between egalitarian and utilitarian outcomes, we consider how well mechanisms approximate the Nash welfare. 

**Abstract (ZH)**: 我们考虑最大化代理人之间公平性的设施定位机制，并采用吉尼指数衡量公平性。我们首先证明了一个简单但基本的不可能性结果，即不存在能够限制单个或多个设施的最优吉尼指数近似比的策略证明机制。我们建议计算效用的补吉尼指数的近似比，并考虑确定性和随机机制对此的逼近程度。此外，由于纳什福利通常被认为是平等主义和功利主义结果的公平妥协，我们研究机制对此的逼近程度。 

---
# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series 

**Title (ZH)**: Time-IMM：不规则多模态多变量时间序列的数据集及基准 

**Authors**: Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wen-Chih Peng, Tien-Fu Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10412)  

**Abstract**: Time series data in real-world applications such as healthcare, climate modeling, and finance are often irregular, multimodal, and messy, with varying sampling rates, asynchronous modalities, and pervasive missingness. However, existing benchmarks typically assume clean, regularly sampled, unimodal data, creating a significant gap between research and real-world deployment. We introduce Time-IMM, a dataset specifically designed to capture cause-driven irregularity in multimodal multivariate time series. Time-IMM represents nine distinct types of time series irregularity, categorized into trigger-based, constraint-based, and artifact-based mechanisms. Complementing the dataset, we introduce IMM-TSF, a benchmark library for forecasting on irregular multimodal time series, enabling asynchronous integration and realistic evaluation. IMM-TSF includes specialized fusion modules, including a timestamp-to-text fusion module and a multimodality fusion module, which support both recency-aware averaging and attention-based integration strategies. Empirical results demonstrate that explicitly modeling multimodality on irregular time series data leads to substantial gains in forecasting performance. Time-IMM and IMM-TSF provide a foundation for advancing time series analysis under real-world conditions. The dataset is publicly available at this https URL, and the benchmark library can be accessed at this https URL. 

**Abstract (ZH)**: 时间序列数据在医疗、气候建模和金融等实际应用中通常是不规则的、多模态的和混乱的，具有变化的采样率、异步的模态性和普遍的数据缺失问题。现有的基准数据集通常假定清洁的、定期采样的、单模态的数据，这在研究与实际部署之间造成了显著的差距。我们引入Time-IMM数据集，专门用于捕捉由因果驱动的多模态多变量时间序列的不规则性。Time-IMM代表九种不同类型的时间序列不规则性，分类为触发机制、约束机制和残余机制。为补充数据集，我们引入了IMM-TSF基准库，用于不规则多模态时间序列的预测，在异步集成和现实评价方面提供支持。IMM-TSF包含专门的融合模块，包括时间戳到文本融合模块和多模态融合模块，支持最近性感知平均策略和基于注意的集成策略。实证结果表明，在不规则时间序列数据中明确建模多模态性能够显著提高预测性能。Time-IMM和IMM-TSF为在实际条件下推进时间序列分析提供了基础。数据集可从以下网址获取：this https URL，基准库可从以下网址访问：this https URL。 

---
# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation 

**Title (ZH)**: 生 PhysioWave: 一种多尺度小波转换器的生理信号表示方法 

**Authors**: Yanlong Chen, Mattia Orlandi, Pierangelo Maria Rapa, Simone Benatti, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10351)  

**Abstract**: Physiological signals are often corrupted by motion artifacts, baseline drift, and other low-SNR disturbances, which pose significant challenges for analysis. Additionally, these signals exhibit strong non-stationarity, with sharp peaks and abrupt changes that evolve continuously, making them difficult to represent using traditional time-domain or filtering methods. To address these issues, a novel wavelet-based approach for physiological signal analysis is presented, aiming to capture multi-scale time-frequency features in various physiological signals. Leveraging this technique, two large-scale pretrained models specific to EMG and ECG are introduced for the first time, achieving superior performance and setting new baselines in downstream tasks. Additionally, a unified multi-modal framework is constructed by integrating pretrained EEG model, where each modality is guided through its dedicated branch and fused via learnable weighted fusion. This design effectively addresses challenges such as low signal-to-noise ratio, high inter-subject variability, and device mismatch, outperforming existing methods on multi-modal tasks. The proposed wavelet-based architecture lays a solid foundation for analysis of diverse physiological signals, while the multi-modal design points to next-generation physiological signal processing with potential impact on wearable health monitoring, clinical diagnostics, and broader biomedical applications. 

**Abstract (ZH)**: 生理信号常受到运动伪影、基线漂移和其他低信噪比干扰的污染，这对分析构成了重大挑战。此外，这些信号表现出强烈的非 Stationarity，拥有不断变化的尖峰和突变，传统的时间域方法或滤波方法难以对其进行表示。为解决这些问题，本文提出了一种基于小波的新颖生理信号分析方法，旨在捕获各种生理信号的多尺度时频特征。利用该技术，首次引入了针对肌电图（EMG）和心电图（ECG）的两个大规模预训练模型，这些模型在下游任务中表现出色，并设立了新的基准。此外，通过将预训练的大脑电图（EEG）模型与多种模态整合构建了一个统一的多模态框架，每种模态通过其专门的分支并借助可学习加权融合进行融合。该设计有效解决了低信噪比、高个体间变异性及设备匹配不良等挑战，多模态任务上优于现有方法。提出的基于小波的架构为各种生理信号的分析奠定了坚实基础，而多模态设计预示着下一代生理信号处理的发展，有可能在穿戴式健康监测、临床诊断以及更广泛的生物医学应用中产生重要影响。 

---
# DUN-SRE: Deep Unrolling Network with Spatiotemporal Rotation Equivariance for Dynamic MRI Reconstruction 

**Title (ZH)**: DUN-SRE：具有时空旋转不变性的深层解卷网络用于动态MRI重建 

**Authors**: Yuliang Zhu, Jing Cheng, Qi Xie, Zhuo-Xu Cui, Qingyong Zhu, Yuanyuan Liu, Xin Liu, Jianfeng Ren, Chengbo Wang, Dong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10309)  

**Abstract**: Dynamic Magnetic Resonance Imaging (MRI) exhibits transformation symmetries, including spatial rotation symmetry within individual frames and temporal symmetry along the time dimension. Explicit incorporation of these symmetry priors in the reconstruction model can significantly improve image quality, especially under aggressive undersampling scenarios. Recently, Equivariant convolutional neural network (ECNN) has shown great promise in exploiting spatial symmetry priors. However, existing ECNNs critically fail to model temporal symmetry, arguably the most universal and informative structural prior in dynamic MRI reconstruction. To tackle this issue, we propose a novel Deep Unrolling Network with Spatiotemporal Rotation Equivariance (DUN-SRE) for Dynamic MRI Reconstruction. The DUN-SRE establishes spatiotemporal equivariance through a (2+1)D equivariant convolutional architecture. In particular, it integrates both the data consistency and proximal mapping module into a unified deep unrolling framework. This architecture ensures rigorous propagation of spatiotemporal rotation symmetry constraints throughout the reconstruction process, enabling more physically accurate modeling of cardiac motion dynamics in cine MRI. In addition, a high-fidelity group filter parameterization mechanism is developed to maintain representation precision while enforcing symmetry constraints. Comprehensive experiments on Cardiac CINE MRI datasets demonstrate that DUN-SRE achieves state-of-the-art performance, particularly in preserving rotation-symmetric structures, offering strong generalization capability to a broad range of dynamic MRI reconstruction tasks. 

**Abstract (ZH)**: 动态磁共振成像（MRI）表现出变换对称性，包括个体框架内的空间旋转对称性和时间维度上的时间对称性。在重建模型中明确 Incorporate 这些对称性先验可以显著提高图像质量，尤其是在激进下采样场景下。最近，空间对称性协变卷积神经网络（ECNN）在利用空间对称性先验方面展现出巨大的潜力。然而，现有的 ECNN 严重无法建模时间对称性，这被认为是动态 MRI 重建中最普遍且最具信息量的结构先验。为解决这一问题，我们提出了一种用于动态 MRI 重建的时空旋转协变 Deep Unrolling 网络（DUN-SRE）。DUN-SRE 通过（2+1）D 协变卷积架构建立了时空协变性。特别地，它将数据一致性模块和邻近映射模块结合进一个统一的 Deep Unrolling 框架。该架构确保在重建过程中严格传播时空旋转对称性约束，从而更准确地建模 cine MRI 中的心脏运动动态。此外，开发了一种高保真群滤波器参数化机制，以在施加对称性约束的同时保持表示精度。全面的实验结果表明，DUN-SRE 在保持旋转对称结构方面达到了最先进的性能，具有较强的泛化能力，能够应对广泛的动态 MRI 重建任务。 

---
# Uncertainty-Aware Deep Learning for Automated Skin Cancer Classification: A Comprehensive Evaluation 

**Title (ZH)**: 基于不确定性感知的深度学习在皮肤癌自动化分类中的全面评估 

**Authors**: Hamzeh Asgharnezhad, Pegah Tabarisaadi, Abbas Khosravi, Roohallah Alizadehsani, U. Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2506.10302)  

**Abstract**: Accurate and reliable skin cancer diagnosis is critical for early treatment and improved patient outcomes. Deep learning (DL) models have shown promise in automating skin cancer classification, but their performance can be limited by data scarcity and a lack of uncertainty awareness. In this study, we present a comprehensive evaluation of DL-based skin lesion classification using transfer learning and uncertainty quantification (UQ) on the HAM10000 dataset. In the first phase, we benchmarked several pre-trained feature extractors-including Contrastive Language-Image Pretraining (CLIP) variants, Residual Network-50 (ResNet50), Densely Connected Convolutional Network (DenseNet121), Visual Geometry Group network (VGG16), and EfficientNet-V2-Large-combined with a range of traditional classifiers such as Support Vector Machine (SVM), eXtreme Gradient Boosting (XGBoost), and logistic regression. Our results show that CLIP-based vision transformers, particularly LAION CLIP ViT-H/14 with SVM, deliver the highest classification performance. In the second phase, we incorporated UQ using Monte Carlo Dropout (MCD), Ensemble, and Ensemble Monte Carlo Dropout (EMCD) to assess not only prediction accuracy but also the reliability of model outputs. We evaluated these models using uncertainty-aware metrics such as uncertainty accuracy(UAcc), uncertainty sensitivity(USen), uncertainty specificity(USpe), and uncertainty precision(UPre). The results demonstrate that ensemble methods offer a good trade-off between accuracy and uncertainty handling, while EMCD is more sensitive to uncertain predictions. This study highlights the importance of integrating UQ into DL-based medical diagnosis to enhance both performance and trustworthiness in real-world clinical applications. 

**Abstract (ZH)**: 准确可靠的皮肤癌诊断对于早期治疗和改善患者预后至关重要。深度学习模型在自动化皮肤癌分类方面显示出潜力，但其性能可能受限于数据稀缺性和不确定性意识不足。在本研究中，我们对基于迁移学习和不确定性量化（UQ）的深度学习皮肤病变分类进行了全面评估，使用了HAM10000数据集。在第一阶段，我们对标了几种预训练特征提取器，包括对比语言-图像预训练（CLIP）变体、残差网络-50（ResNet50）、密集连接卷积网络（DenseNet121）、视觉几何组网络（VGG16）和高效Net-V2-大型（EfficientNet-V2-Large），并结合了传统的分类器，如支持向量机（SVM）、极端梯度提升（XGBoost）和逻辑回归。结果表明，基于CLIP的视觉变换器，特别是LAION CLIP ViT-H/14与SVM结合，提供最高的分类性能。在第二阶段，我们引入了蒙特卡洛丢弃（MCD）、集成（Ensemble）和集成蒙特卡洛丢弃（EMCD），以评估预测准确性和模型输出的可靠性。我们使用不确定性意识度量标准，如不确定性准确度（UAcc）、不确定性敏感性（USen）、不确定性特异性（USpe）和不确定性精确度（UPre）评估这些模型。结果表明，集成方法在准确性和不确定性处理之间提供了良好的权衡，而EMCD对不确定的预测更为敏感。本研究强调了在基于深度学习的医疗诊断中集成不确定性量化的重要性，以提高实际临床应用中的性能和可信度。 

---
# Flick: Few Labels Text Classification using K-Aware Intermediate Learning in Multi-Task Low-Resource Languages 

**Title (ZH)**: Flick：多任务低资源语言中的K- aware中间学习少标签文本分类 

**Authors**: Ali Almutairi, Abdullah Alsuhaibani, Shoaib Jameel, Usman Naseem, Gelareh Mohammadi, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2506.10292)  

**Abstract**: Training deep learning networks with minimal supervision has gained significant research attention due to its potential to reduce reliance on extensive labelled data. While self-training methods have proven effective in semi-supervised learning, they remain vulnerable to errors from noisy pseudo labels. Moreover, most recent approaches to the few-label classification problem are either designed for resource-rich languages such as English or involve complex cascading models that are prone to overfitting. To address the persistent challenge of few-label text classification in truly low-resource linguistic contexts, where existing methods often struggle with noisy pseudo-labels and domain adaptation, we propose Flick. Unlike prior methods that rely on generic multi-cluster pseudo-labelling or complex cascading architectures, Flick leverages the fundamental insight that distilling high-confidence pseudo-labels from a broader set of initial clusters can dramatically improve pseudo-label quality, particularly for linguistically diverse, low-resource settings. Flick introduces a novel pseudo-label refinement component, a departure from traditional pseudo-labelling strategies by identifying and leveraging top-performing pseudo-label clusters. This component specifically learns to distil highly reliable pseudo-labels from an initial broad set by focusing on single-cluster cohesion and leveraging an adaptive top-k selection mechanism. This targeted refinement process is crucial for mitigating the propagation of errors inherent in low-resource data, allowing for robust fine-tuning of pre-trained language models with only a handful of true labels. We demonstrate Flick's efficacy across 14 diverse datasets, encompassing challenging low-resource languages such as Arabic, Urdu, and Setswana, alongside English, showcasing its superior performance and adaptability. 

**Abstract (ZH)**: 使用最少监督训练深度学习网络由于其减少对大量标注数据依赖的潜力而受到广泛关注。不同于以往方法依赖通用多集群伪标签或复杂级联架构，Flick通过从更广泛的初始集群中提取高置信度伪标签，显著提高伪标签质量，特别是在语言多样且资源稀缺的环境中。Flick引入了一种新的伪标签精炼组件，通过识别和利用表现最佳的伪标签集群，该组件专注于单集群凝聚力，并利用自适应top-k选择机制来提取高度可靠的伪标签。这一有针对性的精炼过程对于减轻低资源数据中错误的传播至关重要，允许仅使用少量真实标签对预训练语言模型进行稳健调优。我们展示了Flick在14个多样化的数据集上的有效性，涵盖了如阿拉伯语、乌尔都语和塞茨瓦纳语等具有挑战性的低资源语言，以及英语，展示了其优越的性能和适应能力。 

---
# RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding 

**Title (ZH)**: RT-VC: 实时零样本语音转换与语音articulatory编码 

**Authors**: Yisi Liu, Chenyang Wang, Hanjo Kim, Raniya Khan, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10289)  

**Abstract**: Voice conversion has emerged as a pivotal technology in numerous applications ranging from assistive communication to entertainment. In this paper, we present RT-VC, a zero-shot real-time voice conversion system that delivers ultra-low latency and high-quality performance. Our approach leverages an articulatory feature space to naturally disentangle content and speaker characteristics, facilitating more robust and interpretable voice transformations. Additionally, the integration of differentiable digital signal processing (DDSP) enables efficient vocoding directly from articulatory features, significantly reducing conversion latency. Experimental evaluations demonstrate that, while maintaining synthesis quality comparable to the current state-of-the-art (SOTA) method, RT-VC achieves a CPU latency of 61.4 ms, representing a 13.3\% reduction in latency. 

**Abstract (ZH)**: 零样本实时语音转换系统RT-VC：超低延迟与高质量性能 

---
# Extended Creativity: A Conceptual Framework for Understanding Human-AI Creative Relations 

**Title (ZH)**: 扩展创造力：理解人机创意关系的概念框架 

**Authors**: Andrea Gaggioli, Sabrina Bartolotta, Andrea Ubaldi, Katusha Gerardini, Eleonora Diletta Sarcinella, Alice Chirico  

**Link**: [PDF](https://arxiv.org/pdf/2506.10249)  

**Abstract**: Artificial Intelligence holds significant potential to enhance human creativity. However, achieving this vision requires a clearer understanding of how such enhancement can be effectively realized. Adopting the perspective of distributed creativity, we identify three primary modes through which AI can contribute to creative processes: Support, where AI acts as a tool; Synergy, where AI and humans collaborate in complementary ways; and Symbiosis, where human and AI cognition become so integrated that they form a unified creative system. These modes are defined along two key dimensions: the level of technical autonomy exhibited by the AI system and the degree of perceived agency attributed to it. We examine how each configuration influences different levels of creativity - from everyday problem-solving to paradigm-shifting innovation - and discuss the theoretical, ethical, and design implications. 

**Abstract (ZH)**: 人工智能在增强人类创造力方面具有重要的潜力。然而，实现这一愿景需要对如何有效地实现这种增强有更清晰的理解。从分布式创造力的视角出发，我们确定了人工智能可以通过三种主要方式贡献于创意过程：支持模式，其中人工智能作为工具发挥作用；协同模式，其中人工智能与人类以互补的方式合作；共生模式，其中人类与人工智能的认知深度融合，形成统一的创意系统。这些模式沿着两个关键维度定义：人工智能系统展现的技术自主性水平以及对其感知自主性的程度。我们探讨了每种配置如何影响不同层次的创造力——从日常问题解决到范式转变的创新——并讨论了相关的理论、伦理和设计意义。 

---
# ToxSyn-PT: A Large-Scale Synthetic Dataset for Hate Speech Detection in Portuguese 

**Title (ZH)**: ToxSyn-PT： Portuguese 垃圾言论检测的大规模合成数据集 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2506.10245)  

**Abstract**: We present ToxSyn-PT, the first large-scale Portuguese corpus that enables fine-grained hate-speech classification across nine legally protected minority groups. The dataset contains 53,274 synthetic sentences equally distributed between minorities groups and toxicity labels. ToxSyn-PT is created through a novel four-stage pipeline: (1) a compact, manually curated seed; (2) few-shot expansion with an instruction-tuned LLM; (3) paraphrase-based augmentation; and (4) enrichment, plus additional neutral texts to curb overfitting to group-specific cues. The resulting corpus is class-balanced, stylistically diverse, and free from the social-media domain that dominate existing Portuguese datasets. Despite domain differences with traditional benchmarks, experiments on both binary and multi-label classification on the corpus yields strong results across five public Portuguese hate-speech datasets, demonstrating robust generalization even across domain boundaries. The dataset is publicly released to advance research on synthetic data and hate-speech detection in low-resource settings. 

**Abstract (ZH)**: ToxSyn-PT：首个支持九个法律保护少数群体细粒度仇恨言论分类的大规模葡萄牙语语料库 

---
# Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods 

**Title (ZH)**: 提示攻击揭示去学习方法中浅层知识删除问题 

**Authors**: Yeonwoo Jang, Shariqah Hossain, Ashwin Sreevatsa, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10236)  

**Abstract**: In this work, we show that some machine unlearning methods may fail when subjected to straightforward prompt attacks. We systematically evaluate eight unlearning techniques across three model families, and employ output-based, logit-based, and probe analysis to determine to what extent supposedly unlearned knowledge can be retrieved. While methods like RMU and TAR demonstrate robust unlearning, ELM remains vulnerable to specific prompt attacks (e.g., Hindi filler text in original prompt recovering 57.3% accuracy). Our logit analysis also confirms that unlearned models are generally not hiding knowledge by modifying the way the answer is formatted, as the correlation between output and logit accuracy is strong. These results challenge prevailing assumptions about unlearning effectiveness and highlight the need for evaluation frameworks that can reliably distinguish between true knowledge removal and superficial output suppression. We also publicly make available our evaluation framework to easily evaluate prompting techniques to retrieve unlearning knowledge. 

**Abstract (ZH)**: 在本研究中，我们表明一些机器未学习方法在面对直接提示攻击时可能会失效。我们系统性地评估了八种未学习技术在三种模型家族中的效果，并通过输出分析、logit分析和探针分析来确定已声称未学习的知识能被恢复到何种程度。虽然像RMU和TAR这样的方法显示出较强的未学习能力，但ELM仍然对特定提示攻击（如原始提示中的印地语填充文本恢复57.3%的准确率）保持脆弱性。我们的logit分析也证实，未学习模型通常并不是通过改变答案格式的方式来隐藏知识，因为输出和logit准确率之间的相关性很强。这些结果挑战了现有的未学习效果假设，并强调了需要建立可靠的评估框架来区分真正的知识移除与表面上的输出抑制。我们还公开发布了我们的评估框架，以便于评估提示技术以检索未学习知识。 

---
# Fine-Grained control over Music Generation with Activation Steering 

**Title (ZH)**: 基于激活导向的细粒度音乐生成控制 

**Authors**: Dipanshu Panda, Jayden Koshy Joe, Harshith M R, Swathi Narashiman, Pranay Mathur, Anish Veerakumar, Aniruddh Krishna, Keerthiharan A  

**Link**: [PDF](https://arxiv.org/pdf/2506.10225)  

**Abstract**: We present a method for fine-grained control over music generation through inference-time interventions on an autoregressive generative music transformer called MusicGen. Our approach enables timbre transfer, style transfer, and genre fusion by steering the residual stream using weights of linear probes trained on it, or by steering the attention layer activations in a similar manner. We observe that modelling this as a regression task provides improved performance, hypothesizing that the mean-squared-error better preserve meaningful directional information in the activation space. Combined with the global conditioning offered by text prompts in MusicGen, our method provides both global and local control over music generation. Audio samples illustrating our method are available at our demo page. 

**Abstract (ZH)**: 我们提出了一种通过在自回归生成音乐变压器MusicGen中进行推理时干预以实现细粒度音乐生成控制的方法。该方法通过使用在残差流上训练的线性探针的权重来引导音色转移、风格转移和流派融合，或将注意力层激活以类似方式引导。我们观察到将此建模为回归任务可以提高性能，假设均方误差更好地保留了激活空间中的有意义方向信息。结合MusicGen中文本提示提供的全局条件，该方法为音乐生成提供了全局和局部控制。我们在演示页面上有音频示例展示该方法。 

---
# TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games 

**Title (ZH)**: TTT-Bench：一种基于简单新颖的井字游戏评估推理能力的基准测试 

**Authors**: Prakamya Mishra, Jiang Liu, Jialian Wu, Xiaodong Yu, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.10209)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive reasoning capabilities across a broad range of tasks including Olympiad-level mathematical problems, indicating evidence of their complex reasoning abilities. While many reasoning benchmarks focus on the STEM domain, the ability of LRMs to reason correctly in broader task domains remains underexplored. In this work, we introduce \textbf{TTT-Bench}, a new benchmark that is designed to evaluate basic strategic, spatial, and logical reasoning abilities in LRMs through a suite of four two-player Tic-Tac-Toe-style games that humans can effortlessly solve from a young age. We propose a simple yet scalable programmatic approach for generating verifiable two-player game problems for TTT-Bench. Although these games are trivial for humans, they require reasoning about the intentions of the opponent, as well as the game board's spatial configurations, to ensure a win. We evaluate a diverse set of state-of-the-art LRMs, and \textbf{discover that the models that excel at hard math problems frequently fail at these simple reasoning games}. Further testing reveals that our evaluated reasoning models score on average $\downarrow$ 41\% \& $\downarrow$ 5\% lower on TTT-Bench compared to MATH 500 \& AIME 2024 respectively, with larger models achieving higher performance using shorter reasoning traces, where most of the models struggle on long-term strategic reasoning situations on simple and new TTT-Bench tasks. 

**Abstract (ZH)**: 大型推理模型（LRMs）在包括奥林匹克级别数学问题在内的广泛任务中展示了令人印象深刻的推理能力，表明了它们复杂的推理能力。尽管许多推理基准关注STEM领域，但LRMs在更广泛任务域中的正确推理能力仍然未被充分探索。在本工作中，我们引入了TTT-Bench，这是一个新的基准，旨在通过一系列人类从小就能轻易解决的四款两人对弈的井字游戏来评估LRMs的基本战略、空间和逻辑推理能力。我们提出了一种简单而可扩展的编程方法，用于生成可验证的两人游戏问题以供TTT-Bench使用。虽然这些游戏对人类来说是简单的，但它们要求玩家不仅要考虑对手的意图，还要考虑游戏板的空间配置，以确保胜利。我们评估了一组多样化的最先进的LRMs，并发现擅长解决难题的模型在这些简单的推理游戏中经常表现不佳。进一步测试显示，我们评估的推理模型在TTT-Bench上的得分分别比MATH 500和AIME 2024低$\downarrow$ 41\% 和 $\downarrow$ 5%，其中较大的模型通过较短的推理过程获得更高的性能，而大多数模型在简单和全新的TTT-Bench任务中的长期战略推理情况上挣扎。 

---
# Optimizing Genetic Algorithms with Multilayer Perceptron Networks for Enhancing TinyFace Recognition 

**Title (ZH)**: 使用多层感知器网络优化遗传算法以提高TinyFace识别性能 

**Authors**: Mohammad Subhi Al-Batah, Mowafaq Salem Alzboon, Muhyeeddin Alqaraleh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10184)  

**Abstract**: This study conducts an empirical examination of MLP networks investigated through a rigorous methodical experimentation process involving three diverse datasets: TinyFace, Heart Disease, and Iris. Study Overview: The study includes three key methods: a) a baseline training using the default settings for the Multi-Layer Perceptron (MLP), b) feature selection using Genetic Algorithm (GA) based refinement c) Principal Component Analysis (PCA) based dimension reduction. The results show important information on how such techniques affect performance. While PCA had showed benefits in low-dimensional and noise-free datasets GA consistently increased accuracy in complex datasets by accurately identifying critical features. Comparison reveals that feature selection and dimensionality reduction play interdependent roles in enhancing MLP performance. The study contributes to the literature on feature engineering and neural network parameter optimization, offering practical guidelines for a wide range of machine learning tasks 

**Abstract (ZH)**: 本研究通过严格的实验方法对三种不同数据集（TinyFace、Heart Disease和Iris）下的MLP网络进行了实证考察。研究概述：研究包括三种关键方法：a) 使用默认设置训练Multi-Layer Perceptron (MLP)基线模型，b) 使用遗传算法（GA）进行特征选择精炼，c) 使用主成分分析（PCA）进行维度缩减。结果表明这些技术对性能的影响信息。虽然PCA在低维度和无噪声数据集中显示出优势，但GA在复杂数据集中通过准确识别关键特征持续增加了准确性。比较表明，特征选择和维度缩减在提升MLP性能方面相互依存。本研究为特征工程和神经网络参数优化文献做出了贡献，为广泛领域的机器学习任务提供了实用指南。 

---
# A Comparative Study of Machine Learning Techniques for Early Prediction of Diabetes 

**Title (ZH)**: 机器学习技术在糖尿病早期预测中的比较研究 

**Authors**: Mowafaq Salem Alzboon, Mohammad Al-Batah, Muhyeeddin Alqaraleh, Ahmad Abuashour, Ahmad Fuad Bader  

**Link**: [PDF](https://arxiv.org/pdf/2506.10180)  

**Abstract**: In many nations, diabetes is becoming a significant health problem, and early identification and control are crucial. Using machine learning algorithms to predict diabetes has yielded encouraging results. Using the Pima Indians Diabetes dataset, this study attempts to evaluate the efficacy of several machine-learning methods for diabetes prediction. The collection includes information on 768 patients, such as their ages, BMIs, and glucose levels. The techniques assessed are Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors, Naive Bayes, Support Vector Machine, Gradient Boosting, and Neural Network. The findings indicate that the Neural Network algorithm performed the best, with an accuracy of 78.57 percent, followed by the Random Forest method, with an accuracy of 76.30 percent. The study implies that machine learning algorithms can aid diabetes prediction and be an efficient early detection tool. 

**Abstract (ZH)**: 在许多国家，糖尿病已成为一个重要的健康问题，早期识别和控制至关重要。使用机器学习算法预测糖尿病取得了一些令人鼓舞的结果。本研究使用Pima Indians Diabetes数据集评估了几种机器学习方法在糖尿病预测中的有效性。该数据集包含了768名患者的信息，如年龄、BMI和血糖水平。评估的方法包括逻辑回归、决策树、随机森林、k-近邻、朴素贝叶斯、支持向量机、梯度提升和神经网络。研究结果表明，神经网络算法表现最佳，准确率为78.57%，紧随其后的是随机森林方法，准确率为76.30%。本研究表明，机器学习算法可以辅助糖尿病预测，并成为一种有效的早期检测工具。 

---
# SPARKE: Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score 

**Title (ZH)**: SPARKE: 扩展的基于提示的多样性引导在扩散模型中通过RKE分数 

**Authors**: Mohammad Jalali, Haoyu Lei, Amin Gohari, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2506.10173)  

**Abstract**: Diffusion models have demonstrated remarkable success in high-fidelity image synthesis and prompt-guided generative modeling. However, ensuring adequate diversity in generated samples of prompt-guided diffusion models remains a challenge, particularly when the prompts span a broad semantic spectrum and the diversity of generated data needs to be evaluated in a prompt-aware fashion across semantically similar prompts. Recent methods have introduced guidance via diversity measures to encourage more varied generations. In this work, we extend the diversity measure-based approaches by proposing the Scalable Prompt-Aware Rény Kernel Entropy Diversity Guidance (SPARKE) method for prompt-aware diversity guidance. SPARKE utilizes conditional entropy for diversity guidance, which dynamically conditions diversity measurement on similar prompts and enables prompt-aware diversity control. While the entropy-based guidance approach enhances prompt-aware diversity, its reliance on the matrix-based entropy scores poses computational challenges in large-scale generation settings. To address this, we focus on the special case of Conditional latent RKE Score Guidance, reducing entropy computation and gradient-based optimization complexity from the $O(n^3)$ of general entropy measures to $O(n)$. The reduced computational complexity allows for diversity-guided sampling over potentially thousands of generation rounds on different prompts. We numerically test the SPARKE method on several text-to-image diffusion models, demonstrating that the proposed method improves the prompt-aware diversity of the generated data without incurring significant computational costs. We release our code on the project page: this https URL 

**Abstract (ZH)**: 基于启发式的可扩展提示感知 Rényi 核熵多样性引导方法（SPARKE） 

---
# Measuring Corporate Human Capital Disclosures: Lexicon, Data, Code, and Research Opportunities 

**Title (ZH)**: 测量企业人力资本披露：词汇、数据、代码和研究机会 

**Authors**: Elizabeth Demers, Victor Xiaoqi Wang, Kean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10155)  

**Abstract**: Human capital (HC) is increasingly important to corporate value creation. Unlike other assets, however, HC is not currently subject to well-defined measurement or disclosure rules. We use a machine learning algorithm (word2vec) trained on a confirmed set of HC disclosures to develop a comprehensive list of HC-related keywords classified into five subcategories (DEI; health and safety; labor relations and culture; compensation and benefits; and demographics and other) that capture the multidimensional nature of HC management. We share our lexicon, corporate HC disclosures, and the Python code used to develop the lexicon, and we provide detailed examples of using our data and code, including for fine-tuning a BERT model. Researchers can use our HC lexicon (or modify the code to capture another construct of interest) with their samples of corporate communications to address pertinent HC questions. We close with a discussion of future research opportunities related to HC management and disclosure. 

**Abstract (ZH)**: 人力资本（HC）在企业价值创造中日益重要。然而，与其他资产不同，HC目前并没有明确的计量或披露规则。我们使用一种机器学习算法（word2vec），基于确认的HC披露数据集，开发了一个全面的HC相关关键词列表，按照五个子类别（DEI；健康与安全；劳动关系与文化；薪酬与福利；以及人口统计和其他）进行分类，以捕捉HC管理的多维性质。我们分享了我们的词汇表、公司HC披露数据以及开发词汇表所使用的Python代码，并提供了使用我们数据和代码的详细示例，包括对BERT模型进行微调。研究人员可以使用我们的HC词汇表（或修改代码以捕获另一个感兴趣的构造）与其公司的沟通样本相结合，以解决相关的人力资本问题。我们最后讨论了与HC管理和披露相关的未来研究机会。 

---
# Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban 

**Title (ZH)**: 解析学习到的搜索：在玩索班游戏的RNN中寻找状态转移模型和价值函数 

**Authors**: Mohammad Taufeeque, Aaron David Tucker, Adam Gleave, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2506.10138)  

**Abstract**: We partially reverse-engineer a convolutional recurrent neural network (RNN) trained to play the puzzle game Sokoban with model-free reinforcement learning. Prior work found that this network solves more levels with more test-time compute. Our analysis reveals several mechanisms analogous to components of classic bidirectional search. For each square, the RNN represents its plan in the activations of channels associated with specific directions. These state-action activations are analogous to a value function - their magnitudes determine when to backtrack and which plan branch survives pruning. Specialized kernels extend these activations (containing plan and value) forward and backward to create paths, forming a transition model. The algorithm is also unlike classical search in some ways. State representation is not unified; instead, the network considers each box separately. Each layer has its own plan representation and value function, increasing search depth. Far from being inscrutable, the mechanisms leveraging test-time compute learned in this network by model-free training can be understood in familiar terms. 

**Abstract (ZH)**: 我们部分反向-engineering一种用于使用模型无关强化学习训练的谜题游戏Sokoban玩法规则的卷积循环神经网络（RNN）。先前的研究发现，该网络在更多的测试时间计算资源下可以解决更多的关卡。我们的分析揭示了几种类似于经典双向搜索组件的机制。对于每个方格，RNN通过与特定方向相关的通道激活表示其计划。这些状态-动作激活类似于价值函数——其大小决定何时回溯以及哪些计划分支能够存活。专门的核向前和向后扩展这些包含计划和价值的激活来创建路径，形成转换模型。该算法在某些方面也不同于经典搜索。状态表示不是统一的；相反，网络分别考虑每个箱子。每层都有自己的计划表示和价值函数，从而增加了搜索深度。与普遍认为的难以理解不同，通过模型无关训练学习的利用测试时间计算资源的机制可以以熟悉的方式进行理解。 

---
# Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning 

**Title (ZH)**: 自预测表示在行为克隆中的组合泛化 

**Authors**: Daniel Lawson, Adriana Hugessen, Charlotte Cloutier, Glen Berseth, Khimya Khetarpal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10137)  

**Abstract**: Behavioral cloning (BC) methods trained with supervised learning (SL) are an effective way to learn policies from human demonstrations in domains like robotics. Goal-conditioning these policies enables a single generalist policy to capture diverse behaviors contained within an offline dataset. While goal-conditioned behavior cloning (GCBC) methods can perform well on in-distribution training tasks, they do not necessarily generalize zero-shot to tasks that require conditioning on novel state-goal pairs, i.e. combinatorial generalization. In part, this limitation can be attributed to a lack of temporal consistency in the state representation learned by BC; if temporally related states are encoded to similar latent representations, then the out-of-distribution gap for novel state-goal pairs would be reduced. Hence, encouraging this temporal consistency in the representation space should facilitate combinatorial generalization. Successor representations, which encode the distribution of future states visited from the current state, nicely encapsulate this property. However, previous methods for learning successor representations have relied on contrastive samples, temporal-difference (TD) learning, or both. In this work, we propose a simple yet effective representation learning objective, $\text{BYOL-}\gamma$ augmented GCBC, which is not only able to theoretically approximate the successor representation in the finite MDP case without contrastive samples or TD learning, but also, results in competitive empirical performance across a suite of challenging tasks requiring combinatorial generalization. 

**Abstract (ZH)**: 基于行为克隆的目标条件化方法通过监督学习训练，在 Robotics 等领域从人类演示中学习策略是一种有效的方式。目标条件化这些策略使得单一通用策略能够捕捉到离线数据集中包含的多样行为。虽然目标条件化行为克隆（GCBC）方法在同分布训练任务中表现良好，但在需要以新颖状态-目标对进行条件化的新任务上并不必然实现零样本泛化，即组合泛化。这一局限部分源于行为克隆学习的状态表示缺乏时间一致性；如果相关时间状态被编码到相似的潜在表示中，那么针对新颖状态-目标对的泛化差距将会减小。因此，在表示空间中鼓励这种时间一致性将有助于组合泛化。后继表示，它编码从当前状态访问的未来状态的分布，恰好体现了这一特性。然而，之前学习后继表示的方法依赖于对比样本、时间差分（TD）学习或者两者结合。在本文中，我们提出了一种简单而有效的表示学习目标——$\text{BYOL-}\gamma$增强GCBC，它不仅能够在不使用对比样本或TD学习的情况下理论上逼近在有限MDP情况下的后继表示，还在多种需要组合泛化的挑战性任务中实现了有竞争力的实验性能。 

---
# GRAIL: A Benchmark for GRaph ActIve Learning in Dynamic Sensing Environments 

**Title (ZH)**: GRAIL：动态传感环境中图活性学习的基准评测 

**Authors**: Maryam Khalid, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2506.10120)  

**Abstract**: Graph-based Active Learning (AL) leverages the structure of graphs to efficiently prioritize label queries, reducing labeling costs and user burden in applications like health monitoring, human behavior analysis, and sensor networks. By identifying strategically positioned nodes, graph AL minimizes data collection demands while maintaining model performance, making it a valuable tool for dynamic environments. Despite its potential, existing graph AL methods are often evaluated on static graph datasets and primarily focus on prediction accuracy, neglecting user-centric considerations such as sampling diversity, query fairness, and adaptability to dynamic settings. To bridge this gap, we introduce GRAIL, a novel benchmarking framework designed to evaluate graph AL strategies in dynamic, real-world environments. GRAIL introduces novel metrics to assess sustained effectiveness, diversity, and user burden, enabling a comprehensive evaluation of AL methods under varying conditions. Extensive experiments on datasets featuring dynamic, real-life human sensor data reveal trade-offs between prediction performance and user burden, highlighting limitations in existing AL strategies. GRAIL demonstrates the importance of balancing node importance, query diversity, and network topology, providing an evaluation mechanism for graph AL solutions in dynamic environments. 

**Abstract (ZH)**: 基于图的主动学习（AL）利用图的结构有效优先选择标签查询，减少健康监测、人类行为分析和传感器网络等应用中的标注成本和用户负担。通过识别战略节点，图AL在维持模型性能的同时降低数据收集需求，使其成为动态环境中的 valuable 工具。尽管具有潜力，现有的图AL方法大多在静态图数据集上进行评估，并主要关注预测准确性，忽视了以用户为中心的考虑，如采样多样性、查询公平性和动态环境适应性。为弥补这一差距，我们提出 GRAIL，一种新型基准测试框架，旨在评估图AL策略在动态的真实世界环境中的表现。GRAIL 引入新型指标评估持续有效性、多样性和用户负担，使AL方法在不同条件下进行全面评估。大规模实验结果显示，在动态现实生活中的人体传感器数据集上，预测性能与用户负担之间的权衡关系，突显了现有AL策略的局限性。GRAIL 指出了平衡节点重要性、查询多样性和网络拓扑的重要性，提供了在动态环境中评估图AL解决方案的机制。 

---
# Learning to Collaborate Over Graphs: A Selective Federated Multi-Task Learning Approach 

**Title (ZH)**: 基于图的协同学习：一种选择性联邦多任务学习方法 

**Authors**: Ahmed Elbakary, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.10102)  

**Abstract**: We present a novel federated multi-task learning method that leverages cross-client similarity to enable personalized learning for each client. To avoid transmitting the entire model to the parameter server, we propose a communication-efficient scheme that introduces a feature anchor, a compact vector representation that summarizes the features learned from the client's local classes. This feature anchor is shared with the server to account for local clients' distribution. In addition, the clients share the classification heads, a lightweight linear layer, and perform a graph-based regularization to enable collaboration among clients. By modeling collaboration between clients as a dynamic graph and continuously updating and refining this graph, we can account for any drift from the clients. To ensure beneficial knowledge transfer and prevent negative collaboration, we leverage a community detection-based approach that partitions this dynamic graph into homogeneous communities, maximizing the sum of task similarities, represented as the graph edges' weights, within each community. This mechanism restricts collaboration to highly similar clients within their formed communities, ensuring positive interaction and preserving personalization. Extensive experiments on two heterogeneous datasets demonstrate that our method significantly outperforms state-of-the-art baselines. Furthermore, we show that our method exhibits superior computation and communication efficiency and promotes fairness across clients. 

**Abstract (ZH)**: 我们提出了一种新颖的联邦多任务学习方法，通过利用客户端之间的相似性来实现个性化学习。为了避免传输整个模型到参数服务器，我们提出了一种通信高效的方案，引入了一个特征锚点，这是一种紧凑的向量表示，总结了客户端本地类别的特征学习结果。该特征锚点与服务器共享，以反映局部客户端的数据分布。此外，客户端共享分类头、一个轻量级线性层，并通过图谱正则化实现客户端之间的协作。通过将客户端的协作建模为动态图，并不断更新和完善这一图谱，可以考虑任何客户端可能发生的漂移。为了确保有益的知识迁移并防止负面协作，我们利用基于社区检测的方法将此动态图划分为同质社区，最大化每个社区内任务相似性的总和，用图边权重表示。这一机制限制了高度相似的客户端之间的协作，确保了正面互动并保持个性化。在两个异构数据集上的广泛实验表明，我们的方法显著优于现有最先进的基线方法。此外，我们展示了我们的方法在计算和通信效率方面表现出色，并促进了客户端之间的公平性。 

---
# Test-Time Adaptation for Generalizable Task Progress Estimation 

**Title (ZH)**: 运行时自适应以实现可泛化的任务进度估计 

**Authors**: Christos Ziakas, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10085)  

**Abstract**: We propose a test-time adaptation method that enables a progress estimation model to adapt online to the visual and temporal context of test trajectories by optimizing a learned self-supervised objective. To this end, we introduce a gradient-based meta-learning strategy to train the model on expert visual trajectories and their natural language task descriptions, such that test-time adaptation improves progress estimation relying on semantic content over temporal order. Our test-time adaptation method generalizes from a single training environment to diverse out-of-distribution tasks, environments, and embodiments, outperforming the state-of-the-art in-context learning approach using autoregressive vision-language models. 

**Abstract (ZH)**: 我们提出了一种测试时自适应方法，使进度估计模型能够通过优化一个学习到的自监督目标，在线适应测试轨迹的视觉和时间上下文。为此，我们引入了一种基于梯度的元学习策略，使模型能够在专家视觉轨迹及其自然语言任务描述上进行训练，从而测试时的自适应能够依靠语义内容而不是时间顺序来改善进度估计。我们的测试时自适应方法能够从单一训练环境泛化到多样化的未知分布任务、环境和实体，并且在使用自回归视觉语言模型的现有最佳上下文学习方法上表现出更高的性能。 

---
# FastFLUX: Pruning FLUX with Block-wise Replacement and Sandwich Training 

**Title (ZH)**: FastFLUX: 基于块级替换和三明治训练的FLUX剪枝 

**Authors**: Fuhan Cai, Yong Guo, Jie Li, Wenbo Li, Xiangzhong Fang, Jian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10035)  

**Abstract**: Recent advancements in text-to-image (T2I) generation have led to the emergence of highly expressive models such as diffusion transformers (DiTs), exemplified by FLUX. However, their massive parameter sizes lead to slow inference, high memory usage, and poor deployability. Existing acceleration methods (e.g., single-step distillation and attention pruning) often suffer from significant performance degradation and incur substantial training costs. To address these limitations, we propose FastFLUX, an architecture-level pruning framework designed to enhance the inference efficiency of FLUX. At its core is the Block-wise Replacement with Linear Layers (BRLL) method, which replaces structurally complex residual branches in ResBlocks with lightweight linear layers while preserving the original shortcut connections for stability. Furthermore, we introduce Sandwich Training (ST), a localized fine-tuning strategy that leverages LoRA to supervise neighboring blocks, mitigating performance drops caused by structural replacement. Experiments show that our FastFLUX maintains high image quality under both qualitative and quantitative evaluations, while significantly improving inference speed, even with 20\% of the hierarchy pruned. Our code will be available soon. 

**Abstract (ZH)**: Recent Advancements in Text-to-Image Generation Have Led to Highly Expressive Models Such as Diffusion Transformers (DiTs), Exemplified by FLUX. However, Their Massive Parameter Sizes Lead to Slow Inference, High Memory Usage, and Poor Deployability. To Address These Limitations, We Propose FastFLUX, an Architecture-Level Pruning Framework Designed to Enhance the Inference Efficiency of FLUX. 

---
# A Survey of Automatic Evaluation Methods on Text, Visual and Speech Generations 

**Title (ZH)**: 自动评价方法综述：文本、视觉和语音生成 

**Authors**: Tian Lan, Yang-Hao Zhou, Zi-Ao Ma, Fanshu Sun, Rui-Qing Sun, Junyu Luo, Rong-Cheng Tu, Heyan Huang, Chen Xu, Zhijing Wu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10019)  

**Abstract**: Recent advances in deep learning have significantly enhanced generative AI capabilities across text, images, and audio. However, automatically evaluating the quality of these generated outputs presents ongoing challenges. Although numerous automatic evaluation methods exist, current research lacks a systematic framework that comprehensively organizes these methods across text, visual, and audio modalities. To address this issue, we present a comprehensive review and a unified taxonomy of automatic evaluation methods for generated content across all three modalities; We identify five fundamental paradigms that characterize existing evaluation approaches across these domains. Our analysis begins by examining evaluation methods for text generation, where techniques are most mature. We then extend this framework to image and audio generation, demonstrating its broad applicability. Finally, we discuss promising directions for future research in cross-modal evaluation methodologies. 

**Abstract (ZH)**: 近期深度学习的进展极大地增强了生成型AI在文本、图像和音频方面的能力。然而，自动评估这些生成输出的质量仍然面临挑战。尽管存在大量的自动评估方法，但当前研究缺乏一个系统框架，能够全面组织这些方法，涵盖文本、视觉和音频模态。为解决这一问题，我们提出了一项全面的综述和一种统一的评估方法分类体系，涵盖了所有三个模态；我们确定了五个基本范式，以描述这些领域中现有评估方法的特征。我们的分析始于评估文本生成的方法，这些技术最为成熟。然后将这一框架扩展到图像和音频生成，展示了其广泛的适用性。最后，我们讨论了跨模态评估方法未来研究的有前途的方向。 

---
# Multimodal Large Language Models: A Survey 

**Title (ZH)**: 多模态大型语言模型：一个综述 

**Authors**: Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10016)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved beyond text generation, now spanning diverse output modalities including images, music, video, human motion, and 3D objects, by integrating language with other sensory modalities under unified architectures. This survey categorises six primary generative modalities and examines how foundational techniques, namely Self-Supervised Learning (SSL), Mixture of Experts (MoE), Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) prompting, enable cross-modal capabilities. We analyze key models, architectural trends, and emergent cross-modal synergies, while highlighting transferable techniques and unresolved challenges. Architectural innovations like transformers and diffusion models underpin this convergence, enabling cross-modal transfer and modular specialization. We highlight emerging patterns of synergy, and identify open challenges in evaluation, modularity, and structured reasoning. This survey offers a unified perspective on MLLM development and identifies critical paths toward more general-purpose, adaptive, and interpretable multimodal systems. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已经超越了文本生成，现在涵盖了包括图像、音乐、视频、人体运动和3D对象在内的多种输出模态，通过在统一架构中整合语言与其他感官模态。本文综述分类了六种主要生成模态，并探讨了自监督学习（SSL）、专家混合（MoE）、基于人类反馈的强化学习（RLHF）和思维链（CoT）提示等基础技术如何实现跨模态能力。我们分析了关键模型、架构趋势和新兴的跨模态协同效应，并强调了可转移技术以及未解决的挑战。架构创新如变换器和扩散模型支撑了这一融合过程，使得跨模态转移和模块化专业化成为可能。我们突显了正在形成的协同模式，并指出了评估、模块化和结构化推理等方面的开放挑战。本文综述提供了一种统一的视角，对于开发更通用、自适应和可解释的多模态系统指明了关键路径。 

---
# Immersive Multimedia Communication: State-of-the-Art on eXtended Reality Streaming 

**Title (ZH)**: 沉浸式多媒体通信：扩展现实流媒体的现状 

**Authors**: Haopeng Wang, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2506.10004)  

**Abstract**: Extended reality (XR) is rapidly advancing, and poised to revolutionize content creation and consumption. In XR, users integrate various sensory inputs to form a cohesive perception of the virtual environment. This survey reviews the state-of-the-art in XR streaming, focusing on multiple paradigms. To begin, we define XR and introduce various XR headsets along with their multimodal interaction methods to provide a foundational understanding. We then analyze XR traffic characteristics to highlight the unique data transmission requirements. We also explore factors that influence the quality of experience in XR systems, aiming to identify key elements for enhancing user satisfaction. Following this, we present visual attention-based optimization methods for XR streaming to improve efficiency and performance. Finally, we examine current applications and highlight challenges to provide insights into ongoing and future developments of XR. 

**Abstract (ZH)**: 扩展现实（XR）正在rapidly advancing，并有望革命化内容创作与消费。本综述回顾了XR流传输的前沿技术，重点关注多种 paradigms。首先，我们定义XR并介绍各种XR头显及其多模态交互方法，以提供基础知识。接着，我们分析XR网络流量特性，突显其独特的数据传输需求。我们还探讨影响XR系统体验质量的因素，以识别提升用户满意度的关键要素。随后，我们呈现基于视觉注意力的优化方法以提高XR流传输的效率和性能。最后，我们考察当前应用并突出挑战，以提供有关XR持续和未来发展洞察。 

---
# Semantic Communication-Enabled Cloud-Edge-End-collaborative Metaverse Services Architecure 

**Title (ZH)**: 基于语义通信的云-边-端协作元宇宙服务架构 

**Authors**: Yuxuan Li, Sheng Jinag, Bizhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10001)  

**Abstract**: With technology advancing and the pursuit of new audiovisual experiences strengthening, the metaverse has gained surging enthusiasm. However, it faces practical hurdles as substantial data like high-resolution virtual scenes must be transmitted between cloud platforms and VR devices. Specifically, the VR device's wireless transmission hampered by insufficient bandwidth, causes speed and delay problems. Meanwhile, poor channel quality leads to data errors and worsens user experience. To solve this, we've proposed the Semantic Communication-Enabled Cloud-Edge-End Collaborative Immersive Metaverse Service (SC-CEE-Meta) Architecture, which includes three modules: VR video semantic transmission, video synthesis, and 3D virtual scene reconstruction. By deploying semantic modules on VR devices and edge servers and sending key semantic info instead of focusing on bit-level reconstruction, it can cut latency, resolve the resource-bandwidth conflict, and better withstand channel interference. Also, the cloud deploys video synthesis and 3D scene reconstruction preprocessing, while edge devices host 3D reconstruction rendering modules, all for immersive services. Verified on Meta Quest Pro, the SC-CEE-Meta can reduce wireless transmission delay by 96.05\% and boost image quality by 43.99\% under poor channel condition. 

**Abstract (ZH)**: 随着技术的进步和对新视听体验的追求加强，元宇宙获得了高涨的热情。然而，它面临着实际挑战，因为大量高分辨率虚拟场景数据必须在云平台和VR设备之间传输。具体而言，由于无线传输受限于不足的带宽，导致速度和延迟问题。同时，差的信道质量引起数据错误并恶化用户体验。为了解决这些问题，我们提出了基于语义通信的云-边缘-端协作沉浸式元宇宙服务（SC-CEE-Meta）架构，该架构包括三个模块：VR视频语义传输、视频合成和三维虚拟场景重建。通过在VR设备和边缘服务器上部署语义模块，并发送关键语义信息而非注重位级重构，它可降低延迟、解决资源-带宽冲突，并更能承受信道干扰。此外，云部署视频合成和三维场景重建预处理，而边缘设备承载三维重建渲染模块，以提供沉浸式服务。在Meta Quest Pro上验证，SC-CEE-Meta在差的信道条件下可将无线传输延迟降低96.05%并提升图像质量43.99%。 

---
# Resa: Transparent Reasoning Models via SAEs 

**Title (ZH)**: Resa: 通过SAEs实现透明推理模型 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Deqing Fu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.09967)  

**Abstract**: How cost-effectively can we elicit strong reasoning in language models by leveraging their underlying representations? We answer this question with Resa, a family of 1.5B reasoning models trained via a novel and efficient sparse autoencoder tuning (SAE-Tuning) procedure. This method first trains an SAE to capture reasoning abilities from a source model, and then uses the trained SAE to guide a standard supervised fine-tuning process to elicit such abilities in a target model, all using verified question-answer data without any reasoning traces. Notably, when applied to certain base models before further RL post-training, SAE-Tuning retains >97% of its RL-trained counterpart's reasoning performance while reducing training costs by >2000x to roughly \$1 and training time by >450x to around 20 minutes. Furthermore, when applied to lightly RL-trained models (e.g., within 1 hour on 2 GPUs), it enables reasoning performance such as 43.33% Pass@1 on AIME24 and 90% Pass@1 on AMC23 for only around \$1 additional cost. Surprisingly, the reasoning abilities extracted via SAEs are potentially both generalizable and modular. Generality means abilities extracted from one dataset still elevate performance on a larger and overlapping corpus. Modularity means abilities extracted from Qwen or Qwen-Math can be attached to the R1-Distill model at test time, without any retraining, and yield comparable gains. Extensive ablations validate these findings and all artifacts are fully open-sourced. 

**Abstract (ZH)**: 通过利用语言模型的底层表示，我们如何最有效地激发强推理能力？Resa：一种新型高效的稀疏自编码器调优（SAE-Tuning）方法的研究 

---
