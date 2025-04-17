# Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports 

**Title (ZH)**: 安全与自主：以人为本的安全过滤器及其在智能辅助赛车运动中的应用 

**Authors**: Donggeon David Oh, Justin Lidard, Haimin Hu, Himani Sinhmar, Elle Lazarski, Deepak Gopinath, Emily S. Sumner, Jonathan A. DeCastro, Guy Rosman, Naomi Ehrich Leonard, Jaime Fernández Fisac  

**Link**: [PDF](https://arxiv.org/pdf/2504.11717)  

**Abstract**: We propose a human-centered safety filter (HCSF) for shared autonomy that significantly enhances system safety without compromising human agency. Our HCSF is built on a neural safety value function, which we first learn scalably through black-box interactions and then use at deployment to enforce a novel quality control barrier function (Q-CBF) safety constraint. Since this Q-CBF safety filter does not require any knowledge of the system dynamics for both synthesis and runtime safety monitoring and intervention, our method applies readily to complex, black-box shared autonomy systems. Notably, our HCSF's CBF-based interventions modify the human's actions minimally and smoothly, avoiding the abrupt, last-moment corrections delivered by many conventional safety filters. We validate our approach in a comprehensive in-person user study using Assetto Corsa-a high-fidelity car racing simulator with black-box dynamics-to assess robustness in "driving on the edge" scenarios. We compare both trajectory data and drivers' perceptions of our HCSF assistance against unassisted driving and a conventional safety filter. Experimental results show that 1) compared to having no assistance, our HCSF improves both safety and user satisfaction without compromising human agency or comfort, and 2) relative to a conventional safety filter, our proposed HCSF boosts human agency, comfort, and satisfaction while maintaining robustness. 

**Abstract (ZH)**: 基于人类中心的安全过滤器：提升共享自主性系统的安全性和用户体验 

---
# Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial Simultaneous Localization and Mapping 

**Title (ZH)**: 多普勒-SLAM: 多普勒辅助雷达-惯性与LiDAR-惯性同时定位与建图 

**Authors**: Dong Wang, Hannes Haag, Daniel Casado Herraez, Stefan May, Cyrill Stachniss, Andreas Nuechter  

**Link**: [PDF](https://arxiv.org/pdf/2504.11634)  

**Abstract**: Simultaneous localization and mapping (SLAM) is a critical capability for autonomous systems. Traditional SLAM approaches, which often rely on visual or LiDAR sensors, face significant challenges in adverse conditions such as low light or featureless environments. To overcome these limitations, we propose a novel Doppler-aided radar-inertial and LiDAR-inertial SLAM framework that leverages the complementary strengths of 4D radar, FMCW LiDAR, and inertial measurement units. Our system integrates Doppler velocity measurements and spatial data into a tightly-coupled front-end and graph optimization back-end to provide enhanced ego velocity estimation, accurate odometry, and robust mapping. We also introduce a Doppler-based scan-matching technique to improve front-end odometry in dynamic environments. In addition, our framework incorporates an innovative online extrinsic calibration mechanism, utilizing Doppler velocity and loop closure to dynamically maintain sensor alignment. Extensive evaluations on both public and proprietary datasets show that our system significantly outperforms state-of-the-art radar-SLAM and LiDAR-SLAM frameworks in terms of accuracy and robustness. To encourage further research, the code of our Doppler-SLAM and our dataset are available at: this https URL. 

**Abstract (ZH)**: 基于多传感器融合的 Doppler 辅助雷达-惯导和 LiDAR-惯导 SLAM 框架 

---
# Exploring Video-Based Driver Activity Recognition under Noisy Labels 

**Title (ZH)**: 基于嘈杂标签的视频驱动活动识别探究 

**Authors**: Linjuan Fan, Di Wen, Kunyu Peng, Kailun Yang, Jiaming Zhang, Ruiping Liu, Yufan Chen, Junwei Zheng, Jiamin Wu, Xudong Han, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11966)  

**Abstract**: As an open research topic in the field of deep learning, learning with noisy labels has attracted much attention and grown rapidly over the past ten years. Learning with label noise is crucial for driver distraction behavior recognition, as real-world video data often contains mislabeled samples, impacting model reliability and performance. However, label noise learning is barely explored in the driver activity recognition field. In this paper, we propose the first label noise learning approach for the driver activity recognition task. Based on the cluster assumption, we initially enable the model to learn clustering-friendly low-dimensional representations from given videos and assign the resultant embeddings into clusters. We subsequently perform co-refinement within each cluster to smooth the classifier outputs. Furthermore, we propose a flexible sample selection strategy that combines two selection criteria without relying on any hyperparameters to filter clean samples from the training dataset. We also incorporate a self-adaptive parameter into the sample selection process to enforce balancing across classes. A comprehensive variety of experiments on the public Drive&Act dataset for all granularity levels demonstrates the superior performance of our method in comparison with other label-denoising methods derived from the image classification field. The source code is available at this https URL. 

**Abstract (ZH)**: 深度学习领域中带有噪声标签的学习：一种驾驶活动识别中的噪声标签学习方法 

---
# Linearity, Time Invariance, and Passivity of a Novice Person in Human Teleoperation 

**Title (ZH)**: 新手在人体遥控操作中的线性、时不变性和有界输入有界输出性 

**Authors**: David Black, Septimiu Salcudean  

**Link**: [PDF](https://arxiv.org/pdf/2504.11653)  

**Abstract**: Low-cost teleguidance of medical procedures is becoming essential to provide healthcare to remote and underserved communities. Human teleoperation is a promising new method for guiding a novice person with relatively high precision and efficiency through a mixed reality (MR) interface. Prior work has shown that the novice, or "follower", can reliably track the MR input with performance not unlike a telerobotic system. As a consequence, it is of interest to understand and control the follower's dynamics to optimize the system performance and permit stable and transparent bilateral teleoperation. To this end, linearity, time-invariance, inter-axis coupling, and passivity are important in teleoperation and controller design. This paper therefore explores these effects with regard to the follower person in human teleoperation. It is demonstrated through modeling and experiments that the follower can indeed be treated as approximately linear and time invariant, with little coupling and a large excess of passivity at practical frequencies. Furthermore, a stochastic model of the follower dynamics is derived. These results will permit controller design and analysis to improve the performance of human teleoperation. 

**Abstract (ZH)**: 低成本远程指导在医疗程序中的应用对于提供偏远和未充分服务社区的医疗服务变得至关重要。人类远程操作是一种有前景的新方法，通过混合现实接口，能够相对高精度和高效地引导新手人员。先前的研究表明，新手，即“追随者”，能够可靠地跟踪混合现实输入，其性能类似于遥控机器人系统。因此，了解和控制“追随者”的动力学对于优化系统性能、实现稳定透明的双边远程操作是重要的。本论文因此探讨了这些效应对人类远程操作中“追随者”人员的影响。通过建模和实验表明，“追随者”确实可以近似为线性且时间不变，具有少量耦合和在实用频率下存在大量过剩的耗散性。此外，还推导出“追随者”动力学的随机模型。这些结果将有助于控制器的设计与分析，以提高人类远程操作的性能。 

---
# LANGTRAJ: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation 

**Title (ZH)**: LANGTRAJ：基于语言条件的轨迹模拟扩散模型及数据集 

**Authors**: Wei-Jer Chang, Wei Zhan, Masayoshi Tomizuka, Manmohan Chandraker, Francesco Pittaluga  

**Link**: [PDF](https://arxiv.org/pdf/2504.11521)  

**Abstract**: Evaluating autonomous vehicles with controllability enables scalable testing in counterfactual or structured settings, enhancing both efficiency and safety. We introduce LangTraj, a language-conditioned scene-diffusion model that simulates the joint behavior of all agents in traffic scenarios. By conditioning on natural language inputs, LangTraj provides flexible and intuitive control over interactive behaviors, generating nuanced and realistic scenarios. Unlike prior approaches that depend on domain-specific guidance functions, LangTraj incorporates language conditioning during training, facilitating more intuitive traffic simulation control. We propose a novel closed-loop training strategy for diffusion models, explicitly tailored to enhance stability and realism during closed-loop simulation. To support language-conditioned simulation, we develop Inter-Drive, a large-scale dataset with diverse and interactive labels for training language-conditioned diffusion models. Our dataset is built upon a scalable pipeline for annotating agent-agent interactions and single-agent behaviors, ensuring rich and varied supervision. Validated on the Waymo Motion Dataset, LangTraj demonstrates strong performance in realism, language controllability, and language-conditioned safety-critical simulation, establishing a new paradigm for flexible and scalable autonomous vehicle testing. 

**Abstract (ZH)**: 基于可控性的自主车辆评估方法：在假设或结构化场景中实现可扩展测试，提高效率和安全性。我们引入了LangTraj，一种基于语言的场景扩散模型，用于模拟交通场景中所有代理的联合行为。通过基于自然语言输入，LangTraj提供了灵活且直观的交互行为控制，生成细腻且逼真的场景。与依赖特定领域指导函数的前ethod不同，LangTraj在训练过程中引入了语言条件，使交通模拟控制更加直观。我们提出了一种新的闭环训练策略，专门用于增强闭环仿真过程中的稳定性和真实性。为了支持基于语言的仿真，我们开发了Inter-Drive，一个包含多种互动标签的大型数据集，用于训练基于语言的扩散模型。该数据集基于可扩展的注释管道，确保丰富的监督信息。在Waymo Motion数据集上的验证表明，LangTraj在真实性、语言可控性和语言条件下的关键安全仿真方面表现出强劲性能，建立了灵活和可扩展的自主车辆测试的新范式。 

---
# Advancing Arabic Speech Recognition Through Large-Scale Weakly Supervised Learning 

**Title (ZH)**: 通过大规模弱监督学习推动阿拉伯语音识别进步 

**Authors**: Mahmoud Salhab, Marwan Elghitany, Shameed Sait, Syed Sibghat Ullah, Mohammad Abusheikh, Hasan Abusheikh  

**Link**: [PDF](https://arxiv.org/pdf/2504.12254)  

**Abstract**: Automatic speech recognition (ASR) is crucial for human-machine interaction in diverse applications like conversational agents, industrial robotics, call center automation, and automated subtitling. However, developing high-performance ASR models remains challenging, particularly for low-resource languages like Arabic, due to the scarcity of large, labeled speech datasets, which are costly and labor-intensive to produce. In this work, we employ weakly supervised learning to train an Arabic ASR model using the Conformer architecture. Our model is trained from scratch on 15,000 hours of weakly annotated speech data covering both Modern Standard Arabic (MSA) and Dialectal Arabic (DA), eliminating the need for costly manual transcriptions. Despite the absence of human-verified labels, our approach attains state-of-the-art (SOTA) performance, exceeding all previous efforts in the field of Arabic ASR on the standard benchmarks. By demonstrating the effectiveness of weak supervision as a scalable, cost-efficient alternative to traditional supervised approaches, paving the way for improved ASR systems in low resource settings. 

**Abstract (ZH)**: 自动语音识别（ASR）在对话代理、工业机器人、呼叫中心自动化和自动字幕后等多样应用中的交互至关重要。然而，开发高性能的ASR模型对于低资源语言如阿拉伯语来说仍然具有挑战性，特别是由于缺乏大规模、标注的数据集，这些数据集的获取既昂贵又劳动密集。在本工作中，我们运用弱监督学习训练一种基于Conformer架构的阿拉伯语ASR模型，该模型完全从15,000小时的弱标注语音数据中进行训练，涵盖现代标准阿拉伯语（MSA）和方言阿拉伯语（DA），从而避免了昂贵的手工转录。尽管缺乏人工验证的标签，我们的方法在阿拉伯语ASR领域达到了最先进的（SOTA）性能，超过了所有以往的努力，特别是在标准基准上的表现。通过证明弱监督作为传统监督方法的一种可扩展且经济有效的替代方案的有效性，为低资源环境下的改进ASR系统铺平了道路。 

---
# Leveraging Machine Learning Models to Predict the Outcome of Digital Medical Triage Interviews 

**Title (ZH)**: 利用机器学习模型预测数字医疗分流面试的结果 

**Authors**: Sofia Krylova, Fabian Schmidt, Vladimir Vlassov  

**Link**: [PDF](https://arxiv.org/pdf/2504.11977)  

**Abstract**: Many existing digital triage systems are questionnaire-based, guiding patients to appropriate care levels based on information (e.g., symptoms, medical history, and urgency) provided by the patients answering questionnaires. Such a system often uses a deterministic model with predefined rules to determine care levels. It faces challenges with incomplete triage interviews since it can only assist patients who finish the process. In this study, we explore the use of machine learning (ML) to predict outcomes of unfinished interviews, aiming to enhance patient care and service quality. Predicting triage outcomes from incomplete data is crucial for patient safety and healthcare efficiency. Our findings show that decision-tree models, particularly LGBMClassifier and CatBoostClassifier, achieve over 80\% accuracy in predicting outcomes from complete interviews while having a linear correlation between the prediction accuracy and interview completeness degree. For example, LGBMClassifier achieves 88,2\% prediction accuracy for interviews with 100\% completeness, 79,6\% accuracy for interviews with 80\% completeness, 58,9\% accuracy for 60\% completeness, and 45,7\% accuracy for 40\% completeness. The TabTransformer model demonstrated exceptional accuracy of over 80\% for all degrees of completeness but required extensive training time, indicating a need for more powerful computational resources. The study highlights the linear correlation between interview completeness and predictive power of the decision-tree models. 

**Abstract (ZH)**: 基于机器学习的不完整转诊访谈预测研究 

---
# ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation 

**Title (ZH)**: ADAT：时间序列意识的自适应变换器架构的手语翻译 

**Authors**: Nada Shahin, Leila Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2504.11942)  

**Abstract**: Current sign language machine translation systems rely on recognizing hand movements, facial expressions and body postures, and natural language processing, to convert signs into text. Recent approaches use Transformer architectures to model long-range dependencies via positional encoding. However, they lack accuracy in recognizing fine-grained, short-range temporal dependencies between gestures captured at high frame rates. Moreover, their high computational complexity leads to inefficient training. To mitigate these issues, we propose an Adaptive Transformer (ADAT), which incorporates components for enhanced feature extraction and adaptive feature weighting through a gating mechanism to emphasize contextually relevant features while reducing training overhead and maintaining translation accuracy. To evaluate ADAT, we introduce MedASL, the first public medical American Sign Language dataset. In sign-to-gloss-to-text experiments, ADAT outperforms the encoder-decoder transformer, improving BLEU-4 accuracy by 0.1% while reducing training time by 14.33% on PHOENIX14T and 3.24% on MedASL. In sign-to-text experiments, it improves accuracy by 8.7% and reduces training time by 2.8% on PHOENIX14T and achieves 4.7% higher accuracy and 7.17% faster training on MedASL. Compared to encoder-only and decoder-only baselines in sign-to-text, ADAT is at least 6.8% more accurate despite being up to 12.1% slower due to its dual-stream structure. 

**Abstract (ZH)**: 当前的手语机器翻译系统依赖于识别手势、面部表情和身体姿态，并结合自然语言处理技术，将手语转换为文本。最近的方法使用Transformer架构通过位置编码建模长距离依赖关系，但它们在识别高帧率捕获的手势之间细微的、短距离的时序依赖关系方面缺乏准确性。此外，它们的高计算复杂性导致训练效率低下。为了缓解这些问题，我们提出了一种自适应Transformer（ADAT），它结合了增强特征提取和通过门控机制实现的自适应特征加权组件，强调上下文相关特征，同时减少训练开销并保持翻译准确性。为了评估ADAT，我们引入了MedASL，这是首个公开的医疗美式手语数据集。在手语到图解再到文本的实验中，ADAT在PHOENIX14T上的BLEU-4准确性提高了0.1%，训练时间减少了14.33%，在MedASL上的训练时间减少了3.24%。在手语到文本的实验中，ADAT在PHOENIX14T上的准确性提高了8.7%，训练时间减少了2.8%，在MedASL上的准确性提高了4.7%，训练速度加快了7.17%。与手语到文本中的编码器和解码器基线相比，尽管由于其双流结构导致速度慢了12.1%，但ADAT至少在准确性上提高了6.8%。 

---
# Seeking and leveraging alternative variable dependency concepts in gray-box-elusive bimodal land-use allocation problems 

**Title (ZH)**: 在灰箱逃逸的双模土地利用分配问题中寻求和利用替代变量依赖概念 

**Authors**: J. Maciążek, M. W. Przewozniczek, J. Schwaab  

**Link**: [PDF](https://arxiv.org/pdf/2504.11882)  

**Abstract**: Solving land-use allocation problems can help us to deal with some of the most urgent global environmental issues. Since these problems are NP-hard, effective optimizers are needed to handle them. The knowledge about variable dependencies allows for proposing such tools. However, in this work, we consider a real-world multi-objective problem for which standard variable dependency discovery techniques are inapplicable. Therefore, using linkage-based variation operators is unreachable. To address this issue, we propose a definition of problem-dedicated variable dependency. On this base, we propose obtaining masks of dependent variables. Using them, we construct three novel crossover operators. The results concerning real-world test cases show that introducing our propositions into two well-known optimizers (NSGA-II, MOEA/D) dedicated to multi-objective optimization significantly improves their effectiveness. 

**Abstract (ZH)**: 解决土地利用分配问题有助于应对一些最紧迫的全球环境问题。由于这些问题属于NP-hard问题，需要有效的优化器来处理它们。变量依赖性的知识有助于提出这样的工具。然而，在这项工作中，我们考虑了一个现实世界中的多目标问题，对于该问题，标准的变量依赖性发现技术是不适用的。因此，基于链接的变异操作是不可行的。为解决这一问题，我们提出了一种针对特定问题的变量依赖性的定义，并在此基础上提出了获取依赖变量掩码的方法。利用这些掩码，我们构建了三种新型交叉操作。关于实际案例的结果表明，将我们的提议引入两个著名的多目标优化器（NSGA-II，MOEA/D）中，显著提高了它们的有效性。 

---
# Moving between high-quality optima using multi-satisfiability characteristics in hard-to-solve Max3Sat instances 

**Title (ZH)**: 在难解的Max3Sat实例中利用多重满足性特性在高质量最优解之间移动 

**Authors**: J. Piatek, M. W. Przewozniczek, F. Chicano, R. Tinós  

**Link**: [PDF](https://arxiv.org/pdf/2504.11864)  

**Abstract**: Gray-box optimization proposes effective and efficient optimizers of general use. To this end, it leverages information about variable dependencies and the subfunction-based problem representation. These approaches were already shown effective by enabling \textit{tunnelling} between local optima even if these moves require the modification of many dependent variables. Tunnelling is useful in solving the maximum satisfiability problem (MaxSat), which can be reformulated to Max3Sat. Since many real-world problems can be brought to solving the MaxSat/Max3Sat instances, it is important to solve them effectively and efficiently. Therefore, we focus on Max3Sat instances for which tunnelling fails to introduce improving moves between locally optimal high-quality solutions and the region of globally optimal solutions. We analyze the features of such instances on the ground of phase transitions. Based on these observations, we propose manipulating clause-satisfiability characteristics that allow connecting high-quality solutions distant in the solution space. We utilize multi-satisfiability characteristics in the optimizer built from typical gray-box mechanisms. The experimental study shows that the proposed optimizer can solve those Max3Sat instances that are out of the grasp of state-of-the-art gray-box optimizers. At the same time, it remains effective for instances that have already been successfully solved by gray-box. 

**Abstract (ZH)**: 灰箱优化提出了一类通用且有效的优化器。通过利用变量间依赖关系以及基于子函数的问题表示，这些方法已被证明即使涉及大量依赖变量的修改也能有效实现从局部最优解到高质量全局最优解区域的连接（tunnelling）。鉴于许多实际问题可以归约为求解Max3Sat实例，实现这些实例的有效且高效的求解显得尤为重要。因此，我们关注tunnelling无法在高质局部最优解与全局最优解区域之间引入改进策略的Max3Sat实例。基于相变现象分析这些实例的特性，并提出利用子句满足性特征以连接在解空间中相距较远的高质解。我们在基于典型灰箱机制构建的优化器中利用多满足性特征。实验研究显示，所提出的方法能够解决当前最先进灰箱优化器难以解决的Max3Sat实例，同时对于已由灰箱优化器成功解决的实例依然保持有效。 

---
# Probabilistic causal graphs as categorical data synthesizers: Do they do better than Gaussian Copulas and Conditional Tabular GANs? 

**Title (ZH)**: 基于概率因果图的类别数据合成器：它们比高斯copula和条件表型GAN更优吗？ 

**Authors**: Olha Shaposhnyk, Noor Abid, Mouri Zakir, Svetlana Yanushkevich  

**Link**: [PDF](https://arxiv.org/pdf/2504.11547)  

**Abstract**: This study investigates the generation of high-quality synthetic categorical data, such as survey data, using causal graph models. Generating synthetic data aims not only to create a variety of data for training the models but also to preserve privacy while capturing relationships between the data. The research employs Structural Equation Modeling (SEM) followed by Bayesian Networks (BN). We used the categorical data that are based on the survey of accessibility to services for people with disabilities. We created both SEM and BN models to represent causal relationships and to capture joint distributions between variables. In our case studies, such variables include, in particular, demographics, types of disability, types of accessibility barriers and frequencies of encountering those barriers.
The study compared the SEM-based BN method with alternative approaches, including the probabilistic Gaussian copula technique and generative models like the Conditional Tabular Generative Adversarial Network (CTGAN). The proposed method outperformed others in statistical metrics, including the Chi-square test, Kullback-Leibler divergence, and Total Variation Distance (TVD). In particular, the BN model demonstrated superior performance, achieving the highest TVD, indicating alignment with the original data. The Gaussian Copula ranked second, while CTGAN exhibited moderate performance. These analyses confirmed the ability of the SEM-based BN to produce synthetic data that maintain statistical and relational validity while maintaining confidentiality. This approach is particularly beneficial for research on sensitive data, such as accessibility and disability studies. 

**Abstract (ZH)**: 本研究利用因果图模型探究高质合成分类数据（如调查数据）的生成方法。生成合成数据不仅旨在为模型训练提供多样化数据，而且还能在保护隐私的同时捕捉数据之间的关系。研究采用了结构方程建模（SEM）后接贝叶斯网络（BN）的方法。我们基于残疾人群体服务可及性的调查数据创建了SEM和BN模型，以表示因果关系并捕捉变量间的联合分布。在我们的案例研究中，变量包括但不限于人口统计学特征、残疾类型、可达性障碍类型及其遇到的频率。本研究将基于SEM的BN方法与其他方法，如概率高斯 copula 技术和生成模型（如条件表生成对抗网络CTGAN）进行了比较。基于统计指标（包括卡方检验、KL散度和Total Variation Distance），提出的模型表现优于其他方法。特别是贝叶斯网络模型表现出 superiority，取得最高的TVD值，表明其与原始数据的吻合度较高。高斯 copula 排名第二，CTGAN 表现一般。这些分析证实了基于SEM的BN方法能够生成既符合统计性和关系性要求又能保护隐私的合成数据。该方法特别适用于敏感数据的研究，如可达性和残疾研究。 

---
# REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites 

**Title (ZH)**: REAL：在确定性模拟真实网站环境下的自主代理基准测试 

**Authors**: Divyansh Garg, Shaun VanWeelden, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Tomas Abraham, Michael Lara, Federico Lopez, James Liu, Atharva Gundawar, Prannay Hebbar, Youngchul Joo, Charles London, Christian Schroeder de Witt, Sumeet Motwani  

**Link**: [PDF](https://arxiv.org/pdf/2504.11543)  

**Abstract**: We introduce REAL, a benchmark and framework for multi-turn agent evaluations on deterministic simulations of real-world websites. REAL comprises high-fidelity, deterministic replicas of 11 widely-used websites across domains such as e-commerce, travel, communication, and professional networking. We also release a benchmark consisting of 112 practical tasks that mirror everyday complex user interactions requiring both accurate information retrieval and state-changing actions. All interactions occur within this fully controlled setting, eliminating safety risks and enabling robust, reproducible evaluation of agent capability and reliability. Our novel evaluation framework combines programmatic checks of website state for action-based tasks with rubric-guided LLM-based judgments for information retrieval. The framework supports both open-source and proprietary agent systems through a flexible evaluation harness that accommodates black-box commands within browser environments, allowing research labs to test agentic systems without modification. Our empirical results show that frontier language models achieve at most a 41% success rate on REAL, highlighting critical gaps in autonomous web navigation and task completion capabilities. Our framework supports easy integration of new tasks, reproducible evaluation, and scalable data generation for training web agents. The websites, framework, and leaderboard are available at this https URL and this https URL. 

**Abstract (ZH)**: REAL：一个用于实时网站确定性模拟多轮对话代理评估的基准和框架 

---
# SCENT: Robust Spatiotemporal Learning for Continuous Scientific Data via Scalable Conditioned Neural Fields 

**Title (ZH)**: SCENT: 基于可扩展条件神经场的稳健时空学习用于连续科学数据 

**Authors**: David Keetae Park, Xihaier Luo, Guang Zhao, Seungjun Lee, Miruna Oprescu, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.12262)  

**Abstract**: Spatiotemporal learning is challenging due to the intricate interplay between spatial and temporal dependencies, the high dimensionality of the data, and scalability constraints. These challenges are further amplified in scientific domains, where data is often irregularly distributed (e.g., missing values from sensor failures) and high-volume (e.g., high-fidelity simulations), posing additional computational and modeling difficulties. In this paper, we present SCENT, a novel framework for scalable and continuity-informed spatiotemporal representation learning. SCENT unifies interpolation, reconstruction, and forecasting within a single architecture. Built on a transformer-based encoder-processor-decoder backbone, SCENT introduces learnable queries to enhance generalization and a query-wise cross-attention mechanism to effectively capture multi-scale dependencies. To ensure scalability in both data size and model complexity, we incorporate a sparse attention mechanism, enabling flexible output representations and efficient evaluation at arbitrary resolutions. We validate SCENT through extensive simulations and real-world experiments, demonstrating state-of-the-art performance across multiple challenging tasks while achieving superior scalability. 

**Abstract (ZH)**: 时空学习由于空间依赖性和时间依赖性的复杂交互、数据的高维度以及可扩展性约束而具有挑战性。在科学领域中，这些挑战因数据分布不规则（例如，传感器故障导致的缺失值）和高数据量（例如，高保真仿真）而被进一步放大，带来了额外的计算和建模困难。在这篇论文中，我们提出了一种新颖的框架SCENT，用于可扩展且注重连续性的时空表示学习。SCENT 在单个架构中统一了插值、重建和预测。基于基于变压器的编码器-处理器-解码器架构，SCENT 引入了可学习的查询以增强泛化能力，并引入了查询权wise交叉注意力机制以有效捕获多尺度依赖性。为了确保在数据量和模型复杂性方面的可扩展性，我们结合了稀疏注意力机制，这使得输出表示具有灵活性，并能以任意分辨率进行高效评估。我们通过广泛的仿真和实际实验验证了SCENT，展示了其在多个挑战性任务上的顶级性能，同时实现了卓越的可扩展性。 

---
# Communication Optimization for Decentralized Learning atop Bandwidth-limited Edge Networks 

**Title (ZH)**: 带宽受限边缘网络中去中心化学习的通信优化 

**Authors**: Tingyang Sun, Tuan Nguyen, Ting He  

**Link**: [PDF](https://arxiv.org/pdf/2504.12210)  

**Abstract**: Decentralized federated learning (DFL) is a promising machine learning paradigm for bringing artificial intelligence (AI) capabilities to the network edge. Running DFL on top of edge networks, however, faces severe performance challenges due to the extensive parameter exchanges between agents. Most existing solutions for these challenges were based on simplistic communication models, which cannot capture the case of learning over a multi-hop bandwidth-limited network. In this work, we address this problem by jointly designing the communication scheme for the overlay network formed by the agents and the mixing matrix that controls the communication demands between the agents. By carefully analyzing the properties of our problem, we cast each design problem into a tractable optimization and develop an efficient algorithm with guaranteed performance. Our evaluations based on real topology and data show that the proposed algorithm can reduce the total training time by over $80\%$ compared to the baseline without sacrificing accuracy, while significantly improving the computational efficiency over the state of the art. 

**Abstract (ZH)**: 基于代理形成覆盖网络的联合设计和带宽受限多跳网络上的混合矩阵优化的去中心化联邦学习 

---
# Mapping Controversies Using Artificial Intelligence: An Analysis of the Hamas-Israel Conflict on YouTube 

**Title (ZH)**: 使用人工智能映射争议：巴勒斯坦哈马斯与以色列冲突在YouTube上的分析 

**Authors**: Victor Manuel Hernandez Lopez, Jaime E. Cuellar  

**Link**: [PDF](https://arxiv.org/pdf/2504.12177)  

**Abstract**: This article analyzes the Hamas-Israel controversy through 253,925 Spanish-language YouTube comments posted between October 2023 and January 2024, following the October 7 attack that escalated the conflict. Adopting an interdisciplinary approach, the study combines the analysis of controversies from Science and Technology Studies (STS) with advanced computational methodologies, specifically Natural Language Processing (NLP) using the BERT (Bidirectional Encoder Representations from Transformers) model. Using this approach, the comments were automatically classified into seven categories, reflecting pro-Palestinian, pro-Israeli, anti- Palestinian, anti-Israeli positions, among others. The results show a predominance of pro- Palestinian comments, although pro-Israeli and anti-Palestinian comments received more "likes." This study also applies the agenda-setting theory to demonstrate how media coverage significantly influences public perception, observing a notable shift in public opinion, transitioning from a pro- Palestinian stance to a more critical position towards Israel. This work highlights the importance of combining social science perspectives with technological tools in the analysis of controversies, presenting a methodological innovation by integrating computational analysis with critical social theories to address complex public opinion phenomena and media narratives. 

**Abstract (ZH)**: 本文通过分析2023年10月至2024年1月期间发布的253,925条西班牙语YouTube评论，探讨哈马斯与以色列的争议。采用跨学科的方法，研究结合了科技研究（STS）中的争议分析与先进的计算方法，特别是使用BERT模型的自然语言处理（NLP）。通过这种方法，评论自动被分类为七个类别，反映了亲巴勒斯坦、亲以色列、反巴勒斯坦、反以色列等立场。研究结果表明，亲巴勒斯坦的评论占主导地位，尽管亲以色列和反巴勒斯坦的评论获得了更多的“点赞”。本文还应用议程设置理论，展示媒体 coverage 如何显著影响公众认知，并观察到公众立场出现了从亲巴勒斯坦向更批评以色列立场的转变。本文强调结合社会科学研究视角与技术工具在争议分析中的重要性，并通过将计算分析与批判性社会理论相结合，提出了一种方法论创新，以应对复杂的公众意见现象和媒体叙事。 

---
# Poem Meter Classification of Recited Arabic Poetry: Integrating High-Resource Systems for a Low-Resource Task 

**Title (ZH)**: 阿拉伯口传诗歌的音节格律分类：集成高资源系统以完成低资源任务 

**Authors**: Maged S. Al-Shaibani, Zaid Alyafeai, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2504.12172)  

**Abstract**: Arabic poetry is an essential and integral part of Arabic language and culture. It has been used by the Arabs to spot lights on their major events such as depicting brutal battles and conflicts. They also used it, as in many other languages, for various purposes such as romance, pride, lamentation, etc. Arabic poetry has received major attention from linguistics over the decades. One of the main characteristics of Arabic poetry is its special rhythmic structure as opposed to prose. This structure is referred to as a meter. Meters, along with other poetic characteristics, are intensively studied in an Arabic linguistic field called "\textit{Aroud}". Identifying these meters for a verse is a lengthy and complicated process. It also requires technical knowledge in \textit{Aruod}. For recited poetry, it adds an extra layer of processing. Developing systems for automatic identification of poem meters for recited poems need large amounts of labelled data. In this study, we propose a state-of-the-art framework to identify the poem meters of recited Arabic poetry, where we integrate two separate high-resource systems to perform the low-resource task. To ensure generalization of our proposed architecture, we publish a benchmark for this task for future research. 

**Abstract (ZH)**: 阿拉伯诗歌是阿拉伯语言和文化的重要组成部分，被阿拉伯人用于描绘重要事件，如激烈的战斗和冲突。它还被用于各种目的，如浪漫、自豪、哀悼等。阿拉伯诗歌在几十年里受到了语言学界的广泛关注。阿拉伯诗歌的主要特征之一是其特殊的韵律结构，不同于散文。这种结构被称为“音节格律”。音节格律以及其他诗歌特征在被称为“Aroud”的阿拉伯语言学领域得到了深入研究。识别诗歌的音节格律是一个漫长且复杂的过程，需要具备“Aroud”技术知识。对于朗诵诗歌而言，这一过程还增加了额外的处理层。为了自动识别朗诵阿拉伯诗歌的音节格律，需要大量的标注数据。在本研究中，我们提出了一种最先进的框架来识别朗诵阿拉伯诗歌的音节格律，通过整合两个高资源系统来完成这一低资源任务。为了确保我们提出架构的通用性，我们为该任务发布了基准数据集，供未来研究使用。 

---
# ARCeR: an Agentic RAG for the Automated Definition of Cyber Ranges 

**Title (ZH)**: ARCeR：一种自主型RAG，用于自动化定义网络空间范围 

**Authors**: Matteo Lupinacci, Francesco Blefari, Francesco Romeo, Francesco Aurelio Pironti, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.12143)  

**Abstract**: The growing and evolving landscape of cybersecurity threats necessitates the development of supporting tools and platforms that allow for the creation of realistic IT environments operating within virtual, controlled settings as Cyber Ranges (CRs). CRs can be exploited for analyzing vulnerabilities and experimenting with the effectiveness of devised countermeasures, as well as serving as training environments for building cyber security skills and abilities for IT operators. This paper proposes ARCeR as an innovative solution for the automatic generation and deployment of CRs, starting from user-provided descriptions in a natural language. ARCeR relies on the Agentic RAG paradigm, which allows it to fully exploit state-of-art AI technologies. Experimental results show that ARCeR is able to successfully process prompts even in cases that LLMs or basic RAG systems are not able to cope with. Furthermore, ARCeR is able to target any CR framework provided that specific knowledge is made available to it. 

**Abstract (ZH)**: 不断演变的网络安全威胁 landscape 需要开发支持工具和平台，使能够在虚拟、受控环境中创建现实的 IT 环境，作为 Cyber Ranges (CRs)。CRs 可用于分析漏洞并实验所设计的应对措施的有效性，同时作为培养 IT 运维人员网络安全技能和能力的培训环境。本文提出 ARCeR 作为一种创新解决方案，可以从用户提供的自然语言描述自动生成和部署 CRs。ARCeR 依赖于 Agentic RAG 原理，使其能够充分利用最新的 AI 技术。实验结果表明，ARCeR 能够成功处理即使对于大语言模型 (LLMs) 或基本 RAG 系统也无法应对的提示。此外，只要向 ARCeR 提供特定知识，它就能够针对任何 CR 框架进行操作。 

---
# AttentionDrop: A Novel Regularization Method for Transformer Models 

**Title (ZH)**: AttentionDrop：一种新的Transformer模型正则化方法 

**Authors**: Mirza Samad Ahmed Baig, Syeda Anshrah Gillani, Abdul Akbar Khan, Shahid Munir Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.12088)  

**Abstract**: Transformer-based architectures achieve state-of-the-art performance across a wide range of tasks in natural language processing, computer vision, and speech. However, their immense capacity often leads to overfitting, especially when training data is limited or noisy. We propose AttentionDrop, a unified family of stochastic regularization techniques that operate directly on the self-attention distributions. We introduces three variants: 1. Hard Attention Masking: randomly zeroes out top-k attention logits per query to encourage diverse context utilization. 2. Blurred Attention Smoothing: applies a dynamic Gaussian convolution over attention logits to diffuse overly peaked distributions. 3. Consistency-Regularized AttentionDrop: enforces output stability under multiple independent AttentionDrop perturbations via a KL-based consistency loss. 

**Abstract (ZH)**: 基于Transformer的架构在自然语言处理、计算机视觉和语音处理等多种任务中实现了最先进的性能。然而，其巨大的容量往往会导致过拟合，特别是在训练数据有限或噪声较大的情况下。我们提出了AttentionDrop，这是一种直接作用于自我注意力分布的统一的随机正则化技术家族。我们介绍了三种变体：1. 硬注意力掩码：每次查询随机清零前k个注意力logits以鼓励多样化的上下文利用。2. 模糊注意力平滑：在注意力logits上应用动态高斯卷积以弥散过于尖峰的分布。3. 一致性正则化AttentionDrop：通过基于KL散度的一致性损失强制在多组独立的AttentionDrop扰动下输出的稳定性。 

---
# RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model 

**Title (ZH)**: RadMamba：基于雷达微多普勒导向的anoia状态空间模型的人体活动识别 

**Authors**: Yizhuo Wu, Francesco Fioranelli, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12039)  

**Abstract**: Radar-based HAR has emerged as a promising alternative to conventional monitoring approaches, such as wearable devices and camera-based systems, due to its unique privacy preservation and robustness advantages. However, existing solutions based on convolutional and recurrent neural networks, although effective, are computationally demanding during deployment. This limits their applicability in scenarios with constrained resources or those requiring multiple sensors. Advanced architectures, such as ViT and SSM architectures, offer improved modeling capabilities and have made efforts toward lightweight designs. However, their computational complexity remains relatively high. To leverage the strengths of transformer architectures while simultaneously enhancing accuracy and reducing computational complexity, this paper introduces RadMamba, a parameter-efficient, radar micro-Doppler-oriented Mamba SSM specifically tailored for radar-based HAR. Across three diverse datasets, RadMamba matches the top-performing previous model's 99.8% classification accuracy on Dataset DIAT with only 1/400 of its parameters and equals the leading models' 92.0% accuracy on Dataset CI4R with merely 1/10 of their parameters. In scenarios with continuous sequences of actions evaluated on Dataset UoG2020, RadMamba surpasses other models with significantly higher parameter counts by at least 3%, achieving this with only 6.7k parameters. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于雷达的HAR已成为一种有前景的替代传统监控方法（如可穿戴设备和基于摄像头的系统）的方案，得益于其独特的隐私保护和鲁棒性优势。然而，现有基于卷积神经网络和循环神经网络的解决方案虽然有效，但在部署时计算需求高，限制了其在资源受限场景或需要多传感器的场景中的应用。先进的架构，如ViT和SSM架构，提供了改进的建模能力，并朝着轻量级设计做出了努力。然而，它们的计算复杂度仍然相对较高。为同时利用变换器架构的优势并提高准确性和降低计算复杂度，本文提出了一种参数效率高、雷达微多普勒导向的Mamba SSM——RadMamba，专门适用于基于雷达的HAR。在三个不同的数据集中，RadMamba仅使用前一个最佳模型四百分之一的参数，实现了DIAT数据集99.8%的分类准确率；仅使用前一个最佳模型十分之一的参数，实现了CI4R数据集92.0%的准确率。在对UoG2020数据集中的连续动作序列进行评估时，RadMamba仅用6.7千参数就超过了其他具有更高参数计数的模型，准确率高出至少3%。我们的代码可在以下链接获取：this https URL。 

---
# Proof-Carrying Neuro-Symbolic Code 

**Title (ZH)**: 神经符号推理代码证明 

**Authors**: Ekaterina Komendantskaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.12031)  

**Abstract**: This invited paper introduces the concept of "proof-carrying neuro-symbolic code" and explains its meaning and value, from both the "neural" and the "symbolic" perspectives. The talk outlines the first successes and challenges that this new area of research faces. 

**Abstract (ZH)**: 应邀论文介绍了“证明承载神经符号代码”的概念，从“神经”和“符号”两个视角解释其含义和价值，并概述了这一新研究领域取得的初步成果和面临的影响与挑战。 

---
# Balancing Graph Embedding Smoothness in Self-Supervised Learning via Information-Theoretic Decomposition 

**Title (ZH)**: 基于信息论分解的自监督学习中图嵌入平滑性平衡 

**Authors**: Heesoo Jung, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.12011)  

**Abstract**: Self-supervised learning (SSL) in graphs has garnered significant attention, particularly in employing Graph Neural Networks (GNNs) with pretext tasks initially designed for other domains, such as contrastive learning and feature reconstruction. However, it remains uncertain whether these methods effectively reflect essential graph properties, precisely representation similarity with its neighbors. We observe that existing methods position opposite ends of a spectrum driven by the graph embedding smoothness, with each end corresponding to outperformance on specific downstream tasks. Decomposing the SSL objective into three terms via an information-theoretic framework with a neighbor representation variable reveals that this polarization stems from an imbalance among the terms, which existing methods may not effectively maintain. Further insights suggest that balancing between the extremes can lead to improved performance across a wider range of downstream tasks. A framework, BSG (Balancing Smoothness in Graph SSL), introduces novel loss functions designed to supplement the representation quality in graph-based SSL by balancing the derived three terms: neighbor loss, minimal loss, and divergence loss. We present a theoretical analysis of the effects of these loss functions, highlighting their significance from both the SSL and graph smoothness perspectives. Extensive experiments on multiple real-world datasets across node classification and link prediction consistently demonstrate that BSG achieves state-of-the-art performance, outperforming existing methods. Our implementation code is available at this https URL. 

**Abstract (ZH)**: 图上的自监督学习（SSL）引起了显著关注，特别是在使用初始设计用于其他领域的图神经网络（GNNs），如对比学习和特征重建的先验任务。然而，这些方法是否能有效地反映图的关键属性——与邻居之间的表示相似性——仍不确定。我们观察到，现有方法在网络嵌入平滑性驱动下处于一个光谱的两端，每端在特定下游任务上表现优异。通过信息论框架将SSL目标分解为三个通过邻居表示变量导出的项，表明这种极化源于这些项之间的不平衡，而现有方法可能无法有效维持这种平衡。进一步的洞察表明，在极端之间取得平衡可以提高更广泛范围的下游任务性能。我们提出了一种框架BSG（平衡图SSL中的平滑性），引入了新的损失函数，旨在通过平衡这三个项（邻居损失、最小损失和散度损失）来补充基于图的SSL的表示质量。我们从SSL和平滑性两个角度分析了这些损失函数的效果，展示了它们的重要性。在节点分类和链接预测等多个现实世界数据集上的广泛实验一致证明，BSG达到了最先进的性能，优于现有方法。我们的实现代码可在以下链接获取：this https URL。 

---
# A Computationally Efficient Algorithm for Infinite-Horizon Average-Reward Linear MDPs 

**Title (ZH)**: 无限 horizon 平均奖励线性MDP的计算高效算法 

**Authors**: Kihyuk Hong, Ambuj Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2504.11997)  

**Abstract**: We study reinforcement learning in infinite-horizon average-reward settings with linear MDPs. Previous work addresses this problem by approximating the average-reward setting by discounted setting and employing a value iteration-based algorithm that uses clipping to constrain the span of the value function for improved statistical efficiency. However, the clipping procedure requires computing the minimum of the value function over the entire state space, which is prohibitive since the state space in linear MDP setting can be large or even infinite. In this paper, we introduce a value iteration method with efficient clipping operation that only requires computing the minimum of value functions over the set of states visited by the algorithm. Our algorithm enjoys the same regret bound as the previous work while being computationally efficient, with computational complexity that is independent of the size of the state space. 

**Abstract (ZH)**: 我们在无限_horizon 平均回报环境中研究线性MDP的强化学习。以往的工作通过将平均回报问题近似为折扣问题，并采用一种基于值迭代的算法，利用剪裁来约束值函数的范围以提高统计效率。然而，剪裁过程需要计算状态空间中值函数的最小值，这在线性MDP环境中由于状态空间可能很大甚至无限时是不可行的。在本文中，我们引入了一种高效的剪裁操作值迭代方法，只需计算算法访问的状态集上值函数的最小值。我们的算法享有与以往工作相同的遗憾界，同时具有计算效率，计算复杂度与状态空间的大小无关。 

---
# EngramNCA: a Neural Cellular Automaton Model of Memory Transfer 

**Title (ZH)**: EngramNCA：记忆迁移的神经细胞自动机模型 

**Authors**: Etienne Guichard, Felix Reimers, Mia Kvalsund, Mikkel Lepperød, Stefano Nichele  

**Link**: [PDF](https://arxiv.org/pdf/2504.11855)  

**Abstract**: This study introduces EngramNCA, a neural cellular automaton (NCA) that integrates both publicly visible states and private, cell-internal memory channels, drawing inspiration from emerging biological evidence suggesting that memory storage extends beyond synaptic modifications to include intracellular mechanisms. The proposed model comprises two components: GeneCA, an NCA trained to develop distinct morphologies from seed cells containing immutable "gene" encodings, and GenePropCA, an auxiliary NCA that modulates the private "genetic" memory of cells without altering their visible states. This architecture enables the encoding and propagation of complex morphologies through the interaction of visible and private channels, facilitating the growth of diverse structures from a shared "genetic" substrate. EngramNCA supports the emergence of hierarchical and coexisting morphologies, offering insights into decentralized memory storage and transfer in artificial systems. These findings have potential implications for the development of adaptive, self-organizing systems and may contribute to the broader understanding of memory mechanisms in both biological and synthetic contexts. 

**Abstract (ZH)**: EngramNCA：一种结合公开状态和细胞内私有记忆通道的神经细胞自动机模型 

---
# Learning Strategies in Particle Swarm Optimizer: A Critical Review and Performance Analysis 

**Title (ZH)**: 粒子群优化器中的学习策略：一项关键评述与性能分析 

**Authors**: Dikshit Chauhan, Shivani, P. N. Suganthan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11812)  

**Abstract**: Nature has long inspired the development of swarm intelligence (SI), a key branch of artificial intelligence that models collective behaviors observed in biological systems for solving complex optimization problems. Particle swarm optimization (PSO) is widely adopted among SI algorithms due to its simplicity and efficiency. Despite numerous learning strategies proposed to enhance PSO's performance in terms of convergence speed, robustness, and adaptability, no comprehensive and systematic analysis of these strategies exists. We review and classify various learning strategies to address this gap, assessing their impact on optimization performance. Additionally, a comparative experimental evaluation is conducted to examine how these strategies influence PSO's search dynamics. Finally, we discuss open challenges and future directions, emphasizing the need for self-adaptive, intelligent PSO variants capable of addressing increasingly complex real-world problems. 

**Abstract (ZH)**: 自然界长期启发着 swarm intelligence (群体智能) 的发展，群体智能是人工intelligence的一个关键分支，它通过模仿生物系统中观察到的集体行为来解决复杂优化问题。粒子群优化（PSO）由于其简单性和效率，在群体智能算法中被广泛应用。尽管提出了许多学习策略来提高PSO在收敛速度、鲁棒性和适应性等方面的性能，但这些策略的综合和系统性分析尚不存在。我们回顾并分类了各种学习策略以填补这一空白，评估它们对优化性能的影响。此外，我们还进行了比较实验评估，以考察这些策略如何影响PSO的搜索动力学。最后，我们讨论了开放挑战和未来方向，强调需要能够处理日益复杂现实问题的自适应和智能PSO变体。 

---
# Enhancing Web Agents with Explicit Rollback Mechanisms 

**Title (ZH)**: 增强Web代理的显式回滚机制 

**Authors**: Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11788)  

**Abstract**: With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives the model the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach. 

**Abstract (ZH)**: 随着大型语言模型的 Recent 进展，网络代理得到了极大改进。然而，应对复杂和动态的网络环境需要更高级的规划和搜索能力。前期研究通常采用贪婪的一维搜索策略，这可能难以从错误状态中恢复。在本工作中，我们通过引入显式的回滚机制增强网络代理，使代理能够回退到导航轨迹的先前状态。该机制赋予模型直接控制搜索过程的灵活性，从而实现一种有效且高效的网络导航方法。我们在两个实时网络导航基准上进行了零样本和微调设置的实验。结果证明了我们提出的方法的有效性。 

---
# ACMamba: Fast Unsupervised Anomaly Detection via An Asymmetrical Consensus State Space Model 

**Title (ZH)**: ACMamba: 快速无监督异常检测基于不对称共识状态空间模型 

**Authors**: Guanchun Wang, Xiangrong Zhang, Yifei Zhang, Zelin Peng, Tianyang Zhang, Xu Tang, Licheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.11781)  

**Abstract**: Unsupervised anomaly detection in hyperspectral images (HSI), aiming to detect unknown targets from backgrounds, is challenging for earth surface monitoring. However, current studies are hindered by steep computational costs due to the high-dimensional property of HSI and dense sampling-based training paradigm, constraining their rapid deployment. Our key observation is that, during training, not all samples within the same homogeneous area are indispensable, whereas ingenious sampling can provide a powerful substitute for reducing costs. Motivated by this, we propose an Asymmetrical Consensus State Space Model (ACMamba) to significantly reduce computational costs without compromising accuracy. Specifically, we design an asymmetrical anomaly detection paradigm that utilizes region-level instances as an efficient alternative to dense pixel-level samples. In this paradigm, a low-cost Mamba-based module is introduced to discover global contextual attributes of regions that are essential for HSI reconstruction. Additionally, we develop a consensus learning strategy from the optimization perspective to simultaneously facilitate background reconstruction and anomaly compression, further alleviating the negative impact of anomaly reconstruction. Theoretical analysis and extensive experiments across eight benchmarks verify the superiority of ACMamba, demonstrating a faster speed and stronger performance over the state-of-the-art. 

**Abstract (ZH)**: 无监督高光谱图像异常检测（HSI）旨在从背景中检测未知目标，对于地球表面监测来说具有挑战性。然而，当前研究受限于高维性质导致的高额计算成本以及基于密集采样的训练范式，限制了其快速部署。我们的关键观察是，在训练过程中，并非所有同质区域内的样本都是不可或缺的，而巧妙的采样可以提供一个强大的替代方案来降低成本。基于此，我们提出了一种不对称一致状态空间模型（ACMamba），以显著降低计算成本而不牺牲准确性。具体来说，我们设计了一种利用区域级实例替代密集像素级样本的不对称异常检测范式。在此范式中，引入了一个低成本的Mamba基模块来发现对HSI重建至关重要的全局上下文属性。此外，我们从优化角度开发了一种一致学习策略，以同时促进背景重建和异常压缩，进一步缓解异常重建的负面影响。理论分析和在八个基准上的广泛实验验证了ACMamba的优势，证明其在速度和性能上均超过了现有最佳方法。 

---
# Agile Retrospectives: What went well? What didn't go well? What should we do? 

**Title (ZH)**: 敏捷回顾：什么是好的？什么是不好的？我们应该怎么做？ 

**Authors**: Maria Spichkova, Hina Lee, Kevin Iwan, Madeleine Zwart, Yuwon Yoon, Xiaohan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.11780)  

**Abstract**: In Agile/Scrum software development, the idea of retrospective meetings (retros) is one of the core elements of the project process. In this paper, we present our work in progress focusing on two aspects: analysis of potential usage of generative AI for information interaction within retrospective meetings, and visualisation of retros' information to software development teams. We also present our prototype tool RetroAI++, focusing on retros-related functionalities. 

**Abstract (ZH)**: 在敏捷/Scrum软件开发中，回顾会议（Retros）的理念是项目过程中的核心元素之一。在本文中，我们呈现了我们正在进行的工作，重点关注两个方面：生成式AI在回顾会议中信息交互中的潜在应用分析，以及回顾会议信息的可视化展示给软件开发团队。我们还介绍了我们的原型工具RetroAI++，重点关注与回顾会议相关的功能。 

---
# PCDiff: Proactive Control for Ownership Protection in Diffusion Models with Watermark Compatibility 

**Title (ZH)**: PCDiff: 兼顾水印兼容性的主动控制产权保护方法在扩散模型中 

**Authors**: Keke Gai, Ziyue Shen, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11774)  

**Abstract**: With the growing demand for protecting the intellectual property (IP) of text-to-image diffusion models, we propose PCDiff -- a proactive access control framework that redefines model authorization by regulating generation quality. At its core, PCDIFF integrates a trainable fuser module and hierarchical authentication layers into the decoder architecture, ensuring that only users with valid encrypted credentials can generate high-fidelity images. In the absence of valid keys, the system deliberately degrades output quality, effectively preventing unauthorized this http URL, while the primary mechanism enforces active access control through architectural intervention, its decoupled design retains compatibility with existing watermarking techniques. This satisfies the need of model owners to actively control model ownership while preserving the traceability capabilities provided by traditional watermarking this http URL experimental evaluations confirm a strong dependency between credential verification and image quality across various attack scenarios. Moreover, when combined with typical post-processing operations, PCDIFF demonstrates powerful performance alongside conventional watermarking methods. This work shifts the paradigm from passive detection to proactive enforcement of authorization, laying the groundwork for IP management of diffusion models. 

**Abstract (ZH)**: 文本到图像扩散模型知识产权保护的主动访问控制框架：PCDiff 

---
# Saga: Capturing Multi-granularity Semantics from Massive Unlabelled IMU Data for User Perception 

**Title (ZH)**: Saga：从大量未标注的IMU数据中捕捉多粒度语义以分析用户感知 

**Authors**: Yunzhe Li, Facheng Hu, Hongzi Zhu, Shifan Zhang, Liang Zhang, Shan Chang, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11726)  

**Abstract**: Inertial measurement units (IMUs), have been prevalently used in a wide range of mobile perception applications such as activity recognition and user authentication, where a large amount of labelled data are normally required to train a satisfactory model. However, it is difficult to label micro-activities in massive IMU data due to the hardness of understanding raw IMU data and the lack of ground truth. In this paper, we propose a novel fine-grained user perception approach, called Saga, which only needs a small amount of labelled IMU data to achieve stunning user perception accuracy. The core idea of Saga is to first pre-train a backbone feature extraction model, utilizing the rich semantic information of different levels embedded in the massive unlabelled IMU data. Meanwhile, for a specific downstream user perception application, Bayesian Optimization is employed to determine the optimal weights for pre-training tasks involving different semantic levels. We implement Saga on five typical mobile phones and evaluate Saga on three typical tasks on three IMU datasets. Results show that when only using about 100 training samples per class, Saga can achieve over 90% accuracy of the full-fledged model trained on over ten thousands training samples with no additional system overhead. 

**Abstract (ZH)**: 基于惯性测量单元的细粒度用户感知方法Saga：少量标注数据实现高效用户感知准确性 

---
# Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching 

**Title (ZH)**: 伴随采样：通过伴随匹配实现的高度可扩展扩散采样器 

**Authors**: Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, Guan-Horng Liu, Ricky T. Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11713)  

**Abstract**: We introduce Adjoint Sampling, a highly scalable and efficient algorithm for learning diffusion processes that sample from unnormalized densities, or energy functions. It is the first on-policy approach that allows significantly more gradient updates than the number of energy evaluations and model samples, allowing us to scale to much larger problem settings than previously explored by similar methods. Our framework is theoretically grounded in stochastic optimal control and shares the same theoretical guarantees as Adjoint Matching, being able to train without the need for corrective measures that push samples towards the target distribution. We show how to incorporate key symmetries, as well as periodic boundary conditions, for modeling molecules in both cartesian and torsional coordinates. We demonstrate the effectiveness of our approach through extensive experiments on classical energy functions, and further scale up to neural network-based energy models where we perform amortized conformer generation across many molecular systems. To encourage further research in developing highly scalable sampling methods, we plan to open source these challenging benchmarks, where successful methods can directly impact progress in computational chemistry. 

**Abstract (ZH)**: 我们介绍了一种称为伴随采样的高效算法，该算法适用于从非归一化密度或能量函数中采样的扩散过程的学习。这是首个能够在能量评估和模型样本数量显著多于梯度更新次数的情况下工作的在线策略方法，使我们能够处理比先前类似方法探索的更复杂的问题规模。我们的框架在随机最优控制理论上有坚实的理论基础，并与伴随匹配方法共享相同的理论保证，无需采取纠正措施将样本推送到目标分布即可进行训练。我们展示了如何在笛卡尔坐标和扭转坐标中结合关键对称性和周期边界条件，以用于分子建模。通过在经典能量函数和基于神经网络的能量模型上的广泛实验，我们展示了该方法的有效性，并进一步扩展到多个分子系统的构象生成任务。为促进对高效可扩展采样方法的进一步研究，我们计划开源这些具有挑战性的基准测试，成功的方法可以直接影响计算化学的发展。 

---
# Data driven approach towards more efficient Newton-Raphson power flow calculation for distribution grids 

**Title (ZH)**: 基于数据驱动的方法以提高配电网络牛顿-拉夫逊潮流计算效率 

**Authors**: Shengyuan Yan, Farzad Vazinram, Zeynab Kaseb, Lindsay Spoor, Jochen Stiasny, Betul Mamudi, Amirhossein Heydarian Ardakani, Ugochukwu Orji, Pedro P. Vergara, Yu Xiang, Jerry Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11650)  

**Abstract**: Power flow (PF) calculations are fundamental to power system analysis to ensure stable and reliable grid operation. The Newton-Raphson (NR) method is commonly used for PF analysis due to its rapid convergence when initialized properly. However, as power grids operate closer to their capacity limits, ill-conditioned cases and convergence issues pose significant challenges. This work, therefore, addresses these challenges by proposing strategies to improve NR initialization, hence minimizing iterations and avoiding divergence. We explore three approaches: (i) an analytical method that estimates the basin of attraction using mathematical bounds on voltages, (ii) Two data-driven models leveraging supervised learning or physics-informed neural networks (PINNs) to predict optimal initial guesses, and (iii) a reinforcement learning (RL) approach that incrementally adjusts voltages to accelerate convergence. These methods are tested on benchmark systems. This research is particularly relevant for modern power systems, where high penetration of renewables and decentralized generation require robust and scalable PF solutions. In experiments, all three proposed methods demonstrate a strong ability to provide an initial guess for Newton-Raphson method to converge with fewer steps. The findings provide a pathway for more efficient real-time grid operations, which, in turn, support the transition toward smarter and more resilient electricity networks. 

**Abstract (ZH)**: 基于幂流计算中牛顿-拉夫森方法初始化改进策略的研究 

---
# Achieving Tighter Finite-Time Rates for Heterogeneous Federated Stochastic Approximation under Markovian Sampling 

**Title (ZH)**: 实现更具紧致性的有限时间速率异构联邦随机逼近在马尔可夫采样下的研究 

**Authors**: Feng Zhu, Aritra Mitra, Robert W. Heath  

**Link**: [PDF](https://arxiv.org/pdf/2504.11645)  

**Abstract**: Motivated by collaborative reinforcement learning (RL) and optimization with time-correlated data, we study a generic federated stochastic approximation problem involving $M$ agents, where each agent is characterized by an agent-specific (potentially nonlinear) local operator. The goal is for the agents to communicate intermittently via a server to find the root of the average of the agents' local operators. The generality of our setting stems from allowing for (i) Markovian data at each agent and (ii) heterogeneity in the roots of the agents' local operators. The limited recent work that has accounted for both these features in a federated setting fails to guarantee convergence to the desired point or to show any benefit of collaboration; furthermore, they rely on projection steps in their algorithms to guarantee bounded iterates. Our work overcomes each of these limitations. We develop a novel algorithm titled \texttt{FedHSA}, and prove that it guarantees convergence to the correct point, while enjoying an $M$-fold linear speedup in sample-complexity due to collaboration. To our knowledge, \emph{this is the first finite-time result of its kind}, and establishing it (without relying on a projection step) entails a fairly intricate argument that accounts for the interplay between complex temporal correlations due to Markovian sampling, multiple local steps to save communication, and the drift-effects induced by heterogeneous local operators. Our results have implications for a broad class of heterogeneous federated RL problems (e.g., policy evaluation and control) with function approximation, where the agents' Markov decision processes can differ in their probability transition kernels and reward functions. 

**Abstract (ZH)**: 受协作强化学习（RL）和具有时间相关数据的优化的启发，我们研究了一个通用的联邦随机逼近问题，涉及 $M$ 个代理，每个代理由一个特定于代理（可能是非线性的）局部操作符描述。目标是通过服务器使代理间间歇性通信，找到各局部操作符平均值的根。我们的设置之所以通用，是因为它允许每个代理具有马尔可夫数据，并且代理局部操作符的根具有异质性。近期有限的研究虽然在这两个特征上有所考虑，但在联邦设置中未能保证收敛到所需点，也未能展示合作的好处；此外，它们依赖投影步骤来保证迭代值的有界性。我们的工作克服了这些限制。我们开发了一种新颖的算法，命名为 \texttt{FedHSA}，并证明该算法能够保证收敛到正确的点，同时由于合作而使样本复杂度提升 $M$ 倍。据我们所知，这是首个此类有限时间的结果，并且不依赖于投影步骤建立这一结果需要相当复杂的方法来解决马尔可夫采样引起的复杂时序相关性、节省通信的多步局部操作以及异质局部操作符诱导的漂移效应之间的互动问题。我们的结果对具有函数逼近的异质联邦RL问题（例如策略评估和控制）具有重要意义，其中代理的马尔可夫决策过程可能在转移概率核和奖励函数上有所不同。 

---
# Possibility for Proactive Anomaly Detection 

**Title (ZH)**: 主动性异常检测的可能性 

**Authors**: Jinsung Jeon, Jaehyeon Park, Sewon Park, Jeongwhan Choi, Minjung Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.11623)  

**Abstract**: Time-series anomaly detection, which detects errors and failures in a workflow, is one of the most important topics in real-world applications. The purpose of time-series anomaly detection is to reduce potential damages or losses. However, existing anomaly detection models detect anomalies through the error between the model output and the ground truth (observed) value, which makes them impractical. In this work, we present a \textit{proactive} approach for time-series anomaly detection based on a time-series forecasting model specialized for anomaly detection and a data-driven anomaly detection model. Our proactive approach establishes an anomaly threshold from training data with a data-driven anomaly detection model, and anomalies are subsequently detected by identifying predicted values that exceed the anomaly threshold. In addition, we extensively evaluated the model using four anomaly detection benchmarks and analyzed both predictable and unpredictable anomalies. We attached the source code as supplementary material. 

**Abstract (ZH)**: 基于时间序列预测模型和数据驱动异常检测模型的主动时间序列异常检测方法 

---
# Towards Interpretable Deep Generative Models via Causal Representation Learning 

**Title (ZH)**: 通过因果表示学习实现可解释的深度生成模型 

**Authors**: Gemma E. Moran, Bryon Aragam  

**Link**: [PDF](https://arxiv.org/pdf/2504.11609)  

**Abstract**: Recent developments in generative artificial intelligence (AI) rely on machine learning techniques such as deep learning and generative modeling to achieve state-of-the-art performance across wide-ranging domains. These methods' surprising performance is due in part to their ability to learn implicit "representations'' of complex, multi-modal data. Unfortunately, deep neural networks are notoriously black boxes that obscure these representations, making them difficult to interpret or analyze. To resolve these difficulties, one approach is to build new interpretable neural network models from the ground up. This is the goal of the emerging field of causal representation learning (CRL) that uses causality as a vector for building flexible, interpretable, and transferable generative AI. CRL can be seen as a culmination of three intrinsically statistical problems: (i) latent variable models such as factor analysis; (ii) causal graphical models with latent variables; and (iii) nonparametric statistics and deep learning. This paper reviews recent progress in CRL from a statistical perspective, focusing on connections to classical models and statistical and causal identifiablity results. This review also highlights key application areas, implementation strategies, and open statistical questions in CRL. 

**Abstract (ZH)**: 近期生成人工智能（AI）的发展依赖于深度学习和生成建模等机器学习技术，在广泛的领域中实现了最先进的性能。这些方法令人惊讶的性能部分归因于它们能够学习复杂、多模态数据的隐式“表示”。不幸的是，深度神经网络通常是黑盒子，模糊了这些表示，使其难以解释或分析。为了解决这些问题，一种方法是自底向上构建新的可解释神经网络模型。这正是因果表示学习（CRL）新兴领域的目标，该领域利用因果关系作为构建灵活、可解释和可迁移的生成AI的向量。CRL 可以被视为三个内在统计问题的综合：（i）潜在变量模型，如因子分析；（ii）带有潜在变量的因果图模型；以及（iii）非参数统计和深度学习。本文从统计学的角度回顾了 CRL 的近期进展，重点关注与经典模型的联系以及统计和因果可识别性结果。该回顾还强调了 CRL 的关键应用领域、实施策略以及开放的统计问题。 

---
# Deep Learning Approaches for Medical Imaging Under Varying Degrees of Label Availability: A Comprehensive Survey 

**Title (ZH)**: 深学习方法在不同标注程度下的医学影像应用：综述 

**Authors**: Siteng Ma, Honghui Du, Yu An, Jing Wang, Qinqin Wang, Haochang Wu, Aonghus Lawlor, Ruihai Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11588)  

**Abstract**: Deep learning has achieved significant breakthroughs in medical imaging, but these advancements are often dependent on large, well-annotated datasets. However, obtaining such datasets poses a significant challenge, as it requires time-consuming and labor-intensive annotations from medical experts. Consequently, there is growing interest in learning paradigms such as incomplete, inexact, and absent supervision, which are designed to operate under limited, inexact, or missing labels. This survey categorizes and reviews the evolving research in these areas, analyzing around 600 notable contributions since 2018. It covers tasks such as image classification, segmentation, and detection across various medical application areas, including but not limited to brain, chest, and cardiac imaging. We attempt to establish the relationships among existing research studies in related areas. We provide formal definitions of different learning paradigms and offer a comprehensive summary and interpretation of various learning mechanisms and strategies, aiding readers in better understanding the current research landscape and ideas. We also discuss potential future research challenges. 

**Abstract (ZH)**: 深度学习在医学影像领域的显著突破往往依赖于大规模且well-注释的数据集，然而获取这样的数据集面临着巨大的挑战，因为它需要耗时且劳动密集型的医学专家注释。因此，在有限、不精确或缺失标签的条件下，使用不完备、不精确和无监督学习范式的兴趣日益增长。本文对这些领域的研究进行分类和综述，自2018年以来分析了约600项重要的贡献。涵盖了包括但不限于脑部、胸部和心脏影像在内的各种医学应用领域的图像分类、分割和检测任务。我们试图建立现有研究之间的关系。我们为不同学习范式提供正式定义，并给出各种学习机制和策略的全面总结与解释，帮助读者更好地理解当前的研究景观和理念。我们还讨论了潜在的未来研究挑战。 

---
# MULTI-LF: A Unified Continuous Learning Framework for Real-Time DDoS Detection in Multi-Environment Networks 

**Title (ZH)**: MULTI-LF：多环境网络中实时DDoS检测的统一连续学习框架 

**Authors**: Furqan Rustam, Islam Obaidat, Anca Delia Jurcut  

**Link**: [PDF](https://arxiv.org/pdf/2504.11575)  

**Abstract**: Detecting Distributed Denial of Service (DDoS) attacks in Multi-Environment (M-En) networks presents significant challenges due to diverse malicious traffic patterns and the evolving nature of cyber threats. Existing AI-based detection systems struggle to adapt to new attack strategies and lack real-time attack detection capabilities with high accuracy and efficiency. This study proposes an online, continuous learning methodology for DDoS detection in M-En networks, enabling continuous model updates and real-time adaptation to emerging threats, including zero-day attacks. First, we develop a unique M-En network dataset by setting up a realistic, real-time simulation using the NS-3 tool, incorporating both victim and bot devices. DDoS attacks with varying packet sizes are simulated using the DDoSim application across IoT and traditional IP-based environments under M-En network criteria. Our approach employs a multi-level framework (MULTI-LF) featuring two machine learning models: a lightweight Model 1 (M1) trained on a selective, critical packet dataset for fast and efficient initial detection, and a more complex, highly accurate Model 2 (M2) trained on extensive data. When M1 exhibits low confidence in its predictions, the decision is escalated to M2 for verification and potential fine-tuning of M1 using insights from M2. If both models demonstrate low confidence, the system flags the incident for human intervention, facilitating model updates with human-verified categories to enhance adaptability to unseen attack patterns. We validate the MULTI-LF through real-world simulations, demonstrating superior classification accuracy of 0.999 and low prediction latency of 0.866 seconds compared to established baselines. Furthermore, we evaluate performance in terms of memory usage (3.632 MB) and CPU utilization (10.05%) in real-time scenarios. 

**Abstract (ZH)**: 检测多环境网络中的分布式拒绝服务（DDoS）攻击存在显著挑战，由于恶意流量模式多样和网络威胁的演变性。现有的基于AI的检测系统难以适应新的攻击策略，缺乏高准确性和高效性的实时攻击检测能力。本研究提出了一个在线持续学习方法，用于多环境网络中的DDoS检测，能够实现持续模型更新和对新兴威胁，包括零日攻击的实时适应。首先，我们利用NS-3工具搭建了一个现实且实时的模拟环境，包含了受害者和僵尸设备。使用DDoSim应用程序在符合多环境网络标准的物联网和传统IP环境中模拟了不同包大小的DDoS攻击。我们的方法采用多级框架（MULTI-LF），包含两个机器学习模型：轻量级的模型1（M1），用于快速高效地初始检测，以及更为复杂且准确的模型2（M2），用于全面数据训练。当M1预测不确定性较高时，决策将被提升到M2进行验证和可能的M1微调以利用M2的见解。如果两个模型的不确定性都较高，系统将该事件标记为需要人工干预，并通过人工验证的类别提升模型更新，以增强对未知攻击模式的适应能力。我们通过实际仿真验证了MULTI-LF，结果显示其分类精度高达0.999，预测延迟仅为0.866秒，优于现有基准。此外，我们还在实际场景中评估了其内存使用（3.632 MB）和CPU利用率（10.05%）。 

---
# Error Broadcast and Decorrelation as a Potential Artificial and Natural Learning Mechanism 

**Title (ZH)**: 误差广播与去相关作为潜在的人工和自然学习机制 

**Authors**: Mete Erdogan, Cengiz Pehlevan, Alper T. Erdogan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11558)  

**Abstract**: We introduce the Error Broadcast and Decorrelation (EBD) algorithm, a novel learning framework that addresses the credit assignment problem in neural networks by directly broadcasting output error to individual layers. Leveraging the stochastic orthogonality property of the optimal minimum mean square error (MMSE) estimator, EBD defines layerwise loss functions to penalize correlations between layer activations and output errors, offering a principled approach to error broadcasting without the need for weight transport. The optimization framework naturally leads to the experimentally observed three-factor learning rule and integrates with biologically plausible frameworks to enhance performance and plausibility. Numerical experiments demonstrate that EBD achieves performance comparable to or better than known error-broadcast methods on benchmark datasets. While the scalability of EBD to very large or complex datasets remains to be further explored, our findings suggest it provides a biologically plausible, efficient, and adaptable alternative for neural network training. This approach could inform future advancements in artificial and natural learning paradigms. 

**Abstract (ZH)**: 我们介绍了Error Broadcast and Decorrelation (EBD)算法，这是一种通过直接广播输出误差到各个层来解决神经网络中信用分配问题的新颖学习框架。EBD利用最优最小均方误差（MMSE）估计器的随机正交性属性，通过定义层间损失函数来惩罚层激活与输出误差之间的相关性，从而提供了一种无需权重传输的有原则的误差广播方法。该优化框架自然地导致了实验中观察到的三因素学习规则，并与生物可实现框架集成，以提高性能和可实现性。数值实验表明，EBD在基准数据集上的性能与已知的误差广播方法相当或更好。虽然EBD在非常大规模或复杂数据集上的可扩展性仍需进一步探索，但我们的发现表明，EBD提供了一种生物可实现、高效且灵活的神经网络训练替代方案，该方法可能为未来的人工和自然学习范式提供指导。 

---
# RAID: An In-Training Defense against Attribute Inference Attacks in Recommender Systems 

**Title (ZH)**: RAID：推荐系统中对抗属性推断攻击的在训练防御方法 

**Authors**: Xiaohua Feng, Yuyuan Li, Fengyuan Yu, Ke Xiong, Junjie Fang, Li Zhang, Tianyu Du, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11510)  

**Abstract**: In various networks and mobile applications, users are highly susceptible to attribute inference attacks, with particularly prevalent occurrences in recommender systems. Attackers exploit partially exposed user profiles in recommendation models, such as user embeddings, to infer private attributes of target users, such as gender and political views. The goal of defenders is to mitigate the effectiveness of these attacks while maintaining recommendation performance. Most existing defense methods, such as differential privacy and attribute unlearning, focus on post-training settings, which limits their capability of utilizing training data to preserve recommendation performance. Although adversarial training extends defenses to in-training settings, it often struggles with convergence due to unstable training processes. In this paper, we propose RAID, an in-training defense method against attribute inference attacks in recommender systems. In addition to the recommendation objective, we define a defensive objective to ensure that the distribution of protected attributes becomes independent of class labels, making users indistinguishable from attribute inference attacks. Specifically, this defensive objective aims to solve a constrained Wasserstein barycenter problem to identify the centroid distribution that makes the attribute indistinguishable while complying with recommendation performance constraints. To optimize our proposed objective, we use optimal transport to align users with the centroid distribution. We conduct extensive experiments on four real-world datasets to evaluate RAID. The experimental results validate the effectiveness of RAID and demonstrate its significant superiority over existing methods in multiple aspects. 

**Abstract (ZH)**: 在各种网络和移动应用中，用户高度容易受到属性推理攻击的影响，特别是在推荐系统中尤为普遍。攻击者利用推荐模型中部分暴露的用户画像，如用户嵌入，来推断目标用户的私人属性，例如性别和政治观点。防御方的目标是在保持推荐性能的情况下，减轻这些攻击的有效性。现有的大多数防御方法，如差分隐私和属性遗忘，主要关注于训练后设置，这限制了它们利用训练数据来保持推荐性能的能力。尽管对抗训练将防御扩展到训练中设置，但往往由于不稳定的训练过程而在收敛性方面遇到困难。在本文中，我们提出了一种针对推荐系统中属性推理攻击的训练中防御方法RAID。除了推荐目标外，我们定义了一个防御目标，确保保护属性的分布与类别标签无关，从而使用户在属性推理攻击中不可区分。具体而言，该防御目标旨在解决一个受限的Wasserstein巴尔扎克问题，以确定使得属性不可区分的重心分布，同时遵守推荐性能约束条件。为了优化我们提出的目标，我们使用最优传输将用户与重心分布对齐。我们在四个现实世界的数据集上进行了广泛的实验以评估RAID。实验结果验证了RAID的有效性，并在多个方面展示了其相对于现有方法的显著优越性。 

---
# A Framework for the Private Governance of Frontier Artificial Intelligence 

**Title (ZH)**: 前沿人工智能的私密治理框架 

**Authors**: Dean W. Ball  

**Link**: [PDF](https://arxiv.org/pdf/2504.11501)  

**Abstract**: This paper presents a proposal for the governance of frontier AI systems through a hybrid public-private system. Private bodies, authorized and overseen by government, provide certifications to developers of frontier AI systems on an opt-in basis. In exchange for opting in, frontier AI firms receive protections from tort liability for customer misuse of their models. Before detailing the proposal, the paper explores more commonly discussed approaches to AI governance, analyzing their strengths and flaws. It also examines the nature of frontier AI governance itself. The paper includes consideration of the political economic, institutional, legal, safety, and other merits and tradeoffs inherent in the governance system it proposes. 

**Abstract (ZH)**: 本文提出了一种通过混合公私体系治理前沿人工智能系统的提案。在政府授权和监督下，私营机构为前沿人工智能系统的开发者提供可选认证。作为交换，前沿人工智能企业将获得对其模型客户误用的侵权责任保护。在详细阐述提案之前，本文探讨了更常讨论的人工智能治理方法，分析了它们的优点和缺点。同时，本文还考察了前沿人工智能治理本身的性质。本文还考虑了其所提议的治理体系在政治经济、制度、法律、安全以及其他方面的优势权衡。 

---
# Local Temporal Feature Enhanced Transformer with ROI-rank Based Masking for Diagnosis of ADHD 

**Title (ZH)**: 基于ROI-rank基于掩码的局部时间特征增强变压器用于ADHD诊断 

**Authors**: Byunggun Kim, Younghun Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2504.11474)  

**Abstract**: In modern society, Attention-Deficit/Hyperactivity Disorder (ADHD) is one of the common mental diseases discovered not only in children but also in adults. In this context, we propose a ADHD diagnosis transformer model that can effectively simultaneously find important brain spatiotemporal biomarkers from resting-state functional magnetic resonance (rs-fMRI). This model not only learns spatiotemporal individual features but also learns the correlation with full attention structures specialized in ADHD diagnosis. In particular, it focuses on learning local blood oxygenation level dependent (BOLD) signals and distinguishing important regions of interest (ROI) in the brain. Specifically, the three proposed methods for ADHD diagnosis transformer are as follows. First, we design a CNN-based embedding block to obtain more expressive embedding features in brain region attention. It is reconstructed based on the previously CNN-based ADHD diagnosis models for the transformer. Next, for individual spatiotemporal feature attention, we change the attention method to local temporal attention and ROI-rank based masking. For the temporal features of fMRI, the local temporal attention enables to learn local BOLD signal features with only simple window masking. For the spatial feature of fMRI, ROI-rank based masking can distinguish ROIs with high correlation in ROI relationships based on attention scores, thereby providing a more specific biomarker for ADHD diagnosis. The experiment was conducted with various types of transformer models. To evaluate these models, we collected the data from 939 individuals from all sites provided by the ADHD-200 competition. Through this, the spatiotemporal enhanced transformer for ADHD diagnosis outperforms the performance of other different types of transformer variants. (77.78ACC 76.60SPE 79.22SEN 79.30AUC) 

**Abstract (ZH)**: 现代社会中，注意力缺陷多动障碍（ADHD）是一种不仅在儿童中发现，在成人中也常见的精神疾病。在此背景下，我们提出了一种ADHD诊断变换器模型，能够有效同时从静息状态功能性磁共振成像（rs-fMRI）中发现重要的脑时空生物标志物。该模型不仅学习时空个体内特征，还学习与ADHD诊断专门化全注意结构的相关性。特别是，该模型专注于学习局部血氧水平依赖（BOLD）信号并区分大脑中的重要感兴趣区域（ROI）。具体而言，ADHD诊断变换器提出的三种方法如下。首先，我们设计了一个基于CNN的嵌入块，以获取大脑区域注意力中的更具表达性的嵌入特征。该嵌入块基于之前基于CNN的ADHD诊断模型进行了重构。其次，对于个体时空特征注意力，我们将注意力方法更改为局部时间注意力和基于ROI的掩蔽。对于fMRI的时间特征，局部时间注意力仅通过简单的窗口掩蔽即可学习局部BOLD信号特征。对于fMRI的空间特征，基于ROI的排名掩蔽可以根据注意得分区分ROI关系中具有高相关性的ROI，从而为ADHD诊断提供更具针对性的生物标志物。在不同类型的变换器模型实验中，时空增强的ADHD诊断变换器模型在不同类型的变换器变体中表现最佳（准确率77.78%，特异度76.60%，灵敏度79.22%，AUC 79.30）。 

---
