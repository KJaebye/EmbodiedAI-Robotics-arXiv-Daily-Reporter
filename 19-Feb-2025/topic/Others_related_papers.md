# Memory-updated-based Framework for 100% Reliable Flexible Flat Cables Insertion 

**Title (ZH)**: 基于内存更新的100%可靠柔性扁 cable 插接口技术框架 

**Authors**: Zhengrong Ling, Xiong Yang, Dong Guo, Hongyuan Chang, Tieshan Zhang, Ruijia Zhang, Yajing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12514)  

**Abstract**: Automatic assembly lines have increasingly replaced human labor in various tasks; however, the automation of Flexible Flat Cable (FFC) insertion remains unrealized due to its high requirement for effective feedback and dynamic operation, limiting approximately 11% of global industrial capacity. Despite lots of approaches, like vision-based tactile sensors and reinforcement learning, having been proposed, the implementation of human-like high-reliable insertion (i.e., with a 100% success rate in completed insertion) remains a big challenge. Drawing inspiration from human behavior in FFC insertion, which involves sensing three-dimensional forces, translating them into physical concepts, and continuously improving estimates, we propose a novel framework. This framework includes a sensing module for collecting three-dimensional tactile data, a perception module for interpreting this data into meaningful physical signals, and a memory module based on Bayesian theory for reliability estimation and control. This strategy enables the robot to accurately assess its physical state and generate reliable status estimations and corrective actions. Experimental results demonstrate that the robot using this framework can detect alignment errors of 0.5 mm with an accuracy of 97.92% and then achieve a 100% success rate in all completed tests after a few iterations. This work addresses the challenges of unreliable perception and control in complex insertion tasks, highlighting the path toward the development of fully automated production lines. 

**Abstract (ZH)**: 自动装配线已在各种任务中逐渐取代了人力劳动；然而，由于柔性扁平电缆（FFC）插入对有效反馈和动态操作有高度要求，FFC插入的自动化尚未实现，限制了大约11%的全球工业产能。尽管提出了许多方法，如基于视觉的触觉传感器和强化学习，但实现类似人类的高可靠性插入（即完成插入的100%成功率）仍然是一个巨大挑战。借鉴人类在FFC插入过程中的行为，包括感知三维力、将其转化为物理概念并不断改进估计，我们提出了一种新型框架。该框架包括一个用于收集三维触觉数据的感知模块、一个用于将这些数据解释为有意义的物理信号的认知模块以及基于贝叶斯理论的记忆模块，用于可靠性估计和控制。这种策略使机器人能够准确评估其物理状态并生成可靠的状况估计和纠正措施。实验结果表明，使用该框架的机器人在几次迭代后能够检测到0.5 mm的对准误差，并且在所有完成的测试中实现100%的成功率。本工作解决了复杂插入任务中不可靠感知和控制的挑战，指出了全自动化生产线开发的方向。 

---
# AI-Augmented Metamorphic Testing for Comprehensive Validation of Autonomous Vehicles 

**Title (ZH)**: 基于AI增强的元变换测试对于自主车辆全面验证的研究 

**Authors**: Tony Zhang, Burak Kantarci, Umair Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2502.12208)  

**Abstract**: Self-driving cars have the potential to revolutionize transportation, but ensuring their safety remains a significant challenge. These systems must navigate a variety of unexpected scenarios on the road, and their complexity poses substantial difficulties for thorough testing. Conventional testing methodologies face critical limitations, including the oracle problem determining whether the systems behavior is correct and the inability to exhaustively recreate a range of situations a self-driving car may encounter. While Metamorphic Testing (MT) offers a partial solution to these challenges, its application is often limited by simplistic modifications to test scenarios. In this position paper, we propose enhancing MT by integrating AI-driven image generation tools, such as Stable Diffusion, to improve testing methodologies. These tools can generate nuanced variations of driving scenarios within the operational design domain (ODD)for example, altering weather conditions, modifying environmental elements, or adjusting lane markings while preserving the critical features necessary for system evaluation. This approach enables reproducible testing, efficient reuse of test criteria, and comprehensive evaluation of a self-driving systems performance across diverse scenarios, thereby addressing key gaps in current testing practices. 

**Abstract (ZH)**: 自动驾驶汽车有潜力革新交通方式，但确保其安全仍面临重大挑战。这些系统必须应对道路上的各种意外场景，其复杂性给全面测试带来了巨大困难。传统测试方法存在关键限制，包括确定系统行为是否正确的奥里卡问题以及无法彻底重现自动驾驶汽车可能遇到的各种情况。尽管元型测试（MT）为解决这些挑战提供了一部分解决方案，但其应用往往受限于对测试场景的简单修改。在本文中，我们提议通过集成如Stable Diffusion等AI驱动的图像生成工具来增强MT，以改进测试方法。这些工具可以在操作设计域（ODD）中生成驾驶场景的精细变体，例如改变天气条件、修改环境元素或调整车道标记，同时保留系统评估所需的关键特征。这种方法能够实现可重复的测试、高效地重用测试标准并全面评估自动驾驶系统在不同场景中的性能，从而填补当前测试实践的关键缺口。 

---
# AI-Assisted Decision Making with Human Learning 

**Title (ZH)**: AI辅助决策与人类学习 

**Authors**: Gali Noti, Kate Donahue, Jon Kleinberg, Sigal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2502.13062)  

**Abstract**: AI systems increasingly support human decision-making. In many cases, despite the algorithm's superior performance, the final decision remains in human hands. For example, an AI may assist doctors in determining which diagnostic tests to run, but the doctor ultimately makes the diagnosis. This paper studies such AI-assisted decision-making settings, where the human learns through repeated interactions with the algorithm. In our framework, the algorithm -- designed to maximize decision accuracy according to its own model -- determines which features the human can consider. The human then makes a prediction based on their own less accurate model. We observe that the discrepancy between the algorithm's model and the human's model creates a fundamental tradeoff. Should the algorithm prioritize recommending more informative features, encouraging the human to recognize their importance, even if it results in less accurate predictions in the short term until learning occurs? Or is it preferable to forgo educating the human and instead select features that align more closely with their existing understanding, minimizing the immediate cost of learning? This tradeoff is shaped by the algorithm's time-discounted objective and the human's learning ability. Our results show that optimal feature selection has a surprisingly clean combinatorial characterization, reducible to a stationary sequence of feature subsets that is tractable to compute. As the algorithm becomes more "patient" or the human's learning improves, the algorithm increasingly selects more informative features, enhancing both prediction accuracy and the human's understanding. Notably, early investment in learning leads to the selection of more informative features than a later investment. We complement our analysis by showing that the impact of errors in the algorithm's knowledge is limited as it does not make the prediction directly. 

**Abstract (ZH)**: 人工智能系统日益支持人类决策。在许多情况下，尽管算法在性能上表现出色，最终的决策仍由人类作出。例如，AI可以辅助医生确定需要进行的诊断测试，但最终的诊断还是由医生作出。本文研究了此类由AI辅助的决策设置，在这种设置中，人类通过与算法的反复互动进行学习。在我们的框架中，算法旨在根据自己的模型最大化决策准确性，从而决定人类可以考虑哪些特征。人类基于自己不太准确的模型进行预测。我们观察到，算法模型与人类模型之间的差异性造成了一个基本的权衡。算法应优先推荐更多有信息量的特征，促使人类认识到这些特征的重要性，即使在短期内可能导致预测不那么准确，直到学习发生？还是更宜省去对人类的教育，而是选择与他们现有理解更为一致的特征，从而最小化即时学习的成本？这一权衡由算法的时间折扣目标和人类的学习能力所塑造。我们的研究结果表明，最优特征选择具有出乎意料的简洁组合特性，可归约为一个可计算的稳定特征子集序列。随着算法变得“更有耐心”或人性学习能力提升，算法将越来越多地选择更有信息量的特征，从而同时提高预测准确性和人类的理解。值得注意的是，早期的投资于学习会导致比后期投资选择更多的有信息量的特征。我们通过展示算法知识中的错误对其预测影响有限这一点来补充我们的分析，因为算法并不会直接作出预测。 

---
# Free Argumentative Exchanges for Explaining Image Classifiers 

**Title (ZH)**: 自由argue交流以解释图像分类器 

**Authors**: Avinash Kori, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2502.12995)  

**Abstract**: Deep learning models are powerful image classifiers but their opacity hinders their trustworthiness. Explanation methods for capturing the reasoning process within these classifiers faithfully and in a clear manner are scarce, due to their sheer complexity and size. We provide a solution for this problem by defining a novel method for explaining the outputs of image classifiers with debates between two agents, each arguing for a particular class. We obtain these debates as concrete instances of Free Argumentative eXchanges (FAXs), a novel argumentation-based multi-agent framework allowing agents to internalise opinions by other agents differently than originally stated. We define two metrics (consensus and persuasion rate) to assess the usefulness of FAXs as argumentative explanations for image classifiers. We then conduct a number of empirical experiments showing that FAXs perform well along these metrics as well as being more faithful to the image classifiers than conventional, non-argumentative explanation methods. All our implementations can be found at this https URL. 

**Abstract (ZH)**: 深度学习模型是强大的图像分类器，但其不透明性阻碍了其可信度。由于这些分类器复杂且庞大，忠实且清晰地捕获其推理过程的解释方法相当稀缺。我们通过定义一种新的方法来解决这一问题，即用两个代理之间的辩论来解释图像分类器的输出，每个代理为某个类别进行辩护。我们获得这些辩论的具体实例为一种新的基于论辩的多代理框架——自由论辩交换（FAXs），该框架允许代理以不同于最初陈述的方式内部化其他代理的意见。我们定义了两个指标（共识率和说服率）来评估FAXs作为图像分类器论辩解释的有效性。然后我们进行了一系列实证实验，结果表明FAXs在这些指标上表现良好，并且比传统的非论辩解释方法更忠实于图像分类器。所有我们的实现都可以在以下链接找到：this https URL。 

---
# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions 

**Title (ZH)**: 熬夜之夜，甜食之日：创建具有健康状况的合成用户以实现现实的辅导代理交互 

**Authors**: Taedong Yun, Eric Yang, Mustafa Safdari, Jong Ha Lee, Vaishnavi Vinod Kumar, S. Sara Mahdavi, Jonathan Amar, Derek Peyton, Reut Aharony, Andreas Michaelides, Logan Schneider, Isaac Galatzer-Levy, Yugang Jia, John Canny, Arthur Gretton, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2502.13135)  

**Abstract**: We present an end-to-end framework for generating synthetic users for evaluating interactive agents designed to encourage positive behavior changes, such as in health and lifestyle coaching. The synthetic users are grounded in health and lifestyle conditions, specifically sleep and diabetes management in this study, to ensure realistic interactions with the health coaching agent. Synthetic users are created in two stages: first, structured data are generated grounded in real-world health and lifestyle factors in addition to basic demographics and behavioral attributes; second, full profiles of the synthetic users are developed conditioned on the structured data. Interactions between synthetic users and the coaching agent are simulated using generative agent-based models such as Concordia, or directly by prompting a language model. Using two independently-developed agents for sleep and diabetes coaching as case studies, the validity of this framework is demonstrated by analyzing the coaching agent's understanding of the synthetic users' needs and challenges. Finally, through multiple blinded evaluations of user-coach interactions by human experts, we demonstrate that our synthetic users with health and behavioral attributes more accurately portray real human users with the same attributes, compared to generic synthetic users not grounded in such attributes. The proposed framework lays the foundation for efficient development of conversational agents through extensive, realistic, and grounded simulated interactions. 

**Abstract (ZH)**: 一种用于评估促进正面行为改变的交互式代理的端到端合成用户生成框架 

---
# SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation 

**Title (ZH)**: SongGen: 一种用于文本到歌曲生成的一阶段自回归变压器 

**Authors**: Zihan Liu, Shuangrui Ding, Zhixiong Zhang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13128)  

**Abstract**: Text-to-song generation, the task of creating vocals and accompaniment from textual inputs, poses significant challenges due to domain complexity and data scarcity. Existing approaches often employ multi-stage generation procedures, resulting in cumbersome training and inference pipelines. In this paper, we propose SongGen, a fully open-source, single-stage auto-regressive transformer designed for controllable song generation. The proposed model facilitates fine-grained control over diverse musical attributes, including lyrics and textual descriptions of instrumentation, genre, mood, and timbre, while also offering an optional three-second reference clip for voice cloning. Within a unified auto-regressive framework, SongGen supports two output modes: mixed mode, which generates a mixture of vocals and accompaniment directly, and dual-track mode, which synthesizes them separately for greater flexibility in downstream applications. We explore diverse token pattern strategies for each mode, leading to notable improvements and valuable insights. Furthermore, we design an automated data preprocessing pipeline with effective quality control. To foster community engagement and future research, we will release our model weights, training code, annotated data, and preprocessing pipeline. The generated samples are showcased on our project page at this https URL , and the code will be available at this https URL . 

**Abstract (ZH)**: 文本到歌曲生成：一种基于单阶段自回归变压器的可控歌曲生成方法 

---
# Near-Optimal Private Learning in Linear Contextual Bandits 

**Title (ZH)**: 近最优私密学习在线性上下文臂bandit中 

**Authors**: Fan Chen, Jiachun Li, Alexander Rakhlin, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13115)  

**Abstract**: We analyze the problem of private learning in generalized linear contextual bandits. Our approach is based on a novel method of re-weighted regression, yielding an efficient algorithm with regret of order $\sqrt{T}+\frac{1}{\alpha}$ and $\sqrt{T}/\alpha$ in the joint and local model of $\alpha$-privacy, respectively. Further, we provide near-optimal private procedures that achieve dimension-independent rates in private linear models and linear contextual bandits. In particular, our results imply that joint privacy is almost "for free" in all the settings we consider, partially addressing the open problem posed by Azize and Basu (2024). 

**Abstract (ZH)**: 我们分析了广义线性上下文_bandits中的私人学习问题。我们的方法基于一种新型加权回归方法，从而得到了在$\alpha$-隐私的联合模型和局部模型中分别具有$\sqrt{T}+\frac{1}{\alpha}$和$\sqrt{T}/\alpha$遗憾界的高效算法。此外，我们提供了接近最优的私人学习程序，在私人线性模型和线性上下文_bandits中实现了与维数无关的速度。特别是，我们的结果表明，在我们考虑的所有设置中，联合隐私几乎“免费”，部分解决了Azize和Basu（2024）提出的开放问题。 

---
# Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization 

**Title (ZH)**: 基于多任务学习改进临床问题回答：一种结合答案提取与医疗分类的方法 

**Authors**: Priyaranjan Pattnayak, Hitesh Laxmichand Patel, Amit Agarwal, Bhargava Kumar, Srikant Panda, Tejaswini Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.13108)  

**Abstract**: Clinical Question Answering (CQA) plays a crucial role in medical decision-making, enabling physicians to extract relevant information from Electronic Medical Records (EMRs). While transformer-based models such as BERT, BioBERT, and ClinicalBERT have demonstrated state-of-the-art performance in CQA, existing models lack the ability to categorize extracted answers, which is critical for structured retrieval, content filtering, and medical decision support.
To address this limitation, we introduce a Multi-Task Learning (MTL) framework that jointly trains CQA models for both answer extraction and medical categorization. In addition to predicting answer spans, our model classifies responses into five standardized medical categories: Diagnosis, Medication, Symptoms, Procedure, and Lab Reports. This categorization enables more structured and interpretable outputs, making clinical QA models more useful in real-world healthcare settings.
We evaluate our approach on emrQA, a large-scale dataset for medical question answering. Results show that MTL improves F1-score by 2.2% compared to standard fine-tuning, while achieving 90.7% accuracy in answer categorization. These findings suggest that MTL not only enhances CQA performance but also introduces an effective mechanism for categorization and structured medical information retrieval. 

**Abstract (ZH)**: 多任务学习在医疗问答中的应用：结合答案提取与医学分类 

---
# BOLIMES: Boruta and LIME optiMized fEature Selection for Gene Expression Classification 

**Title (ZH)**: BOLIMES: Boruta和LIME优化的特征选择用于基因表达分类 

**Authors**: Bich-Chung Phan, Thanh Ma, Huu-Hoa Nguyen, and Thanh-Nghi Do  

**Link**: [PDF](https://arxiv.org/pdf/2502.13080)  

**Abstract**: Gene expression classification is a pivotal yet challenging task in bioinformatics, primarily due to the high dimensionality of genomic data and the risk of overfitting. To bridge this gap, we propose BOLIMES, a novel feature selection algorithm designed to enhance gene expression classification by systematically refining the feature subset. Unlike conventional methods that rely solely on statistical ranking or classifier-specific selection, we integrate the robustness of Boruta with the interpretability of LIME, ensuring that only the most relevant and influential genes are retained. BOLIMES first employs Boruta to filter out non-informative genes by comparing each feature against its randomized counterpart, thus preserving valuable information. It then uses LIME to rank the remaining genes based on their local importance to the classifier. Finally, an iterative classification evaluation determines the optimal feature subset by selecting the number of genes that maximizes predictive accuracy. By combining exhaustive feature selection with interpretability-driven refinement, our solution effectively balances dimensionality reduction with high classification performance, offering a powerful solution for high-dimensional gene expression analysis. 

**Abstract (ZH)**: 基因表达分类是生物informatics中的一个关键但极具挑战性的任务，主要由于基因组数据的高维度和过拟合的风险。为解决这一问题，我们提出了一种新颖的特征选择算法BOLIMES，旨在通过系统性地精炼特征子集来提高基因表达分类。BOLIMES融合了Boruta的稳健性和LIME的可解释性，确保仅保留最相关的基因。该算法首先使用Boruta通过将每个特征与其随机化对应物进行比较来筛选出非信息性基因，从而保留有价值的信息。随后，使用LIME基于其对分类器的局部重要性对剩余基因进行排序。最后，迭代的分类评估通过选择最大化预测准确性的基因数量来确定最优特征子集。通过结合 exhaustive 特征选择与可解释性驱动的精炼，我们的解决方案有效地平衡了维度缩减与高分类性能，为高维基因表达分析提供了强大解决方案。 

---
# Natural Language Generation from Visual Sequences: Challenges and Future Directions 

**Title (ZH)**: 基于视觉序列的自然语言生成：挑战与未来方向 

**Authors**: Aditya K Surikuchi, Raquel Fernández, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2502.13034)  

**Abstract**: The ability to use natural language to talk about visual content is at the core of human intelligence and a crucial feature of any artificial intelligence system. Various studies have focused on generating text for single images. In contrast, comparatively little attention has been paid to exhaustively analyzing and advancing work on multiple-image vision-to-text settings. In this position paper, we claim that any task dealing with temporally ordered sequences of multiple images or frames is an instance of a broader, more general problem involving the understanding of intricate relationships between the visual content and the corresponding text. We comprehensively analyze five tasks that are instances of this problem and argue that they pose a common set of challenges and share similarities in terms of modeling and evaluation approaches. Based on the insights from these various aspects and stages of multi-image-to-text generation, we highlight several open questions and suggest future research directions. We believe that these directions can advance the understanding of complex phenomena in this domain and the development of better models. 

**Abstract (ZH)**: 自然语言描述多张图像的能力是人类智能的核心，并且是任何人工智能系统的关键特征。尽管已有研究集中在生成单张图像的文本描述上，但对多张图像视觉到文本转换的全面分析和工作进展关注相对较少。在本文中，我们提出，任何涉及时间顺序多张图像或帧的任务都是更广泛、更一般问题的一个实例，该问题涉及视觉内容与相应文本之间复杂关系的理解。我们全面分析了五个此类问题的具体任务，并认为它们面临着一组共同的挑战，并且在建模和评估方法上具有相似性。基于这些多图像到文本生成不同方面和阶段的见解，我们强调了若干开放问题，并建议未来的研究方向。我们认为，这些方向可以促进对该领域复杂现象的理解以及更优模型的发展。 

---
# Likelihood-Ratio Regularized Quantile Regression: Adapting Conformal Prediction to High-Dimensional Covariate Shifts 

**Title (ZH)**: 基于似然比正则化的分位数回归：适应高维协变量偏移的同伏预测调整 

**Authors**: Sunay Joshi, Shayan Kiyani, George Pappas, Edgar Dobriban, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2502.13030)  

**Abstract**: We consider the problem of conformal prediction under covariate shift. Given labeled data from a source domain and unlabeled data from a covariate shifted target domain, we seek to construct prediction sets with valid marginal coverage in the target domain. Most existing methods require estimating the unknown likelihood ratio function, which can be prohibitive for high-dimensional data such as images. To address this challenge, we introduce the likelihood ratio regularized quantile regression (LR-QR) algorithm, which combines the pinball loss with a novel choice of regularization in order to construct a threshold function without directly estimating the unknown likelihood ratio. We show that the LR-QR method has coverage at the desired level in the target domain, up to a small error term that we can control. Our proofs draw on a novel analysis of coverage via stability bounds from learning theory. Our experiments demonstrate that the LR-QR algorithm outperforms existing methods on high-dimensional prediction tasks, including a regression task for the Communities and Crime dataset, and an image classification task from the WILDS repository. 

**Abstract (ZH)**: 我们考虑在协变量偏移下的保函呈现值预测问题。给定源领域带标签的数据和目标领域协变量偏移的未标记数据，我们寻求构建在目标领域的有效边际覆盖预测集。现有大多数方法需要估计未知的似然比函数，在处理高维数据如图像时可能具有挑战性。为应对这一挑战，我们引入了似然比正则化分位数回归（LR-QR）算法，该算法结合了尖球损失和一种新颖的选择正则化方法，以构建阈值函数，而不直接估计未知的似然比。我们证明了LR-QR方法在其目标领域达到了所期望的覆盖率，而仅在可以控制的小误差项范围内有所不同。我们的证明基于学习理论中的一种新颖的覆盖分析和稳定性界。我们的实验表明，LR-QR算法在高维预测任务中优于现有方法，包括Communities and Crime数据集上的回归任务和WILDS仓库中的图像分类任务。 

---
# Time-series attribution maps with regularized contrastive learning 

**Title (ZH)**: 正则化对比学习下的时间序列归因图 

**Authors**: Steffen Schneider, Rodrigo González Laiz, Anastasiia Filippova, Markus Frey, Mackenzie Weygandt Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12977)  

**Abstract**: Gradient-based attribution methods aim to explain decisions of deep learning models but so far lack identifiability guarantees. Here, we propose a method to generate attribution maps with identifiability guarantees by developing a regularized contrastive learning algorithm trained on time-series data plus a new attribution method called Inverted Neuron Gradient (collectively named xCEBRA). We show theoretically that xCEBRA has favorable properties for identifying the Jacobian matrix of the data generating process. Empirically, we demonstrate robust approximation of zero vs. non-zero entries in the ground-truth attribution map on synthetic datasets, and significant improvements across previous attribution methods based on feature ablation, Shapley values, and other gradient-based methods. Our work constitutes a first example of identifiable inference of time-series attribution maps and opens avenues to a better understanding of time-series data, such as for neural dynamics and decision-processes within neural networks. 

**Abstract (ZH)**: 基于梯度的归因方法旨在解释深度学习模型的决策，但迄今为止缺乏可识别性保证。在这里，我们提出了一种通过在时间序列数据上开发正则化的对比学习算法并结合一种新的归因方法（称为反转神经元梯度，统称为xceBRA）来生成具有可识别性保证的归因图的方法。我们理论证明xceBRA具有识别数据生成过程雅可比矩阵的良好性质。在实验中，我们在合成数据集上展示了对真实归因图中零值与非零值的鲁棒近似，并显著改进了基于特征消融、Shapley值和其他基于梯度的方法的先前归因方法。我们的工作构成了时间序列归因图可识别推断的第一个示例，并为更好地理解时间序列数据，如神经动力学和神经网络内的决策过程提供了途径。 

---
# A Survey of Text Classification Under Class Distribution Shift 

**Title (ZH)**: 文本分类中的类别分布偏移综述 

**Authors**: Adriana Valentina Costache, Silviu Florin Gheorghe, Eduard Gabriel Poesina, Paul Irofti, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12965)  

**Abstract**: The basic underlying assumption of machine learning (ML) models is that the training and test data are sampled from the same distribution. However, in daily practice, this assumption is often broken, i.e.~the distribution of the test data changes over time, which hinders the application of conventional ML models. One domain where the distribution shift naturally occurs is text classification, since people always find new topics to discuss. To this end, we survey research articles studying open-set text classification and related tasks. We divide the methods in this area based on the constraints that define the kind of distribution shift and the corresponding problem formulation, i.e.~learning with the Universum, zero-shot learning, and open-set learning. We next discuss the predominant mitigation approaches for each problem setup. Finally, we identify several future work directions, aiming to push the boundaries beyond the state of the art. Interestingly, we find that continual learning can solve many of the issues caused by the shifting class distribution. We maintain a list of relevant papers at this https URL. 

**Abstract (ZH)**: 机器学习模型的基本假设是训练数据和测试数据来自相同的分布。然而，在实际应用中，这种假设常常被打破，即测试数据的分布会随时间变化，这阻碍了传统机器学习模型的应用。文本分类领域自然会遇到分布偏移的问题，因为人们总能找到新的讨论主题。为此，我们综述了研究开放集文本分类及相关任务的文章。我们将该领域的方法根据界定分布偏移类型及其相应的研究问题进行分类，即使用Universum学习、零样本学习和开放集学习。接着，我们讨论了每种问题设置的主要缓解方法。最后，我们确定了若干未来研究方向，旨在超越现有技术水平。有趣的是，我们发现连续学习可以解决由类分布偏移引起的一些问题。我们在此维护了一份相关论文的列表：https://this.url/ 

---
# Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text 

**Title (ZH)**: 基于掩码的任务导向反 curriculum 提高文本下游性能 

**Authors**: Andrei Jarca, Florinel Alin Croitoru, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12953)  

**Abstract**: Masked language modeling has become a widely adopted unsupervised technique to pre-train language models. However, the process of selecting tokens for masking is random, and the percentage of masked tokens is typically fixed for the entire training process. In this paper, we propose to adjust the masking ratio and to decide which tokens to mask based on a novel task-informed anti-curriculum learning scheme. First, we harness task-specific knowledge about useful and harmful tokens in order to determine which tokens to mask. Second, we propose a cyclic decaying masking ratio, which corresponds to an anti-curriculum schedule (from hard to easy). We exemplify our novel task-informed anti-curriculum by masking (TIACBM) approach across three diverse downstream tasks: sentiment analysis, text classification by topic, and authorship attribution. Our findings suggest that TIACBM enhances the ability of the model to focus on key task-relevant features, contributing to statistically significant performance gains across tasks. We release our code at this https URL. 

**Abstract (ZH)**: 基于任务信息的反梯度学习掩码语言模型方法 

---
# Keep what you need : extracting efficient subnetworks from large audio representation models 

**Title (ZH)**: 保留所需部分：从大型音频表示模型中提取高效子网络 

**Authors**: David Genova, Philippe Esling, Tom Hurlin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12925)  

**Abstract**: Recently, research on audio foundation models has witnessed notable advances, as illustrated by the ever improving results on complex downstream tasks. Subsequently, those pretrained networks have quickly been used for various audio applications. These improvements have however resulted in a considerable increase both in size and complexity of these models. Along the environmental concerns this issue raises, this prevents the deployment of such networks on consumer-level devices, and precludes their use for real-time applications. Moreover, this appears contradictory with the specificity of the tasks for which these models are used, which are often simpler compared to extracting a rich, multi-purpose representation from any type of audio data. In this paper, we address this issue with a simple, yet effective method to extract lightweight specialist subnetworks from large foundation models. Specifically, we introduce learnable binary masks in-between the layers of a pretrained representation model. When training the end-to-end model on a downstream task, we add a sparsity-inducing loss to the overall objective, hence learning a compact subnetwork specialized on a single task. Importantly, the weights of the foundation model are kept frozen, resulting into low additional training costs. Once trained, the masked computational units can then be removed from the network, implying significant performance gains. We assess our method on three widespread audio foundation models, each based on a different backbone architecture, and illustrate its effectiveness on common audio representation evaluation tasks, as well as its versatility on both speech, music, and general audio. Code for reproducing the results and supporting webpage are available at this https URL 

**Abstract (ZH)**: 近年来，音频基础模型的研究取得了显著进展，这体现在其在复杂下游任务上的持续改进结果上。随后，这些预训练网络被迅速应用于各种音频应用中。然而，这些改进导致了模型规模和复杂性的显著增加。这一问题不仅引发环境关切，还阻碍了在消费级设备上的部署，并限制了其用于实时应用。此外，这似乎与这些模型所使用的特定任务的复杂性相矛盾，后者的复杂性往往低于从任何类型音频数据中提取丰富、多功能表示的需求。在本文中，我们提出了一种简单而有效的方法，从大模型中提取轻量级的专业子网络。具体而言，我们引入了可学习的二值掩码，位于预训练表示模型的层之间。在针对下游任务训练端到端模型时，我们添加了一个促进稀疏性的损失函数，从而学习一个专门针对单一任务的精简子网络。重要的是，基础模型的权重保持冻结状态，从而降低了额外的训练成本。训练完成后，可以移除带有掩码的计算单元，这带来了显著的性能提升。我们在三个广泛使用的音频基础模型上评估了我们的方法，每个模型基于不同的骨干架构，并通过常见的音频表示评估任务展示了其有效性，以及在语音、音乐和通用音频上的 versatility。用于复现结果的代码和支持页面可在以下链接获取：this https URL。 

---
# Graph Neural Networks for Databases: A Survey 

**Title (ZH)**: 图神经网络在数据库中的应用：一个综述 

**Authors**: Ziming Li, Youhuan Li, Yuyu Luo, Guoliang Li, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12908)  

**Abstract**: Graph neural networks (GNNs) are powerful deep learning models for graph-structured data, demonstrating remarkable success across diverse domains. Recently, the database (DB) community has increasingly recognized the potentiality of GNNs, prompting a surge of researches focusing on improving database systems through GNN-based approaches. However, despite notable advances, There is a lack of a comprehensive review and understanding of how GNNs could improve DB systems. Therefore, this survey aims to bridge this gap by providing a structured and in-depth overview of GNNs for DB systems. Specifically, we propose a new taxonomy that classifies existing methods into two key categories: (1) Relational Databases, which includes tasks like performance prediction, query optimization, and text-to-SQL, and (2) Graph Databases, addressing challenges like efficient graph query processing and graph similarity computation. We systematically review key methods in each category, highlighting their contributions and practical implications. Finally, we suggest promising avenues for integrating GNNs into Database systems. 

**Abstract (ZH)**: 图神经网络（GNNs）是处理图结构数据的强大力量深学习模型，已在多个领域取得显著成果。最近，数据库（DB）社区越来越认识到GNNs的潜力，促使通过GNN基础方法改进数据库系统的研究呈上升趋势。然而，尽管取得了显著进展，但仍缺乏对GNNs如何改善数据库系统进行全面的综述和理解。因此，本文旨在通过提供GNNs在数据库系统中的结构化和深入概述来填补这一空白。具体而言，我们提出了一种新的分类法，将现有方法分为两大类：（1）关系型数据库，包括性能预测、查询优化和文本到SQL等任务；（2）图数据库，解决高效图查询处理和图相似性计算等挑战。我们系统地审查了每个类别的关键方法，强调了它们的贡献和实际影响。最后，我们提出了将GNNs集成到数据库系统中的有前途的途径。 

---
# Integrating Arithmetic Learning Improves Mathematical Reasoning in Smaller Models 

**Title (ZH)**: 整合算术学习提高小模型的数学推理能力 

**Authors**: Neeraj Gangwar, Suma P Bhat, Nickvash Kani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12855)  

**Abstract**: While large models pre-trained on high-quality data exhibit excellent performance across various reasoning tasks, including mathematical reasoning (e.g. GSM8k, MultiArith), specializing smaller models to excel at mathematical reasoning remains a challenging problem. Common approaches to address this challenge include knowledge distillation, where smaller student models learn from large pre-trained teacher models, and data augmentation, such as rephrasing questions. Despite these efforts, smaller models struggle with arithmetic computations, leading to errors in mathematical reasoning. In this work, we focus on leveraging a programmatically generated arithmetic dataset to enhance the reasoning capabilities of smaller models. We investigate two key approaches to incorporate this dataset -- (1) intermediate fine-tuning, where a model is fine-tuned on the arithmetic dataset before being trained on a reasoning dataset, and (2) integrating the arithmetic dataset into the instruction-tuning mixture, allowing the model to learn arithmetic skills alongside general instruction-following abilities. Our experiments on multiple reasoning benchmarks demonstrate that incorporating an arithmetic dataset, whether through targeted fine-tuning or within the instruction-tuning mixture, enhances the models' arithmetic capabilities, which in turn improves their mathematical reasoning performance. 

**Abstract (ZH)**: 尽管预训练于高质量数据的大模型在各种推理任务中表现出色，包括数学推理（如GSM8k、MultiArith），但使较小的模型在数学推理方面表现出色仍然是一个具有挑战性的问题。解决这一挑战的常见方法包括知识蒸馏，其中较小的学生模型从大型预训练教师模型中学习，以及数据增强，例如重新表述问题。尽管如此，较小的模型在进行算术计算时仍然存在问题，导致数学推理中的错误。在本工作中，我们重点利用程序生成的算术数据集来增强较小模型的推理能力。我们考察了两种关键方法来融入这个数据集——（1）中间微调，即在模型在推理数据集上训练之前先在算术数据集上进行微调；（2）将算术数据集整合到指令调优混合模型中，使模型在学习算术技能的同时也能够掌握一般的指令遵循能力。在多个推理基准上的实验表明，无论是通过定向微调还是将其整合到指令调优混合模型中，引入算术数据集都能提高模型的算术能力，进而提升其数学推理性能。 

---
# Envious Explore and Exploit 

**Title (ZH)**: 嫉妒性探索与利用 

**Authors**: Omer Ben-Porat, Yotam Gafni, Or Markovetzki  

**Link**: [PDF](https://arxiv.org/pdf/2502.12798)  

**Abstract**: Explore-and-exploit tradeoffs play a key role in recommendation systems (RSs), aiming at serving users better by learning from previous interactions. Despite their commercial success, the societal effects of explore-and-exploit mechanisms are not well understood, especially regarding the utility discrepancy they generate between different users. In this work, we measure such discrepancy using the economic notion of envy. We present a multi-armed bandit-like model in which every round consists of several sessions, and rewards are realized once per round. We call the latter property reward consistency, and show that the RS can leverage this property for better societal outcomes. On the downside, doing so also generates envy, as late-to-arrive users enjoy the information gathered by early-to-arrive users. We examine the generated envy under several arrival order mechanisms and virtually any anonymous algorithm, i.e., any algorithm that treats all similar users similarly without leveraging their identities. We provide tight envy bounds on uniform arrival and upper bound the envy for nudged arrival, in which the RS can affect the order of arrival by nudging its users. Furthermore, we study the efficiency-fairness trade-off by devising an algorithm that allows constant envy and approximates the optimal welfare in restricted settings. Finally, we validate our theoretical results empirically using simulations. 

**Abstract (ZH)**: 探索与利用权衡在推荐系统中的作用：通过学习先前的交互更好地服务用户，尽管探索与利用机制在商业上取得了成功，但它们对社会的影响尚未得到充分理解，特别是它们为不同用户带来的效用偏差。在本工作中，我们使用经济概念中的嫉妒度量这种偏差。我们提出了一种类似于多臂_bandit_模型，在其中每个轮次包含若干会话，并且奖励在每个轮次结束时实现。我们称后一种性质为奖励一致性，并表明推荐系统可以利用这一性质以实现更好的社会效益。然而，这样做也会导致嫉妒，因为后来到达的用户可以享受到早期到达的用户收集的信息。我们研究了在几种不同的到达顺序机制下生成的嫉妒，并考虑了任何不利用用户身份信息的任何匿名算法。我们为均匀到达提供了精确的嫉妒边界，并为通过影响到达顺序推荐系统生成的嫉妒提供了上界。此外，我们通过设计一个允许恒定嫉妒并近似在受限条件下最优福利的算法研究了效率与公平之间的权衡。最后，我们通过仿真验证了我们的理论结果。 

---
# Unsupervised Anomaly Detection through Mass Repulsing Optimal Transport 

**Title (ZH)**: 无监督异常检测通过质量排斥最优传输 

**Authors**: Eduardo Fernandes Montesuma, Adel El Habazi, Fred Ngole Mboula  

**Link**: [PDF](https://arxiv.org/pdf/2502.12793)  

**Abstract**: Detecting anomalies in datasets is a longstanding problem in machine learning. In this context, anomalies are defined as a sample that significantly deviates from the remaining data. Meanwhile, optimal transport (OT) is a field of mathematics concerned with the transportation, between two probability measures, at least effort. In classical OT, the optimal transportation strategy of a measure to itself is the identity. In this paper, we tackle anomaly detection by forcing samples to displace its mass, while keeping the least effort objective. We call this new transportation problem Mass Repulsing Optimal Transport (MROT). Naturally, samples lying in low density regions of space will be forced to displace mass very far, incurring a higher transportation cost. We use these concepts to design a new anomaly score. Through a series of experiments in existing benchmarks, and fault detection problems, we show that our algorithm improves over existing methods. 

**Abstract (ZH)**: 在机器学习中检测数据集中的异常是一个长期存在的问题。在这一背景下，异常被定义为显著偏离其余数据的一个样本。同时，最优运输（OT）是一门数学领域，关注的是在两个概率测度之间的运输，以最小的努力进行。在古典最优运输中，一个测度到自身的最优运输策略是恒等映射。在本文中，我们通过强制样本移动其质量，同时保持最小努力目标来解决异常检测问题。我们将这种新的运输问题称为质量排斥最优运输（MROT）。自然地，位于空间低密度区域的样本将被迫移动很远的质量，导致更高的运输成本。我们利用这些概念设计了一个新的异常值评分方法。通过一系列在现有基准和故障检测问题中的实验，我们展示了我们的算法优于现有方法。 

---
# Evaluating link prediction: New perspectives and recommendations 

**Title (ZH)**: 评价链接预测：新视角与建议 

**Authors**: Bhargavi Kalyani I, A Rama Prasad Mathi, Niladri Sett  

**Link**: [PDF](https://arxiv.org/pdf/2502.12777)  

**Abstract**: Link prediction (LP) is an important problem in network science and machine learning research. The state-of-the-art LP methods are usually evaluated in a uniform setup, ignoring several factors associated with the data and application specific needs. We identify a number of such factors, such as, network-type, problem-type, geodesic distance between the end nodes and its distribution over the classes, nature and applicability of LP methods, class imbalance and its impact on early retrieval, evaluation metric, etc., and present an experimental setup which allows us to evaluate LP methods in a rigorous and controlled manner. We perform extensive experiments with a variety of LP methods over real network datasets in this controlled setup, and gather valuable insights on the interactions of these factors with the performance of LP through an array of carefully designed hypotheses. Following the insights, we provide recommendations to be followed as best practice for evaluating LP methods. 

**Abstract (ZH)**: 链接预测（LP）是网络科学和机器学习研究中的一个重要问题。最前沿的LP方法通常在统一的评估框架下进行评估，忽视了与数据和应用需求相关的多个因素。我们识别出若干此类因素，如网络类型、问题类型、端节点的测地距离及其在类别的分布、LP方法的性质及其适用性、类不平衡及其对早期检索的影响、评估指标等，并提出一个严格的控制实验框架，以评估LP方法。我们在这一控制框架下使用多种LP方法对实际网络数据集进行广泛的实验，并通过一系列精心设计的假设，收集了关于这些因素与LP性能之间相互作用的宝贵见解。基于这些见解，我们提供了作为最佳实践遵循的评估LP方法的建议。 

---
# Beyond Seen Data: Improving KBQA Generalization Through Schema-Guided Logical Form Generation 

**Title (ZH)**: 超越已见数据：通过基于方案的逻辑形式生成提高KBQA泛化能力 

**Authors**: Shengxiang Gao, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12737)  

**Abstract**: Knowledge base question answering (KBQA) aims to answer user questions in natural language using rich human knowledge stored in large KBs. As current KBQA methods struggle with unseen knowledge base elements at test time,we introduce SG-KBQA: a novel model that injects schema contexts into entity retrieval and logical form generation to tackle this issue. It uses the richer semantics and awareness of the knowledge base structure provided by schema contexts to enhance generalizability. We show that SG-KBQA achieves strong generalizability, outperforming state-of-the-art models on two commonly used benchmark datasets across a variety of test settings. Code will be released upon paper publication. 

**Abstract (ZH)**: 基于知识库的问答（KBQA）旨在利用大型知识库中存储的丰富人类知识以自然语言回答用户问题。由于当前的KBQA方法在测试时难以处理未见过的知识库元素，我们提出了SG-KBQA：一种将模式上下文注入实体检索和逻辑形式生成的新模型，以应对这一问题。通过利用模式上下文提供的更丰富的语义和知识库结构意识，提高其泛化能力。实验表明，SG-KBQA 在泛化能力方面表现出色，在两个常用基准数据集的多种测试设置下均优于现有最佳模型。论文发表后将释放代码。 

---
# TREND: A Whitespace Replacement Information Hiding Method 

**Title (ZH)**: 趋势：一种空白字符替换信息隐藏方法 

**Authors**: Malte Hellmeier, Hendrik Norkowski, Ernst-Christoph Schrewe, Haydar Qarawlus, Falk Howar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12710)  

**Abstract**: Large Language Models (LLMs) have gained significant popularity in recent years. Differentiating between a text written by a human and a text generated by an LLM has become almost impossible. Information hiding techniques such as digital watermarking or steganography can help by embedding information inside text without being noticed. However, existing techniques, such as linguistic-based or format-based methods, change the semantics or do not work on pure, unformatted text. In this paper, we introduce a novel method for information hiding termed TREND, which is able to conceal any byte-encoded sequence within a cover text. The proposed method is implemented as a multi-platform library using the Kotlin programming language, accompanied by a command-line tool and a web interface provided as examples of usage. By substituting conventional whitespace characters with visually similar Unicode whitespace characters, our proposed scheme preserves the semantics of the cover text without increasing the number of characters. Furthermore, we propose a specified structure for secret messages that enables configurable compression, encryption, hashing, and error correction. Our experimental benchmark comparison on a dataset of one million Wikipedia articles compares ten algorithms from literature and practice. It proves the robustness of our proposed method in various applications while remaining imperceptible to humans. We discuss the limitations of limited embedding capacity and further robustness, which guide implications for future work. 

**Abstract (ZH)**: 大型语言模型中的信息隐藏方法：TREND 

---
# Fast Data Aware Neural Architecture Search via Supernet Accelerated Evaluation 

**Title (ZH)**: 基于超网络加速评估的快速数据感知神经架构搜索 

**Authors**: Emil Njor, Colby Banbury, Xenofon Fafoutis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12690)  

**Abstract**: Tiny machine learning (TinyML) promises to revolutionize fields such as healthcare, environmental monitoring, and industrial maintenance by running machine learning models on low-power embedded systems. However, the complex optimizations required for successful TinyML deployment continue to impede its widespread adoption. A promising route to simplifying TinyML is through automatic machine learning (AutoML), which can distill elaborate optimization workflows into accessible key decisions. Notably, Hardware Aware Neural Architecture Searches - where a computer searches for an optimal TinyML model based on predictive performance and hardware metrics - have gained significant traction, producing some of today's most widely used TinyML models. Nevertheless, limiting optimization solely to neural network architectures can prove insufficient. Because TinyML systems must operate under extremely tight resource constraints, the choice of input data configuration, such as resolution or sampling rate, also profoundly impacts overall system efficiency. Achieving truly optimal TinyML systems thus requires jointly tuning both input data and model architecture. Despite its importance, this "Data Aware Neural Architecture Search" remains underexplored. To address this gap, we propose a new state-of-the-art Data Aware Neural Architecture Search technique and demonstrate its effectiveness on the novel TinyML ``Wake Vision'' dataset. Our experiments show that across varying time and hardware constraints, Data Aware Neural Architecture Search consistently discovers superior TinyML systems compared to purely architecture-focused methods, underscoring the critical role of data-aware optimization in advancing TinyML. 

**Abstract (ZH)**: 基于数据感知的神经架构搜索在TinyML中的应用 

---
# Multi-Step Alignment as Markov Games: An Optimistic Online Gradient Descent Approach with Convergence Guarantees 

**Title (ZH)**: 多步对齐作为马尔可夫博弈：具有收敛保证的乐观在线梯度下降方法 

**Authors**: Yongtao Wu, Luca Viano, Yihang Chen, Zhenyu Zhu, Kimon Antonakopoulos, Quanquan Gu, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2502.12678)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has been highly successful in aligning large language models with human preferences. While prevalent methods like DPO have demonstrated strong performance, they frame interactions with the language model as a bandit problem, which limits their applicability in real-world scenarios where multi-turn conversations are common. Additionally, DPO relies on the Bradley-Terry model assumption, which does not adequately capture the non-transitive nature of human preferences. In this paper, we address these challenges by modeling the alignment problem as a two-player constant-sum Markov game, where each player seeks to maximize their winning rate against the other across all steps of the conversation. Our approach Multi-step Preference Optimization (MPO) is built upon the natural actor-critic framework~\citep{peters2008natural}. We further develop OMPO based on the optimistic online gradient descent algorithm~\citep{rakhlin2013online,joulani17a}. Theoretically, we provide a rigorous analysis for both algorithms on convergence and show that OMPO requires $\mathcal{O}(\epsilon^{-1})$ policy updates to converge to an $\epsilon$-approximate Nash equilibrium. We also validate the effectiveness of our method on multi-turn conversations dataset and math reasoning dataset. 

**Abstract (ZH)**: 基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）在使大型语言模型与人类偏好一致方面取得了巨大成功。尽管像DPO这样的广泛方法展示了强大的性能，但它们将与语言模型的交互视为一个拉普拉斯问题，这限制了它们在常见多轮对话的现实场景中的适用性。此外，DPO依赖于Bradley-Terry模型假设，该假设不能充分捕捉人类偏好中的非传递性。在这篇论文中，我们通过将对齐问题建模为两名玩家的常和马尔可夫游戏来解决这些挑战，在该游戏中，每个玩家试图在整个对话步骤中最大限度地提高自己相对于另一方的胜率。我们的方法多步偏好优化（Multi-step Preference Optimization, MPO）基于自然演员-评论家框架~\citep{peters2008natural}。我们在此基础上进一步开发了OMPO，基于乐观在线梯度下降算法~\citep{rakhlin2013online,joulani17a}。理论上，我们对这两种算法的收敛性进行了严格的分析，并证明了OMPO需要$\mathcal{O}(\epsilon^{-1})$次策略更新以收敛到$\epsilon$-近似纳什均衡。我们还在多轮对话数据集和数学推理数据集上验证了我们方法的有效性。 

---
# Speech-FT: A Fine-tuning Strategy for Enhancing Speech Representation Models Without Compromising Generalization Ability 

**Title (ZH)**: Speech-FT: 一种在不牺牲泛化能力的情况下增强语音表示模型的微调策略 

**Authors**: Tzu-Quan Lin, Wei-Ping Huang, Hao Tang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12672)  

**Abstract**: Speech representation models are highly effective at extracting general features for various tasks. While fine-tuning can enhance these representations for specific applications, it often compromises their generalization ability. To address this challenge, we propose Speech-FT, a fine-tuning strategy for speech representation models that leverages model merging to preserve generalization ability while still benefiting from fine-tuning. Speech-FT is effective across different fine-tuning scenarios and is compatible with various types of speech representation models, providing a versatile solution. Speech-FT offers an efficient and practical approach to further improving general speech representations after pre-training. 

**Abstract (ZH)**: 基于模型融合的Speech-FT：一种保持泛化能力的同时增强特定应用的语音表示微调策略 

---
# Label Drop for Multi-Aspect Relation Modeling in Universal Information Extraction 

**Title (ZH)**: 面向通用信息提取的多方面关系建模中的标签掉落方法 

**Authors**: Lu Yang, Jiajia Li, En Ci, Lefei Zhang, Zuchao Li, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12614)  

**Abstract**: Universal Information Extraction (UIE) has garnered significant attention due to its ability to address model explosion problems effectively. Extractive UIE can achieve strong performance using a relatively small model, making it widely adopted. Extractive UIEs generally rely on task instructions for different tasks, including single-target instructions and multiple-target instructions. Single-target instruction UIE enables the extraction of only one type of relation at a time, limiting its ability to model correlations between relations and thus restricting its capability to extract complex relations. While multiple-target instruction UIE allows for the extraction of multiple relations simultaneously, the inclusion of irrelevant relations introduces decision complexity and impacts extraction accuracy. Therefore, for multi-relation extraction, we propose LDNet, which incorporates multi-aspect relation modeling and a label drop mechanism. By assigning different relations to different levels for understanding and decision-making, we reduce decision confusion. Additionally, the label drop mechanism effectively mitigates the impact of irrelevant relations. Experiments show that LDNet outperforms or achieves competitive performance with state-of-the-art systems on 9 tasks, 33 datasets, in both single-modal and multi-modal, few-shot and zero-shot settings.\footnote{this https URL} 

**Abstract (ZH)**: 通用信息提取（UIE）因其有效解决模型爆炸问题的能力而引起了广泛关注。提取式UIE可以使用相对较小的模型实现较强的性能，因此被广泛采用。提取式UIEs通常依赖于不同的任务指令，包括单目标指令和多目标指令。单目标指令的UIE一次只能提取一种关系，限制了其建模关系之间关联的能力，从而限制了其提取复杂关系的能力。而多目标指令的UIE可以同时提取多种关系，但由于包含无关关系，增加了决策复杂度并影响提取精度。因此，为了进行多关系提取，我们提出了一种LDNet方法，该方法结合了多方面关系建模和标签丢弃机制。通过将不同关系分配到不同的层级进行理解和决策，我们减少了决策混淆。此外，标签丢弃机制有效地减轻了无关关系的影响。实验结果显示，LDNet在9个任务、33个数据集中，在单模态和多模态、少样本和零样本设置中均优于或达到了最先进的系统水平。 

---
# Unveiling Mode Connectivity in Graph Neural Networks 

**Title (ZH)**: 揭示图神经网络中的模式连通性 

**Authors**: Bingheng Li, Zhikai Chen, Haoyu Han, Shenglai Zeng, Jingzhe Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12608)  

**Abstract**: A fundamental challenge in understanding graph neural networks (GNNs) lies in characterizing their optimization dynamics and loss landscape geometry, critical for improving interpretability and robustness. While mode connectivity, a lens for analyzing geometric properties of loss landscapes has proven insightful for other deep learning architectures, its implications for GNNs remain unexplored. This work presents the first investigation of mode connectivity in GNNs. We uncover that GNNs exhibit distinct non-linear mode connectivity, diverging from patterns observed in fully-connected networks or CNNs. Crucially, we demonstrate that graph structure, rather than model architecture, dominates this behavior, with graph properties like homophily correlating with mode connectivity patterns. We further establish a link between mode connectivity and generalization, proposing a generalization bound based on loss barriers and revealing its utility as a diagnostic tool. Our findings further bridge theoretical insights with practical implications: they rationalize domain alignment strategies in graph learning and provide a foundation for refining GNN training paradigms. 

**Abstract (ZH)**: 理解图神经网络优化动力学和损失landscape几何结构的基本挑战在于提高其可解释性和鲁棒性。虽然模式连通性作为一种分析损失landscape几何性质的视角在其他深度学习架构中展现了洞见，但其对图神经网络的影响尚未被探索。本工作首次调查了图神经网络中的模式连通性。我们发现图神经网络表现出与完全连接网络或CNNs中观察到的模式不同的非线性模式连通性。关键的是，我们证明了图结构而非模型架构主导了这种行为，图属性如同质性与模式连通性模式相关。我们还建立了模式连通性与泛化的联系，提出基于损失屏障的泛化界，并揭示其作为诊断工具的实用性。我们的发现进一步将理论洞察与实际应用联系起来：它们为图学习中的领域对齐策略提供了理据，并为改进图神经网络训练范式奠定了基础。 

---
# Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting 

**Title (ZH)**: 在未知干预下的长短期状态解耦在线时间序列预测 

**Authors**: Ruichu Cai, Haiqin Huang, Zhifang Jiang, Zijian Li, Changze Zhou, Yuequn Liu, Yuming Liu, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12603)  

**Abstract**: Current methods for time series forecasting struggle in the online scenario, since it is difficult to preserve long-term dependency while adapting short-term changes when data are arriving sequentially. Although some recent methods solve this problem by controlling the updates of latent states, they cannot disentangle the long/short-term states, leading to the inability to effectively adapt to nonstationary. To tackle this challenge, we propose a general framework to disentangle long/short-term states for online time series forecasting. Our idea is inspired by the observations where short-term changes can be led by unknown interventions like abrupt policies in the stock market. Based on this insight, we formalize a data generation process with unknown interventions on short-term states. Under mild assumptions, we further leverage the independence of short-term states led by unknown interventions to establish the identification theory to achieve the disentanglement of long/short-term states. Built on this theory, we develop a long short-term disentanglement model (LSTD) to extract the long/short-term states with long/short-term encoders, respectively. Furthermore, the LSTD model incorporates a smooth constraint to preserve the long-term dependencies and an interrupted dependency constraint to enforce the forgetting of short-term dependencies, together boosting the disentanglement of long/short-term states. Experimental results on several benchmark datasets show that our \textbf{LSTD} model outperforms existing methods for online time series forecasting, validating its efficacy in real-world applications. 

**Abstract (ZH)**: 一种在线时间序列预测的长短期状态解耦框架 

---
# Enhancing Semi-supervised Learning with Noisy Zero-shot Pseudolabels 

**Title (ZH)**: 增强半监督学习中的 noisy 零-shot 假标签 

**Authors**: Jichan Chung, Irene Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12584)  

**Abstract**: Semi-supervised learning (SSL) leverages limited labeled data alongside abundant unlabeled data to address labeling costs in machine learning. While recent foundation models enable zero-shot inference, attempts to integrate these capabilities into SSL through pseudo-labeling have shown mixed results due to unreliable zero-shot predictions. We present ZMT (Zero-Shot Multi-Task Learning), a framework that jointly optimizes zero-shot pseudo-labels and unsupervised representation learning objectives from contemporary SSL approaches. Our method introduces a multi-task learning-based mechanism that incorporates pseudo-labels while ensuring robustness to varying pseudo-label quality. Experiments across 8 datasets in vision, language, and audio domains demonstrate that ZMT reduces error by up to 56% compared to traditional SSL methods, with particularly compelling results when pseudo-labels are noisy and unreliable. ZMT represents a significant step toward making semi-supervised learning more effective and accessible in resource-constrained environments. 

**Abstract (ZH)**: 半监督学习（SSL）利用有限的标注数据和丰富的未标注数据来解决机器学习中的标注成本问题。尽管最近的预训练模型能够实现零-shot推理，但试图通过伪标签将这些能力与SSL结合所产生的结果参差不齐，因为这些零-shot预测存在不可靠性。我们提出了ZMT（零-shot多任务学习），这是一种框架，可以同时优化零-shot伪标签和来自现代SSL方法的无监督表示学习目标。我们的方法引入了一种基于多任务学习的机制，该机制在确保伪标签质量变化时的鲁棒性的同时引入了伪标签。在视觉、语言和音频领域8个数据集上的实验表明，与传统SSL方法相比，ZMT可以将错误率降低多达56%，特别是在伪标签嘈杂和不可靠时表现尤为明显。ZMT代表了朝着使半监督学习在资源受限环境中更加有效和可行的重要一步。 

---
# The Majority Vote Paradigm Shift: When Popular Meets Optimal 

**Title (ZH)**: 多数投票范式的转变：流行与最优的交汇 

**Authors**: Antonio Purificato, Maria Sofia Bucarelli, Anil Kumar Nelakanti, Andrea Bacciu, Fabrizio Silvestri, Amin Mantrach  

**Link**: [PDF](https://arxiv.org/pdf/2502.12581)  

**Abstract**: Reliably labelling data typically requires annotations from multiple human workers. However, humans are far from being perfect. Hence, it is a common practice to aggregate labels gathered from multiple annotators to make a more confident estimate of the true label. Among many aggregation methods, the simple and well known Majority Vote (MV) selects the class label polling the highest number of votes. However, despite its importance, the optimality of MV's label aggregation has not been extensively studied. We address this gap in our work by characterising the conditions under which MV achieves the theoretically optimal lower bound on label estimation error. Our results capture the tolerable limits on annotation noise under which MV can optimally recover labels for a given class distribution. This certificate of optimality provides a more principled approach to model selection for label aggregation as an alternative to otherwise inefficient practices that sometimes include higher experts, gold labels, etc., that are all marred by the same human uncertainty despite huge time and monetary costs. Experiments on both synthetic and real world data corroborate our theoretical findings. 

**Abstract (ZH)**: 可靠地标注数据通常需要多个人工工作者的注释。然而，人类远非完美。因此，汇总多名注释者获得的标签以更自信地估计真实标签是一种常见做法。在众多汇总方法中，简单的多数投票（MV）方法选择获得最高票数的类别标签。尽管其重要性不容忽视，但MV标签汇总的最优性尚未得到充分研究。我们在工作中通过分析MV在哪些条件下能达到标签估计误差的理论最优下界，填补了这一空白。我们的结果捕捉了在给定类别分布下MV能最优恢复标签的可容忍注释噪声的限度。这种最优性证书为标签汇总的模型选择提供了一种更原则的方法，替代以往有时包括更高专家、黄金标签等虽然成本高昂但同样存在人类不确定性的做法。我们在合成数据和真实数据上的实验结果验证了我们的理论发现。 

---
# A Fuzzy Evaluation of Sentence Encoders on Grooming Risk Classification 

**Title (ZH)**: 模糊评价句编码器在 grooming 风险分类中的表现 

**Authors**: Geetanjali Bihani, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12576)  

**Abstract**: With the advent of social media, children are becoming increasingly vulnerable to the risk of grooming in online settings. Detecting grooming instances in an online conversation poses a significant challenge as the interactions are not necessarily sexually explicit, since the predators take time to build trust and a relationship with their victim. Moreover, predators evade detection using indirect and coded language. While previous studies have fine-tuned Transformers to automatically identify grooming in chat conversations, they overlook the impact of coded and indirect language on model predictions, and how these align with human perceptions of grooming. In this paper, we address this gap and evaluate bi-encoders on the task of classifying different degrees of grooming risk in chat contexts, for three different participant groups, i.e. law enforcement officers, real victims, and decoys. Using a fuzzy-theoretic framework, we map human assessments of grooming behaviors to estimate the actual degree of grooming risk. Our analysis reveals that fine-tuned models fail to tag instances where the predator uses indirect speech pathways and coded language to evade detection. Further, we find that such instances are characterized by a higher presence of out-of-vocabulary (OOV) words in samples, causing the model to misclassify. Our findings highlight the need for more robust models to identify coded language from noisy chat inputs in grooming contexts. 

**Abstract (ZH)**: 随着社交媒体的兴起，儿童在在线环境中面临的诱骗风险越来越高。检测在线对话中的诱骗实例具有显著挑战性，因为互动不一定包含性暗示，因为贩子会花时间建立信任和与受害者的关系。此外，贩子利用间接和编码语言规避检测。尽管以往的研究已将Transformer微调以自动识别聊天对话中的诱骗行为，但它们忽视了编码和间接语言对模型预测的影响，以及这些影响与人类对诱骗行为的认知如何一致。在本文中，我们弥补了这一空白，并评估双编码器在确定不同参与者组（即执法官员、真实受害者和诱饵）在聊天情境中面临的不同程度诱骗风险方面的性能。利用模糊理论框架，我们将人类对诱骗行为的评估映射到估计实际的诱骗风险程度。我们的分析表明，微调后的模型无法识别贩子利用间接言辞路径和编码语言规避检测的实例。此外，我们发现这类实例表征出样本文本中未登录词（OOV词）的更高出现频率，导致模型误分类。本文的研究结果强调了在诱骗情境中需要更 robust 的模型来识别来自嘈杂聊天输入的编码语言的重要性。 

---
# Improving the Stability of GNN Force Field Models by Reducing Feature Correlation 

**Title (ZH)**: 通过降低特征相关性提高GNN力场模型的稳定性 

**Authors**: Yujie Zeng, Wenlong He, Ihor Vasyltsov, Jiaxin Wei, Ying Zhang, Lin Chen, Yuehua Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.12548)  

**Abstract**: Recently, Graph Neural Network based Force Field (GNNFF) models are widely used in Molecular Dynamics (MD) simulation, which is one of the most cost-effective means in semiconductor material research. However, even such models provide high accuracy in energy and force Mean Absolute Error (MAE) over trained (in-distribution) datasets, they often become unstable during long-time MD simulation when used for out-of-distribution datasets. In this paper, we propose a feature correlation based method for GNNFF models to enhance the stability of MD simulation. We reveal the negative relationship between feature correlation and the stability of GNNFF models, and design a loss function with a dynamic loss coefficient scheduler to reduce edge feature correlation that can be applied in general GNNFF training. We also propose an empirical metric to evaluate the stability in MD simulation. Experiments show our method can significantly improve stability for GNNFF models especially in out-of-distribution data with less than 3% computational overhead. For example, we can ensure the stable MD simulation time from 0.03ps to 10ps for Allegro model. 

**Abstract (ZH)**: 基于特征相关性的图神经网络力场模型在分子动力学模拟中的稳定性增强方法 

---
# Computing Voting Rules with Improvement Feedback 

**Title (ZH)**: 基于改进反馈的投票规则计算 

**Authors**: Evi Micha, Vasilis Varsamis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12542)  

**Abstract**: Aggregating preferences under incomplete or constrained feedback is a fundamental problem in social choice and related domains. While prior work has established strong impossibility results for pairwise comparisons, this paper extends the inquiry to improvement feedback, where voters express incremental adjustments rather than complete preferences. We provide a complete characterization of the positional scoring rules that can be computed given improvement feedback. Interestingly, while plurality is learnable under improvement feedback--unlike with pairwise feedback--strong impossibility results persist for many other positional scoring rules. Furthermore, we show that improvement feedback, unlike pairwise feedback, does not suffice for the computation of any Condorcet-consistent rule. We complement our theoretical findings with experimental results, providing further insights into the practical implications of improvement feedback for preference aggregation. 

**Abstract (ZH)**: 在不完整或受约束反馈下聚合偏好是社会选择及相关领域中的一个基本问题。虽然先前的工作在成对比较方面建立了强有力的不可能结果，但本文将研究扩展到改进反馈，其中选民表达增量调整而非完整的偏好。我们提供了在给定改进反馈的情况下可以计算的职位评分规则的完整characterization。有趣的是，虽然在改进反馈下可以学习 plurality（占优），而不同于成对反馈，许多其他职位评分规则仍然面临着强大的不可能结果。此外，我们证明了改进反馈不足以计算任何 Condorcet-一致规则，而不同于成对反馈。我们通过理论发现和实验结果进一步探讨了改进反馈对偏好聚合的实用影响。 

---
# Finding Optimal Trading History in Reinforcement Learning for Stock Market Trading 

**Title (ZH)**: 在股票市场交易中通过强化学习寻找最优交易历史 

**Authors**: Sina Montazeria, Haseebullah Jumakhanb, Amir Mirzaeinia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12537)  

**Abstract**: This paper investigates the optimization of temporal windows in Financial Deep Reinforcement Learning (DRL) models using 2D Convolutional Neural Networks (CNNs). We introduce a novel approach to treating the temporal field as a hyperparameter and examine its impact on model performance across various datasets and feature arrangements. We introduce a new hyperparameter for the CNN policy, proposing that this temporal field can and should be treated as a hyperparameter for these models. We examine the significance of this temporal field by iteratively expanding the window of observations presented to the CNN policy during the deep reinforcement learning process. Our iterative process involves progressively increasing the observation period from two weeks to twelve weeks, allowing us to examine the effects of different temporal windows on the model's performance. This window expansion is implemented in two settings. In one setting, we rearrange the features in the dataset to group them by company, allowing the model to have a full view of company data in its observation window and CNN kernel. In the second setting, we do not group the features by company, and features are arranged by category. Our study reveals that shorter temporal windows are most effective when no feature rearrangement to group per company is in effect. However, the model will utilize longer temporal windows and yield better performance once we introduce the feature rearrangement. To examine the consistency of our findings, we repeated our experiment on two datasets containing the same thirty companies from the Dow Jones Index but with different features in each dataset and consistently observed the above-mentioned patterns. The result is a trading model significantly outperforming global financial services firms such as the Global X Guru by the established Mirae Asset. 

**Abstract (ZH)**: 本文研究了在金融深度强化学习（DRL）模型中使用2D卷积神经网络（CNNs）优化时间窗口的问题。我们引入了一种新颖的方法将时间场视为超参数，并考察了它在不同数据集和特征排列下的模型性能影响。我们为CNN策略引入了一个新的超参数，建议时间场可以并且应该被视为这些模型的超参数。通过逐步扩大呈现给CNN策略观察的时间窗口，我们反复检查了不同时间窗口对模型性能的影响。这一扩增过程在两种设置中实现。在一种设置中，我们将数据集中的特征重新排列以按公司分组，使模型在其观察窗口和CNN核中能够全面查看公司数据。在另一种设置中，我们不按公司对特征进行分组，而是按类别对特征进行排列。我们的研究表明，在未对特征进行重新排列以按公司分组的情况下，较短的时间窗口最有效。但在引入特征重新排列后，模型将使用较长的时间窗口并获得更好的性能。为了验证我们的发现的一致性，我们在包含道琼斯指数中相同三十家公司的两个数据集中重复了实验，并且观察到了相同模式。结果是一个交易模型显著优于包括全球X导师在内的全球金融服务公司，由Mirae Asset公司验证。 

---
# Myna: Masking-Based Contrastive Learning of Musical Representations 

**Title (ZH)**: Myna：基于遮掩的对比学习音乐表示方法 

**Authors**: Ori Yonay, Tracy Hammond, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12511)  

**Abstract**: We present Myna, a simple yet effective approach for self-supervised musical representation learning. Built on a contrastive learning framework, Myna introduces two key innovations: (1) the use of a Vision Transformer (ViT) on mel-spectrograms as the backbone and (2) a novel data augmentation strategy, token masking, that masks 90 percent of spectrogram tokens. These innovations deliver both effectiveness and efficiency: (i) Token masking enables a significant increase in per-GPU batch size, from 48 or 120 in prior methods (CLMR, MULE) to 4096. (ii) By avoiding traditional augmentations, Myna retains pitch sensitivity, enhancing performance in tasks like key detection. (iii) The use of vertical patches allows the model to better capture critical features for key detection. Our hybrid model, Myna-22M-Hybrid, processes both 16x16 and 128x2 patches, achieving state-of-the-art results. Trained on a single GPU, it outperforms MULE (62M) on average and rivals MERT-95M, which was trained on 16 and 64 GPUs, respectively. Additionally, it surpasses MERT-95M-public, establishing itself as the best-performing model trained on publicly available data. We release our code and models to promote reproducibility and facilitate future research. 

**Abstract (ZH)**: 我们提出Myna，一种简单而有效的自监督音乐表示学习方法。基于对比学习框架，Myna引入了两项关键创新：（1）使用音色图上的视觉变换器（ViT）作为骨干网络；（2）一种新的数据增强策略——token掩蔽，掩蔽了音色图上90%的token。这些创新在有效性和效率上都取得了显著效果：（i）token掩蔽使得每块GPU的批次大小显著增加，从先前方法（CLMR, MULE）的48或120增加到4096。（ii）通过避免传统的数据增强，Myna保留了音高敏感性，提升了调性检测等任务的性能。（iii）使用垂直补丁使模型能够更好地捕捉关键特征以进行调性检测。我们的混合模型Myna-22M-Hybrid同时处理16x16和128x2补丁，实现了最先进的结果。在单块GPU上训练时，其平均性能优于MULE（62M），并且接近MERT-95M，后者在16和64块GPU上分别训练。此外，它还超越了MERT-95M-public，成为在公开可用数据上训练性能最佳的模型。我们发布我们的代码和模型以促进可重复性和未来研究。 

---
# Mixture of Attention Yields Accurate Results for Tabular Data 

**Title (ZH)**: 混合注意力机制适用于表格数据的准确结果 

**Authors**: Xuechen Li, Yupeng Li, Jian Liu, Xiaolin Jin, Tian Yang, Xin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12507)  

**Abstract**: Tabular data inherently exhibits significant feature heterogeneity, but existing transformer-based methods lack specialized mechanisms to handle this property. To bridge the gap, we propose MAYA, an encoder-decoder transformer-based framework. In the encoder, we design a Mixture of Attention (MOA) that constructs multiple parallel attention branches and averages the features at each branch, effectively fusing heterogeneous features while limiting parameter growth. Additionally, we employ collaborative learning with a dynamic consistency weight constraint to produce more robust representations. In the decoder stage, cross-attention is utilized to seamlessly integrate tabular data with corresponding label features. This dual-attention mechanism effectively captures both intra-instance and inter-instance interactions. We evaluate the proposed method on a wide range of datasets and compare it with other state-of-the-art transformer-based methods. Extensive experiments demonstrate that our model achieves superior performance among transformer-based methods in both tabular classification and regression tasks. 

**Abstract (ZH)**: Tabular 数据固有地表现出显著的特征异质性，但现有的基于变压器的方法缺乏专门处理这一特性的机制。为了弥合这一差距，我们提出了 MAYA，一个基于编码器-解码器的变压器框架。在编码器中，我们设计了混合注意力（MOA），构建了多个并行的注意力分支并在每个分支上平均特征，从而有效地融合异质特征同时限制参数增长。此外，我们采用了具有动态一致性权重约束的合作学习，以生成更稳健的表示。在解码阶段，我们使用交叉注意力无缝地将表结构数据与相应的标签特征集成起来。这种双注意力机制有效地捕捉了实例内和实例间的交互。我们使用多种数据集评估了提出的模型，并将其与其它最先进的基于变压器的方法进行了比较。广泛的实验表明，我们的模型在表结构分类和回归任务中均表现出优于基于变压器方法的性能。 

---
# LocalEscaper: A Weakly-supervised Framework with Regional Reconstruction for Scalable Neural TSP Solvers 

**Title (ZH)**: LocalEscaper: 一种基于区域重建的弱监督框架，用于可扩展的神经TSP求解器 

**Authors**: Junrui Wen, Yifei Li, Bart Selman, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2502.12484)  

**Abstract**: Neural solvers have shown significant potential in solving the Traveling Salesman Problem (TSP), yet current approaches face significant challenges. Supervised learning (SL)-based solvers require large amounts of high-quality labeled data, while reinforcement learning (RL)-based solvers, though less dependent on such data, often suffer from inefficiencies. To address these limitations, we propose LocalEscaper, a novel weakly-supervised learning framework for large-scale TSP. LocalEscaper effectively combines the advantages of both SL and RL, enabling effective training on datasets with low-quality labels. To further enhance solution quality, we introduce a regional reconstruction strategy, which mitigates the problem of local optima, a common issue in existing local reconstruction methods. Additionally, we propose a linear-complexity attention mechanism that reduces computational overhead, enabling the efficient solution of large-scale TSPs without sacrificing performance. Experimental results on both synthetic and real-world datasets demonstrate that LocalEscaper outperforms existing neural solvers, achieving state-of-the-art results. Notably, it sets a new benchmark for scalability and efficiency, solving TSP instances with up to 50,000 cities. 

**Abstract (ZH)**: 局部逃逸者：一种新型弱监督学习框架用于大规模旅行商问题 

---
# UniMatch: Universal Matching from Atom to Task for Few-Shot Drug Discovery 

**Title (ZH)**: UniMatch: 从原子到任务的通用 few-shot 药物发现匹配方法 

**Authors**: Ruifeng Li, Mingqian Li, Wei Liu, Yuhua Zhou, Xiangxin Zhou, Yuan Yao, Qiang Zhang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12453)  

**Abstract**: Drug discovery is crucial for identifying candidate drugs for various this http URL, its low success rate often results in a scarcity of annotations, posing a few-shot learning problem. Existing methods primarily focus on single-scale features, overlooking the hierarchical molecular structures that determine different molecular properties. To address these issues, we introduce Universal Matching Networks (UniMatch), a dual matching framework that integrates explicit hierarchical molecular matching with implicit task-level matching via meta-learning, bridging multi-level molecular representations and task-level generalization. Specifically, our approach explicitly captures structural features across multiple levels, such as atoms, substructures, and molecules, via hierarchical pooling and matching, facilitating precise molecular representation and comparison. Additionally, we employ a meta-learning strategy for implicit task-level matching, allowing the model to capture shared patterns across tasks and quickly adapt to new ones. This unified matching framework ensures effective molecular alignment while leveraging shared meta-knowledge for fast adaptation. Our experimental results demonstrate that UniMatch outperforms state-of-the-art methods on the MoleculeNet and FS-Mol benchmarks, achieving improvements of 2.87% in AUROC and 6.52% in delta AUPRC. UniMatch also shows excellent generalization ability on the Meta-MolNet benchmark. 

**Abstract (ZH)**: 药物发现对于识别各种候选药物至关重要，但由于其较低的成功率常常导致标注数据稀缺，从而产生少量样本学习问题。现有方法主要关注单尺度特征，忽视了决定不同分子性质的层级分子结构。为解决这些问题，我们引入了统一匹配网络（UniMatch），这是一种结合显式层级分子匹配和隐式任务级匹配的双重匹配框架，通过元学习将多层次分子表示与任务级泛化联系起来。具体而言，我们的方法通过层级聚合和匹配明确捕捉多层级的结构特征，如原子、亚结构和分子，促进精确的分子表示和比较。此外，我们采用了元学习策略进行隐式任务级匹配，使模型能够捕捉任务间的共享模式并快速适应新任务。这种统一匹配框架确保了有效的分子对齐，同时通过共享的元知识实现快速适应。实验结果表明，UniMatch在MoleculeNet和FS-Mol基准上优于现有方法，在AUROC上提升了2.87%，在delta AUPRC上提升了6.52%。UniMatch还展示了在Meta-MolNet基准上的强大泛化能力。 

---
# Bridge the Gaps between Machine Unlearning and AI Regulation 

**Title (ZH)**: 机器卸载与人工智能规制之间的差距弥合 

**Authors**: Bill Marino, Meghdad Kurmanji, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2502.12430)  

**Abstract**: The "right to be forgotten" and the data privacy laws that encode it have motivated machine unlearning since its earliest days. Now, an inbound wave of artificial intelligence regulations - like the European Union's Artificial Intelligence Act (AIA) - potentially offer important new use cases for machine unlearning. However, this position paper argues, this opportunity will only be realized if researchers, aided by policymakers, proactively bridge the (sometimes sizable) gaps between machine unlearning's state of the art and its potential applications to AI regulation. To demonstrate this point, we use the AIA as an example. Specifically, we deliver a "state of the union" as regards machine unlearning's current potential for aiding compliance with the AIA. This starts with a precise cataloging of the potential applications of machine unlearning to AIA compliance. For each, we flag any legal ambiguities clouding the potential application and, moreover, flag the technical gaps that exist between the potential application and the state of the art of machine unlearning. Finally, we end with a call to action: for both machine learning researchers and policymakers, to, respectively, solve the open technical and legal questions that will unlock machine unlearning's potential to assist compliance with the AIA - and other AI regulation like it. 

**Abstract (ZH)**: “被遗忘权”和编码其中的数据隐私法律促使机器遗忘技术自其最早期开始发展。现在，一股入境的人工智能法规浪潮——如欧盟的《人工智能法案》——可能为机器遗忘技术提供了重要的新应用场景。然而，本文认为，这种机会只有在研究者得到政策制定者支持的情况下，积极弥合机器遗忘技术的最新进展与其在人工智能法规中的潜在应用之间的（有时较大的）差距后，才能实现。为了证明这一点，我们以《人工智能法案》为例。具体而言，我们提供了一种关于机器遗忘技术当前如何帮助遵守《人工智能法案》的“国情咨文”。这始于一份精确列出机器遗忘技术在《人工智能法案》合规中潜在应用的清单。对于每一种潜在应用，我们指出了任何可能模糊潜在应用的法律 ambiguities，并进一步指出了存在于潜在应用与机器遗忘技术的最新进展之间的技术差距。最后，我们发出呼吁：对于机器学习研究人员和政策制定者来说，分别解决解锁机器遗忘技术协助遵守《人工智能法案》及其他类似人工智能法规的开放技术与法律问题。 

---
# Solving the Cold Start Problem on One's Own as an End User via Preference Transfer 

**Title (ZH)**: 独自解决作为终端用户的冷启动问题通过偏好转移 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2502.12398)  

**Abstract**: We propose a new approach that enables end users to directly solve the cold start problem by themselves. The cold start problem is a common issue in recommender systems, and many methods have been proposed to address the problem on the service provider's side. However, when the service provider does not take action, users are left with poor recommendations and no means to improve their experience. We propose an algorithm, Pretender, that allows end users to proactively solve the cold start problem on their own. Pretender does not require any special support from the service provider and can be deployed independently by users. We formulate the problem as minimizing the distance between the source and target distributions and optimize item selection from the target service accordingly. Furthermore, we establish theoretical guarantees for Pretender based on a discrete quadrature problem. We conduct experiments on real-world datasets to demonstrate the effectiveness of Pretender. 

**Abstract (ZH)**: 我们提出一种新方法，使最终用户能够直接解决冷启动问题。冷启动问题是在推荐系统中常见的问题，许多方法已经提出，以在服务提供商一方解决该问题。然而，当服务提供商不采取行动时，用户将收到糟糕的推荐，而无任何手段来改善其体验。我们提出了一种名为Pretender的算法，该算法允许最终用户主动在自己一方解决冷启动问题。Pretender不需要服务提供商的特殊支持，可以独立部署由用户完成。我们将问题形式化为最小化源分布和目标分布之间的距离，并相应地优化目标服务中的项目选择。此外，我们基于离散矩量问题为Pretender建立了理论保证。我们在真实世界的数据集上进行了实验，以展示Pretender的有效性。 

---
# Could AI Leapfrog the Web? Evidence from Teachers in Sierra Leone 

**Title (ZH)**: AI能超越互联网吗？来自塞拉利昂教师的证据 

**Authors**: Daniel Björkegren, Jun Ho Choi, Divya Budihal, Dominic Sobhani, Oliver Garrod, Paul Atherton  

**Link**: [PDF](https://arxiv.org/pdf/2502.12397)  

**Abstract**: Access to digital information is a driver of economic development. But although 85% of sub-Saharan Africa's population is covered by mobile broadband signal, only 37% use the internet, and those who do seldom use the web. We investigate whether AI can bridge this gap by analyzing how 469 teachers use an AI chatbot in Sierra Leone. The chatbot, accessible via a common messaging app, is compared against traditional web search. Teachers use AI more frequently than web search for teaching assistance. Data cost is the most frequently cited reason for low internet usage across Africa. The average web search result consumes 3,107 times more data than an AI response, making AI 87% less expensive than web search. Additionally, only 2% of results for corresponding web searches contain content from Sierra Leone. In blinded evaluations, an independent sample of teachers rate AI responses as more relevant, helpful, and correct than web search results. These findings suggest that AI-driven solutions can cost-effectively bridge information gaps in low-connectivity regions. 

**Abstract (ZH)**: 数字信息的获取是经济增长的驱动力。尽管撒哈拉以南非洲地区85%的人口覆盖了移动宽带信号，但只有37%的人使用互联网，而且其中多数人很少上网。我们通过分析469名塞拉利昂教师如何使用AI聊天机器人来研究AI能否弥合这一差距。该聊天机器人可通过一款常用的消息应用访问，并与传统的网络搜索进行了对比。教师们比网络搜索更频繁地使用AI进行教学辅助。数据成本是非洲各地互联网使用率低频次最高的原因。平均每条网络搜索结果的数据消耗量是AI回复的3,107倍，使AI比网络搜索便宜87%。此外，针对相应网络搜索结果中仅有2%包含塞拉利昂的内容。在盲测评估中，一组独立教师样本认为AI回复比网络搜索结果更具相关性、帮助性和准确性。这些发现表明，AI驱动的解决方案可以在低连接区域成本效益地弥合信息鸿沟。 

---
# Time Series Treatment Effects Analysis with Always-Missing Controls 

**Title (ZH)**: 始终缺失的控制组时间序列治疗效应分析 

**Authors**: Juan Shu, Qiyu Han, George Chen, Xihao Cao, Kangming Luo, Dan Pallotta, Shivam Agrawal, Yuping Lu, Xiaoyu Zhang, Jawad Mansoor, Jyoti Anand  

**Link**: [PDF](https://arxiv.org/pdf/2502.12393)  

**Abstract**: Estimating treatment effects in time series data presents a significant challenge, especially when the control group is always unobservable. For example, in analyzing the effects of Christmas on retail sales, we lack direct observation of what would have occurred in late December without the Christmas impact. To address this, we try to recover the control group in the event period while accounting for confounders and temporal dependencies. Experimental results on the M5 Walmart retail sales data demonstrate robust estimation of the potential outcome of the control group as well as accurate predicted holiday effect. Furthermore, we provided theoretical guarantees for the estimated treatment effect, proving its consistency and asymptotic normality. The proposed methodology is applicable not only to this always-missing control scenario but also in other conventional time series causal inference settings. 

**Abstract (ZH)**: 在时间序列数据中估计治疗效果存在显著挑战，特别是在控制组始终不可观测的情况下。为了应对这一挑战，我们尝试在事件期间恢复控制组，并考虑到混杂因素和时间依赖性。M5 Walmart零售销售数据的实验结果表明，该方法能够稳健地估计控制组的潜在结果以及准确预测假期效应。此外，我们提供了所估计治疗效果的理论保证，证明了其一致性与渐近正态性。所提出的方法不仅适用于控制组始终缺失的情况，还适用于其他传统的时序因果推断场景。 

---
# Bridging the Data Gap in AI Reliability Research and Establishing DR-AIR, a Comprehensive Data Repository for AI Reliability 

**Title (ZH)**: 填补AI可靠性研究中的数据缺口并建立DR-AIR，一个全面的AI可靠性数据仓库 

**Authors**: Simin Zheng, Jared M. Clark, Fatemeh Salboukh, Priscila Silva, Karen da Mata, Fenglian Pan, Jie Min, Jiayi Lian, Caleb B. King, Lance Fiondella, Jian Liu, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.12386)  

**Abstract**: Artificial intelligence (AI) technology and systems have been advancing rapidly. However, ensuring the reliability of these systems is crucial for fostering public confidence in their use. This necessitates the modeling and analysis of reliability data specific to AI systems. A major challenge in AI reliability research, particularly for those in academia, is the lack of readily available AI reliability data. To address this gap, this paper focuses on conducting a comprehensive review of available AI reliability data and establishing DR-AIR: a data repository for AI reliability. Specifically, we introduce key measurements and data types for assessing AI reliability, along with the methodologies used to collect these data. We also provide a detailed description of the currently available datasets with illustrative examples. Furthermore, we outline the setup of the DR-AIR repository and demonstrate its practical applications. This repository provides easy access to datasets specifically curated for AI reliability research. We believe these efforts will significantly benefit the AI research community by facilitating access to valuable reliability data and promoting collaboration across various academic domains within AI. We conclude our paper with a call to action, encouraging the research community to contribute and share AI reliability data to further advance this critical field of study. 

**Abstract (ZH)**: 人工智能技术与系统正迅速发展。然而，确保这些系统的可靠性对于培养公众对其使用的信心至关重要。这需要特定于人工智能系统的可靠xing数据的建模与分析。人工智能可靠性研究中的一个主要挑战，尤其是在学术界，是可用的人工智能可靠性数据的缺乏。为了解决这一差距，本文侧重于进行全面的人工智能可靠性数据审查，并建立DR-AIR数据 repository：一个人工智能可靠性数据存储库。具体而言，我们介绍了评估人工智能可靠xing的关键测量和数据类型，以及收集这些数据的方法论。我们还详细描述了目前可用的数据集，并提供了示例。此外，我们概述了DR-AIR存储库的设置，并展示了其实际应用。该存储库为人工智能可靠性研究专门提供了便捷的数据访问。我们相信这些努力将显著惠及人工智能研究社区，通过提供有价值的可靠xing数据促进各个学术领域之间的合作。在论文结尾，我们呼吁研究界贡献并分享人工智能可靠性数据，以进一步推动这一关键领域的研究。 

---
# Hybrid Machine Learning Models for Intrusion Detection in IoT: Leveraging a Real-World IoT Dataset 

**Title (ZH)**: 物联网中入侵检测的混合机器学习模型：利用实际物联网数据集 

**Authors**: Md Ahnaf Akif, Ismail Butun, Andre Williams, Imadeldin Mahgoub  

**Link**: [PDF](https://arxiv.org/pdf/2502.12382)  

**Abstract**: The rapid growth of the Internet of Things (IoT) has revolutionized industries, enabling unprecedented connectivity and functionality. However, this expansion also increases vulnerabilities, exposing IoT networks to increasingly sophisticated cyberattacks. Intrusion Detection Systems (IDS) are crucial for mitigating these threats, and recent advancements in Machine Learning (ML) offer promising avenues for improvement. This research explores a hybrid approach, combining several standalone ML models such as Random Forest (RF), XGBoost, K-Nearest Neighbors (KNN), and AdaBoost, in a voting-based hybrid classifier for effective IoT intrusion detection. This ensemble method leverages the strengths of individual algorithms to enhance accuracy and address challenges related to data complexity and scalability. Using the widely-cited IoT-23 dataset, a prominent benchmark in IoT cybersecurity research, we evaluate our hybrid classifiers for both binary and multi-class intrusion detection problems, ensuring a fair comparison with existing literature. Results demonstrate that our proposed hybrid models, designed for robustness and scalability, outperform standalone approaches in IoT environments. This work contributes to the development of advanced, intelligent IDS frameworks capable of addressing evolving cyber threats. 

**Abstract (ZH)**: 物联网(IoT)的快速发展已经革新了诸多行业，实现了前所未有的连接性和功能性。然而，这种扩展也增加了脆弱性，使得IoT网络面临更为复杂的 cybersecurity攻击。入侵检测系统(IDS)对于缓解这些威胁至关重要，而Recent机器学习(ML)的最新进展提供了改进的有希望的途径。本研究探索了一种混合方法，结合了独立的机器学习模型，如随机森林(RF)、XGBoost、K-最近邻(KNN)和AdaBoost，在基于投票的混合分类器中，以有效地进行IoT入侵检测。这种方法利用了单一算法的优势，以提高准确性并解决与数据复杂性和可扩展性相关的问题。使用广泛引用的IoT-23数据集，在IoT网络安全研究中的一个权威基准中，我们评估了我们的混合分类器在二元和多分类入侵检测问题上的性能，确保与现有文献进行公平比较。结果表明，本研究提出的设计稳健性和可扩展性的混合模型，在IoT环境中优于单一方法。本工作为能够应对不断演化的网络安全威胁的先进、智能IDS框架的发展做出了贡献。 

---
# Classifiers of Data Sharing Statements in Clinical Trial Records 

**Title (ZH)**: 临床试验记录中数据共享声明的分类器 

**Authors**: Saber Jelodari Mamaghani, Cosima Strantz, Dennis Toddenroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12362)  

**Abstract**: Digital individual participant data (IPD) from clinical trials are increasingly distributed for potential scientific reuse. The identification of available IPD, however, requires interpretations of textual data-sharing statements (DSS) in large databases. Recent advancements in computational linguistics include pre-trained language models that promise to simplify the implementation of effective classifiers based on textual inputs. In a subset of 5,000 textual DSS from this http URL, we evaluate how well classifiers based on domain-specific pre-trained language models reproduce original availability categories as well as manually annotated labels. Typical metrics indicate that classifiers that predicted manual annotations outperformed those that learned to output the original availability categories. This suggests that the textual DSS descriptions contain applicable information that the availability categories do not, and that such classifiers could thus aid the automatic identification of available IPD in large trial databases. 

**Abstract (ZH)**: 临床试验的数字个体参与者数据（IPD）越来越多地被分散用于潜在的科学研究 reuse。然而，识别可用的 IPD 需要对大型数据库中的文本型数据共享声明（DSS）进行解释。近年来，计算语言学的进步包括预训练语言模型，这些模型承诺简化基于文本输入的有效分类器的实现。在从该网站获取的 5,000 个文本型 DSS 子集中，我们评估基于领域特定预训练语言模型的分类器如何准确地再现原始可用性类别以及手动标注的标签。通常使用的指标表明，预测手动标注的分类器优于学习输出原始可用性类别的分类器。这表明文本型 DSS 描述中包含适用于现有可用性类别描述的信息，并且这样的分类器可以帮助自动识别大型试验数据库中的可用 IPD。 

---
# Human-centered explanation does not fit all: The interplay of sociotechnical, cognitive, and individual factors in the effect AI explanations in algorithmic decision-making 

**Title (ZH)**: 以人为本的解释并不适用于所有人：社会技术、认知及个人因素在AI解释于算法决策中的作用互动 

**Authors**: Yongsu Ahn, Yu-Run Lin, Malihe Alikhani, Eunjeong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2502.12354)  

**Abstract**: Recent XAI studies have investigated what constitutes a \textit{good} explanation in AI-assisted decision-making. Despite the widely accepted human-friendly properties of explanations, such as contrastive and selective, existing studies have yielded inconsistent findings. To address these gaps, our study focuses on the cognitive dimensions of explanation evaluation, by evaluating six explanations with different contrastive strategies and information selectivity and scrutinizing factors behind their valuation process. Our analysis results find that contrastive explanations are not the most preferable or understandable in general; Rather, different contrastive and selective explanations were appreciated to a different extent based on who they are, when, how, and what to explain -- with different level of cognitive load and engagement and sociotechnical contexts. Given these findings, we call for a nuanced view of explanation strategies, with implications for designing AI interfaces to accommodate individual and contextual differences in AI-assisted decision-making. 

**Abstract (ZH)**: 最近关于XAI的研究探讨了构成良好解释的要素，特别是在AI辅助决策中的解释。尽管现有的解释普遍具备人性化特性，如对比性和选择性，但现有研究结果并不一致。为了解决这些缺口，我们的研究集中在解释评估的认知维度上，通过评估六种具有不同对比策略和信息选择性的解释，并审查其评价过程背后的因素。我们的分析结果表明，对比解释通常不是最可接受或最容易理解的；相反，基于解释的对象、时间、方式和内容，不同类型的对比和选择性解释在认知负载和参与度以及社会技术背景下被评价的程度各不相同。鉴于这些发现，我们需要对解释策略采取更为细致的观点，并为AI辅助决策设计接口，以适应个体和情境差异。 

---
# Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs 

**Title (ZH)**: 通过注意力图 toward 图Transformer的机制可解释性研究 

**Authors**: Batu El, Deepro Choudhury, Pietro Liò, Chaitanya K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12352)  

**Abstract**: We introduce Attention Graphs, a new tool for mechanistic interpretability of Graph Neural Networks (GNNs) and Graph Transformers based on the mathematical equivalence between message passing in GNNs and the self-attention mechanism in Transformers. Attention Graphs aggregate attention matrices across Transformer layers and heads to describe how information flows among input nodes. Through experiments on homophilous and heterophilous node classification tasks, we analyze Attention Graphs from a network science perspective and find that: (1) When Graph Transformers are allowed to learn the optimal graph structure using all-to-all attention among input nodes, the Attention Graphs learned by the model do not tend to correlate with the input/original graph structure; and (2) For heterophilous graphs, different Graph Transformer variants can achieve similar performance while utilising distinct information flow patterns. Open source code: this https URL 

**Abstract (ZH)**: 我们介绍了一种基于图神经网络（GNN）和图变换器中消息传递与变换器自注意力机制数学等价性的 Attention Graphs，作为一种新的工具，用于机械解释 GNN 和图变换器的可 interpretability。通过在网络科学视角下对同质性和异质性节点分类任务的实验分析，我们发现：(1) 当允许图变换器利用输入节点之间的全连接注意力学习最优图结构时，模型学习到的 Attention Graphs 不倾向于与输入/原始图结构相关；(2) 对于异质性图，不同的图变换器变种可以在利用不同的信息流模式的同时达到相似的性能。开源代码：见链接。 

---
# A Novel Unified Parametric Assumption for Nonconvex Optimization 

**Title (ZH)**: 一种新型统一参数假设方法用于非凸优化 

**Authors**: Artem Riabinin, Ahmed Khaled, Peter Richtárik  

**Link**: [PDF](https://arxiv.org/pdf/2502.12329)  

**Abstract**: Nonconvex optimization is central to modern machine learning, but the general framework of nonconvex optimization yields weak convergence guarantees that are too pessimistic compared to practice. On the other hand, while convexity enables efficient optimization, it is of limited applicability to many practical problems. To bridge this gap and better understand the practical success of optimization algorithms in nonconvex settings, we introduce a novel unified parametric assumption. Our assumption is general enough to encompass a broad class of nonconvex functions while also being specific enough to enable the derivation of a unified convergence theorem for gradient-based methods. Notably, by tuning the parameters of our assumption, we demonstrate its versatility in recovering several existing function classes as special cases and in identifying functions amenable to efficient optimization. We derive our convergence theorem for both deterministic and stochastic optimization, and conduct experiments to verify that our assumption can hold practically over optimization trajectories. 

**Abstract (ZH)**: 非凸优化是现代机器学习的核心，但非凸优化的一般框架提供的收敛保证过于悲观，与实践不符。另一方面，虽然凸性能够使优化变得更有效率，但在许多实际问题中的应用却是有限的。为了弥合这一差距并更好地理解优化算法在非凸设置下的实际成功，我们引入了一种新型的统一参数假设。该假设既足够广泛以涵盖广泛的非凸函数类，又足够具体以推导出基于梯度的方法的统一收敛定理。值得注意的是，通过调整我们假设的参数，我们展示了其在恢复多个现有函数类的特殊案例以及识别易优化函数方面的灵活性。我们为确定性和随机优化都推导了收敛定理，并进行了实验以验证我们的假设在优化轨迹上可以实际成立。 

---
# Learning Plasma Dynamics and Robust Rampdown Trajectories with Predict-First Experiments at TCV 

**Title (ZH)**: 基于预测先行实验在TCV上的等离子体动力学学习与鲁棒降功率轨迹研究 

**Authors**: Allen M. Wang, Alessandro Pau, Cristina Rea, Oswin So, Charles Dawson, Olivier Sauter, Mark D. Boyer, Anna Vu, Cristian Galperti, Chuchu Fan, Antoine Merle, Yoeri Poels, Cristina Venturini, Stefano Marchioni, TCV Team  

**Link**: [PDF](https://arxiv.org/pdf/2502.12327)  

**Abstract**: The rampdown in tokamak operations is a difficult to simulate phase during which the plasma is often pushed towards multiple instability limits. To address this challenge, and reduce the risk of disrupting operations, we leverage recent advances in Scientific Machine Learning (SciML) to develop a neural state-space model (NSSM) that predicts plasma dynamics during Tokamak à Configuration Variable (TCV) rampdowns. By integrating simple physics structure and data-driven models, the NSSM efficiently learns plasma dynamics during the rampdown from a modest dataset of 311 pulses with only five pulses in the reactor relevant high performance regime. The NSSM is parallelized across uncertainties, and reinforcement learning (RL) is applied to design trajectories that avoid multiple instability limits with high probability. Experiments at TCV ramping down high performance plasmas show statistically significant improvements in current and energy at plasma termination, with improvements in speed through continuous re-training. A predict-first experiment, increasing plasma current by 20\% from baseline, demonstrates the NSSM's ability to make small extrapolations with sufficient accuracy to design trajectories that successfully terminate the pulse. The developed approach paves the way for designing tokamak controls with robustness to considerable uncertainty, and demonstrates the relevance of the SciML approach to learning plasma dynamics for rapidly developing robust trajectories and controls during the incremental campaigns of upcoming burning plasma tokamaks. 

**Abstract (ZH)**: 托卡马克运行减载期间等离子体动态的科学机器学习模型研究 

---
# Warmup Generations: A Task-Agnostic Approach for Guiding Sequence-to-Sequence Learning with Unsupervised Initial State Generation 

**Title (ZH)**: warmup 生成：一种任务无关的方法，用于通过无监督初始状态生成指导序列到序列学习 

**Authors**: Senyu Li, Zipeng Sun, Jiayi Wang, Xue Liu, Pontus Stenetorp, Siva Reddy, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12304)  

**Abstract**: Traditional supervised fine-tuning (SFT) strategies for sequence-to-sequence tasks often train models to directly generate the target output. Recent work has shown that guiding models with intermediate steps, such as keywords, outlines, or reasoning chains, can significantly improve performance, coherence, and interpretability. However, these methods often depend on predefined intermediate formats and annotated data, limiting their scalability and generalizability. In this work, we introduce a task-agnostic framework that enables models to generate intermediate "warmup" sequences. These warmup sequences, serving as an initial state for subsequent generation, are optimized to enhance the probability of generating the target sequence without relying on external supervision or human-designed structures. Drawing inspiration from reinforcement learning principles, our method iteratively refines these intermediate steps to maximize their contribution to the final output, similar to reward-driven optimization in reinforcement learning with human feedback. Experimental results across tasks such as translation, summarization, and multi-choice question answering for logical reasoning show that our approach outperforms traditional SFT methods, and offers a scalable and flexible solution for sequence-to-sequence tasks. 

**Abstract (ZH)**: 传统的序列到序列任务的监督微调策略通常直接生成目标输出。最近的工作表明，使用中间步骤，如关键词、大纲或推理链，可以显著提高性能、连贯性和可解释性。然而，这些方法往往依赖于预定义的中间格式和标注数据，限制了它们的可扩展性和通用性。在本工作中，我们提出了一种任务无关的框架，使模型能够生成中间的“热身”序列。这些热身序列作为后续生成的初始状态，通过优化增强生成目标序列的概率，而不依赖于外部监督或人类设计的结构。受强化学习原理的启发，我们的方法迭代优化这些中间步骤，使其最大化对最终输出的贡献，类似于具有人类反馈的奖励驱动优化。在翻译、总结以及逻辑推理的多选题回答等任务上的实验结果表明，我们的方法优于传统的监督微调方法，并提供了一种适用于序列到序列任务的可扩展和灵活的解决方案。 

---
# Towards Practical First-Order Model Counting 

**Title (ZH)**: 面向 practical 的一阶模型计数 

**Authors**: Ananth K. Kidambi, Guramrit Singh, Paulius Dilkas, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2502.12278)  

**Abstract**: First-order model counting (FOMC) is the problem of counting the number of models of a sentence in first-order logic. Since lifted inference techniques rely on reductions to variants of FOMC, the design of scalable methods for FOMC has attracted attention from both theoreticians and practitioners over the past decade. Recently, a new approach based on first-order knowledge compilation was proposed. This approach, called Crane, instead of simply providing the final count, generates definitions of (possibly recursive) functions that can be evaluated with different arguments to compute the model count for any domain size. However, this approach is not fully automated, as it requires manual evaluation of the constructed functions. The primary contribution of this work is a fully automated compilation algorithm, called Gantry, which transforms the function definitions into C++ code equipped with arbitrary-precision arithmetic. These additions allow the new FOMC algorithm to scale to domain sizes over 500,000 times larger than the current state of the art, as demonstrated through experimental results. 

**Abstract (ZH)**: 基于一阶知识编译的一阶模型计数自动化编译算法 

---
# Identifying the Best Transition Law 

**Title (ZH)**: 识别最佳转移定律 

**Authors**: Mehrasa Ahmadipour, élise Crepon, Aurélien Garivier  

**Link**: [PDF](https://arxiv.org/pdf/2502.12227)  

**Abstract**: Motivated by recursive learning in Markov Decision Processes, this paper studies best-arm identification in bandit problems where each arm's reward is drawn from a multinomial distribution with a known support. We compare the performance { reached by strategies including notably LUCB without and with use of this knowledge. } In the first case, we use classical non-parametric approaches for the confidence intervals. In the second case, where a probability distribution is to be estimated, we first use classical deviation bounds (Hoeffding and Bernstein) on each dimension independently, and then the Empirical Likelihood method (EL-LUCB) on the joint probability vector. The effectiveness of these methods is demonstrated through simulations on scenarios with varying levels of structural complexity. 

**Abstract (ZH)**: 受马尔可夫决策过程中的递归学习启发，本文研究了每根杆的奖励服从具有已知支持的多项分布的臂的选择问题中的最优臂识别。我们比较了包括LUCB在内的策略性能，包括利用这种知识的情况。在第一种情况下，我们使用经典的非参数方法来构建置信区间。在第二种情况下，我们首先在每个维度上独立使用经典的偏差界（霍夫丁和伯恩斯坦），然后对联合概率向量使用经验似然方法（EL-LUCB）。这些方法的有效性通过不同结构复杂度水平的模拟场景得到了验证。 

---
# On Creating a Causally Grounded Usable Rating Method for Assessing the Robustness of Foundation Models Supporting Time Series 

**Title (ZH)**: 基于因果 grounding 的可用地评方法以评估支持时间序列的基础模型的鲁棒性 

**Authors**: Kausik Lakkaraju, Rachneet Kaur, Parisa Zehtabi, Sunandita Patra, Siva Likitha Valluru, Zhen Zeng, Biplav Srivastava, Marco Valtorta  

**Link**: [PDF](https://arxiv.org/pdf/2502.12226)  

**Abstract**: Foundation Models (FMs) have improved time series forecasting in various sectors, such as finance, but their vulnerability to input disturbances can hinder their adoption by stakeholders, such as investors and analysts. To address this, we propose a causally grounded rating framework to study the robustness of Foundational Models for Time Series (FMTS) with respect to input perturbations. We evaluate our approach to the stock price prediction problem, a well-studied problem with easily accessible public data, evaluating six state-of-the-art (some multi-modal) FMTS across six prominent stocks spanning three industries. The ratings proposed by our framework effectively assess the robustness of FMTS and also offer actionable insights for model selection and deployment. Within the scope of our study, we find that (1) multi-modal FMTS exhibit better robustness and accuracy compared to their uni-modal versions and, (2) FMTS pre-trained on time series forecasting task exhibit better robustness and forecasting accuracy compared to general-purpose FMTS pre-trained across diverse settings. Further, to validate our framework's usability, we conduct a user study showcasing FMTS prediction errors along with our computed ratings. The study confirmed that our ratings reduced the difficulty for users in comparing the robustness of different systems. 

**Abstract (ZH)**: 基于因果原理的评估框架：研究输入扰动下Foundational Models for Time Series的鲁棒性 

---
# Subjective Logic Encodings 

**Title (ZH)**: 主观逻辑编码 

**Authors**: Jake Vasilakes  

**Link**: [PDF](https://arxiv.org/pdf/2502.12225)  

**Abstract**: Many existing approaches for learning from labeled data assume the existence of gold-standard labels. According to these approaches, inter-annotator disagreement is seen as noise to be removed, either through refinement of annotation guidelines, label adjudication, or label filtering. However, annotator disagreement can rarely be totally eradicated, especially on more subjective tasks such as sentiment analysis or hate speech detection where disagreement is natural. Therefore, a new approach to learning from labeled data, called data perspectivism, seeks to leverage inter-annotator disagreement to learn models that stay true to the inherent uncertainty of the task by treating annotations as opinions of the annotators, rather than gold-standard facts. Despite this conceptual grounding, existing methods under data perspectivism are limited to using disagreement as the sole source of annotation uncertainty. To expand the possibilities of data perspectivism, we introduce Subjective Logic Encodings (SLEs), a flexible framework for constructing classification targets that explicitly encodes annotations as opinions of the annotators. Based on Subjective Logic Theory, SLEs encode labels as Dirichlet distributions and provide principled methods for encoding and aggregating various types of annotation uncertainty -- annotator confidence, reliability, and disagreement -- into the targets. We show that SLEs are a generalization of other types of label encodings as well as how to estimate models to predict SLEs using a distribution matching objective. 

**Abstract (ZH)**: 标签数据中的主观逻辑编码：一种利用注释员分歧进行学习的新方法 

---
# Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting 

**Title (ZH)**: 空间时间aware趋势-季节性分解网络用于交通流量预测 

**Authors**: Lingxiao Cao, Bin Wang, Guiyuan Jiang, Yanwei Yu, Junyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.12213)  

**Abstract**: Traffic prediction is critical for optimizing travel scheduling and enhancing public safety, yet the complex spatial and temporal dynamics within traffic data present significant challenges for accurate forecasting. In this paper, we introduce a novel model, the Spatiotemporal-aware Trend-Seasonality Decomposition Network (STDN). This model begins by constructing a dynamic graph structure to represent traffic flow and incorporates novel spatio-temporal embeddings to jointly capture global traffic dynamics. The representations learned are further refined by a specially designed trend-seasonality decomposition module, which disentangles the trend-cyclical component and seasonal component for each traffic node at different times within the graph. These components are subsequently processed through an encoder-decoder network to generate the final predictions. Extensive experiments conducted on real-world traffic datasets demonstrate that STDN achieves superior performance with remarkable computation cost. Furthermore, we have released a new traffic dataset named JiNan, which features unique inner-city dynamics, thereby enriching the scenario comprehensiveness in traffic prediction evaluation. 

**Abstract (ZH)**: 时空感知趋势-季节性分解网络（STDN）：面向真实交通数据的高效预测 

---
# Enhancing Frame Detection with Retrieval Augmented Generation 

**Title (ZH)**: 增强帧检测的检索增强生成 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2502.12210)  

**Abstract**: Recent advancements in Natural Language Processing have significantly improved the extraction of structured semantic representations from unstructured text, especially through Frame Semantic Role Labeling (FSRL). Despite this progress, the potential of Retrieval-Augmented Generation (RAG) models for frame detection remains under-explored. In this paper, we present the first RAG-based approach for frame detection called RCIF (Retrieve Candidates and Identify Frames). RCIF is also the first approach to operate without the need for explicit target span and comprises three main stages: (1) generation of frame embeddings from various representations ; (2) retrieval of candidate frames given an input text; and (3) identification of the most suitable frames. We conducted extensive experiments across multiple configurations, including zero-shot, few-shot, and fine-tuning settings. Our results show that our retrieval component significantly reduces the complexity of the task by narrowing the search space thus allowing the frame identifier to refine and complete the set of candidates. Our approach achieves state-of-the-art performance on FrameNet 1.5 and 1.7, demonstrating its robustness in scenarios where only raw text is provided. Furthermore, we leverage the structured representation obtained through this method as a proxy to enhance generalization across lexical variations in the task of translating natural language questions into SPARQL queries. 

**Abstract (ZH)**: Recent advancements in Natural Language Processing have significantly improved the extraction of structured semantic representations from unstructured text, especially through Frame Semantic Role Labeling (FSRL). Despite this progress, the potential of Retrieval-Augmented Generation (RAG) models for frame detection remains under-explored. In this paper, we present the first RAG-based approach for frame detection called RCIF (Retrieve Candidates and Identify Frames). RCIF is also the first approach to operate without the need for explicit target span and comprises three main stages: (1) generation of frame embeddings from various representations; (2) retrieval of candidate frames given an input text; and (3) identification of the most suitable frames. We conducted extensive experiments across multiple configurations, including zero-shot, few-shot, and fine-tuning settings. Our results show that our retrieval component significantly reduces the complexity of the task by narrowing the search space thus allowing the frame identifier to refine and complete the set of candidates. Our approach achieves state-of-the-art performance on FrameNet 1.5 and 1.7, demonstrating its robustness in scenarios where only raw text is provided. Furthermore, we leverage the structured representation obtained through this method as a proxy to enhance generalization across lexical variations in the task of translating natural language questions into SPARQL queries. 

---
# Suboptimal Shapley Value Explanations 

**Title (ZH)**: 次优Shapley值解释 

**Authors**: Xiaolei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12209)  

**Abstract**: Deep Neural Networks (DNNs) have demonstrated strong capacity in supporting a wide variety of applications. Shapley value has emerged as a prominent tool to analyze feature importance to help people understand the inference process of deep neural models. Computing Shapley value function requires choosing a baseline to represent feature's missingness. However, existing random and conditional baselines could negatively influence the explanation. In this paper, by analyzing the suboptimality of different baselines, we identify the problematic baseline where the asymmetric interaction between $\bm{x}'_i$ (the replacement of the faithful influential feature) and other features has significant directional bias toward the model's output, and conclude that $p(y|\bm{x}'_i) = p(y)$ potentially minimizes the asymmetric interaction involving $\bm{x}'_i$. We further generalize the uninformativeness of $\bm{x}'_i$ toward the label space $L$ to avoid estimating $p(y)$ and design a simple uncertainty-based reweighting mechanism to accelerate the computation process. We conduct experiments on various NLP tasks and our quantitative analysis demonstrates the effectiveness of the proposed uncertainty-based reweighting mechanism. Furthermore, by measuring the consistency of explanations generated by explainable methods and human, we highlight the disparity between model inference and human understanding. 

**Abstract (ZH)**: 深层神经网络（DNNs）在支持多种应用方面展现了强大的能力。Shapley值已经成为分析特征重要性、帮助人们理解深度神经模型推理过程的重要工具。计算Shapley值需要选择一个基准来代表特征的缺失性。然而，现有的随机和条件基准可能会负面影响解释的效果。在本文中，通过对不同基准的亚最优性进行分析，我们识别出了一个有问题的基准，在该基准中，替代理性影响特征$\bm{x}'_i$与其他特征之间的不对称交互具有显著的方向性偏差指向模型的输出，并得出结论认为$p(y|\bm{x}'_i) = p(y)$可能最小化涉及$\bm{x}'_i$的不对称交互。我们进一步将$\bm{x}'_i$不对特征空间$L$提供信息推广为避免估算$p(y)$的方式，并设计了一个简单的基于不确定性加权机制来加速计算过程。我们在多种NLP任务上进行了实验，定量分析表明所提出基于不确定性的加权机制的有效性。此外，通过衡量可解释方法生成的解释与人类的一致性，我们突显了模型推理与人类理解之间的差异。 

---
# PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN 

**Title (ZH)**: PAR-AdvGAN: 逐步自回归AdvGAN以提高对抗攻击能力 

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12207)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original this http URL, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: this https URL 

**Abstract (ZH)**: 深度神经网络在各个领域都展现出了卓越的性能。然而，它们对对抗样本尤为脆弱，可能导致错误的预测。生成式对抗网络（GANs）可以通过生成器和判别器模型快速生成高质量的对抗样本。由于两个模块以竞争性和同步性的方式进行训练，基于GAN的算法如AdvGAN能够生成具有更好迁移性的对抗样本，相较于传统方法。然而，扰动的生成通常局限于单次迭代，这限制了这些样本充分发挥方法的潜力。为解决这一问题，我们提出了一种名为渐进自回归AdvGAN（PAR-AdvGAN）的新方法。该方法在渐进生成网络中引入了自回归迭代机制，以生成具有增强攻击能力的对抗样本。我们通过大规模实验全面评估了我们的PAR-AdvGAN方法，表明其在各种最新的黑盒对抗攻击方法中具有优越的性能，同时显著加速了对抗样本的生成，如Inception-v3模型的生成速度可达每秒335.5帧，超越基于梯度的可移植攻击算法。我们的代码可在以下链接获取：this https URL。 

---
# Predicting Depression in Screening Interviews from Interactive Multi-Theme Collaboration 

**Title (ZH)**: 基于互动多主题合作的抑郁筛查访谈中抑郁预测 

**Authors**: Xianbing Zhao, Yiqing Lyu, Di Wang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12204)  

**Abstract**: Automatic depression detection provides cues for early clinical intervention by clinicians. Clinical interviews for depression detection involve dialogues centered around multiple themes. Existing studies primarily design end-to-end neural network models to capture the hierarchical structure of clinical interview dialogues. However, these methods exhibit defects in modeling the thematic content of clinical interviews: 1) they fail to capture intra-theme and inter-theme correlation explicitly, and 2) they do not allow clinicians to intervene and focus on themes of interest. To address these issues, this paper introduces an interactive depression detection framework. This framework leverages in-context learning techniques to identify themes in clinical interviews and then models both intra-theme and inter-theme correlation. Additionally, it employs AI-driven feedback to simulate the interests of clinicians, enabling interactive adjustment of theme importance. PDIMC achieves absolute improvements of 35\% and 12\% compared to the state-of-the-art on the depression detection dataset DAIC-WOZ, which demonstrates the effectiveness of modeling theme correlation and incorporating interactive external feedback. 

**Abstract (ZH)**: 自动抑郁检测为临床早期干预提供线索：一种交互式抑郁检测框架 

---
# Efficient and Effective Prompt Tuning via Prompt Decomposition and Compressed Outer Product 

**Title (ZH)**: 基于提示分解和压缩外积的高效且有效的提示调优 

**Authors**: Pengxiang Lan, Haoyu Xu, Enneng Yang, Yuliang Liang, Guibing Guo, Jianzhe Zhao, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12200)  

**Abstract**: Prompt tuning (PT) offers a cost-effective alternative to fine-tuning large-scale pre-trained language models (PLMs), requiring only a few parameters in soft prompt tokens added before the input text. However, existing PT approaches face two significant issues: (i) They overlook intrinsic semantic associations between soft prompt tokens, leading to high discreteness and limited interactions, thus reducing the model's comprehension and effectiveness in complex tasks. (ii) Due to the complexity of downstream tasks, long soft prompt is necessitated to improve performance, but prompt length correlates positively with memory usage and computational costs. Achieving high efficiency and performance remains an ongoing challenge. To address these issues, we propose a novel Low-parameters prompt tuning (LAMP) method, which leverages prompt decomposition and compressed outer product. Specifically, the prompt decomposition module employs Truncated SVD to reduce training parameters and significantly lower the dimensionality of the soft prompt parameter space. It then utilizes a compressed outer product module to facilitate multiple interactions among prompt tokens, exploring their intrinsic associations to enhance knowledge representation. Finally, LAMP uses average pooling to reduce memory usage and training/inference time. Extensive experiments across six architectures and eight datasets demonstrate that LAMP outperforms state-of-the-art PT-based and LoRA-based methods in performance and efficiency. 

**Abstract (ZH)**: 低参数提示调谐：基于提示分解和压缩外积的方法（LAMP） 

---
# AI and the Law: Evaluating ChatGPT's Performance in Legal Classification 

**Title (ZH)**: AI与法律：评估ChatGPT在法律分类中的表现 

**Authors**: Pawel Weichbroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12193)  

**Abstract**: The use of ChatGPT to analyze and classify evidence in criminal proceedings has been a topic of ongoing discussion. However, to the best of our knowledge, this issue has not been studied in the context of the Polish language. This study addresses this research gap by evaluating the effectiveness of ChatGPT in classifying legal cases under the Polish Penal Code. The results show excellent binary classification accuracy, with all positive and negative cases correctly categorized. In addition, a qualitative evaluation confirms that the legal basis provided for each case, along with the relevant legal content, was appropriate. The results obtained suggest that ChatGPT can effectively analyze and classify evidence while applying the appropriate legal rules. In conclusion, ChatGPT has the potential to assist interested parties in the analysis of evidence and serve as a valuable legal resource for individuals with less experience or knowledge in this area. 

**Abstract (ZH)**: ChatGPT在分析和分类刑事诉讼证据中的应用研究：以波兰法律为视角 

---
# Self-supervised Attribute-aware Dynamic Preference Ranking Alignment 

**Title (ZH)**: 自我监督的属性感知动态偏好排序对齐 

**Authors**: Hongyu Yang, Qi Zhao, Zhenhua hu, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12189)  

**Abstract**: Reinforcement Learning from Human Feedback and its variants excel in aligning with human intentions to generate helpful, harmless, and honest responses. However, most of them rely on costly human-annotated pairwise comparisons for supervised alignment, which is not suitable for list-level scenarios, such as community question answering. Additionally, human preferences are influenced by multiple intrinsic factors in responses, leading to decision-making inconsistencies. Therefore, we propose \textbf{Se}lf-supervised \textbf{A}ttribute-aware \textbf{d}ynamic \textbf{p}reference \textbf{ra}nking, called \shortname. \ It quantifies preference differences between responses based on Attribute-Perceptual Distance Factors (APDF) and dynamically determines the list-wise alignment order. Furthermore, it achieves fine-grained preference difference learning and enables precise alignment with the optimal one. We specifically constructed a challenging code preference dataset named StaCoCoQA, and introduced more cost-effective and scalable preference evaluation metrics: PrefHit and PrefRecall. Extensive experimental results show that SeAdpra exhibits superior performance and generalizability on both StaCoCoQA and preference datasets from eight popular domains. 

**Abstract (ZH)**: 自我监督属性感知动态偏好排序：SeAdpra及其在人类反馈强化学习中的应用 

---
# Boosting Generalization in Diffusion-Based Neural Combinatorial Solver via Energy-guided Sampling 

**Title (ZH)**: 基于能量导向采样的扩散型神经组合求解器泛化能力增强方法 

**Authors**: Haoyu Lei, Kaiwen Zhou, Yinchuan Li, Zhitang Chen, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12188)  

**Abstract**: Diffusion-based Neural Combinatorial Optimization (NCO) has demonstrated effectiveness in solving NP-complete (NPC) problems by learning discrete diffusion models for solution generation, eliminating hand-crafted domain knowledge. Despite their success, existing NCO methods face significant challenges in both cross-scale and cross-problem generalization, and high training costs compared to traditional solvers. While recent studies have introduced training-free guidance approaches that leverage pre-defined guidance functions for zero-shot conditional generation, such methodologies have not been extensively explored in combinatorial optimization. To bridge this gap, we propose a general energy-guided sampling framework during inference time that enhances both the cross-scale and cross-problem generalization capabilities of diffusion-based NCO solvers without requiring additional training. We provide theoretical analysis that helps understanding the cross-problem transfer capability. Our experimental results demonstrate that a diffusion solver, trained exclusively on the Traveling Salesman Problem (TSP), can achieve competitive zero-shot solution generation on TSP variants, such as Prize Collecting TSP (PCTSP) and the Orienteering Problem (OP), through energy-guided sampling across different problem scales. 

**Abstract (ZH)**: 基于扩散的神经组合优化（NCO）通过学习离散扩散模型来生成解决方案，证明了在解决NP完全（NPC）问题时的有效性，无需手工构建领域知识。尽管取得了成功，现有的NCO方法在跨尺度和跨问题泛化方面仍面临重大挑战，并且与传统求解器相比，训练成本较高。虽然近期的研究引入了无需训练的引导方法，利用预定义的引导函数进行零样本条件生成，但此类方法在组合优化中的应用尚未得到广泛探索。为弥补这一空白，我们提出了一种在推理时能源引导采样框架，该框架在无需额外训练的情况下增强基于扩散的NCO求解器的跨尺度和跨问题泛化能力。我们提供了理论分析，有助于理解跨问题迁移能力。实验结果表明，仅在旅行商问题（TSP）上训练的扩散求解器，可以通过跨不同问题规模的能源引导采样实现Prize Collecting TSP（PCTSP）和旅者问题（OP）上具有竞争力的零样本解决方案生成。 

---
# E2CB2former: Effecitve and Explainable Transformer for CB2 Receptor Ligand Activity Prediction 

**Title (ZH)**: E2CB2former: 有效可解释的变换器模型用于CB2受体配体活性预测 

**Authors**: Jiacheng Xie, Yingrui Ji, Linghuan Zeng, Xi Xiao, Gaofei Chen, Lijing Zhu, Joyanta Jyoti Mondal, Jiansheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12186)  

**Abstract**: Accurate prediction of CB2 receptor ligand activity is pivotal for advancing drug discovery targeting this receptor, which is implicated in inflammation, pain management, and neurodegenerative conditions. Although conventional machine learning and deep learning techniques have shown promise, their limited interpretability remains a significant barrier to rational drug design. In this work, we introduce CB2former, a framework that combines a Graph Convolutional Network with a Transformer architecture to predict CB2 receptor ligand activity. By leveraging the Transformer's self attention mechanism alongside the GCN's structural learning capability, CB2former not only enhances predictive performance but also offers insights into the molecular features underlying receptor activity. We benchmark CB2former against diverse baseline models including Random Forest, Support Vector Machine, K Nearest Neighbors, Gradient Boosting, Extreme Gradient Boosting, Multilayer Perceptron, Convolutional Neural Network, and Recurrent Neural Network and demonstrate its superior performance with an R squared of 0.685, an RMSE of 0.675, and an AUC of 0.940. Moreover, attention weight analysis reveals key molecular substructures influencing CB2 receptor activity, underscoring the model's potential as an interpretable AI tool for drug discovery. This ability to pinpoint critical molecular motifs can streamline virtual screening, guide lead optimization, and expedite therapeutic development. Overall, our results showcase the transformative potential of advanced AI approaches exemplified by CB2former in delivering both accurate predictions and actionable molecular insights, thus fostering interdisciplinary collaboration and innovation in drug discovery. 

**Abstract (ZH)**: 准确预测CB2受体配体活性对于针对该受体的药物发现具有重要意义，CB2受体与炎症、疼痛管理和神经退行性疾病有关。尽管传统的机器学习和深度学习技术显示出了希望，但它们的解释性限制仍然是合理药物设计的一个重大障碍。在本研究中，我们引入了CB2former，这是一种将图卷积网络与Transformer架构结合的框架，用于预测CB2受体配体活性。通过利用Transformer的自注意力机制和GCN的结构学习能力，CB2former不仅能提升预测性能，还能揭示受体活性背后的分子特征。我们将CB2former与随机森林、支持向量机、K近邻、梯度提升、极端梯度提升、多层感知机、卷积神经网络和递归神经网络等不同基准模型进行对比，并展示其优越性能，其中R平方值为0.685，均方根误差为0.675，AUC为0.940。此外，注意力权重分析揭示了关键的分子亚结构对CB2受体活性的影响，表明该模型具有作为药物发现中可解释AI工具的潜力。能够准确定位关键分子模式的能力可以简化虚拟筛选，指导先导化合物的优化，加速治疗性药物的开发。总的来说，我们的结果展示了CB2former等先进AI方法在提供准确预测和可操作的分子洞察方面的变革潜力，从而促进药物发现领域的跨学科合作与创新。 

---
# Towards Transparent and Accurate Plasma State Monitoring at JET 

**Title (ZH)**: 向JET等离子体状态透明且准确监测迈进 

**Authors**: Andrin Bürli, Alessandro Pau, Thomas Koller, Olivier Sauter, JET Contributors  

**Link**: [PDF](https://arxiv.org/pdf/2502.12182)  

**Abstract**: Controlling and monitoring plasma within a tokamak device is complex and challenging. Plasma off-normal events, such as disruptions, are hindering steady-state operation. For large devices, they can even endanger the machine's integrity and it represents in general one of the most serious concerns for the exploitation of the tokamak concept for future power plants. Effective plasma state monitoring carries the potential to enable an understanding of such phenomena and their evolution which is crucial for the successful operation of tokamaks. This paper presents the application of a transparent and data-driven methodology to monitor the plasma state in a tokamak. Compared to previous studies in the field, supervised and unsupervised learning techniques are combined. The dataset consisted of 520 expert-validated discharges from JET. The goal was to provide an interpretable plasma state representation for the JET operational space by leveraging multi-task learning for the first time in the context of plasma state monitoring. When evaluated as disruption predictors, a sequence-based approach showed significant improvements compared to the state-based models. The best resulting network achieved a promising cross-validated success rate when combined with a physical indicator and accounting for nearby instabilities. Qualitative evaluations of the learned latent space uncovered operational and disruptive regions as well as patterns related to learned dynamics and global feature importance. The applied methodology provides novel possibilities for the definition of triggers to switch between different control scenarios, data analysis, and learning as well as exploring latent dynamics for plasma state monitoring. It also showed promising quantitative and qualitative results with warning times suitable for avoidance purposes and distributions that are consistent with known physical mechanisms. 

**Abstract (ZH)**: 基于透明和数据驱动方法的托卡马克等离子体状态监控应用 

---
# 3D ReX: Causal Explanations in 3D Neuroimaging Classification 

**Title (ZH)**: 3D ReX: 3D神经 Imaging 分类中的因果解释 

**Authors**: Melane Navaratnarajah, Sophie A. Martin, David A. Kelly, Nathan Blake, Hana Chocker  

**Link**: [PDF](https://arxiv.org/pdf/2502.12181)  

**Abstract**: Explainability remains a significant problem for AI models in medical imaging, making it challenging for clinicians to trust AI-driven predictions. We introduce 3D ReX, the first causality-based post-hoc explainability tool for 3D models. 3D ReX uses the theory of actual causality to generate responsibility maps which highlight the regions most crucial to the model's decision. We test 3D ReX on a stroke detection model, providing insight into the spatial distribution of features relevant to stroke. 

**Abstract (ZH)**: 医学成像中的人工智能模型可解释性依然存在显著问题，使得临床医生难以信任基于人工智能的预测。我们引入了3D ReX，这是一种基于因果性的后验可解释性工具，适用于3D模型。3D ReX 使用实际因果理论生成责任图，突出显示对模型决策至关重要的区域。我们以中风检测模型为测试对象，揭示了与中风相关的空间特征分布。 

---
# Ten Challenging Problems in Federated Foundation Models 

**Title (ZH)**: 联邦基础模型中的十个挑战性问题 

**Authors**: Tao Fan, Hanlin Gu, Xuemei Cao, Chee Seng Chan, Qian Chen, Yiqiang Chen, Yihui Feng, Yang Gu, Jiaxiang Geng, Bing Luo, Shuoling Liu, Win Kent Ong, Chao Ren, Jiaqi Shao, Chuan Sun, Xiaoli Tang, Hong Xi Tae, Yongxin Tong, Shuyue Wei, Fan Wu, Wei Xi, Mingcong Xu, He Yang, Xin Yang, Jiangpeng Yan, Hao Yu, Han Yu, Teng Zhang, Yifei Zhang, Xiaojin Zhang, Zhenzhe Zheng, Lixin Fan, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12176)  

**Abstract**: Federated Foundation Models (FedFMs) represent a distributed learning paradigm that fuses general competences of foundation models as well as privacy-preserving capabilities of federated learning. This combination allows the large foundation models and the small local domain models at the remote clients to learn from each other in a teacher-student learning setting. This paper provides a comprehensive summary of the ten challenging problems inherent in FedFMs, encompassing foundational theory, utilization of private data, continual learning, unlearning, Non-IID and graph data, bidirectional knowledge transfer, incentive mechanism design, game mechanism design, model watermarking, and efficiency. The ten challenging problems manifest in five pivotal aspects: ``Foundational Theory," which aims to establish a coherent and unifying theoretical framework for FedFMs. ``Data," addressing the difficulties in leveraging domain-specific knowledge from private data while maintaining privacy; ``Heterogeneity," examining variations in data, model, and computational resources across clients; ``Security and Privacy," focusing on defenses against malicious attacks and model theft; and ``Efficiency," highlighting the need for improvements in training, communication, and parameter efficiency. For each problem, we offer a clear mathematical definition on the objective function, analyze existing methods, and discuss the key challenges and potential solutions. This in-depth exploration aims to advance the theoretical foundations of FedFMs, guide practical implementations, and inspire future research to overcome these obstacles, thereby enabling the robust, efficient, and privacy-preserving FedFMs in various real-world applications. 

**Abstract (ZH)**: 联邦基础模型（FedFMs）代表了一种分布式学习范式，融合了基础模型的一般能力和联邦学习的隐私保护能力。这种结合使得大型基础模型和远程客户端的小规模本地领域模型能够在教师-学生学习设置中相互学习。本文全面总结了FedFMs内在的十个挑战性问题，涵盖了基础理论、私有数据利用、持续学习、遗忘、非IID和图数据、双向知识迁移、激励机制设计、博弈机制设计、模型水印以及效率等方面。这十个挑战性问题在五个关键方面得到体现：“基础理论”，旨在为FedFMs建立一个协调和统一的理论框架；“数据”，解决在利用私有数据中的领域特定知识时保持隐私的难题；“异质性”，考察客户端在数据、模型和计算资源方面的差异；“安全与隐私”，关注抵御恶意攻击和模型盗窃的防御措施；“效率”，强调在训练、通信和参数效率方面的改进。对于每个问题，我们提供了清晰的数学定义，分析现有方法，并讨论关键挑战和潜在解决方案。这种深入探索旨在推动FedFMs的理论基础发展，指导实际实施，并激发未来研究以克服这些障碍，从而实现各种实际应用场景下的稳健、高效和隐私保护的FedFMs。 

---
# Spatiotemporal Graph Neural Networks in short term load forecasting: Does adding Graph Structure in Consumption Data Improve Predictions? 

**Title (ZH)**: 时空图神经网络在短期负荷预测中的应用：在消费数据中加入图结构能改善预测吗？ 

**Authors**: Quoc Viet Nguyen, Joaquin Delgado Fernandez, Sergio Potenciano Menci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12175)  

**Abstract**: Short term Load Forecasting (STLF) plays an important role in traditional and modern power systems. Most STLF models predominantly exploit temporal dependencies from historical data to predict future consumption. Nowadays, with the widespread deployment of smart meters, their data can contain spatiotemporal dependencies. In particular, their consumption data is not only correlated to historical values but also to the values of neighboring smart meters. This new characteristic motivates researchers to explore and experiment with new models that can effectively integrate spatiotemporal interrelations to increase forecasting performance. Spatiotemporal Graph Neural Networks (STGNNs) can leverage such interrelations by modeling relationships between smart meters as a graph and using these relationships as additional features to predict future energy consumption. While extensively studied in other spatiotemporal forecasting domains such as traffic, environments, or renewable energy generation, their application to load forecasting remains relatively unexplored, particularly in scenarios where the graph structure is not inherently available. This paper overviews the current literature focusing on STGNNs with application in STLF. Additionally, from a technical perspective, it also benchmarks selected STGNN models for STLF at the residential and aggregate levels. The results indicate that incorporating graph features can improve forecasting accuracy at the residential level; however, this effect is not reflected at the aggregate level 

**Abstract (ZH)**: 短期负荷预测（STLF）在传统和现代电力系统中发挥着重要作用。大多数STLF模型主要通过历史数据的时间依赖性来预测未来的消费需求。随着智能电表的广泛应用，其数据可能包含时空依赖性，特别是其用电数据不仅与历史值相关，还与相邻智能电表的值相关。这种新特性激发了研究人员探索并试验能够有效整合时空相互关系的新模型，以提高预测性能。时空图神经网络（STGNN）可以通过将智能电表的关系建模为图，并利用这些关系作为额外特征来预测未来的能源消耗，从而利用这些相互关系。尽管STGNN在交通、环境或可再生能源生成等其他时空预测领域受到了广泛研究，但在负荷预测中的应用仍然相对未被探索，特别是在图结构本身不可用的情况下。本文综述了现有的关于应用于STLF的STGNN的相关文献。从技术层面来看，本文还对选定的STGNN模型在住宅和汇总层面的STLF进行了基准测试。结果显示，在住宅层面引入图特征可以提高预测准确性；但在汇总层面，这种效果并未显现。 

---
# TastepepAI, An artificial intelligence platform for taste peptide de novo design 

**Title (ZH)**: TastepepAI，一种用于味肽从头设计的人工智能平台 

**Authors**: Jianda Yue, Tingting Li, Jian Ouyang, Jiawei Xu, Hua Tan, Zihui Chen, Changsheng Han, Huanyu Li, Songping Liang, Zhonghua Liu, Zhonghua Liu, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12167)  

**Abstract**: Taste peptides have emerged as promising natural flavoring agents attributed to their unique organoleptic properties, high safety profile, and potential health benefits. However, the de novo identification of taste peptides derived from animal, plant, or microbial sources remains a time-consuming and resource-intensive process, significantly impeding their widespread application in the food industry. Here, we present TastePepAI, a comprehensive artificial intelligence framework for customized taste peptide design and safety assessment. As the key element of this framework, a loss-supervised adaptive variational autoencoder (LA-VAE) is implemented to efficiently optimizes the latent representation of sequences during training and facilitates the generation of target peptides with desired taste profiles. Notably, our model incorporates a novel taste-avoidance mechanism, allowing for selective flavor exclusion. Subsequently, our in-house developed toxicity prediction algorithm (SpepToxPred) is integrated in the framework to undergo rigorous safety evaluation of generated peptides. Using this integrated platform, we successfully identified 73 peptides exhibiting sweet, salty, and umami, significantly expanding the current repertoire of taste peptides. This work demonstrates the potential of TastePepAI in accelerating taste peptide discovery for food applications and provides a versatile framework adaptable to broader peptide engineering challenges. 

**Abstract (ZH)**: 味肽作为一种因其独特的感官特性、高安全性概况和潜在健康益处而被视为有前景的天然调味剂，已逐渐受到关注。然而，从动物、植物或微生物源中鉴定新的味肽仍是一个耗时且资源密集的过程，严重阻碍了其在食品工业中的广泛应用。为此，我们提出了一种全面的人工智能框架——TastePepAI，用于定制化味肽设计和安全性评估。作为该框架的关键组成部分，我们实现了一种损失监督自适应变分自动编码器（LA-VAE），以在训练期间高效优化序列的潜在表示，并促进生成具有特定味觉轮廓的目标肽。值得注意的是，我们的模型集成了一个新颖的味觉回避机制，允许选择性地排除特定风味。随后，我们自主研发的毒性预测算法（SpepToxPred）被整合到该框架中，用于对生成的肽进行严格的安全性评估。通过该集成平台，我们成功鉴定了73种具有甜、咸、鲜味特征的肽，显著扩展了现有味肽的范围。本研究展示了TastePepAI在加速食品应用中味肽发现方面的潜力，并提供了一种广泛适应于肽工程挑战的多功能框架。 

---
# Integrating Artificial Intelligence and Geophysical Insights for Earthquake Forecasting: A Cross-Disciplinary Review 

**Title (ZH)**: 将人工智能与地质物理洞察整合于地震预测：一门跨学科综述 

**Authors**: Zhang Ying, Wen Congcong, Sornette Didier, Zhan Chengxiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12161)  

**Abstract**: Earthquake forecasting remains a significant scientific challenge, with current methods falling short of achieving the performance necessary for meaningful societal benefits. Traditional models, primarily based on past seismicity and geomechanical data, struggle to capture the complexity of seismic patterns and often overlook valuable non-seismic precursors such as geophysical, geochemical, and atmospheric anomalies. The integration of such diverse data sources into forecasting models, combined with advancements in AI technologies, offers a promising path forward. AI methods, particularly deep learning, excel at processing complex, large-scale datasets, identifying subtle patterns, and handling multidimensional relationships, making them well-suited for overcoming the limitations of conventional approaches.
This review highlights the importance of combining AI with geophysical knowledge to create robust, physics-informed forecasting models. It explores current AI methods, input data types, loss functions, and practical considerations for model development, offering guidance to both geophysicists and AI researchers. While many AI-based studies oversimplify earthquake prediction, neglecting critical features such as data imbalance and spatio-temporal clustering, the integration of specialized geophysical insights into AI models can address these shortcomings.
We emphasize the importance of interdisciplinary collaboration, urging geophysicists to experiment with AI architectures thoughtfully and encouraging AI experts to deepen their understanding of seismology. By bridging these disciplines, we can develop more accurate, reliable, and societally impactful earthquake forecasting tools. 

**Abstract (ZH)**: 地震预测仍然是一个重大的科学挑战，现有方法尚未达到实现有意义社会收益所需的效果。传统模型主要基于过去的地震活动和地质力学数据，难以捕捉地震模式的复杂性，往往忽视了诸如地球物理、地球化学和大气异常等有价值的非地震前兆信号。将如此多样的数据源整合到预测模型中，并结合AI技术的进步，为解决这一挑战提供了有希望的途径。特别是深度学习方法，擅长处理复杂的、大规模的数据集，识别细微的模式，并处理多维度关系，使它们适合克服传统方法的局限性。
本文强调了将AI与地球物理知识相结合以构建健壮、符合物理原理的预测模型的重要性。它探讨了当前的AI方法、输入数据类型、损失函数，并提供了模型开发的实际考量，为地球物理学家和AI研究人员提供了指导。尽管许多基于AI的研究简化了地震预测，忽视了数据不平衡和时空聚集等关键特征，将专门的地球物理洞见融入AI模型可以解决这些问题。
我们强调跨学科合作的重要性，敦促地球物理学家明智地尝试AI架构，并鼓励AI专家加深对地震学的理解。通过这些学科的交叉融合，我们可以开发出更准确、更可靠并更具社会影响力的地震预测工具。 

---
