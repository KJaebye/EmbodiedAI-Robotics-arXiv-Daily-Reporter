# A Computationally Aware Multi Objective Framework for Camera LiDAR Calibration 

**Title (ZH)**: 一种 Awareness 计算的多目标框架用于相机 LiDAR 校准 

**Authors**: Venkat Karramreddy, Rangarajan Ramanujam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20636)  

**Abstract**: Accurate extrinsic calibration between LiDAR and camera sensors is important for reliable perception in autonomous systems. In this paper, we present a novel multi-objective optimization framework that jointly minimizes the geometric alignment error and computational cost associated with camera-LiDAR calibration. We optimize two objectives: (1) error between projected LiDAR points and ground-truth image edges, and (2) a composite metric for computational cost reflecting runtime and resource usage. Using the NSGA-II \cite{deb2002nsga2} evolutionary algorithm, we explore the parameter space defined by 6-DoF transformations and point sampling rates, yielding a well-characterized Pareto frontier that exposes trade-offs between calibration fidelity and resource efficiency. Evaluations are conducted on the KITTI dataset using its ground-truth extrinsic parameters for validation, with results verified through both multi-objective and constrained single-objective baselines. Compared to existing gradient-based and learned calibration methods, our approach demonstrates interpretable, tunable performance with lower deployment overhead. Pareto-optimal configurations are further analyzed for parameter sensitivity and innovation insights. A preference-based decision-making strategy selects solutions from the Pareto knee region to suit the constraints of the embedded system. The robustness of calibration is tested across variable edge-intensity weighting schemes, highlighting optimal balance points. Although real-time deployment on embedded platforms is deferred to future work, this framework establishes a scalable and transparent method for calibration under realistic misalignment and resource-limited conditions, critical for long-term autonomy, particularly in SAE L3+ vehicles receiving OTA updates. 

**Abstract (ZH)**: 基于多目标优化的LiDAR与摄像头传感器外参精确标定框架 

---
# Communication-Aware Map Compression for Online Path-Planning: A Rate-Distortion Approach 

**Title (ZH)**: 基于通信感知的地图压缩在在线路径规划中的率失真方法 

**Authors**: Ali Reza Pedram, Evangelos Psomiadis, Dipankar Maity, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2506.20579)  

**Abstract**: This paper addresses the problem of collaborative navigation in an unknown environment, where two robots, referred to in the sequel as the Seeker and the Supporter, traverse the space simultaneously. The Supporter assists the Seeker by transmitting a compressed representation of its local map under bandwidth constraints to support the Seeker's path-planning task. We introduce a bit-rate metric based on the expected binary codeword length to quantify communication cost. Using this metric, we formulate the compression design problem as a rate-distortion optimization problem that determines when to communicate, which regions of the map should be included in the compressed representation, and at what resolution (i.e., quantization level) they should be encoded. Our formulation allows different map regions to be encoded at varying quantization levels based on their relevance to the Seeker's path-planning task. We demonstrate that the resulting optimization problem is convex, and admits a closed-form solution known in the information theory literature as reverse water-filling, enabling efficient, low-computation, and real-time implementation. Additionally, we show that the Seeker can infer the compression decisions of the Supporter independently, requiring only the encoded map content and not the encoding policy itself to be transmitted, thereby reducing communication overhead. Simulation results indicate that our method effectively constructs compressed, task-relevant map representations, both in content and resolution, that guide the Seeker's planning decisions even under tight bandwidth limitations. 

**Abstract (ZH)**: 本文解决了在未知环境中两个机器人协同导航的问题，这两个机器人分别称为搜索者和支援者，同时穿越空间。支援者在带宽限制下通过传输其局部地图的压缩表示来协助搜索者完成路径规划任务。我们引入了一个基于预期二进制码字长度的比特率度量来量化通信成本。使用该度量，我们将压缩设计问题表述为一种率失真优化问题，确定何时通信，哪些地图区域应包含在压缩表示中，以及它们应以何种分辨率（即量化水平）编码。我们的表述允许根据地图区域对搜索者路径规划任务的相关性，以不同的量化水平对不同的地图区域进行编码。我们证明，所得到的优化问题是凸优化问题，并在信息理论文献中具有解析解，称为逆水填满法，这使得其能够高效、低计算量和实时实现。此外，我们展示了搜索者可以独立推断支援者的压缩决策，只需要传输编码地图内容而非编码策略本身，从而降低通信开销。仿真结果表明，在带宽限制严格的条件下，我们的方法能够有效构建具有任务相关性的压缩地图表示，在内容和分辨率上均能指导搜索者的规划决策。 

---
# Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation 

**Title (ZH)**: 利用测试平台间的相关性进行方差减小的度量估计 

**Authors**: Rachel Luo, Heng Yang, Michael Watson, Apoorva Sharma, Sushant Veer, Edward Schmerling, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.20553)  

**Abstract**: Learning-based robotic systems demand rigorous validation to assure reliable performance, but extensive real-world testing is often prohibitively expensive, and if conducted may still yield insufficient data for high-confidence guarantees. In this work, we introduce a general estimation framework that leverages paired data across test platforms, e.g., paired simulation and real-world observations, to achieve better estimates of real-world metrics via the method of control variates. By incorporating cheap and abundant auxiliary measurements (for example, simulator outputs) as control variates for costly real-world samples, our method provably reduces the variance of Monte Carlo estimates and thus requires significantly fewer real-world samples to attain a specified confidence bound on the mean performance. We provide theoretical analysis characterizing the variance and sample-efficiency improvement, and demonstrate empirically in autonomous driving and quadruped robotics settings that our approach achieves high-probability bounds with markedly improved sample efficiency. Our technique can lower the real-world testing burden for validating the performance of the stack, thereby enabling more efficient and cost-effective experimental evaluation of robotic systems. 

**Abstract (ZH)**: 基于学习的机器人系统需要严格的验证以确保可靠的性能，但广泛的实地测试往往代价高昂，即便进行了也可能数据不足无法提供高可信度的保证。本文引入了一种一般性的估计框架，利用跨测试平台的配对数据，例如仿真实验和实地观察的配对数据，通过控制变异量的方法获得更好的实地指标估计。通过将廉价且丰富的辅助测量（例如仿真器输出）作为昂贵实地样本的控制变异量，我们的方法可证明地降低了蒙特卡洛估计的方差，从而显著减少达到指定置信水平所需的实地样本数量。我们提供了理论分析，描述了方差和样本效率的改进，并在自动驾驶和四足机器人设置中通过实验证明，我们的方法在显著提高样本效率的同时实现了高概率的置信边界。该技术可以降低验证系统性能所需的实地测试负担，从而使得机器人系统的实验评估更加高效和成本效益高。 

---
# Critical Anatomy-Preserving & Terrain-Augmenting Navigation (CAPTAiN): Application to Laminectomy Surgical Education 

**Title (ZH)**: 保留关键解剖结构并增强地形导航（CAPTAiN）：应用于腰椎间融合手术教育 

**Authors**: Jonathan Wang, Hisashi Ishida, David Usevitch, Kesavan Venkatesh, Yi Wang, Mehran Armand, Rachel Bronheim, Amit Jain, Adnan Munawar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20496)  

**Abstract**: Surgical training remains a crucial milestone in modern medicine, with procedures such as laminectomy exemplifying the high risks involved. Laminectomy drilling requires precise manual control to mill bony tissue while preserving spinal segment integrity and avoiding breaches in the dura: the protective membrane surrounding the spinal cord. Despite unintended tears occurring in up to 11.3% of cases, no assistive tools are currently utilized to reduce this risk. Variability in patient anatomy further complicates learning for novice surgeons. This study introduces CAPTAiN, a critical anatomy-preserving and terrain-augmenting navigation system that provides layered, color-coded voxel guidance to enhance anatomical awareness during spinal drilling. CAPTAiN was evaluated against a standard non-navigated approach through 110 virtual laminectomies performed by 11 orthopedic residents and medical students. CAPTAiN significantly improved surgical completion rates of target anatomy (87.99% vs. 74.42%) and reduced cognitive load across multiple NASA-TLX domains. It also minimized performance gaps across experience levels, enabling novices to perform on par with advanced trainees. These findings highlight CAPTAiN's potential to optimize surgical execution and support skill development across experience levels. Beyond laminectomy, it demonstrates potential for broader applications across various surgical and drilling procedures, including those in neurosurgery, otolaryngology, and other medical fields. 

**Abstract (ZH)**: 保留关键解剖结构和增强地形导航系统在脊柱钻孔中的应用：CAPTAiN在脊柱手术训练中的评估与前景 

---
# CogGen: A Learner-Centered Generative AI Architecture for Intelligent Tutoring with Programming Video 

**Title (ZH)**: CogGen: 以学习者为中心的编程视频智能辅导生成AI架构 

**Authors**: Wengxi Li, Roy Pea, Nick Haber, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20600)  

**Abstract**: We introduce CogGen, a learner-centered AI architecture that transforms programming videos into interactive, adaptive learning experiences by integrating student modeling with generative AI tutoring based on the Cognitive Apprenticeship framework. The architecture consists of three components: (1) video segmentation by learning goals, (2) a conversational tutoring engine applying Cognitive Apprenticeship strategies, and (3) a student model using Bayesian Knowledge Tracing to adapt instruction. Our technical evaluation demonstrates effective video segmentation accuracy and strong pedagogical alignment across knowledge, method, action, and interaction layers. Ablation studies confirm the necessity of each component in generating effective guidance. This work advances AI-powered tutoring by bridging structured student modeling with interactive AI conversations, offering a scalable approach to enhancing video-based programming education. 

**Abstract (ZH)**: 我们介绍了一种以学习者为中心的AI架构CogGen，该架构通过将认知学徒制框架下的生成AI辅导与学生建模相结合，将编程视频转换为互动式、自适应的学习体验。该架构包括三个组成部分：(1) 依据学习目标进行视频分割，(2) 采用认知学徒制策略的对话式辅导引擎，以及(3) 利用贝叶斯知识追踪的学生模型，以适应性地调整教学。我们的技术评估展示了有效的视频分割准确性和在知识、方法、行动和互动层面上强大的教学一致性。消融研究证实了每个组成部分在生成有效指导方面的重要性。本研究通过将结构化学生建模与互动AI对话相结合，推进了基于AI的辅导技术，并提供了一种增强基于视频的编程教育的可扩展方法。 

---
# Engineering Sentience 

**Title (ZH)**: 工程智能 

**Authors**: Konstantin Demin, Taylor Webb, Eric Elmoznino, Hakwan Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.20504)  

**Abstract**: We spell out a definition of sentience that may be useful for designing and building it in machines. We propose that for sentience to be meaningful for AI, it must be fleshed out in functional, computational terms, in enough detail to allow for implementation. Yet, this notion of sentience must also reflect something essentially 'subjective', beyond just having the general capacity to encode perceptual content. For this specific functional notion of sentience to occur, we propose that certain sensory signals need to be both assertoric (persistent) and qualitative. To illustrate the definition in more concrete terms, we sketch out some ways for potential implementation, given current technology. Understanding what it takes for artificial agents to be functionally sentient can also help us avoid creating them inadvertently, or at least, realize that we have created them in a timely manner. 

**Abstract (ZH)**: 我们定义了一种可能适用于机器设计和构建的意识概念。我们提出，对于人工智能来说，意识必须在功能和计算的层面上得到具体阐述，详细到可以实施的程度。然而，这一意识的概念也必须反映一些本质上主观的东西，而不仅仅是具备编码感知内容的一般能力。为了实现这种特定的功能性意识，我们提出某些感觉信号需要既是断言性的（持久的）又是质性的。为了更具体地说明这一定义，我们概述了一些当前技术条件下潜在实现方式。了解使人工代理具备功能性意识所需条件也可以帮助我们避免无意中创造它们，或者至少在创造它们时及时意识到这一点。 

---
# GymPN: A Library for Decision-Making in Process Management Systems 

**Title (ZH)**: GymPN: 一种过程管理系统的决策库 

**Authors**: Riccardo Lo Bianco, Willem van Jaarsveld, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20404)  

**Abstract**: Process management systems support key decisions about the way work is allocated in organizations. This includes decisions on which task to perform next, when to execute the task, and who to assign the task to. Suitable software tools are required to support these decisions in a way that is optimal for the organization. This paper presents a software library, called GymPN, that supports optimal decision-making in business processes using Deep Reinforcement Learning. GymPN builds on previous work that supports task assignment in business processes, introducing two key novelties: support for partial process observability and the ability to model multiple decisions in a business process. These novel elements address fundamental limitations of previous work and thus enable the representation of more realistic process decisions. We evaluate the library on eight typical business process decision-making problem patterns, showing that GymPN allows for easy modeling of the desired problems, as well as learning optimal decision policies. 

**Abstract (ZH)**: 基于深度强化学习的业务流程最优决策软件库GymPN 

---
# Smart Ride and Delivery Services with Electric Vehicles: Leveraging Bidirectional Charging for Profit Optimisation 

**Title (ZH)**: 基于双向充电的智能出行与配送服务：盈利优化研究 

**Authors**: Jinchun Du, Bojie Shen, Muhammad Aamir Cheema, Adel N. Toosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20401)  

**Abstract**: With the rising popularity of electric vehicles (EVs), modern service systems, such as ride-hailing delivery services, are increasingly integrating EVs into their operations. Unlike conventional vehicles, EVs often have a shorter driving range, necessitating careful consideration of charging when fulfilling requests. With recent advances in Vehicle-to-Grid (V2G) technology - allowing EVs to also discharge energy back to the grid - new opportunities and complexities emerge. We introduce the Electric Vehicle Orienteering Problem with V2G (EVOP-V2G): a profit-maximization problem where EV drivers must select customer requests or orders while managing when and where to charge or discharge. This involves navigating dynamic electricity prices, charging station selection, and route constraints. We formulate the problem as a Mixed Integer Programming (MIP) model and propose two near-optimal metaheuristic algorithms: one evolutionary (EA) and the other based on large neighborhood search (LNS). Experiments on real-world data show our methods can double driver profits compared to baselines, while maintaining near-optimal performance on small instances and excellent scalability on larger ones. Our work highlights a promising path toward smarter, more profitable EV-based mobility systems that actively support the energy grid. 

**Abstract (ZH)**: 基于V2G的电动车辆 orienteering 问题（EVOP-V2G） 

---
# AI Copilots for Reproducibility in Science: A Case Study 

**Title (ZH)**: AI副驾助力科学的可重复性：一个案例研究 

**Authors**: Adrien Bibal, Steven N. Minton, Deborah Khider, Yolanda Gil  

**Link**: [PDF](https://arxiv.org/pdf/2506.20130)  

**Abstract**: Open science initiatives seek to make research outputs more transparent, accessible, and reusable, but ensuring that published findings can be independently reproduced remains a persistent challenge. This paper introduces OpenPub, an AI-powered platform that supports researchers, reviewers, and readers through a suite of modular copilots focused on key open science tasks. In this work, we present the Reproducibility Copilot, which analyzes manuscripts, code, and supplementary materials to generate structured Jupyter Notebooks and recommendations aimed at facilitating computational, or "rote", reproducibility. We conducted feasibility tests using previously studied research papers with known reproducibility benchmarks. Results indicate that OpenPub can substantially reduce reproduction time - from over 30 hours to about 1 hour - while achieving high coverage of figures, tables, and results suitable for computational reproduction. The system systematically detects barriers to reproducibility, including missing hyperparameters, undocumented preprocessing steps, and incomplete or inaccessible datasets. These findings suggest that AI-driven tools can meaningfully reduce the burden of reproducibility efforts and contribute to more transparent and verifiable scientific communication. The modular copilot architecture also provides a foundation for extending AI assistance to additional open science objectives beyond reproducibility. 

**Abstract (ZH)**: 开源科学倡议旨在使研究成果更加透明、可访问和可重用，但确保已发表的研究结果可以独立重现仍然是一个持续的挑战。本文介绍了OpenPub，这是一个基于AI的平台，通过专注于关键开源任务的一系列模块化副驾驶，支持研究人员、审稿人和读者。在这项工作中，我们提出了重现性副驾驶，它可以分析手稿、代码和补充材料，生成结构化Jupyter Notebook和建议，以促进计算性或“机械性”的重现性。我们在具有已知重现性基准的先前研究论文上进行了可行性测试。结果表明，OpenPub可以显著减少重现时间——从超过30小时减少到约1小时——同时实现对适合计算性重现的图形、表格和结果的高水平覆盖。该系统系统地检测重现性障碍，包括缺失的超参数、未记录的预处理步骤以及不完整或不可访问的数据集。这些发现表明，以AI驱动的工具可以在减少重现性努力的负担方面发挥实质性作用，并有助于更透明和可验证的科学交流。模块化副驾驶架构也为将AI辅助扩展到重现性之外的其他开源目标奠定了基础。 

---
# Prover Agent: An Agent-based Framework for Formal Mathematical Proofs 

**Title (ZH)**: 证明代理：基于代理的正式数学证明框架 

**Authors**: Kaito Baba, Chaoran Liu, Shuhei Kurita, Akiyoshi Sannai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19923)  

**Abstract**: We present Prover Agent, a novel AI agent for automated theorem proving that integrates large language models (LLMs) with a formal proof assistant, Lean. Prover Agent coordinates an informal reasoning LLM, a formal prover model, and feedback from Lean while also generating auxiliary lemmas to assist in discovering the overall proof strategy. It achieves an 86.1% success rate on the MiniF2F benchmark, establishing a new state-of-the-art among methods using small language models (SLMs) with a much lower sample budget than previous approaches. We also present case studies illustrating how these generated lemmas contribute to solving challenging problems. 

**Abstract (ZH)**: 我们介绍了Prover Agent，这是一种将大型语言模型与形式证明助手Lean集成的新型AI代理，用于自动化定理证明。Prover Agent协调非形式推理的大语言模型、形式证明模型以及来自Lean的反馈，同时生成辅助引理以协助发现整体证明策略。它在MiniF2F基准测试中的成功率达到86.1%，是使用小型语言模型的方法中新的前沿，同时比之前的方法使用了更低的数据样本预算。我们还介绍了案例研究，说明这些生成的引理如何有助于解决复杂问题。 

---
# Define-ML: An Approach to Ideate Machine Learning-Enabled Systems 

**Title (ZH)**: Define-ML：一种机器学习赋能系统构思的方法 

**Authors**: Silvio Alonso, Antonio Pedro Santos Alves, Lucas Romao, Hélio Lopes, Marcos Kalinowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.20621)  

**Abstract**: [Context] The increasing adoption of machine learning (ML) in software systems demands specialized ideation approaches that address ML-specific challenges, including data dependencies, technical feasibility, and alignment between business objectives and probabilistic system behavior. Traditional ideation methods like Lean Inception lack structured support for these ML considerations, which can result in misaligned product visions and unrealistic expectations. [Goal] This paper presents Define-ML, a framework that extends Lean Inception with tailored activities - Data Source Mapping, Feature-to-Data Source Mapping, and ML Mapping - to systematically integrate data and technical constraints into early-stage ML product ideation. [Method] We developed and validated Define-ML following the Technology Transfer Model, conducting both static validation (with a toy problem) and dynamic validation (in a real-world industrial case study). The analysis combined quantitative surveys with qualitative feedback, assessing utility, ease of use, and intent of adoption. [Results] Participants found Define-ML effective for clarifying data concerns, aligning ML capabilities with business goals, and fostering cross-functional collaboration. The approach's structured activities reduced ideation ambiguity, though some noted a learning curve for ML-specific components, which can be mitigated by expert facilitation. All participants expressed the intention to adopt Define-ML. [Conclusion] Define-ML provides an openly available, validated approach for ML product ideation, building on Lean Inception's agility while aligning features with available data and increasing awareness of technical feasibility. 

**Abstract (ZH)**: 定义-机器学习：Lean Inception的扩展框架以系统地将数据和技术约束纳入早期机器学习产品构想 

---
# Weighted Mean Frequencies: a handcraft Fourier feature for 4D Flow MRI segmentation 

**Title (ZH)**: 加权平均频率：一种手工制作的傅里叶特征用于4D流MRI分割 

**Authors**: Simon Perrin, Sébastien Levilly, Huajun Sun, Harold Mouchère, Jean-Michel Serfaty  

**Link**: [PDF](https://arxiv.org/pdf/2506.20614)  

**Abstract**: In recent decades, the use of 4D Flow MRI images has enabled the quantification of velocity fields within a volume of interest and along the cardiac cycle. However, the lack of resolution and the presence of noise in these biomarkers are significant issues. As indicated by recent studies, it appears that biomarkers such as wall shear stress are particularly impacted by the poor resolution of vessel segmentation. The Phase Contrast Magnetic Resonance Angiography (PC-MRA) is the state-of-the-art method to facilitate segmentation. The objective of this work is to introduce a new handcraft feature that provides a novel visualisation of 4D Flow MRI images, which is useful in the segmentation task. This feature, termed Weighted Mean Frequencies (WMF), is capable of revealing the region in three dimensions where a voxel has been passed by pulsatile flow. Indeed, this feature is representative of the hull of all pulsatile velocity voxels. The value of the feature under discussion is illustrated by two experiments. The experiments involved segmenting 4D Flow MRI images using optimal thresholding and deep learning methods. The results obtained demonstrate a substantial enhancement in terms of IoU and Dice, with a respective increase of 0.12 and 0.13 in comparison with the PC-MRA feature, as evidenced by the deep learning task. This feature has the potential to yield valuable insights that could inform future segmentation processes in other vascular regions, such as the heart or the brain. 

**Abstract (ZH)**: 近几十年来，4D Flow MRI图像的使用使得在感兴趣体积和整个心脏周期内定量分析速度场成为可能。然而，这些生物标志物分辨率低且存在噪声是重要问题。最近的研究表明，诸如壁剪应力等生物标志物特别受到血管分割低分辨率的影响。相位对比磁共振血管造影（PC-MRA）是目前最先进的分割方法。本文的目的在于介绍一种新的手工特征，该特征为4D Flow MRI图像提供了一种新颖的可视化方法，有助于分割任务。该特征称为加权均值频率（WMF），能够揭示三维空间中脉动流体通过的体元区域。实际上，该特征代表了所有脉动速度体元的包络。通过两组实验说明了该特征的价值。实验使用了最优阈值分割和深度学习方法来分割4D Flow MRI图像。结果显示，相较于PC-MRA特征，在深度学习任务中IoU和Dice值分别提高了0.12和0.13，证明了该特征的有效性。该特征在其他血管区域，如心脏或大脑的分割过程中有可能提供有价值的见解。 

---
# Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings 

**Title (ZH)**: 通过枪声录音的声学分析 deciphering 枪型层次结构 

**Authors**: Ankit Shah, Rita Singh, Bhiksha Raj, Alexander Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.20609)  

**Abstract**: The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders. 

**Abstract (ZH)**: escalating 枪击事件和群体性枪击事件频率的上升对公共安全构成了重大威胁。及时准确的信息对于执法机构减轻这些事件的影响至关重要。当前的商业枪声检测系统虽然有效，但往往成本高昂。本研究通过利用来自常见设备（如手机）的枪声录音声学分析探索了一种成本效益更高的替代方案，不仅用于检测枪声，还用于识别所使用的枪械类型。本文详细介绍了使用3459条录音制作的精心挑选数据集研究枪械类型层次结构的研究。我们探讨了枪声的基本声学特征，包括枪口爆裂和冲击波，这些特征根据枪械类型、弹药和射击方向而变化。我们提出了机器学习框架，包括支持向量机（SVM）作为基线和更先进的卷积神经网络（CNN）结构，用于联合枪声检测和枪械类型分类。结果表明，在干净标签数据上，我们的深度学习方法平均准确率（mAP）达到0.58，优于SVM基线（mAP 0.39）。还讨论了数据质量、环境噪音以及在使用嘈杂的网络来源数据时的泛化能力（mAP 0.35）带来的挑战。长期愿景是开发一种高度准确的实时系统，可部署在常见录音设备上，显著降低检测成本并为一线紧急响应人员提供关键情报。 

---
# AI in the Writing Process: How Purposeful AI Support Fosters Student Writing 

**Title (ZH)**: AI在写作过程中的应用：目标导向的AI支持对提升学生写作能力的影响 

**Authors**: Momin N. Siddiqui, Roy Pea, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20595)  

**Abstract**: The ubiquity of technologies like ChatGPT has raised concerns about their impact on student writing, particularly regarding reduced learner agency and superficial engagement with content. While standalone chat-based LLMs often produce suboptimal writing outcomes, evidence suggests that purposefully designed AI writing support tools can enhance the writing process. This paper investigates how different AI support approaches affect writers' sense of agency and depth of knowledge transformation. Through a randomized control trial with 90 undergraduate students, we compare three conditions: (1) a chat-based LLM writing assistant, (2) an integrated AI writing tool to support diverse subprocesses, and (3) a standard writing interface (control). Our findings demonstrate that, among AI-supported conditions, students using the integrated AI writing tool exhibited greater agency over their writing process and engaged in deeper knowledge transformation overall. These results suggest that thoughtfully designed AI writing support targeting specific aspects of the writing process can help students maintain ownership of their work while facilitating improved engagement with content. 

**Abstract (ZH)**: 像ChatGPT这样的技术的普遍存在引发了对其对学生写作影响的担忧，特别是关于学习者自主性的降低和内容的浅层参与。虽然独立的基于聊天的大型语言模型往往产生次优的写作效果，但证据表明，目的设计的AI写作支持工具可以增强写作过程。本文探讨不同AI支持方法如何影响作者的自主感和知识深度转换。通过一项包含90名本科生的随机对照试验，我们比较了三种条件：（1）基于聊天的LLM写作助手，（2）集成AI写作工具以支持多样化的次过程，以及（3）标准写作界面（对照）。研究发现，在AI支持的条件下，使用集成AI写作工具的学生在写作过程中表现出更大的自主性，并且总体上参与了更深层次的知识转换。这些结果表明，针对写作过程特定方面的精心设计的AI写作支持可以帮助学生保持对自己作品的所有权，同时促进他们与内容的更好参与。 

---
# Causal Representation Learning with Observational Grouping for CXR Classification 

**Title (ZH)**: 基于观测分组的因果表示学习在胸部X光分类中的应用 

**Authors**: Rajat Rasal, Avinash Kori, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20582)  

**Abstract**: Identifiable causal representation learning seeks to uncover the true causal relationships underlying a data generation process. In medical imaging, this presents opportunities to improve the generalisability and robustness of task-specific latent features. This work introduces the concept of grouping observations to learn identifiable representations for disease classification in chest X-rays via an end-to-end framework. Our experiments demonstrate that these causal representations improve generalisability and robustness across multiple classification tasks when grouping is used to enforce invariance w.r.t race, sex, and imaging views. 

**Abstract (ZH)**: 可识别的因果表示学习旨在揭示数据生成过程中真实的因果关系。在医学成像领域，这为提高特定任务的潜变量特征的一般化能力和稳健性提供了机会。本工作通过端到端框架引入了将观测分组以学习可识别表示的思想，用于胸片疾病分类。我们的实验表明，在分组时强制不变性（针对种族、性别和成像视角），这些因果表示能够提高多种分类任务的一般化能力和稳健性。 

---
# Vulnerability Disclosure through Adaptive Black-Box Adversarial Attacks on NIDS 

**Title (ZH)**: 基于自适应黑盒 adversarial 攻击的 NIDS 漏洞披露 

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi V. Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2506.20576)  

**Abstract**: Adversarial attacks, wherein slight inputs are carefully crafted to mislead intelligent models, have attracted increasing attention. However, a critical gap persists between theoretical advancements and practical application, particularly in structured data like network traffic, where interdependent features complicate effective adversarial manipulations. Moreover, ambiguity in current approaches restricts reproducibility and limits progress in this field. Hence, existing defenses often fail to handle evolving adversarial attacks. This paper proposes a novel approach for black-box adversarial attacks, that addresses these limitations. Unlike prior work, which often assumes system access or relies on repeated probing, our method strictly respect black-box constraints, reducing interaction to avoid detection and better reflect real-world scenarios. We present an adaptive feature selection strategy using change-point detection and causality analysis to identify and target sensitive features to perturbations. This lightweight design ensures low computational cost and high deployability. Our comprehensive experiments show the attack's effectiveness in evading detection with minimal interaction, enhancing its adaptability and applicability in real-world scenarios. By advancing the understanding of adversarial attacks in network traffic, this work lays a foundation for developing robust defenses. 

**Abstract (ZH)**: adversarial 攻击中的黑盒攻击方法：针对结构化数据的新型策略 

---
# DeepQuark: deep-neural-network approach to multiquark bound states 

**Title (ZH)**: DeepQuark：深度神经网络方法研究多夸克束缚态 

**Authors**: Wei-Lin Wu, Lu Meng, Shi-Lin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20555)  

**Abstract**: For the first time, we implement the deep-neural-network-based variational Monte Carlo approach for the multiquark bound states, whose complexity surpasses that of electron or nucleon systems due to strong SU(3) color interactions. We design a novel and high-efficiency architecture, DeepQuark, to address the unique challenges in multiquark systems such as stronger correlations, extra discrete quantum numbers, and intractable confinement interaction. Our method demonstrates competitive performance with state-of-the-art approaches, including diffusion Monte Carlo and Gaussian expansion method, in the nucleon, doubly heavy tetraquark, and fully heavy tetraquark systems. Notably, it outperforms existing calculations for pentaquarks, exemplified by the triply heavy pentaquark. For the nucleon, we successfully incorporate three-body flux-tube confinement interactions without additional computational costs. In tetraquark systems, we consistently describe hadronic molecule $T_{cc}$ and compact tetraquark $T_{bb}$ with an unbiased form of wave function ansatz. In the pentaquark sector, we obtain weakly bound $\bar D^*\Xi_{cc}^*$ molecule $P_{cc\bar c}(5715)$ with $S=\frac{5}{2}$ and its bottom partner $P_{bb\bar b}(15569)$. They can be viewed as the analogs of the molecular $T_{cc}$. We recommend experimental search of $P_{cc\bar c}(5715)$ in the D-wave $J/\psi \Lambda_c$ channel. DeepQuark holds great promise for extension to larger multiquark systems, overcoming the computational barriers in conventional methods. It also serves as a powerful framework for exploring confining mechanism beyond two-body interactions in multiquark states, which may offer valuable insights into nonperturbative QCD and general many-body physics. 

**Abstract (ZH)**: 基于深度神经网络的变分蒙特卡罗方法首次应用于多夸克束缚态，因其复杂性超出电子或核子系统，主要原因在于强烈的SU(3)色相互作用。我们设计了一种新颖且高效的架构DeepQuark，以解决多夸克系统中更强的相关性、额外的离散量子数以及难以处理的束缚相互作用等独特挑战。我们的方法在核子、双重重夸克四夸克系统和完全重夸克四夸克系统中展示了与最先进的方法（包括扩散蒙特卡罗和高斯展开方法）竞争的性能。特别是在五夸克系统中，DeepQuark在三重重夸克五夸克$P_{cc\bar{c}}(5715)$的计算中超越了现有方法，$S=\frac{5}{2}$，以及其底夸克伙伴$P_{bb\bar{b}}(15569)$。它们可以视为四夸克分子$T_{cc}$的类比。我们建议在$D$波$J/\psi \Lambda_c$通道中寻找$P_{cc\bar{c}}(5715)$的实验。DeepQuark在扩展到更大规模的多夸克系统以及探索多夸克态中超越二体相互作用的束缚机制方面具有巨大潜力，这可能为非微扰量子色动力学和广义多体物理学提供宝贵见解。 

---
# WattsOnAI: Measuring, Analyzing, and Visualizing Energy and Carbon Footprint of AI Workloads 

**Title (ZH)**: WattsOnAI：测量、分析和可视化人工智能工作负载的能耗与碳足迹 

**Authors**: Hongzhen Huang, Kunming Zhang, Hanlong Liao, Kui Wu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20535)  

**Abstract**: The rapid advancement of AI, particularly large language models (LLMs), has raised significant concerns about the energy use and carbon emissions associated with model training and inference. However, existing tools for measuring and reporting such impacts are often fragmented, lacking systematic metric integration and offering limited support for correlation analysis among them. This paper presents WattsOnAI, a comprehensive software toolkit for the measurement, analysis, and visualization of energy use, power draw, hardware performance, and carbon emissions across AI workloads. By seamlessly integrating with existing AI frameworks, WattsOnAI offers standardized reports and exports fine-grained time-series data to support benchmarking and reproducibility in a lightweight manner. It further enables in-depth correlation analysis between hardware metrics and model performance and thus facilitates bottleneck identification and performance enhancement. By addressing critical limitations in existing tools, WattsOnAI encourages the research community to weigh environmental impact alongside raw performance of AI workloads and advances the shift toward more sustainable "Green AI" practices. The code is available at this https URL. 

**Abstract (ZH)**: AI的迅猛发展，尤其是大型语言模型（LLMs），引起了人们对模型训练和推理过程中能源使用和碳排放的关注。然而，现有的测量和报告此类影响的工具往往支离破碎，缺乏系统性的度量整合，并且对它们之间的相关性分析支持有限。本文介绍了WattsOnAI，一个全面的软件工具包，用于测量、分析和可视化跨AI工作负载的能源使用、功耗、硬件性能和碳排放。通过与现有AI框架无缝集成，WattsOnAI提供了标准化的报告并导出细粒度的时间序列数据，以轻量级的方式支持基准测试和可重复性。该工具还能够深入分析硬件指标与模型性能之间的相关性，从而促进瓶颈识别和性能提升。通过解决现有工具的关键局限性，WattsOnAI鼓励研究社区在评估AI工作负载的原始性能的同时考虑其环境影响，并推动向更可持续的“绿色AI”实践转变。代码可在以下链接获取：this https URL。 

---
# Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation 

**Title (ZH)**: 基于数字孪生生成数据集和高效数据增强的工业能源拆分 

**Authors**: Christian Internò, Andrea Castellani, Sebastian Schmitt, Fabio Stella, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.20525)  

**Abstract**: Industrial Non-Intrusive Load Monitoring (NILM) is limited by the scarcity of high-quality datasets and the complex variability of industrial energy consumption patterns. To address data scarcity and privacy issues, we introduce the Synthetic Industrial Dataset for Energy Disaggregation (SIDED), an open-source dataset generated using Digital Twin simulations. SIDED includes three types of industrial facilities across three different geographic locations, capturing diverse appliance behaviors, weather conditions, and load profiles. We also propose the Appliance-Modulated Data Augmentation (AMDA) method, a computationally efficient technique that enhances NILM model generalization by intelligently scaling appliance power contributions based on their relative impact. We show in experiments that NILM models trained with AMDA-augmented data significantly improve the disaggregation of energy consumption of complex industrial appliances like combined heat and power systems. Specifically, in our out-of-sample scenarios, models trained with AMDA achieved a Normalized Disaggregation Error of 0.093, outperforming models trained without data augmentation (0.451) and those trained with random data augmentation (0.290). Data distribution analyses confirm that AMDA effectively aligns training and test data distributions, enhancing model generalization. 

**Abstract (ZH)**: 工业非侵入式负荷监测（NILM）受限于高质量数据集的稀缺性和工业能源消耗模式的复杂多变性。为了解决数据稀缺性和隐私问题，我们引入了基于数字孪生模拟生成的开源合成工业分解数据集（SIDED）。SIDED 包含三种不同类型工业设施，覆盖三个不同的地理区域，涵盖了多样化的电器行为、天气条件和负载特性。我们还提出了电器调制数据增强（AMDA）方法，这是一种计算高效的增强技术，通过智能调整电器功率贡献的比例，提高 NILM 模型的泛化能力。实验结果表明，使用 AMDA 增强的数据训练的 NILM 模型显著提高了复杂工业电器（如热电联供系统）的分解性能。具体来说，在我们的泛化场景中，使用 AMDA 增强的数据训练的模型取得了归一化分解误差为 0.093 的成绩，优于未使用数据增强（0.451）和随机数据增强（0.290）的模型。数据分析证实，AMDA 有效地对齐了训练和测试数据分布，提升了模型的泛化能力。 

---
# Counterfactual Influence as a Distributional Quantity 

**Title (ZH)**: 反事实影响作为分布量纲 

**Authors**: Matthieu Meeus, Igor Shilov, Georgios Kaissis, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.20481)  

**Abstract**: Machine learning models are known to memorize samples from their training data, raising concerns around privacy and generalization. Counterfactual self-influence is a popular metric to study memorization, quantifying how the model's prediction for a sample changes depending on the sample's inclusion in the training dataset. However, recent work has shown memorization to be affected by factors beyond self-influence, with other training samples, in particular (near-)duplicates, having a large impact. We here study memorization treating counterfactual influence as a distributional quantity, taking into account how all training samples influence how a sample is memorized. For a small language model, we compute the full influence distribution of training samples on each other and analyze its properties. We find that solely looking at self-influence can severely underestimate tangible risks associated with memorization: the presence of (near-)duplicates seriously reduces self-influence, while we find these samples to be (near-)extractable. We observe similar patterns for image classification, where simply looking at the influence distributions reveals the presence of near-duplicates in CIFAR-10. Our findings highlight that memorization stems from complex interactions across training data and is better captured by the full influence distribution than by self-influence alone. 

**Abstract (ZH)**: 机器学习模型由于记忆训练数据中的样本而存在隐私和泛化方面的顾虑。事实上的影响反事实自影响是研究记忆的一种流行度量，量化模型对样本的预测如何依赖于该样本是否包含在训练数据集中。然而，近期研究表明，记忆不仅受自影响因素的影响，其他训练样本，尤其是（准）副本，也对其有重大影响。我们通过将反事实影响视为分布量来研究记忆，考虑到所有训练样本如何影响一个样本的记忆过程。对于一个小语言模型，我们计算了每个训练样本之间的影响分布，并分析了其性质。我们发现，仅关注自影响严重低估了记忆所带来的实际风险：存在（准）副本会显著降低自影响，而我们发现这些样本是（准）可提取的。我们在图像分类中也观察到类似模式，在 CIFAR-10 中通过观察影响分布揭示了存在近似副本的现象。我们的研究结果强调，记忆源于训练数据间的复杂交互，而完整的影響分布能更好地捕捉记忆现象，而不仅仅是依赖于自影响。 

---
# Off-Policy Evaluation and Learning for the Future under Non-Stationarity 

**Title (ZH)**: 未来非稳态下的离策评估与学习 

**Authors**: Tatsuhiro Shimizu, Kazuki Kawamura, Takanori Muroi, Yusuke Narita, Kei Tateno, Takuma Udagawa, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.20417)  

**Abstract**: We study the novel problem of future off-policy evaluation (F-OPE) and learning (F-OPL) for estimating and optimizing the future value of policies in non-stationary environments, where distributions vary over time. In e-commerce recommendations, for instance, our goal is often to estimate and optimize the policy value for the upcoming month using data collected by an old policy in the previous month. A critical challenge is that data related to the future environment is not observed in the historical data. Existing methods assume stationarity or depend on restrictive reward-modeling assumptions, leading to significant bias. To address these limitations, we propose a novel estimator named \textit{\textbf{O}ff-\textbf{P}olicy Estimator for the \textbf{F}uture \textbf{V}alue (\textbf{\textit{OPFV}})}, designed for accurately estimating policy values at any future time point. The key feature of OPFV is its ability to leverage the useful structure within time-series data. While future data might not be present in the historical log, we can leverage, for example, seasonal, weekly, or holiday effects that are consistent in both the historical and future data. Our estimator is the first to exploit these time-related structures via a new type of importance weighting, enabling effective F-OPE. Theoretical analysis identifies the conditions under which OPFV becomes low-bias. In addition, we extend our estimator to develop a new policy-gradient method to proactively learn a good future policy using only historical data. Empirical results show that our methods substantially outperform existing methods in estimating and optimizing the future policy value under non-stationarity for various experimental setups. 

**Abstract (ZH)**: 未来离策评估与学习（F-OPE和F-OPL）及其在非平稳环境下的未来策略价值估计与优化 

---
# Client Clustering Meets Knowledge Sharing: Enhancing Privacy and Robustness in Personalized Peer-to-Peer Learning 

**Title (ZH)**: 客户端聚类与知识共享：增强个性化peer-to-peer学习中的隐私性和稳健性 

**Authors**: Mohammad Mahdi Maheri, Denys Herasymuk, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20413)  

**Abstract**: The growing adoption of Artificial Intelligence (AI) in Internet of Things (IoT) ecosystems has intensified the need for personalized learning methods that can operate efficiently and privately across heterogeneous, resource-constrained devices. However, enabling effective personalized learning in decentralized settings introduces several challenges, including efficient knowledge transfer between clients, protection of data privacy, and resilience against poisoning attacks. In this paper, we address these challenges by developing P4 (Personalized, Private, Peer-to-Peer) -- a method designed to deliver personalized models for resource-constrained IoT devices while ensuring differential privacy and robustness against poisoning attacks. Our solution employs a lightweight, fully decentralized algorithm to privately detect client similarity and form collaborative groups. Within each group, clients leverage differentially private knowledge distillation to co-train their models, maintaining high accuracy while ensuring robustness to the presence of malicious clients. We evaluate P4 on popular benchmark datasets using both linear and CNN-based architectures across various heterogeneity settings and attack scenarios. Experimental results show that P4 achieves 5% to 30% higher accuracy than leading differentially private peer-to-peer approaches and maintains robustness with up to 30% malicious clients. Additionally, we demonstrate its practicality by deploying it on resource-constrained devices, where collaborative training between two clients adds only ~7 seconds of overhead. 

**Abstract (ZH)**: 人工智能在物联网生态系统中的广泛应用加剧了对高效私密个性化学习方法的需求。然而，在去中心化的环境中实现有效的个性化学习带来了若干挑战，包括在客户端之间高效的知识转移、保护数据隐私以及抵御投毒攻击的鲁棒性。本文通过开发P4（个性化、私密、点对点）方法来应对这些挑战，该方法旨在为资源受限的物联网设备提供个性化模型，同时确保差分隐私并抵御投毒攻击。我们的解决方案采用一种轻量级的完全去中心化算法，以私密方式检测客户端相似性并形成协作组。在每个组内，客户端利用差分隐私的知识蒸馏共同训练模型，在保证高准确率的同时确保在恶意客户端存在时具备鲁棒性。我们在多种异构设置和攻击场景下使用线性架构和CNN架构的流行基准数据集评估了P4。实验结果表明，P4的准确率比领先的差分隐私点对点方法高出5%至30%，并能够容忍高达30%的恶意客户端。此外，我们通过部署在资源受限的设备上展示了其实用性，其中两个客户端之间的协作训练仅增加了约7秒的额外开销。 

---
# Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations 

**Title (ZH)**: 基于谱自助强化和拉普拉斯基扩增的自监督图学习 

**Authors**: Lorenzo Bini, Stephane Marchand-Maillet  

**Link**: [PDF](https://arxiv.org/pdf/2506.20362)  

**Abstract**: We present LaplaceGNN, a novel self-supervised graph learning framework that bypasses the need for negative sampling by leveraging spectral bootstrapping techniques. Our method integrates Laplacian-based signals into the learning process, allowing the model to effectively capture rich structural representations without relying on contrastive objectives or handcrafted augmentations. By focusing on positive alignment, LaplaceGNN achieves linear scaling while offering a simpler, more efficient, self-supervised alternative for graph neural networks, applicable across diverse domains. Our contributions are twofold: we precompute spectral augmentations through max-min centrality-guided optimization, enabling rich structural supervision without relying on handcrafted augmentations, then we integrate an adversarial bootstrapped training scheme that further strengthens feature learning and robustness. Our extensive experiments on different benchmark datasets show that LaplaceGNN achieves superior performance compared to state-of-the-art self-supervised graph methods, offering a promising direction for efficiently learning expressive graph representations. 

**Abstract (ZH)**: LaplaceGNN：一种基于谱增强的自监督图学习框架 

---
# A foundation model with multi-variate parallel attention to generate neuronal activity 

**Title (ZH)**: 具有多变量并行注意力的基础模型生成神经活动 

**Authors**: Francesco Carzaniga, Michael Hersche, Abu Sebastian, Kaspar Schindler, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20354)  

**Abstract**: Learning from multi-variate time-series with heterogeneous channel configurations remains a fundamental challenge for deep neural networks (DNNs), particularly in clinical domains such as intracranial electroencephalography (iEEG), where channel setups vary widely across subjects. In this work, we introduce multi-variate parallel attention (MVPA), a novel self-attention mechanism that disentangles content, temporal, and spatial attention, enabling flexible, generalizable, and efficient modeling of time-series data with varying channel counts and configurations. We use MVPA to build MVPFormer, a generative foundation model for human electrophysiology, trained to predict the evolution of iEEG signals across diverse subjects. To support this and future effort by the community, we release the SWEC iEEG dataset, the largest publicly available iEEG dataset to date, comprising nearly 10,000 hours of recordings from heterogeneous clinical sources. MVPFormer leverages MVPA to achieve strong generalization across subjects, demonstrating expert-level performance in seizure detection and outperforming state-of-the-art Transformer baselines on our SWEC, the MAYO, and the FNUSA dataset. We further validate MVPA on standard time-series forecasting and classification tasks, where it matches or exceeds existing attention-based models. Together, our contributions establish MVPA as a general-purpose attention mechanism for heterogeneous time-series and MVPFormer as the first open-source, open-weights, and open-data iEEG foundation model with state-of-the-art clinical performance. The code is available at this https URL. The SWEC iEEG dataset is available at this https URL. 

**Abstract (ZH)**: 多变量时间序列在异构通道配置下的学习仍然是深度神经网络（DNNs）的基本挑战，尤其是在如颅内电encephalography (iEEG)等临床领域，其中各个被试的通道设置差异很大。在这项工作中，我们介绍了多变量并行注意（MVPA）机制，这是一种新型的自注意力机制，能够解耦内容、时间和空间注意力，从而实现对具有不同通道数量和配置的时间序列数据的灵活、泛化和高效建模。我们使用MVPA构建了MVPFormer，一个用于人类电生理学的生成基础模型，该模型经过训练，可以预测跨不同类型被试的iEEG信号演变。为了支持这一工作中以及未来的社区工作，我们发布了SWEC iEEG数据集，这是迄今为止最大的公开可用的iEEG数据集，包含来自异构临床来源的近10000小时的记录。MVPFormer利用MVPA实现了跨被试的强泛化能力，在癫痫检测方面达到专家级性能，并在我们的SWEC、MAYO和FNUSA数据集上优于最先进的Transformer基线。我们进一步在标准的时间序列预测和分类任务上验证了MVPA，结果显示它与现有的注意机制模型相当或更优。我们的贡献确立了MVPA作为异构时间序列的通用注意机制的地位，同时将MVPFormer确立为首个具有先进临床性能的开源、开放权重、开放数据的iEEG基础模型。代码可在以下链接获取：这个 https URL 数据集可在以下链接获取：这个 https URL 

---
# Comparative Analysis of Deep Learning Models for Crop Disease Detection: A Transfer Learning Approach 

**Title (ZH)**: 基于迁移学习的作物病害检测深度学习模型比较分析 

**Authors**: Saundarya Subramaniam, Shalini Majumdar, Shantanu Nadar, Kaustubh Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20323)  

**Abstract**: This research presents the development of an Artificial Intelligence (AI) - driven crop disease detection system designed to assist farmers in rural areas with limited resources. We aim to compare different deep learning models for a comparative analysis, focusing on their efficacy in transfer learning. By leveraging deep learning models, including EfficientNet, ResNet101, MobileNetV2, and our custom CNN, which achieved a validation accuracy of 95.76%, the system effectively classifies plant diseases. This research demonstrates the potential of transfer learning in reshaping agricultural practices, improving crop health management, and supporting sustainable farming in rural environments. 

**Abstract (ZH)**: 本研究提出了一种基于人工智能的作物疾病检测系统，旨在帮助资源有限的农村地区农民。我们旨在比较不同深度学习模型，侧重于它们在迁移学习中的有效性。通过利用包括EfficientNet、ResNet101、MobileNetV2以及我们自定义的CNN在内的深度学习模型，系统实现了95.76%的验证准确率，有效地分类植物疾病。本研究展示了迁移学习在重塑农业实践、改善作物健康管理以及支持农村可持续农业方面的潜力。 

---
# Beyond-Expert Performance with Limited Demonstrations: Efficient Imitation Learning with Double Exploration 

**Title (ZH)**: 在有限示范下的超越专家性能：双探索增强的高效 imitation 学习 

**Authors**: Heyang Zhao, Xingrui Yu, David M. Bossens, Ivor W. Tsang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20307)  

**Abstract**: Imitation learning is a central problem in reinforcement learning where the goal is to learn a policy that mimics the expert's behavior. In practice, it is often challenging to learn the expert policy from a limited number of demonstrations accurately due to the complexity of the state space. Moreover, it is essential to explore the environment and collect data to achieve beyond-expert performance. To overcome these challenges, we propose a novel imitation learning algorithm called Imitation Learning with Double Exploration (ILDE), which implements exploration in two aspects: (1) optimistic policy optimization via an exploration bonus that rewards state-action pairs with high uncertainty to potentially improve the convergence to the expert policy, and (2) curiosity-driven exploration of the states that deviate from the demonstration trajectories to potentially yield beyond-expert performance. Empirically, we demonstrate that ILDE outperforms the state-of-the-art imitation learning algorithms in terms of sample efficiency and achieves beyond-expert performance on Atari and MuJoCo tasks with fewer demonstrations than in previous work. We also provide a theoretical justification of ILDE as an uncertainty-regularized policy optimization method with optimistic exploration, leading to a regret growing sublinearly in the number of episodes. 

**Abstract (ZH)**: 模仿学习是强化学习中的一个核心问题，目标是学习一个模仿专家行为的策略。在实践中，由于状态空间的复杂性，从有限的演示中准确学习专家策略往往颇具挑战。此外，探索环境和收集数据以实现超越专家的表现至关重要。为克服这些挑战，我们提出了一种新颖的模仿学习算法——双重探索模仿学习（ILDE），该算法从两个方面实现探索：（1）通过探索奖励来乐观的政策优化，奖励具有高不确定性的状态-动作对，以潜在地提高向专家策略收敛的速度；（2）针对与演示轨迹偏差的状态进行好奇心驱动的探索，以潜在地实现超越专家的表现。实验结果表明，ILDE 在样本效率方面优于现有的模仿学习算法，并在使用较少演示的情况下实现了超越专家的表现，特别是在Atari和MuJoCo任务上。我们还从理论上证明了ILDE 是一种正则化不确定性优化方法，具有乐观探索性，其后悔的增长率在episode数量增加时呈次线性增长。 

---
# Argumentative Ensembling for Robust Recourse under Model Multiplicity 

**Title (ZH)**: 基于模型多样性的情况下稳健反事实生成的论证集成方法 

**Authors**: Junqi Jiang, Antonio Rago, Francesco Leofante, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20260)  

**Abstract**: In machine learning, it is common to obtain multiple equally performing models for the same prediction task, e.g., when training neural networks with different random seeds. Model multiplicity (MM) is the situation which arises when these competing models differ in their predictions for the same input, for which ensembling is often employed to determine an aggregation of the outputs. Providing recourse recommendations via counterfactual explanations (CEs) under MM thus becomes complex, since the CE may not be valid across all models, i.e., the CEs are not robust under MM. In this work, we formalise the problem of providing recourse under MM, which we name recourse-aware ensembling (RAE). We propose the idea that under MM, CEs for each individual model should be considered alongside their predictions so that the aggregated prediction and recourse are decided in tandem. Centred around this intuition, we introduce six desirable properties for solutions to this problem. For solving RAE, we propose a novel argumentative ensembling method which guarantees the robustness of CEs under MM. Specifically, our method leverages computational argumentation to explicitly represent the conflicts between models and counterfactuals regarding prediction results and CE validity. It then uses argumentation semantics to resolve the conflicts and obtain the final solution, in a manner which is parametric to the chosen semantics. Our method also allows for the specification of preferences over the models under MM, allowing further customisation of the ensemble. In a comprehensive theoretical analysis, we characterise the behaviour of argumentative ensembling with four different argumentation semantics. We then empirically demonstrate the effectiveness of our approach in satisfying desirable properties with eight instantiations of our method. (Abstract is shortened for arXiv.) 

**Abstract (ZH)**: 在机器学习中，当使用不同的随机种子训练神经网络时，常常会得到多个等效模型。模型多样性（MD）是指这些竞争模型对同一输入的预测结果不一致的情况，常常通过集成方法来确定预测结果的聚合。在模型多样性的情况下，通过反事实解释（CEs）提供纠正建议变得复杂，因为CE可能不适用于所有模型，即CE在MD下缺乏鲁棒性。在本文中，我们形式化了在MD情况下提供纠正建议的问题，并将其命名为意识多样性的集成（RAE）。我们提出，在模型多样性的情况下，每个模型的CE应与其预测一起考虑，以便同时决定聚合预测和纠正建议。围绕这一直觉，我们提出了六种解决方案应具备的 desirable 属性。为了实现RAE，我们提出了一种新的论证型集成方法，确保在模型多样性的情况下CE的鲁棒性。具体而言，我们的方法利用计算论证来明确表示模型和反事实之间的冲突，包括预测结果和CE有效性。然后通过使用论证语义来解决冲突并获得最终解决方案，该过程对所选语义是参数化的。我们的方法还允许指定模型多样性下的偏好，从而进一步定制集成。在全面的理论分析中，我们使用四种不同的论证语义来刻画论证型集成的行为。然后通过八个实例的实验证明，我们的方法能够满足 desirable 属性。 

---
# Time-series surrogates from energy consumers generated by machine learning approaches for long-term forecasting scenarios 

**Title (ZH)**: 由机器学习方法生成的能源消费者时序替代数据用于长期预测场景 

**Authors**: Ben Gerhards, Nikita Popkov, Annekatrin König, Marcel Arpogaus, Bastian Schäfermeier, Leonie Riedl, Stephan Vogt, Philip Hehlert  

**Link**: [PDF](https://arxiv.org/pdf/2506.20253)  

**Abstract**: Forecasting attracts a lot of research attention in the electricity value chain. However, most studies concentrate on short-term forecasting of generation or consumption with a focus on systems and less on individual consumers. Even more neglected is the topic of long-term forecasting of individual power consumption.
Here, we provide an in-depth comparative evaluation of data-driven methods for generating synthetic time series data tailored to energy consumption long-term forecasting. High-fidelity synthetic data is crucial for a wide range of applications, including state estimations in energy systems or power grid planning. In this study, we assess and compare the performance of multiple state-of-the-art but less common techniques: a hybrid Wasserstein Generative Adversarial Network (WGAN), Denoising Diffusion Probabilistic Model (DDPM), Hidden Markov Model (HMM), and Masked Autoregressive Bernstein polynomial normalizing Flows (MABF). We analyze the ability of each method to replicate the temporal dynamics, long-range dependencies, and probabilistic transitions characteristic of individual energy consumption profiles. Our comparative evaluation highlights the strengths and limitations of: WGAN, DDPM, HMM and MABF aiding in selecting the most suitable approach for state estimations and other energy-related tasks. Our generation and analysis framework aims to enhance the accuracy and reliability of synthetic power consumption data while generating data that fulfills criteria like anonymisation - preserving privacy concerns mitigating risks of specific profiling of single customers. This study utilizes an open-source dataset from households in Germany with 15min time resolution. The generated synthetic power profiles can readily be used in applications like state estimations or consumption forecasting. 

**Abstract (ZH)**: 基于数据驱动方法的电力消费长周期预测合成时间序列数据生成对比研究 

---
# FedBKD: Distilled Federated Learning to Embrace Gerneralization and Personalization on Non-IID Data 

**Title (ZH)**: FedBKD：提炼联邦学习以拥抱非IID数据上的泛化能力和个性化能力 

**Authors**: Yushan Zhao, Jinyuan He, Donglai Chen, Weijie Luo, Chong Xie, Ri Zhang, Yonghong Chen, Yan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20245)  

**Abstract**: Federated learning (FL) is a decentralized collaborative machine learning (ML) technique. It provides a solution to the issues of isolated data islands and data privacy leakage in industrial ML practices. One major challenge in FL is handling the non-identical and independent distributed (non-IID) data. Current solutions either focus on constructing an all-powerful global model, or customizing personalized local models. Few of them can provide both a well-generalized global model and well-performed local models at the same time. Additionally, many FL solutions to the non-IID problem are benefited from introducing public datasets. However, this will also increase the risk of data leakage. To tackle the problems, we propose a novel data-free distillation framework, Federated Bidirectional Knowledge Distillation (FedBKD). Specifically, we train Generative Adversarial Networks (GAN) for synthetic data. During the GAN training, local models serve as discriminators and their parameters are frozen. The synthetic data is then used for bidirectional distillation between global and local models to achieve knowledge interactions so that performances for both sides are improved. We conduct extensive experiments on 4 benchmarks under different non-IID settings. The results show that FedBKD achieves SOTA performances in every case. 

**Abstract (ZH)**: 联邦学习（FL）是一种去中心化的协作机器学习（ML）技术。它提供了一种解决工业ML实践中孤立的数据孤岛和数据隐私泄露问题的方案。FL面临的一个主要挑战是如何处理非同质独立分布（non-IID）数据。目前的解决方案要么侧重于构建全能的全局模型，要么定制个性化的局部模型。很少有方法能在同时提供泛化良好的全局模型和性能良好的局部模型方面同时取得成功。此外，许多解决非-IID问题的FL方案得益于引入公共数据集，但这也增加了数据泄露的风险。为了解决这些问题，我们提出了一种新颖的数据免费蒸馏框架，联邦双向知识蒸馏（FedBKD）。具体而言，我们训练生成对抗网络（GAN）生成合成数据。在GAN训练过程中，局部模型充当鉴别器且参数被冻结。合成数据随后用于全局模型和局部模型之间的双向蒸馏，以实现知识交互，从而提高双方的性能。我们在不同非-IID设置下的4个基准上进行了广泛的实验。结果表明，FedBKD在所有情况下都取得了最佳性能。 

---
# Directed Link Prediction using GNN with Local and Global Feature Fusion 

**Title (ZH)**: 基于局部和全局特征融合的图神经网络定向链接预测 

**Authors**: Yuyang Zhang, Xu Shen, Yu Xie, Ka-Chun Wong, Weidun Xie, Chengbin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20235)  

**Abstract**: Link prediction is a classical problem in graph analysis with many practical applications. For directed graphs, recently developed deep learning approaches typically analyze node similarities through contrastive learning and aggregate neighborhood information through graph convolutions. In this work, we propose a novel graph neural network (GNN) framework to fuse feature embedding with community information. We theoretically demonstrate that such hybrid features can improve the performance of directed link prediction. To utilize such features efficiently, we also propose an approach to transform input graphs into directed line graphs so that nodes in the transformed graph can aggregate more information during graph convolutions. Experiments on benchmark datasets show that our approach outperforms the state-of-the-art in most cases when 30%, 40%, 50%, and 60% of the connected links are used as training data, respectively. 

**Abstract (ZH)**: 基于图神经网络的特征嵌入与社区信息融合的有向链接预测方法 

---
# Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems 

**Title (ZH)**: 玩的视角：一种更加包容的NLP系统多视角方法 

**Authors**: Benedetta Muscato, Lucia Passaro, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.20209)  

**Abstract**: In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions. 

**Abstract (ZH)**: 在自然语言处理（NLP）领域，处理人类分歧的常见方法是汇总标注者的观点以建立单一的Ground Truth。然而，先前的研究表明，忽略个人意见可能会导致少数派视角的代表性不足，尤其是在主观任务中，标注者可能会由于其偏好系统性地产生分歧。鉴于标签反映了个体的多样背景、生活经历和价值观，本研究提出了一种新的多视角方法，利用软标签来促进下一代具有视角意识模型的发展，更具包容性和多元性。我们在包括仇恨言论、讽刺、攻击性语言和观点检测在内的多种主观文本分类任务中进行了广泛分析，以强调捕捉人类分歧的重要性，这往往是传统汇总方法忽略的。结果表明，多视角方法不仅在 Jensen-Shannon 散度（JSD）衡量的人类标签分布近似方面更优，同时在分类性能（更高的 F1 分数）方面也优于传统方法。然而，在讽刺和观点检测等任务中，我们的方法表现出较低的置信度，这可能归因于文本中存在的固有主观性。最后，利用可解释人工智能（XAI），我们探索了模型的不确定性并揭示了有关模型预测的有意义见解。 

---
# Affective Priming Score: A Data-Driven Method to Detect Priming in Sequential Datasets 

**Title (ZH)**: 情感启动分数：一种基于数据的方法用于检测序列数据中的启动效应 

**Authors**: Eduardo Gutierrez Maestro, Hadi Banaee, Amy Loutfi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20204)  

**Abstract**: Affective priming exemplifies the challenge of ambiguity in affective computing. While the community has largely addressed this issue from a label-based perspective, identifying data points in the sequence affected by the priming effect, the impact of priming on data itself, particularly in physiological signals, remains underexplored. Data affected by priming can lead to misclassifications when used in learning models. This study proposes the Affective Priming Score (APS), a data-driven method to detect data points influenced by the priming effect. The APS assigns a score to each data point, quantifying the extent to which it is affected by priming. To validate this method, we apply it to the SEED and SEED-VII datasets, which contain sufficient transitions between emotional events to exhibit priming effects. We train models with the same configuration using both the original data and priming-free sequences. The misclassification rate is significantly reduced when using priming-free sequences compared to the original data. This work contributes to the broader challenge of ambiguity by identifying and mitigating priming effects at the data level, enhancing model robustness, and offering valuable insights for the design and collection of affective computing datasets. 

**Abstract (ZH)**: 情感启动体现了情感计算中不确定性的挑战。尽管社区主要从标签的角度来解决这一问题，标识序列中受启动效应影响的数据点，启动效应对数据本身的影响，尤其是在生理信号方面，仍需进一步探索。受启动效应影响的数据可能导致在学习模型中出现误分类。本研究提出情感启动评分（APS），这是一种数据驱动的方法，用于检测受启动效应影响的数据点。APS为每个数据点分配一个得分，量化其受启动效应影响的程度。为验证该方法，我们在SEED和SEED-VII数据集上应用此方法，这两个数据集包含足够的情绪事件转换，以展示启动效应。使用去启动序列与原始数据训练相同配置的模型。使用去启动序列的错误分类率显著低于使用原始数据的模型。本研究通过在数据层面识别和缓解启动效应，为更广泛的不确定性挑战做出了贡献，增强了模型的稳健性，并提供了有关情感计算数据集设计和收集的宝贵见解。 

---
# COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees 

**Title (ZH)**: COIN：具有可证明风险保证的不确定性保护选择性问答基础模型 

**Authors**: Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20178)  

**Abstract**: Uncertainty quantification (UQ) for foundation models is essential to identify and mitigate potential hallucinations in automatically generated text. However, heuristic UQ approaches lack formal guarantees for key metrics such as the false discovery rate (FDR) in selective prediction. Previous work adopts the split conformal prediction (SCP) framework to ensure desired coverage of admissible answers by constructing prediction sets, but these sets often contain incorrect candidates, limiting their practical utility. To address this, we propose COIN, an uncertainty-guarding selection framework that calibrates statistically valid thresholds to filter a single generated answer per question under user-specified FDR constraints. COIN estimates the empirical error rate on a calibration set and applies confidence interval methods such as Clopper-Pearson to establish a high-probability upper bound on the true error rate (i.e., FDR). This enables the selection of the largest uncertainty threshold that ensures FDR control on test data while significantly increasing sample retention. We demonstrate COIN's robustness in risk control, strong test-time power in retaining admissible answers, and predictive efficiency under limited calibration data across both general and multimodal text generation tasks. Furthermore, we show that employing alternative upper bound constructions and UQ strategies can further boost COIN's power performance, which underscores its extensibility and adaptability to diverse application scenarios. 

**Abstract (ZH)**: 基础模型的不确定性量化（UQ）对于识别和缓解自动生成文本中的幻觉至关重要。然而，启发式的UQ方法在确保选择性预测中的假发现率（FDR）等方面缺乏正式保证。先前工作采用分割置信预测（SCP）框架通过构建预测集来确保可接受答案的覆盖范围，但这些集往往包含错误的候选项，限制了其实用价值。为解决这一问题，我们提出COIN，这是一种不确定性保护的选择框架，通过在用户指定的FDR约束下校准统计有效的阈值来筛选每个问题的单个生成答案。COIN在校准集上估计经验错误率，并应用Clopper-Pearson等置信区间方法来建立真实错误率（即FDR）的高概率上限。这使得在显著增加样本保留的情况下，通过选择确保测试数据中的FDR控制的不确定性阈值成为可能。我们展示了COIN在风险控制中的稳健性、在保留可接受答案方面的强大测试时效能以及在有限校准数据下对通用和多模态文本生成任务的预测效率。此外，我们表明采用替代的上限构建方法和不确定性量化策略可以进一步提升COIN的效能，这突显了其在多种应用场景中的扩展性和适应性。 

---
# Valid Selection among Conformal Sets 

**Title (ZH)**: 在同构集中的有效选择 

**Authors**: Mahmoud Hegazy, Liviu Aolaritei, Michael I. Jordan, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20173)  

**Abstract**: Conformal prediction offers a distribution-free framework for constructing prediction sets with coverage guarantees. In practice, multiple valid conformal prediction sets may be available, arising from different models or methodologies. However, selecting the most desirable set, such as the smallest, can invalidate the coverage guarantees. To address this challenge, we propose a stability-based approach that ensures coverage for the selected prediction set. We extend our results to the online conformal setting, propose several refinements in settings where additional structure is available, and demonstrate its effectiveness through experiments. 

**Abstract (ZH)**: 基于稳定性的可覆盖性预测集选择方法及其在线 conformal 设置的拓展 

---
# Do psychic cells generate consciousness? 

**Title (ZH)**: 灵性细胞是否产生意识？ 

**Authors**: Mototaka Suzuki, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2506.20164)  

**Abstract**: Technological advances in the past decades have begun to enable neuroscientists to address fundamental questions about consciousness in an unprecedented way. Here we review remarkable recent progress in our understanding of cellular-level mechanisms of conscious processing in the brain. Of particular interest are the cortical pyramidal neurons -- or "psychic cells" called by Ramón y Cajal more than 100 years ago -- which have an intriguing cellular mechanism that accounts for selective disruption of feedback signaling in the brain upon anesthetic-induced loss of consciousness. Importantly, a particular class of metabotropic receptors distributed over the dendrites of pyramidal cells are highlighted as the key cellular mechanism. After all, Cajal's instinct over a century ago may turn out to be correct -- we may have just begun to understand whether and how psychic cells indeed generate and control our consciousness. 

**Abstract (ZH)**: 过去的几十年中，技术的进步已经开始使神经科学家以前所未有的方式探究意识的基本问题。本文回顾了对大脑中意识处理细胞机制的最新理解进展。特别是皮层尖*spiny*神经元——拉蒙·耶·卡哈尔一百多年前称其为“心理细胞”——具有一个引人注目的细胞机制，可以解释在麻醉导致意识丧失时反馈信号的选择性中断。重要的是，分布在尖神经元树突上的特定类型代谢型受体被突出显示为关键的细胞机制。毕竟，一个多世纪前卡哈尔的直觉可能是正确的——我们可能才刚刚开始理解心理细胞是否以及如何生成和控制我们的意识。 

---
# AI and Agile Software Development: From Frustration to Success -- XP2025 Workshop Summary 

**Title (ZH)**: AI和敏捷软件开发：从挫折到成功——XP2025研讨会摘要 

**Authors**: Tomas Herda, Victoria Pichler, Zheying Zhang, Pekka Abrahamsson, Geir K. Hanssen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20159)  

**Abstract**: The full-day workshop on AI and Agile at XP 2025 convened a diverse group of researchers and industry practitioners to address the practical challenges and opportunities of integrating Artificial Intelligence into Agile software development. Through interactive sessions, participants identified shared frustrations related to integrating AI into Agile Software Development practices, including challenges with tooling, governance, data quality, and critical skill gaps. These challenges were systematically prioritized and analyzed to uncover root causes. The workshop culminated in the collaborative development of a research roadmap that pinpoints actionable directions for future work, including both immediate solutions and ambitious long-term goals. The key outcome is a structured agenda designed to foster joint industry-academic efforts to move from identified frustrations to successful implementation. 

**Abstract (ZH)**: 全日工作坊：AI与极限编程2025中的敏捷开发 

---
# Irec: A Metacognitive Scaffolding for Self-Regulated Learning through Just-in-Time Insight Recall: A Conceptual Framework and System Prototype 

**Title (ZH)**: Irec：一种即时洞察回忆的元认知支架以促进自我调节学习：一个概念框架及系统原型 

**Authors**: Xuefei Hou, Xizhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20156)  

**Abstract**: The core challenge in learning has shifted from knowledge acquisition to effective Self-Regulated Learning (SRL): planning, monitoring, and reflecting on one's learning. Existing digital tools, however, inadequately support metacognitive reflection. Spaced Repetition Systems (SRS) use de-contextualized review, overlooking the role of context, while Personal Knowledge Management (PKM) tools require high manual maintenance.
To address these challenges, this paper introduces "Insight Recall," a novel paradigm that conceptualizes the context-triggered retrieval of personal past insights as a metacognitive scaffold to promote SRL. We formalize this paradigm using the Just-in-Time Adaptive Intervention (JITAI) framework and implement a prototype system, Irec, to demonstrate its feasibility. At its core, Irec uses a dynamic knowledge graph of the user's learning history. When a user faces a new problem, a hybrid retrieval engine recalls relevant personal "insights." Subsequently, a large language model (LLM) performs a deep similarity assessment to filter and present the most relevant scaffold in a just-in-time manner. To reduce cognitive load, Irec features a human-in-the-loop pipeline for LLM-based knowledge graph construction. We also propose an optional "Guided Inquiry" module, where users can engage in a Socratic dialogue with an expert LLM, using the current problem and recalled insights as context. The contribution of this paper is a solid theoretical framework and a usable system platform for designing next-generation intelligent learning systems that enhance metacognition and self-regulation. 

**Abstract (ZH)**: 学习的核心挑战已从知识获取转变为有效的自我调节学习（SRL）：规划、监控和反思自己的学习。然而，现有数字工具在支持元认知反思方面做得不够。间隔重复系统（SRS）采用脱嵌式的复习，忽视了情境的作用，而个人知识管理（PKM）工具需要大量的手动维护。

为应对这些挑战，本文提出了“洞察回忆”这一新的范式，将情境触发的个人过往洞察回忆概念化为元认知支架，以促进自我调节学习（SRL）。本文使用即时适配干预（JITAI）框架形式化了这一范式，并构建了一个原型系统Irec，以证明其实用性。Irec的核心在于使用用户学习历史的动态知识图谱。当用户遇到新问题时，混合检索引擎会召回相关个人“洞察”。随后，大型语言模型（LLM）执行深度相似性评估，以实时方式过滤并呈现最相关的支架。为减轻认知负担，Irec配备了一个包含人类在环的流水线，用于基于LLM的知识图谱构建。此外，本文还提出了一个可选的“引导性探究”模块，用户可以在其中与专家LLM进行苏格拉底式的对话，以当前问题和召回的洞察作为背景。本文的贡献在于提供了一个坚实理论框架和一个实用系统平台，用于设计增强元认知和自我调节能力的下一代智能学习系统。 

---
# Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural Network Acceleration 

**Title (ZH)**: 基于损失感知的结构化剪枝标准自动选择方法以加速深度神经网络 

**Authors**: Deepak Ghimire, Kilho Lee, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.20152)  

**Abstract**: Structured pruning is a well-established technique for compressing neural networks, making it suitable for deployment in resource-limited edge devices. This paper presents an efficient Loss-Aware Automatic Selection of Structured Pruning Criteria (LAASP) for slimming and accelerating deep neural networks. The majority of pruning methodologies employ a sequential process consisting of three stages: 1) training, 2) pruning, and 3) fine-tuning, whereas the proposed pruning technique adopts a pruning-while-training approach that eliminates the first stage and integrates the second and third stages into a single cycle. The automatic selection of magnitude or similarity-based filter pruning criteria from a specified pool of criteria and the specific pruning layer at each pruning iteration is guided by the network's overall loss on a small subset of the training data. To mitigate the abrupt accuracy drop due to pruning, the network is retrained briefly after each reduction of a predefined number of floating-point operations (FLOPs). The optimal pruning rates for each layer in the network are automatically determined, eliminating the need for manual allocation of fixed or variable pruning rates for each layer. Experiments on the VGGNet and ResNet models on the CIFAR-10 and ImageNet benchmark datasets demonstrate the effectiveness of the proposed method. In particular, the ResNet56 and ResNet110 models on the CIFAR-10 dataset significantly improve the top-1 accuracy compared to state-of-the-art methods while reducing the network FLOPs by 52\%. Furthermore, the ResNet50 model on the ImageNet dataset reduces FLOPs by more than 42\% with a negligible 0.33\% drop in top-5 accuracy. The source code of this paper is publicly available online - this https URL. 

**Abstract (ZH)**: 基于损失感知的结构剪枝自动选择标准（LAASP）: 一种高效 slimming 和加速深度神经网络的方法 

---
# SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization 

**Title (ZH)**: SACL：通过语义增强重排和定位理解并对抗代码检索中的文本偏见 

**Authors**: Dhruv Gupta, Gayathri Ganesh Lakshmy, Yiqing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20081)  

**Abstract**: Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is this http URL on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval). 

**Abstract (ZH)**: 检索增强代码生成（RACG）是通过检索相关信息来提升代码生成的关键技术。在本工作中，我们系统地屏蔽特定特征以保留代码功能，进行了深入的代码检索分析。我们的发现包括：(1) 尽管基于代码训练，当前的检索器严重依赖表面级文本特征（例如，文档字符串、标识符名称），(2) 并且它们强烈偏向于有良好文档的代码，即使这些文档存在问题。鉴于上述发现，我们提出了SACL框架，该框架通过用语义信息丰富文本信息并增加代码或结构知识来减轻偏见。广泛的实验表明，SACL显著改进了代码检索（例如，在HumanEval / MBPP / SWE-Bench-Lite上的Recall@1分别提高了12.8% / 9.4% / 7.0%），这也导致了更好的代码生成性能（例如，在HumanEval上的Pass@1提高了4.88%）。 

---
# Beyond Autocomplete: Designing CopilotLens Towards Transparent and Explainable AI Coding Agents 

**Title (ZH)**: 超越自动补全：设计透明可解释的CopilotLens代码伴侣agents 

**Authors**: Runlong Ye, Zeling Zhang, Boushra Almazroua, Michael Liut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20062)  

**Abstract**: AI-powered code assistants are widely used to generate code completions, significantly boosting developer productivity. However, these tools typically present suggestions without explaining their rationale, leaving their decision-making process inscrutable. This opacity hinders developers' ability to critically evaluate the output, form accurate mental models, and build calibrated trust in the system. To address this, we introduce CopilotLens, a novel interactive framework that reframes code completion from a simple suggestion into a transparent, explainable event. CopilotLens operates as an explanation layer that reveals the AI agent's "thought process" through a dynamic two-level interface, surfacing everything from its reconstructed high-level plans to the specific codebase context influencing the code. This paper presents the design and rationale of CopilotLens, offering a concrete framework for building future agentic code assistants that prioritize clarity of reasoning over speed of suggestion, thereby fostering deeper comprehension and more robust human-AI collaboration. 

**Abstract (ZH)**: 基于AI的代码助手广泛用于生成代码补全，显著提升开发者生产力。然而，这些工具通常不解释其建议的理由，使开发者难以评估输出，形成准确的心理模型，并建立对系统的校准信任。为解决这一问题，我们介绍了CopilotLens，一个新颖的交互框架，将代码补全从简单的建议重新定义为透明可解释的事件。CopilotLens 作为一个解释层，通过动态的双层用户界面揭示AI代理的“思维过程”，从其重建的高阶计划到具体代码上下文的影响机制。本文阐述了CopilotLens的设计与 rationale，提供了一个具体的框架，用于构建优先考虑推理清晰度而非建议速度的未来代理型代码助手，从而促进更深入的理解和更稳健的人机协作。 

---
# GNN's Uncertainty Quantification using Self-Distillation 

**Title (ZH)**: GNN的不确定性量化研究：基于自我蒸馏的方法 

**Authors**: Hirad Daneshvar, Reza Samavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20046)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in the healthcare domain. However, what remained challenging is quantifying the predictive uncertainty of GNNs, which is an important aspect of trustworthiness in clinical settings. While Bayesian and ensemble methods can be used to quantify uncertainty, they are computationally expensive. Additionally, the disagreement metric used by ensemble methods to compute uncertainty cannot capture the diversity of models in an ensemble network. In this paper, we propose a novel method, based on knowledge distillation, to quantify GNNs' uncertainty more efficiently and with higher precision. We apply self-distillation, where the same network serves as both the teacher and student models, thereby avoiding the need to train several networks independently. To ensure the impact of self-distillation, we develop an uncertainty metric that captures the diverse nature of the network by assigning different weights to each GNN classifier. We experimentally evaluate the precision, performance, and ability of our approach in distinguishing out-of-distribution data on two graph datasets: MIMIC-IV and Enzymes. The evaluation results demonstrate that the proposed method can effectively capture the predictive uncertainty of the model while having performance similar to that of the MC Dropout and ensemble methods. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于知识蒸馏的图神经网络不确定性量化方法 

---
# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification 

**Title (ZH)**: LSH-DynED：一种基于LSH下采样的动态集成框架用于 evolving 多类别不平衡分类 

**Authors**: Soheil Abadifard, Fazli Can  

**Link**: [PDF](https://arxiv.org/pdf/2506.20041)  

**Abstract**: The classification of imbalanced data streams, which have unequal class distributions, is a key difficulty in machine learning, especially when dealing with multiple classes. While binary imbalanced data stream classification tasks have received considerable attention, only a few studies have focused on multi-class imbalanced data streams. Effectively managing the dynamic imbalance ratio is a key challenge in this domain. This study introduces a novel, robust, and resilient approach to address these challenges by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. To the best of our knowledge, we present the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams. The proposed method undersamples the majority classes by utilizing LSH-RHP, provides a balanced training set, and improves the ensemble's prediction performance. We conduct comprehensive experiments on 23 real-world and ten semi-synthetic datasets and compare LSH-DynED with 15 state-of-the-art methods. The results reveal that LSH-DynED outperforms other approaches in terms of both Kappa and mG-Mean effectiveness measures, demonstrating its capability in dealing with multi-class imbalanced non-stationary data streams. Notably, LSH-DynED performs well in large-scale, high-dimensional datasets with considerable class imbalances and demonstrates adaptation and robustness in real-world circumstances. To motivate our design, we review existing methods for imbalanced data streams, outline key challenges, and offer guidance for future work. For the reproducibility of our results, we have made our implementation available on GitHub. 

**Abstract (ZH)**: 不平衡数据流的分类：一种基于局部敏感哈希与随机超平面投影的动态ensemble多样化方法 

---
# Learning Bilateral Team Formation in Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 学习合作多智能体强化学习中的双边团队形成 

**Authors**: Koorosh Moslemi, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20039)  

**Abstract**: Team formation and the dynamics of team-based learning have drawn significant interest in the context of Multi-Agent Reinforcement Learning (MARL). However, existing studies primarily focus on unilateral groupings, predefined teams, or fixed-population settings, leaving the effects of algorithmic bilateral grouping choices in dynamic populations underexplored. To address this gap, we introduce a framework for learning two-sided team formation in dynamic multi-agent systems. Through this study, we gain insight into what algorithmic properties in bilateral team formation influence policy performance and generalization. We validate our approach using widely adopted multi-agent scenarios, demonstrating competitive performance and improved generalization in most scenarios. 

**Abstract (ZH)**: 多Agent强化学习（MARL）背景下动态团队形成及动态团队学习的机制和动态性引起了广泛关注。然而，现有研究主要集中在单边分组、预定义团队或固定群体设置上，忽视了动态群体中算法双边分组选择的影响。为填补这一空白，我们提出了一种学习动态多Agent系统中双边团队形成的方法。通过本研究，我们探究了双边团队形成中算法属性如何影响策略性能和泛化能力。我们利用广泛采用的多Agent场景验证了该方法，展示了在大多数场景中的竞争性能和增强的泛化能力。 

---
# Automated Generation of Diverse Courses of Actions for Multi-Agent Operations using Binary Optimization and Graph Learning 

**Title (ZH)**: 使用二元优化和图学习生成多智能体操作的多样化行动方案的自动化生成 

**Authors**: Prithvi Poddar, Ehsan Tarkesh Esfahani, Karthik Dantu, Souma Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.20031)  

**Abstract**: Operations in disaster response, search \& rescue, and military missions that involve multiple agents demand automated processes to support the planning of the courses of action (COA). Moreover, traverse-affecting changes in the environment (rain, snow, blockades, etc.) may impact the expected performance of a COA, making it desirable to have a pool of COAs that are diverse in task distributions across agents. Further, variations in agent capabilities, which could be human crews and/or autonomous systems, present practical opportunities and computational challenges to the planning process. This paper presents a new theoretical formulation and computational framework to generate such diverse pools of COAs for operations with soft variations in agent-task compatibility. Key to the problem formulation is a graph abstraction of the task space and the pool of COAs itself to quantify its diversity. Formulating the COAs as a centralized multi-robot task allocation problem, a genetic algorithm is used for (order-ignoring) allocations of tasks to each agent that jointly maximize diversity within the COA pool and overall compatibility of the agent-task mappings. A graph neural network is trained using a policy gradient approach to then perform single agent task sequencing in each COA, which maximizes completion rates adaptive to task features. Our tests of the COA generation process in a simulated environment demonstrate significant performance gain over a random walk baseline, small optimality gap in task sequencing, and execution time of about 50 minutes to plan up to 20 COAs for 5 agent/100 task operations. 

**Abstract (ZH)**: 灾害响应、搜索与救援及军事任务中涉及多agent的操作需要自动化流程支持行动方案（COA）的规划。环境变化（如降雨、降雪、封锁等）可能影响COA的预期性能，因此需要一个在agent间任务分配上多样化的COA池。此外，agent能力的差异，包括人类班组和/or自主系统，为规划过程带来了实际机遇和计算挑战。本文提出了一种新的理论框架和计算方法，以生成在agent-task兼容性软变化情况下的多样化COA池。问题的核心在于任务空间及COA池的图抽象表示，以量化多样性。将COA表示为集中式多机器人任务分配问题，使用遗传算法进行忽略顺序的任务分配，以在COA池内最大化多样性，并优化agent-task映射的整体兼容性。通过策略梯度方法训练图神经网络，以在每个COA中进行单个agent的任务序列，从而最大化适应任务特征的完成率。在模拟环境中测试COA生成过程显示，相比于随机漫步基准，显著提高了性能，任务序列的优化差距较小，规划20个COA的执行时间为约50分钟。 

---
# Elucidated Rolling Diffusion Models for Probabilistic Weather Forecasting 

**Title (ZH)**: 阐述性滚动扩散模型在概率天气预报中的应用 

**Authors**: Salva Rühling Cachay, Miika Aittala, Karsten Kreis, Noah Brenowitz, Arash Vahdat, Morteza Mardani, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20024)  

**Abstract**: Diffusion models are a powerful tool for probabilistic forecasting, yet most applications in high-dimensional chaotic systems predict future snapshots one-by-one. This common approach struggles to model complex temporal dependencies and fails to explicitly account for the progressive growth of uncertainty inherent to such systems. While rolling diffusion frameworks, which apply increasing noise to forecasts at longer lead times, have been proposed to address this, their integration with state-of-the-art, high-fidelity diffusion techniques remains a significant challenge. We tackle this problem by introducing Elucidated Rolling Diffusion Models (ERDM), the first framework to successfully unify a rolling forecast structure with the principled, performant design of Elucidated Diffusion Models (EDM). To do this, we adapt the core EDM components-its noise schedule, network preconditioning, and Heun sampler-to the rolling forecast setting. The success of this integration is driven by three key contributions: (i) a novel loss weighting scheme that focuses model capacity on the mid-range forecast horizons where determinism gives way to stochasticity; (ii) an efficient initialization strategy using a pre-trained EDM for the initial window; and (iii) a bespoke hybrid sequence architecture for robust spatiotemporal feature extraction under progressive denoising. On 2D Navier-Stokes simulations and ERA5 global weather forecasting at 1.5^\circ resolution, ERDM consistently outperforms key diffusion-based baselines, including conditional autoregressive EDM. ERDM offers a flexible and powerful general framework for tackling diffusion-based sequence generation problems where modeling escalating uncertainty is paramount. Code is available at: this https URL 

**Abstract (ZH)**: 阐明滚动扩散模型（ERDM）: 首个结合滚动预测结构与原则性高性能设计的扩散模型框架 

---
# New Insights on Unfolding and Fine-tuning Quantum Federated Learning 

**Title (ZH)**: 新见解：展开与细调量子联邦学习 

**Authors**: Shanika Iroshi Nanayakkara, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2506.20016)  

**Abstract**: Client heterogeneity poses significant challenges to the performance of Quantum Federated Learning (QFL). To overcome these limitations, we propose a new approach leveraging deep unfolding, which enables clients to autonomously optimize hyperparameters, such as learning rates and regularization factors, based on their specific training behavior. This dynamic adaptation mitigates overfitting and ensures robust optimization in highly heterogeneous environments where standard aggregation methods often fail. Our framework achieves approximately 90% accuracy, significantly outperforming traditional methods, which typically yield around 55% accuracy, as demonstrated through real-time training on IBM quantum hardware and Qiskit Aer simulators. By developing self adaptive fine tuning, the proposed method proves particularly effective in critical applications such as gene expression analysis and cancer detection, enhancing diagnostic precision and predictive modeling within quantum systems. Our results are attributed to convergence-aware, learnable optimization steps intrinsic to the deep unfolded framework, which maintains the generalization. Hence, this study addresses the core limitations of conventional QFL, streamlining its applicability to any complex challenges such as healthcare and genomic research. 

**Abstract (ZH)**: 客户端异质性对量子联邦学习（QFL）的性能构成了重大挑战。为克服这些限制，我们提出了一种基于深度展开的新方法，该方法使客户端能够根据其特定的训练行为自主优化超参数，如学习率和正则化因子。这种动态适应性减轻了过拟合，确保在标准聚合方法常失败的高度异质环境中实现稳健优化。我们的框架实现了约90%的准确率，显著优于传统的约55%的准确率，这已在IBM量子硬件和Qiskit Aer仿真器上的实时训练中得到验证。通过开发自我适应的精细调整，所提出的方法尤其有效于基因表达分析和癌症检测等关键应用中，增强了量子系统中的诊断精确度和预测建模。本研究将归因于深度展开框架中固有的、关注收敛性的可学习优化步骤，从而保持泛化能力。因此，这项研究解决了传统QFL的核心局限，使其更易于应用于任何复杂的挑战，如医疗保健和基因组研究。 

---
# Quantum Neural Networks for Propensity Score Estimation and Survival Analysis in Observational Biomedical Studies 

**Title (ZH)**: 量子神经网络在观察 biomedical 研究中倾向评分估计和生存分析中的应用 

**Authors**: Vojtěch Novák, Ivan Zelinka, Lenka Přibylová, Lubomír Martínek  

**Link**: [PDF](https://arxiv.org/pdf/2506.19973)  

**Abstract**: This study investigates the application of quantum neural networks (QNNs) for propensity score estimation to address selection bias in comparing survival outcomes between laparoscopic and open surgical techniques in a cohort of 1177 colorectal carcinoma patients treated at University Hospital Ostrava (2001-2009). Using a dataset with 77 variables, including patient demographics and tumor characteristics, we developed QNN-based propensity score models focusing on four key covariates (Age, Sex, Stage, BMI). The QNN architecture employed a linear ZFeatureMap for data encoding, a SummedPaulis operator for predictions, and the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for robust, gradient-free optimization in noisy quantum environments. Variance regularization was integrated to mitigate quantum measurement noise, with simulations conducted under exact, sampling (1024 shots), and noisy hardware (FakeManhattanV2) conditions. QNNs, particularly with simulated hardware noise, outperformed classical logistic regression and gradient boosted machines in small samples (AUC up to 0.750 for n=100), with noise modeling enhancing predictive stability. Propensity score matching and weighting, optimized via genetic matching and matching weights, achieved covariate balance with standardized mean differences of 0.0849 and 0.0869, respectively. Survival analyses using Kaplan-Meier estimation, Cox proportional hazards, and Aalen additive regression revealed no significant survival differences post-adjustment (p-values 0.287-0.851), indicating confounding bias in unadjusted outcomes. These results highlight QNNs' potential, enhanced by CMA-ES and noise-aware strategies, to improve causal inference in biomedical research, particularly for small-sample, high-dimensional datasets. 

**Abstract (ZH)**: 量子神经网络在队列研究中用于腹腔镜与开放手术技术生存结果比较中的倾向评分估计应用：大学医院奥strava（2001-2009年）1177例结直肠癌患者的倾向评分建模研究 

---
# An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking 

**Title (ZH)**: 从头算原理模型的波函数基础：准确描述化学键断裂 

**Authors**: Adam Foster, Zeno Schätzle, P. Bernát Szabó, Lixue Cheng, Jonas Köhler, Gino Cassella, Nicholas Gao, Jiawei Li, Frank Noé, Jan Hermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.19960)  

**Abstract**: Reliable description of bond breaking remains a major challenge for quantum chemistry due to the multireferential character of the electronic structure in dissociating species. Multireferential methods in particular suffer from large computational cost, which under the normal paradigm has to be paid anew for each system at a full price, ignoring commonalities in electronic structure across molecules. Quantum Monte Carlo with deep neural networks (deep QMC) uniquely offers to exploit such commonalities by pretraining transferable wavefunction models, but all such attempts were so far limited in scope. Here, we bring this new paradigm to fruition with Orbformer, a novel transferable wavefunction model pretrained on 22,000 equilibrium and dissociating structures that can be fine-tuned on unseen molecules reaching an accuracy-cost ratio rivalling classical multireferential methods. On established benchmarks as well as more challenging bond dissociations and Diels-Alder reactions, Orbformer is the only method that consistently converges to chemical accuracy (1 kcal/mol). This work turns the idea of amortizing the cost of solving the Schrödinger equation over many molecules into a practical approach in quantum chemistry. 

**Abstract (ZH)**: 可靠的键断裂描述仍然是量子化学中的一个重大挑战，尤其是在解离物种的多重参量电子结构特性下。特别是多重参量方法受到大量计算成本的困扰，通常在处理每个系统时都需要重新计算，无视分子间电子结构的共同性。通过深度神经网络进行量子蒙特卡罗（深度QMC）唯一地提供了通过先验训练可转移波函数模型利用这些共同性的机会，但迄今为止此类尝试的范围都有限。在此，我们通过Orbformer——一种在22,000个平衡和解离结构上预训练的新型可转移波函数模型，实现了这一新范式，该模型可以对未见过的分子进行微调，达到与经典多重参量方法媲美的准确度-成本比。在标准基准以及更具挑战性的键解离和狄耳 erklärt-阿尔德反应中，Orbformer是唯一一种能够一致地达到化学精度（1 kcal/mol）的方法。这项工作将解决薛定谔方程的计算成本在多个分子上的分摊理念转变为量子化学中的实用方法。 

---
# A Framework for Uncertainty Quantification Based on Nearest Neighbors Across Layers 

**Title (ZH)**: 基于跨层最近邻的不确定性量化框架 

**Authors**: Miguel N. Font, José L. Jorro-Aragoneses, Carlos M. Alaíz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19895)  

**Abstract**: Neural Networks have high accuracy in solving problems where it is difficult to detect patterns or create a logical model. However, these algorithms sometimes return wrong solutions, which become problematic in high-risk domains like medical diagnosis or autonomous driving. One strategy to detect and mitigate these errors is the measurement of the uncertainty over neural network decisions. In this paper, we present a novel post-hoc framework for measuring the uncertainty of a decision based on retrieved training cases that have a similar activation vector to the query for each layer. Based on these retrieved cases, we propose two new metrics: Decision Change and Layer Uncertainty, which capture changes in nearest-neighbor class distributions across layers. We evaluated our approach in a classification model for two datasets: CIFAR-10 and MNIST. The results show that these metrics enhance uncertainty estimation, especially in challenging classification tasks, outperforming softmax-based confidence. 

**Abstract (ZH)**: 神经网络在难以检测模式或创建逻辑模型的问题上具有高精度，但在医疗诊断或自动驾驶等高风险领域中有时会返回错误的解，这成为问题。一种检测和减轻这些错误的策略是对神经网络决策的不确定性进行测量。本文提出了一种新的后处理框架，基于查询在同一层中具有相似激活向量的训练案例来衡量决策的不确定性。基于这些检索到的案例，我们提出了两个新的度量标准：决策变化和层不确定性，它们捕捉各层之间最近邻类别分布的变化。我们在CIFAR-10和MNIST两个数据集上的分类模型中评估了该方法。结果表明，这些指标在困难的分类任务中增强了不确定性估计，优于基于softmax的置信度。 

---
# Explaining deep neural network models for electricity price forecasting with XAI 

**Title (ZH)**: 使用XAI解释深度神经网络模型的电力价格预测 

**Authors**: Antoine Pesenti, Aidan OSullivan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19894)  

**Abstract**: Electricity markets are highly complex, involving lots of interactions and complex dependencies that make it hard to understand the inner workings of the market and what is driving prices. Econometric methods have been developed for this, white-box models, however, they are not as powerful as deep neural network models (DNN). In this paper, we use a DNN to forecast the price and then use XAI methods to understand the factors driving the price dynamics in the market. The objective is to increase our understanding of how different electricity markets work. To do that, we apply explainable methods such as SHAP and Gradient, combined with visual techniques like heatmaps (saliency maps) to analyse the behaviour and contributions of various features across five electricity markets. We introduce the novel concepts of SSHAP values and SSHAP lines to enhance the complex representation of high-dimensional tabular models. 

**Abstract (ZH)**: 电市场极为复杂，涉及众多交互和复杂依赖，使得理解市场的内在运作机制和价格驱动因素颇具挑战。虽然已经发展了计量经济学方法，但白盒模型的效力不如深度神经网络模型（DNN）。本文利用DNN进行价格预测，并结合XAI方法理解市场中价格动态的因素。旨在增加我们对不同电力市场运作机制的理解。为此，我们应用可解释方法，如SHAP和梯度，结合热图（可注意力图）等可视化技术，分析五个电力市场中各种特征的行为和贡献。我们引入了SSHAP值和SSHAP线的概念，以增强高维表型模型的复杂表示。 

---
# Distillation-Enabled Knowledge Alignment for Generative Semantic Communications in AIGC Provisioning Tasks 

**Title (ZH)**: 基于蒸馏的知识对齐在AIGC生成语义通信任务中的应用 

**Authors**: Jingzhi Hu, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19893)  

**Abstract**: Due to the surging amount of AI-generated content (AIGC), its provisioning to edges and mobile users from the cloud incurs substantial traffic on networks. Generative semantic communication (GSC) offers a promising solution by transmitting highly compact information, i.e., prompt text and latent representations, instead of high-dimensional AIGC data. However, GSC relies on the alignment between the knowledge in the cloud generative AI (GAI) and that possessed by the edges and users, and between the knowledge for wireless transmission and that of actual channels, which remains challenging. In this paper, we propose DeKA-g, a distillation-enabled knowledge alignment algorithm for GSC systems. The core idea is to distill the generation knowledge from the cloud-GAI into low-rank matrices, which can be incorporated by the edge and used to adapt the transmission knowledge to diverse wireless channel conditions. DeKA-g comprises two novel methods: metaword-aided knowledge distillation (MAKD) and variable-rate grouped SNR adaptation (VGSA). For MAKD, an optimized metaword is employed to enhance the efficiency of knowledge distillation, while VGSA enables efficient adaptation to diverse compression rates and SNR ranges. From simulation results, DeKA-g improves the alignment between the edge-generated images and the cloud-generated ones by 44%. Moreover, it adapts to compression rates with 116% higher efficiency than the baseline and enhances the performance in low-SNR conditions by 28%. 

**Abstract (ZH)**: 由于生成式人工智能内容（AIGC）的数量激增，将其从云端传输到边缘和移动用户导致网络流量显著增加。生成式语义通信（GSC）通过传输高度压缩的信息，即提示文本和潜在表示，而非高维AIGC数据，提供了一种有前景的解决方案。然而，GSC依赖于云端生成式人工智能（GAI）的知识与边缘设备和用户所具备的知识之间的对齐，以及无线传输知识与实际信道知识之间的对齐，这仍然是一个挑战。在本文中，我们提出了一种名为DeKA-g的蒸馏增强知识对齐算法，用于GSC系统。核心思想是从云端GAI中蒸馏生成知识并将其转换为低秩矩阵，边缘设备可以采用这些矩阵来适应各种无线信道条件。DeKA-g包括两种新颖的方法：元词辅助知识蒸馏（MAKD）和可变速率分组信噪比适应（VGSA）。MAKD通过优化元词增强知识蒸馏的效率，VGSA允许高效地适应不同的压缩率和信噪比范围。从仿真实验结果来看，DeKA-g将边缘生成的图像与云端生成的图像之间的对齐提高了44%。此外，它在压缩率适应性上比基线提高了116%的效率，并在低信噪比条件下提升了28%的性能。 

---
# RepuNet: A Reputation System for Mitigating Malicious Clients in DFL 

**Title (ZH)**: RepuNet：一种缓解DFL中恶意客户端的声誉系统 

**Authors**: Isaac Marroqui Penalva, Enrique Tomás Martínez Beltrán, Manuel Gil Pérez, Alberto Huertas Celdrán  

**Link**: [PDF](https://arxiv.org/pdf/2506.19892)  

**Abstract**: Decentralized Federated Learning (DFL) enables nodes to collaboratively train models without a central server, introducing new vulnerabilities since each node independently selects peers for model aggregation. Malicious nodes may exploit this autonomy by sending corrupted models (model poisoning), delaying model submissions (delay attack), or flooding the network with excessive messages, negatively affecting system performance. Existing solutions often depend on rigid configurations or additional infrastructures such as blockchain, leading to computational overhead, scalability issues, or limited adaptability. To overcome these limitations, this paper proposes RepuNet, a decentralized reputation system that categorizes threats in DFL and dynamically evaluates node behavior using metrics like model similarity, parameter changes, message latency, and communication volume. Nodes' influence in model aggregation is adjusted based on their reputation scores. RepuNet was integrated into the Nebula DFL platform and experimentally evaluated with MNIST and CIFAR-10 datasets under non-IID distributions, using federations of up to 25 nodes in both fully connected and random topologies. Different attack intensities, frequencies, and activation intervals were tested. Results demonstrated that RepuNet effectively detects and mitigates malicious behavior, achieving F1 scores above 95% for MNIST scenarios and approximately 76% for CIFAR-10 cases. These outcomes highlight RepuNet's adaptability, robustness, and practical potential for mitigating threats in decentralized federated learning environments. 

**Abstract (ZH)**: 去中心化联邦学习中的信誉网络（RepuNet）：一种动态评估节点行为的去中心化声誉系统 

---
# Orthogonal Soft Pruning for Efficient Class Unlearning 

**Title (ZH)**: 正交软剪枝以实现高效类遗忘 

**Authors**: Qinghui Gong, Xue Yang, Xiaohu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19891)  

**Abstract**: Machine unlearning aims to selectively remove class-specific knowledge from pretrained neural networks to satisfy privacy regulations such as the GDPR. Existing methods typically face a trade-off between unlearning speed and preservation of predictive accuracy, often incurring either high computational overhead or significant performance degradation on retained classes. In this paper, we propose a novel class-aware soft pruning framework leveraging orthogonal convolutional kernel regularization to achieve rapid and precise forgetting with millisecond-level response times. By enforcing orthogonality constraints during training, our method decorrelates convolutional filters and disentangles feature representations, while efficiently identifying class-specific channels through activation difference analysis. Extensive evaluations across multiple architectures and datasets demonstrate stable pruning with near-instant execution, complete forgetting of targeted classes, and minimal accuracy loss on retained data. Experiments on CIFAR-10, CIFAR-100, and TinyImageNet confirm that our approach substantially reduces membership inference attack risks and accelerates unlearning by orders of magnitude compared to state-of-the-art baselines. This framework provides an efficient, practical solution for real-time machine unlearning in Machine Learning as a Service (MLaaS) scenarios. 

**Abstract (ZH)**: 机器遗忘旨在从预训练神经网络中选择性地移除特定类别的知识，以满足GDPR等隐私法规的要求。现有方法通常在遗忘速度和预测准确性的保留之间存在权衡，往往导致较高的计算开销或在保留类别的显著性能下降。本文提出了一种新颖的类自意识软剪枝框架，利用正交卷积核正则化来实现毫秒级响应时间的快速且精确的遗忘。通过在训练过程中施加正交约束，我们的方法解耦卷积滤波器并分离特征表示，同时通过激活差异分析高效地识别特定类别的通道。在多个架构和数据集上的广泛评估表明，该方法具有稳定的剪枝性能，近乎即时的执行速度，目标类别的完全遗忘，并且在保留数据上的准确率损失最小。在CIFAR-10、CIFAR-100和TinyImageNet上的实验确认，与最先进的基线相比，我们的方法大大降低了成员推理攻击的风险并极大地加速了遗忘过程。该框架为机器学习即服务（MLaaS）场景中的实时机器遗忘提供了一种高效且实用的解决方案。 

---
# FlightKooba: A Fast Interpretable FTP Model 

**Title (ZH)**: FlightKooba：一种快速可解释的FTP模型 

**Authors**: Jing Lu, Xuan Wu, Yizhun Tian, Songhan Fan, Yali Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19885)  

**Abstract**: The Koopman theory is a powerful and effective modeling tool for converting nonlinear systems into linear representations, and flight trajectory prediction (FTP) is a complex nonlinear system. However, current models applying the Koopman theory to FTP tasks are not very effective, model interpretability is indeed an issue, and the Koopman operators are computationally intensive, resulting in long training times. To address this issue, this paper proposes a new modeling and control framework based on the HIPPO method, the Koopman theory, and state space equations from cybernetics: FlightKooba. Inspired by the idea of structural state space equations, FlightKooba directly constructs the Koopman operators from data. This makes the framework highly interpretable and significantly reduces the number of trainable parameters in the module, thereby greatly reducing training time. Experiments have demonstrated the superiority of the FlightKooba modeling method in terms of time and memory consumption (training time comparable to the Mamba module without using CUDA-level acceleration; memory reduced by more than 50% on most datasets, with a tenfold reduction in the number of parameters), essentially completing the FTP task. It provides a new method for the fast computation of the Koopman operators, opening up new possibilities for the combination of time series forecasting and control. 

**Abstract (ZH)**: 基于HIPPO方法、库普曼理论和控制论状态空间方程的FlightKooba建模与控制框架 

---
# STIMULUS: Achieving Fast Convergence and Low Sample Complexity in Stochastic Multi-Objective Learning 

**Title (ZH)**: STIMULUS: 实现随机多目标学习的快速收敛和低样本复杂性 

**Authors**: Zhuqing Liu, Chaosheng Dong, Michinari Momma, Simone Shao, Shaoyuan Xu, Yan Gao, Haibo Yang, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19883)  

**Abstract**: Recently, multi-objective optimization (MOO) has gained attention for its broad applications in ML, operations research, and engineering. However, MOO algorithm design remains in its infancy and many existing MOO methods suffer from unsatisfactory convergence rate and sample complexity performance. To address this challenge, in this paper, we propose an algorithm called STIMULUS( stochastic path-integrated multi-gradient recursive e\ulstimator), a new and robust approach for solving MOO problems. Different from the traditional methods, STIMULUS introduces a simple yet powerful recursive framework for updating stochastic gradient estimates to improve convergence performance with low sample complexity. In addition, we introduce an enhanced version of STIMULUS, termed STIMULUS-M, which incorporates a momentum term to further expedite convergence. We establish $O(1/T)$ convergence rates of the proposed methods for non-convex settings and $O (\exp{-\mu T})$ for strongly convex settings, where $T$ is the total number of iteration rounds. Additionally, we achieve the state-of-the-art $O \left(n+\sqrt{n}\epsilon^{-1}\right)$ sample complexities for non-convex settings and $O\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ for strongly convex settings, where $\epsilon>0$ is a desired stationarity error. Moreover, to alleviate the periodic full gradient evaluation requirement in STIMULUS and STIMULUS-M, we further propose enhanced versions with adaptive batching called STIMULUS+/ STIMULUS-M+ and provide their theoretical analysis. 

**Abstract (ZH)**: 最近，多目标优化（MOO）因其在机器学习、运筹学和工程领域的广泛应用而受到关注。然而，MOO算法设计仍处于初级阶段，许多现有的MOO方法在收敛速度和样本复杂性方面表现不佳。为了解决这一挑战，本文 propose 一种名为 STIMULUS（随机路径集成多梯度递归估计器）的新颖且稳健的解决多目标优化问题的方法。与传统方法不同，STIMULUS 引入了一种简单的高效递归框架，用于更新随机梯度估计，从而在低样本复杂性的情况下提高收敛性能。此外，我们还提出了一种增强的 STIMULUS 版本，称为 STIMULUS-M，该版本引入动量项以进一步加快收敛速度。我们建立了非凸环境下提出方法的 $O(1/T)$ 收敛率和强凸环境下 $O(\exp{-\mu T})$ 的收敛率，其中 $T$ 是总的迭代轮数。此外，我们在非凸环境下实现了最先进的 $O \left(n+\sqrt{n}\epsilon^{-1}\right)$ 样本复杂性和强凸环境下 $O\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ 的样本复杂性，其中 $\epsilon>0$ 是期望的稳定误差。此外，为进一步缓解 STIMULUS 和 STIMULUS-M 中周期性的完整梯度评估要求，我们还提出了具有自适应批次的增强版本，称为 STIMULUS+/STIMULUS-M+，并提供了它们的理论分析。 

---
# Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track 

**Title (ZH)**: 机器学习会议应设立“反驳与批评”track 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch, Brando Miranda, Matthias Gerstgrasser, Susan Zhang, Andreas Haupt, Isha Gupta, Elyas Obbad, Jesse Dodge, Jessica Zosa Forde, Koustuv Sinha, Francesco Orabona, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2506.19882)  

**Abstract**: Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are this http URL position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R & C) Track. This R & C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct. 

**Abstract (ZH)**: 科学通过迭代地推进和纠正人类对世界的理解而发展。在机器学习（ML）研究中，快速的进步导致了大量出版物的涌现，但也导致了一些误导性、不正确、有缺陷甚至可能是虚假的研究被接受并在机器学习会议上受到关注，这反映了同行评审的局限性。虽然这些错误是可以理解的，但机器学习会议缺乏系统纠正这类错误的 robust 过程。本文建议机器学习会议应设立专门的“反驳与批判”（R & C）赛道。该R & C赛道将提供一个高知名度、信誉良好的平台，支持对先前研究进行批判性挑战的重要研究，从而促进动态的自我纠正研究生态系统。我们讨论了赛道设计、评审原则、潜在陷阱，并提供了关于ICLR 2025 Oral的一项实例提交。我们得出结论，机器学习会议应创建正式的、可信赖的机制，以帮助机器学习研究自我纠正。 

---
# Physics-Guided Radiotherapy Treatment Planning with Deep Learning 

**Title (ZH)**: 基于物理引导的放射治疗规划深度学习方法 

**Authors**: Stefanos Achlatis, Efstratios Gavves, Jan-Jakob Sonke  

**Link**: [PDF](https://arxiv.org/pdf/2506.19880)  

**Abstract**: Radiotherapy (RT) is a critical cancer treatment, with volumetric modulated arc therapy (VMAT) being a commonly used technique that enhances dose conformity by dynamically adjusting multileaf collimator (MLC) positions and monitor units (MU) throughout gantry rotation. Adaptive radiotherapy requires frequent modifications to treatment plans to account for anatomical variations, necessitating time-efficient solutions. Deep learning offers a promising solution to automate this process. To this end, we propose a two-stage, physics-guided deep learning pipeline for radiotherapy planning. In the first stage, our network is trained with direct supervision on treatment plan parameters, consisting of MLC and MU values. In the second stage, we incorporate an additional supervision signal derived from the predicted 3D dose distribution, integrating physics-based guidance into the training process. We train and evaluate our approach on 133 prostate cancer patients treated with a uniform 2-arc VMAT protocol delivering a dose of 62 Gy to the planning target volume (PTV). Our results demonstrate that the proposed approach, implemented using both 3D U-Net and UNETR architectures, consistently produces treatment plans that closely match clinical ground truths. Our method achieves a mean difference of D95% = 0.42 +/- 1.83 Gy and V95% = -0.22 +/- 1.87% at the PTV while generating dose distributions that reduce radiation exposure to organs at risk. These findings highlight the potential of physics-guided deep learning in RT planning. 

**Abstract (ZH)**: 基于物理引导的深度学习在放射治疗规划中的两阶段方法 

---
# Robust Anomaly Detection in Network Traffic: Evaluating Machine Learning Models on CICIDS2017 

**Title (ZH)**: 网络流量中鲁棒异常检测：评估CICIDS2017上的机器学习模型 

**Authors**: Zhaoyang Xu, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19877)  

**Abstract**: Identifying suitable machine learning paradigms for intrusion detection remains critical for building effective and generalizable security solutions. In this study, we present a controlled comparison of four representative models - Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (CNN), One-Class Support Vector Machine (OCSVM) and Local Outlier Factor (LOF) - on the CICIDS2017 dataset under two scenarios: detecting known attack types and generalizing to previously unseen threats. Our results show that supervised MLP and CNN achieve near-perfect accuracy on familiar attacks but suffer drastic recall drops on novel attacks. Unsupervised LOF attains moderate overall accuracy and high recall on unknown threats at the cost of elevated false alarms, while boundary-based OCSVM balances precision and recall best, demonstrating robust detection across both scenarios. These findings offer practical guidance for selecting IDS models in dynamic network environments. 

**Abstract (ZH)**: 识别适合入侵检测的机器学习范式对于构建有效且具泛化性的安全解决方案仍至关重要。在本研究中，我们在两种情景下对比了四种代表性模型——多层感知机（MLP）、一维卷积神经网络（CNN）、一类支持向量机（OCSVM）和局部异常因子（LOF）——在CICIDS2017数据集上的性能：检测已知攻击类型和泛化到之前未见过的威胁。研究结果表明，监督MLP和CNN在熟悉攻击上的准确率接近完美，但在新型攻击上的召回率显著下降。无监督的LOF在未知威胁上总体准确率和召回率较高，但伴随着较高的误报率，而基于边界的一类支持向量机在精确率和召回率之间取得了最佳平衡，在两种情景中均展现出稳健的检测能力。这些发现为在动态网络环境中选择IDS模型提供了实用指导。 

---
# Speaker Embeddings to Improve Tracking of Intermittent and Moving Speakers 

**Title (ZH)**: 基于演讲者嵌入提高间歇性和移动演讲者跟踪性能 

**Authors**: Taous Iatariene, Can Cui, Alexandre Guérin, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.19875)  

**Abstract**: Speaker tracking methods often rely on spatial observations to assign coherent track identities over time. This raises limits in scenarios with intermittent and moving speakers, i.e., speakers that may change position when they are inactive, thus leading to discontinuous spatial trajectories. This paper proposes to investigate the use of speaker embeddings, in a simple solution to this issue. We propose to perform identity reassignment post-tracking, using speaker embeddings. We leverage trajectory-related information provided by an initial tracking step and multichannel audio signal. Beamforming is used to enhance the signal towards the speakers' positions in order to compute speaker embeddings. These are then used to assign new track identities based on an enrollment pool. We evaluate the performance of the proposed speaker embedding-based identity reassignment method on a dataset where speakers change position during inactivity periods. Results show that it consistently improves the identity assignment performance of neural and standard tracking systems. In particular, we study the impact of beamforming and input duration for embedding extraction. 

**Abstract (ZH)**: 基于说话人嵌入的身份重新指派方法研究 

---
# Towards Provable (In)Secure Model Weight Release Schemes 

**Title (ZH)**: 关于可验证的安全性模型权重发布方案 

**Authors**: Xing Yang, Bingtao Wang, Yuhao Wang, Zimo Ji, Terry Jingchen Zhang, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19874)  

**Abstract**: Recent secure weight release schemes claim to enable open-source model distribution while protecting model ownership and preventing misuse. However, these approaches lack rigorous security foundations and provide only informal security guarantees. Inspired by established works in cryptography, we formalize the security of weight release schemes by introducing several concrete security definitions. We then demonstrate our definition's utility through a case study of TaylorMLP, a prominent secure weight release scheme. Our analysis reveals vulnerabilities that allow parameter extraction thus showing that TaylorMLP fails to achieve its informal security goals. We hope this work will advocate for rigorous research at the intersection of machine learning and security communities and provide a blueprint for how future weight release schemes should be designed and evaluated. 

**Abstract (ZH)**: 近期的 secure weight release 方案声称能够在保护模型所有权和防止滥用的情况下实现开源模型分发，但这些方法缺乏严格的.security 基础，并仅提供非正式的安全保证。受密码学中现有工作的启发，我们通过引入几种具体的 security 定义来正式化 weight release 方案的安全性。我们通过一个典型的 secure weight release 方案 TaylorMLP 的案例研究展示了我们定义的实用性。我们的分析揭示了漏洞，允许参数提取，从而证明 TaylorMLP 未能实现其非正式的安全目标。我们希望这项工作能够促进机器学习与安全社区之间的严谨研究，并为未来 weight release 方案的设计与评估提供蓝图。 

---
# An Attack Method for Medical Insurance Claim Fraud Detection based on Generative Adversarial Network 

**Title (ZH)**: 基于生成对抗网络的医疗保险理赔欺诈检测攻击方法 

**Authors**: Yining Pang, Chenghan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19871)  

**Abstract**: Insurance fraud detection represents a pivotal advancement in modern insurance service, providing intelligent and digitalized monitoring to enhance management and prevent fraud. It is crucial for ensuring the security and efficiency of insurance systems. Although AI and machine learning algorithms have demonstrated strong performance in detecting fraudulent claims, the absence of standardized defense mechanisms renders current systems vulnerable to emerging adversarial threats. In this paper, we propose a GAN-based approach to conduct adversarial attacks on fraud detection systems. Our results indicate that an attacker, without knowledge of the training data or internal model details, can generate fraudulent cases that are classified as legitimate with a 99\% attack success rate (ASR). By subtly modifying real insurance records and claims, adversaries can significantly increase the fraud risk, potentially bypassing compromised detection systems. These findings underscore the urgent need to enhance the robustness of insurance fraud detection models against adversarial manipulation, thereby ensuring the stability and reliability of different insurance systems. 

**Abstract (ZH)**: 基于GAN的保险欺诈检测系统 adversarial 攻击研究 

---
# Secure Energy Transactions Using Blockchain Leveraging AI for Fraud Detection and Energy Market Stability 

**Title (ZH)**: 利用AI进行 fraud detection 和能源市场稳定性的区块链安全能源交易 

**Authors**: Md Asif Ul Hoq Khan, MD Zahedul Islam, Istiaq Ahmed, Md Masud Karim Rabbi, Farhana Rahman Anonna, MD Abdul Fahim Zeeshan, Mehedi Hasan Ridoy, Bivash Ranjan Chowdhury, Md Nazmul Shakir Rabbi, GM Alamin Sadnan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19870)  

**Abstract**: Peer-to-peer trading and the move to decentralized grids have reshaped the energy markets in the United States. Notwithstanding, such developments lead to new challenges, mainly regarding the safety and authenticity of energy trade. This study aimed to develop and build a secure, intelligent, and efficient energy transaction system for the decentralized US energy market. This research interlinks the technological prowess of blockchain and artificial intelligence (AI) in a novel way to solve long-standing challenges in the distributed energy market, specifically those of security, fraudulent behavior detection, and market reliability. The dataset for this research is comprised of more than 1.2 million anonymized energy transaction records from a simulated peer-to-peer (P2P) energy exchange network emulating real-life blockchain-based American microgrids, including those tested by LO3 Energy and Grid+ Labs. Each record contains detailed fields of transaction identifier, timestamp, energy volume (kWh), transaction type (buy/sell), unit price, prosumer/consumer identifier (hashed for privacy), smart meter readings, geolocation regions, and settlement confirmation status. The dataset also includes system-calculated behavior metrics of transaction rate, variability of energy production, and historical pricing patterns. The system architecture proposed involves the integration of two layers, namely a blockchain layer and artificial intelligence (AI) layer, each playing a unique but complementary function in energy transaction securing and market intelligence improvement. The machine learning models used in this research were specifically chosen for their established high performance in classification tasks, specifically in the identification of energy transaction fraud in decentralized markets. 

**Abstract (ZH)**: P2P交易和去中心化电网的兴起重塑了美国能源市场：一种安全、智能、高效的去中心化能源交易系统的研究 

---
# Scalable and Cost-Efficient de Novo Template-Based Molecular Generation 

**Title (ZH)**: 基于模板的分子生成的可扩展和低成本从头设计方法 

**Authors**: Piotr Gaiński, Oussama Boussif, Andrei Rekesh, Dmytro Shevchuk, Ali Parviz, Mike Tyers, Robert A. Batey, Michał Koziarski  

**Link**: [PDF](https://arxiv.org/pdf/2506.19865)  

**Abstract**: Template-based molecular generation offers a promising avenue for drug design by ensuring generated compounds are synthetically accessible through predefined reaction templates and building blocks. In this work, we tackle three core challenges in template-based GFlowNets: (1) minimizing synthesis cost, (2) scaling to large building block libraries, and (3) effectively utilizing small fragment sets. We propose \textbf{Recursive Cost Guidance}, a backward policy framework that employs auxiliary machine learning models to approximate synthesis cost and viability. This guidance steers generation toward low-cost synthesis pathways, significantly enhancing cost-efficiency, molecular diversity, and quality, especially when paired with an \textbf{Exploitation Penalty} that balances the trade-off between exploration and exploitation. To enhance performance in smaller building block libraries, we develop a \textbf{Dynamic Library} mechanism that reuses intermediate high-reward states to construct full synthesis trees. Our approach establishes state-of-the-art results in template-based molecular generation. 

**Abstract (ZH)**: 基于模板的分子生成为通过预定义反应模板和构建块确保生成化合物的合成可行性提供了有前途的设计途径。本文攻克了基于模板的GFlowNets中的三大核心挑战：(1) 减少合成成本，(2) 扩展到大型构建块库，(3) 有效利用小片段集。我们提出了一种递归成本指导方法，这是一种.backward策略框架，利用辅助机器学习模型来近似合成成本和可行性。这种指导使生成偏向低成本合成路径，大幅提升了成本效率、分子多样性和质量，特别是在与探索与利用之间的平衡惩罚项（Exploitation Penalty）结合使用时更为显著。为了在较小的构建块库中增强性能，我们开发了一种动态库机制，通过重用中间高奖励状态来构建完整的合成树。我们的方法在基于模板的分子生成中达到了最先进的成果。 

---
# DualEquiNet: A Dual-Space Hierarchical Equivariant Network for Large Biomolecules 

**Title (ZH)**: DualEquiNet: 一种双空间分层等变网络，用于大生物分子 

**Authors**: Junjie Xu, Jiahao Zhang, Mangal Prakash, Xiang Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19862)  

**Abstract**: Geometric graph neural networks (GNNs) that respect E(3) symmetries have achieved strong performance on small molecule modeling, but they face scalability and expressiveness challenges when applied to large biomolecules such as RNA and proteins. These systems require models that can simultaneously capture fine-grained atomic interactions, long-range dependencies across spatially distant components, and biologically relevant hierarchical structure, such as atoms forming residues, which in turn form higher-order domains. Existing geometric GNNs, which typically operate exclusively in either Euclidean or Spherical Harmonics space, are limited in their ability to capture both the fine-scale atomic details and the long-range, symmetry-aware dependencies required for modeling the multi-scale structure of large biomolecules. We introduce DualEquiNet, a Dual-Space Hierarchical Equivariant Network that constructs complementary representations in both Euclidean and Spherical Harmonics spaces to capture local geometry and global symmetry-aware features. DualEquiNet employs bidirectional cross-space message passing and a novel Cross-Space Interaction Pooling mechanism to hierarchically aggregate atomic features into biologically meaningful units, such as residues, enabling efficient and expressive multi-scale modeling for large biomolecular systems. DualEquiNet achieves state-of-the-art performance on multiple existing benchmarks for RNA property prediction and protein modeling, and outperforms prior methods on two newly introduced 3D structural benchmarks demonstrating its broad effectiveness across a range of large biomolecule modeling tasks. 

**Abstract (ZH)**: 几何图神经网络（GNNs）若符合E(3)对称性，在小分子建模中取得了显著成果，但在应用于大型生物分子如RNA和蛋白质时面临可扩展性和表达能力的挑战。现有的几何GNNs通常只能在同一空间（欧几里得空间或球谐空间）中操作，这限制了它们捕捉精细原子细节和长程、对称性 Awareness 的依赖性的能力，这两者对于建模大型生物分子的多层次结构至关重要。我们提出了DualEquiNet，这是一个在欧几里得和球谐空间构建互补表示的层次等变网络，用于捕捉局部几何结构和全局对称性敏感的特征。DualEquiNet 使用双向跨空间消息传递和一种新颖的跨空间交互聚池机制，分层聚合原子特征为生物意义单位，如残基，从而实现对大型生物分子系统的高效且表达性强的多层次建模。DualEquiNet 在多个现有基准上的RNA性质预测和蛋白质建模任务中取得了最先进的性能，并在两个新引入的三维结构基准上优于先前方法，展示了其在大型生物分子建模任务范围内的广泛有效性。 

---
