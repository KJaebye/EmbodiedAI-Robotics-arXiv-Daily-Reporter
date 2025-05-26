# Classification of assembly tasks combining multiple primitive actions using Transformers and xLSTMs 

**Title (ZH)**: 使用Transformer和xLSTM结合多种基本动作进行装配任务分类 

**Authors**: Miguel Neves, Pedro Neto  

**Link**: [PDF](https://arxiv.org/pdf/2505.18012)  

**Abstract**: The classification of human-performed assembly tasks is essential in collaborative robotics to ensure safety, anticipate robot actions, and facilitate robot learning. However, achieving reliable classification is challenging when segmenting tasks into smaller primitive actions is unfeasible, requiring us to classify long assembly tasks that encompass multiple primitive actions. In this study, we propose classifying long assembly sequential tasks based on hand landmark coordinates and compare the performance of two well-established classifiers, LSTM and Transformer, as well as a recent model, xLSTM. We used the HRC scenario proposed in the CT benchmark, which includes long assembly tasks that combine actions such as insertions, screw fastenings, and snap fittings. Testing was conducted using sequences gathered from both the human operator who performed the training sequences and three new operators. The testing results of real-padded sequences for the LSTM, Transformer, and xLSTM models was 72.9%, 95.0% and 93.2% for the training operator, and 43.5%, 54.3% and 60.8% for the new operators, respectively. The LSTM model clearly underperformed compared to the other two approaches. As expected, both the Transformer and xLSTM achieved satisfactory results for the operator they were trained on, though the xLSTM model demonstrated better generalization capabilities to new operators. The results clearly show that for this type of classification, the xLSTM model offers a slight edge over Transformers. 

**Abstract (ZH)**: 基于手部关键点坐标的人工装配任务分类在协作机器人中的应用：LSTM、Transformer和xLSTM模型的性能比较 

---
# CU-Multi: A Dataset for Multi-Robot Data Association 

**Title (ZH)**: CU-Multi: 多机器人数据关联数据集 

**Authors**: Doncey Albin, Miles Mena, Annika Thomas, Harel Biggie, Xuefei Sun, Dusty Woods, Steve McGuire, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17576)  

**Abstract**: Multi-robot systems (MRSs) are valuable for tasks such as search and rescue due to their ability to coordinate over shared observations. A central challenge in these systems is aligning independently collected perception data across space and time, i.e., multi-robot data association. While recent advances in collaborative SLAM (C-SLAM), map merging, and inter-robot loop closure detection have significantly progressed the field, evaluation strategies still predominantly rely on splitting a single trajectory from single-robot SLAM datasets into multiple segments to simulate multiple robots. Without careful consideration to how a single trajectory is split, this approach will fail to capture realistic pose-dependent variation in observations of a scene inherent to multi-robot systems. To address this gap, we present CU-Multi, a multi-robot dataset collected over multiple days at two locations on the University of Colorado Boulder campus. Using a single robotic platform, we generate four synchronized runs with aligned start times and deliberate percentages of trajectory overlap. CU-Multi includes RGB-D, GPS with accurate geospatial heading, and semantically annotated LiDAR data. By introducing controlled variations in trajectory overlap and dense lidar annotations, CU-Multi offers a compelling alternative for evaluating methods in multi-robot data association. Instructions on accessing the dataset, support code, and the latest updates are publicly available at this https URL 

**Abstract (ZH)**: 多机器人系统（MRSs）在搜索与救援等任务中因其在共享观察信息上的协同能力而具有价值。这些系统中的核心挑战在于对独立收集的感知数据进行空间和时间上的对齐，即多机器人数据关联。虽然在协作SLAM（C-SLAM）、地图合并以及机器人间循环闭合检测方面取得了显著进展，但评估策略仍未摆脱将单个机器人SLAM数据集中的单个轨迹拆分成多个段落来模拟多个机器人的方法。若不仔细考虑轨迹拆分的方式，这种方法将无法捕捉多机器人系统中场景观察中固有的、依赖于姿态的真实变化。为解决这一问题，我们介绍了CU-Multi，这是一个在科罗拉多大学博尔德分校两个地点多天收集的多机器人数据集。使用单一机器人平台，我们生成了四个同步运行，具有对齐的起始时间和故意的轨迹重叠比例。CU-Multi 包含RGB-D、GPS（带精确地理方向性）和语义标注的LiDAR数据。通过引入轨迹重叠的可控变化和密集的LiDAR注释，CU-Multi 为评估多机器人数据关联方法提供了极具吸引力的替代方案。有关数据集的访问说明、支持代码以及最新更新的详细信息，请访问 <https://this.is.public>。 

---
# What Do You Need for Diverse Trajectory Stitching in Diffusion Planning? 

**Title (ZH)**: 你需要什么来进行多样化的轨迹缝合以规划扩散？ 

**Authors**: Quentin Clark, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2505.18083)  

**Abstract**: In planning, stitching is an ability of algorithms to piece together sub-trajectories of data they are trained on to generate new and diverse behaviours. While stitching is historically a strength of offline reinforcement learning, recent generative behavioural cloning (BC) methods have also shown proficiency at stitching. However, the main factors behind this are poorly understood, hindering the development of new algorithms that can reliably stitch. Focusing on diffusion planners trained via BC, we find two properties are needed to compose: \emph{positional equivariance} and \emph{local receptiveness}. We use these two properties to explain architecture, data, and inference choices in existing generative BC methods based on diffusion planning, including replanning frequency, data augmentation, and data scaling. Experimental comparisions show that (1) while locality is more important than positional equivariance in creating a diffusion planner capable of composition, both are crucial (2) enabling these properties through relatively simple architecture choices can be competitive with more computationally expensive methods such as replanning or scaling data, and (3) simple inpainting-based guidance can guide architecturally compositional models to enable generalization in goal-conditioned settings. 

**Abstract (ZH)**: 规划中的缝合能力：基于扩散规划的生成行为克隆方法的综合研究 

---
# Linear Mixture Distributionally Robust Markov Decision Processes 

**Title (ZH)**: 线性混合分布鲁棒马尔可夫决策过程 

**Authors**: Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18044)  

**Abstract**: Many real-world decision-making problems face the off-dynamics challenge: the agent learns a policy in a source domain and deploys it in a target domain with different state transitions. The distributionally robust Markov decision process (DRMDP) addresses this challenge by finding a robust policy that performs well under the worst-case environment within a pre-specified uncertainty set of transition dynamics. Its effectiveness heavily hinges on the proper design of these uncertainty sets, based on prior knowledge of the dynamics. In this work, we propose a novel linear mixture DRMDP framework, where the nominal dynamics is assumed to be a linear mixture model. In contrast with existing uncertainty sets directly defined as a ball centered around the nominal kernel, linear mixture DRMDPs define the uncertainty sets based on a ball around the mixture weighting parameter. We show that this new framework provides a more refined representation of uncertainties compared to conventional models based on $(s,a)$-rectangularity and $d$-rectangularity, when prior knowledge about the mixture model is present. We propose a meta algorithm for robust policy learning in linear mixture DRMDPs with general $f$-divergence defined uncertainty sets, and analyze its sample complexities under three divergence metrics instantiations: total variation, Kullback-Leibler, and $\chi^2$ divergences. These results establish the statistical learnability of linear mixture DRMDPs, laying the theoretical foundation for future research on this new setting. 

**Abstract (ZH)**: 一种基于线性混合模型的分布鲁棒马尔可夫决策过程及其学习算法研究 

---
# Embracing Contradiction: Theoretical Inconsistency Will Not Impede the Road of Building Responsible AI Systems 

**Title (ZH)**: 拥抱对立：理论不一致不会妨碍负责任的人工智能系统建设之路 

**Authors**: Gordon Dai, Yunze Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18139)  

**Abstract**: This position paper argues that the theoretical inconsistency often observed among Responsible AI (RAI) metrics, such as differing fairness definitions or tradeoffs between accuracy and privacy, should be embraced as a valuable feature rather than a flaw to be eliminated. We contend that navigating these inconsistencies, by treating metrics as divergent objectives, yields three key benefits: (1) Normative Pluralism: Maintaining a full suite of potentially contradictory metrics ensures that the diverse moral stances and stakeholder values inherent in RAI are adequately represented. (2) Epistemological Completeness: The use of multiple, sometimes conflicting, metrics allows for a more comprehensive capture of multifaceted ethical concepts, thereby preserving greater informational fidelity about these concepts than any single, simplified definition. (3) Implicit Regularization: Jointly optimizing for theoretically conflicting objectives discourages overfitting to one specific metric, steering models towards solutions with enhanced generalization and robustness under real-world complexities. In contrast, efforts to enforce theoretical consistency by simplifying or pruning metrics risk narrowing this value diversity, losing conceptual depth, and degrading model performance. We therefore advocate for a shift in RAI theory and practice: from getting trapped in inconsistency to characterizing acceptable inconsistency thresholds and elucidating the mechanisms that permit robust, approximated consistency in practice. 

**Abstract (ZH)**: This Position Paper argues that the theoretical inconsistency often observed among Responsible AI (RAI) metrics should be embraced as a valuable feature rather than a flaw to be eliminated. We contend that navigating these inconsistencies yields three key benefits: Normative Pluralism, Epistemological Completeness, and Implicit Regularization. We advocate for a shift in RAI theory and practice from eliminating inconsistency to characterizing acceptable inconsistency thresholds and elucidating the mechanisms that permit robust, approximated consistency in practice. 

---
# Automata Learning of Preferences over Temporal Logic Formulas from Pairwise Comparisons 

**Title (ZH)**: 基于成对比较学习时序逻辑公式下的偏好自动机 

**Authors**: Hazhar Rahmani, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18030)  

**Abstract**: Many preference elicitation algorithms consider preference over propositional logic formulas or items with different attributes. In sequential decision making, a user's preference can be a preorder over possible outcomes, each of which is a temporal sequence of events. This paper considers a class of preference inference problems where the user's unknown preference is represented by a preorder over regular languages (sets of temporal sequences), referred to as temporal goals. Given a finite set of pairwise comparisons between finite words, the objective is to learn both the set of temporal goals and the preorder over these goals. We first show that a preference relation over temporal goals can be modeled by a Preference Deterministic Finite Automaton (PDFA), which is a deterministic finite automaton augmented with a preorder over acceptance conditions. The problem of preference inference reduces to learning the PDFA. This problem is shown to be computationally challenging, with the problem of determining whether there exists a PDFA of size smaller than a given integer $k$, consistent with the sample, being NP-Complete. We formalize the properties of characteristic samples and develop an algorithm that guarantees to learn, given a characteristic sample, the minimal PDFA equivalent to the true PDFA from which the sample is drawn. We present the method through a running example and provide detailed analysis using a robotic motion planning problem. 

**Abstract (ZH)**: 一类时序目标的偏好推断问题：从有限词对的比较中学习时序偏好确定有限自动机 

---
# ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback 

**Title (ZH)**: ComfyMind: 基于树状规划和反应性反馈的通用生成方法 

**Authors**: Litao Guo, Xinli Xu, Luozhou Wang, Jiantao Lin, Jinsong Zhou, Zixin Zhang, Bolan Su, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17908)  

**Abstract**: With the rapid advancement of generative models, general-purpose generation has gained increasing attention as a promising approach to unify diverse tasks across modalities within a single system. Despite this progress, existing open-source frameworks often remain fragile and struggle to support complex real-world applications due to the lack of structured workflow planning and execution-level feedback. To address these limitations, we present ComfyMind, a collaborative AI system designed to enable robust and scalable general-purpose generation, built on the ComfyUI platform. ComfyMind introduces two core innovations: Semantic Workflow Interface (SWI) that abstracts low-level node graphs into callable functional modules described in natural language, enabling high-level composition and reducing structural errors; Search Tree Planning mechanism with localized feedback execution, which models generation as a hierarchical decision process and allows adaptive correction at each stage. Together, these components improve the stability and flexibility of complex generative workflows. We evaluate ComfyMind on three public benchmarks: ComfyBench, GenEval, and Reason-Edit, which span generation, editing, and reasoning tasks. Results show that ComfyMind consistently outperforms existing open-source baselines and achieves performance comparable to GPT-Image-1. ComfyMind paves a promising path for the development of open-source general-purpose generative AI systems. Project page: this https URL 

**Abstract (ZH)**: 随着生成模型的迅速发展，通用生成在单一系统内统一多种模态任务方面获得了越来越多的关注，作为有前景的方法。尽管取得了这些进展，现有的开源框架往往仍不稳定，并因缺乏结构化的 workflows 规划和执行级反馈而难以支持复杂的现实世界应用。为解决这些局限，我们提出了 ComfyMind，一个协作式 AI 系统，旨在实现稳健和可扩展的通用生成，基于 ComfyUI 平台构建。ComfyMind 引入了两项核心创新：语义工作流接口（SWI），将低层节点图抽象为用自然语言描述的可调用功能模块，促进高级组合并减少结构错误；局部反馈执行的搜索树规划机制，将生成建模为分层决策过程，并允许在每个阶段进行适应性校正。这些组件共同提高了复杂生成 workflows 的稳定性和灵活性。我们使用三个公开基准测试 ComfyMind：ComfyBench、GenEval 和 Reason-Edit，涵盖生成、编辑和推理任务。结果表明，ComfyMind 在所有基准测试中均优于现有开源基线系统，并且性能接近 GPT-Image-1。ComfyMind 为开源通用生成 AI 系统的发展铺平了前景。项目页面：this https URL 

---
# PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions 

**Title (ZH)**: PatientSim：以个性为导向的现实医生-患者交互模拟器 

**Authors**: Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim, Jae Ho Sohn, Taerim Kim, Soo Kyung Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17818)  

**Abstract**: Doctor-patient consultations require multi-turn, context-aware communication tailored to diverse patient personas. Training or evaluating doctor LLMs in such settings requires realistic patient interaction systems. However, existing simulators often fail to reflect the full range of personas seen in clinical practice. To address this, we introduce PatientSim, a patient simulator that generates realistic and diverse patient personas for clinical scenarios, grounded in medical expertise. PatientSim operates using: 1) clinical profiles, including symptoms and medical history, derived from real-world data in the MIMIC-ED and MIMIC-IV datasets, and 2) personas defined by four axes: personality, language proficiency, medical history recall level, and cognitive confusion level, resulting in 37 unique combinations. We evaluated eight LLMs for factual accuracy and persona consistency. The top-performing open-source model, Llama 3.3, was validated by four clinicians to confirm the robustness of our framework. As an open-source, customizable platform, PatientSim provides a reproducible and scalable solution that can be customized for specific training needs. Offering a privacy-compliant environment, it serves as a robust testbed for evaluating medical dialogue systems across diverse patient presentations and shows promise as an educational tool for healthcare. 

**Abstract (ZH)**: 医生-患者咨询需要针对不同患者人格特征进行多轮、情境感知的沟通。在such设置下训练或评估医生大语言模型需要真实的患者交互系统。然而，现有模拟器往往无法反映临床实践中见到的全部患者人格范围。为此，我们引入了PatientSim，这是一种基于医疗专家知识生成真实且多样的患者人格的患者模拟器。PatientSim的操作基于：1) 包括症状和医疗历史的临床档案，来自MIMIC-ED和MIMIC-IV数据集的现实世界数据，和2) 由四个维度定义的人格：个性、语言熟练度、对医疗历史的回忆水平和认知混乱程度，产生37种独特的组合。我们评估了八种语言模型的事实准确性与人格一致性。开源模型Llama 3.3在四名临床医生的验证下确认了我们框架的稳健性。作为开源且可定制的平台，PatientSim提供了一个可重复且可扩展的解决方案，可满足特定的培训需求。通过提供隐私合规的环境，它为评估在多样患者表现的医疗对话系统提供了坚实的测试平台，并有望作为医疗教育工具。 

---
# Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour 

**Title (ZH)**: 将反事实模拟与语言模型结合以解释多agent行为 

**Authors**: Bálint Gyevnár, Christopher G. Lucas, Stefano V. Albrecht, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17801)  

**Abstract**: Autonomous multi-agent systems (MAS) are useful for automating complex tasks but raise trust concerns due to risks like miscoordination and goal misalignment. Explainability is vital for trust calibration, but explainable reinforcement learning for MAS faces challenges in state/action space complexity, stakeholder needs, and evaluation. Using the counterfactual theory of causation and LLMs' summarisation capabilities, we propose Agentic eXplanations via Interrogative Simulation (AXIS). AXIS generates intelligible causal explanations for pre-trained multi-agent policies by having an LLM interrogate an environment simulator using queries like 'whatif' and 'remove' to observe and synthesise counterfactual information over multiple rounds. We evaluate AXIS on autonomous driving across 10 scenarios for 5 LLMs with a novel evaluation methodology combining subjective preference, correctness, and goal/action prediction metrics, and an external LLM as evaluator. Compared to baselines, AXIS improves perceived explanation correctness by at least 7.7% across all models and goal prediction accuracy by 23% for 4 models, with improved or comparable action prediction accuracy, achieving the highest scores overall. 

**Abstract (ZH)**: 自主多agent系统中的Agentic eXplanations via Interrogative Simulation（AXIS）：基于反事实因果理论和大语言模型的可解释性生成 

---
# Enhancing AI System Resiliency: Formulation and Guarantee for LSTM Resilience Based on Control Theory 

**Title (ZH)**: 基于控制理论的LSTM鲁棒性建模与保证：增强AI系统韧性 

**Authors**: Sota Yoshihara, Ryousuke Yamamoto, Hiroyuki Kusumoto, Masanari Shimura  

**Link**: [PDF](https://arxiv.org/pdf/2505.17696)  

**Abstract**: This research proposes methods for formulating and guaranteeing the resilience of long short-term memory (LSTM) networks, which can serve as a key technology in AI system quality assurance. We introduce a novel methodology applying incremental input-to-state stability ($\delta$ISS) to mathematically define and evaluate the resilience of LSTM against input perturbations. Key achievements include the development of a data-independent evaluation method and the demonstration of resilience control through adjustments to training parameters. This research presents concrete solutions to AI quality assurance from a control theory perspective, which can advance AI applications in control systems. 

**Abstract (ZH)**: 本研究提出了一种制定和保证长短期记忆（LSTM）网络韧性的方法，该方法可作为AI系统质量保证的关键技术。我们引入了一种新颖的方法，利用增量输入到状态稳定性（$\delta$ISS）来数学上定义和评估LSTM在网络输入扰动下的韧性。关键成果包括开发了一种数据无关的评估方法，并通过调整训练参数展示了韧性控制。从控制理论的角度，本研究提出了AI质量保证的 concrete 解决方案，有助于推动AI在控制系统中的应用。 

---
# Does Chain-of-Thought Reasoning Really Reduce Harmfulness from Jailbreaking? 

**Title (ZH)**: 链式思考推理真能减少 Jailbreaking 的危害性？ 

**Authors**: Chengda Lu, Xiaoyu Fan, Yu Huang, Rongwu Xu, Jijie Li, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17650)  

**Abstract**: Jailbreak attacks have been observed to largely fail against recent reasoning models enhanced by Chain-of-Thought (CoT) reasoning. However, the underlying mechanism remains underexplored, and relying solely on reasoning capacity may raise security concerns. In this paper, we try to answer the question: Does CoT reasoning really reduce harmfulness from jailbreaking? Through rigorous theoretical analysis, we demonstrate that CoT reasoning has dual effects on jailbreaking harmfulness. Based on the theoretical insights, we propose a novel jailbreak method, FicDetail, whose practical performance validates our theoretical findings. 

**Abstract (ZH)**: Jailbreak攻击被观察到在近期通过链式思考（CoT）增强的推理模型中大量失效。然而，其背后的机制仍待深入探究，单纯依赖推理能力可能引发安全担忧。本文试图回答的问题是：链式思考（CoT）推理是否真的减少了 Jailbreak 的危害性？通过严格的理论分析，我们证明链式思考（CoT）推理对 Jailbreak 的危害性具有双重影响。基于理论见解，我们提出了一种新颖的 Jailbreak 方法 FicDetail，其实用性能验证了我们的理论发现。 

---
# Transparency and Proportionality in Post-Processing Algorithmic Bias Correction 

**Title (ZH)**: 后处理算法偏见纠正中的透明度与比例原则 

**Authors**: Juliett Suárez Ferreira, Marija Slavkovik, Jorge Casillas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17525)  

**Abstract**: Algorithmic decision-making systems sometimes produce errors or skewed predictions toward a particular group, leading to unfair results. Debiasing practices, applied at different stages of the development of such systems, occasionally introduce new forms of unfairness or exacerbate existing inequalities. We focus on post-processing techniques that modify algorithmic predictions to achieve fairness in classification tasks, examining the unintended consequences of these interventions. To address this challenge, we develop a set of measures that quantify the disparity in the flips applied to the solution in the post-processing stage. The proposed measures will help practitioners: (1) assess the proportionality of the debiasing strategy used, (2) have transparency to explain the effects of the strategy in each group, and (3) based on those results, analyze the possibility of the use of some other approaches for bias mitigation or to solve the problem. We introduce a methodology for applying the proposed metrics during the post-processing stage and illustrate its practical application through an example. This example demonstrates how analyzing the proportionality of the debiasing strategy complements traditional fairness metrics, providing a deeper perspective to ensure fairer outcomes across all groups. 

**Abstract (ZH)**: 算法决策系统有时会产生针对特定群体的错误或偏差预测，导致不公平的结果。去偏见实践在这些系统开发的不同阶段的应用有时会引入新的不公平形式或加剧现有不平等。我们重点关注调整算法预测以在分类任务中实现公平性的后处理技术，研究这些干预措施的意外后果。为此，我们开发了一套度量标准，量化后处理阶段对解决方案调整的不平等程度。所提出的度量标准将帮助从业者：(1)评估所使用去偏见策略的比例性，(2)增强透明度以解释策略在各群体中的效应，(3)根据这些结果分析使用其他方法减少偏差或解决问题的可能性。我们介绍了在后处理阶段应用所提度量标准的方法，并通过一个示例说明其实用应用。该示例展示了分析去偏见策略的比例性如何补充传统公平性度量，提供更深入的视角以确保所有群体的更公平结果。 

---
# PD$^3$: A Project Duplication Detection Framework via Adapted Multi-Agent Debate 

**Title (ZH)**: PD$^3$: 一种基于适应性多agent辩论的项目复制检测框架 

**Authors**: Dezheng Bao, Yueci Yang, Xin Chen, Zhengxuan Jiang, Zeguo Fei, Daoze Zhang, Xuanwen Huang, Junru Chen, Chutian Yu, Xiang Yuan, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17492)  

**Abstract**: Project duplication detection is critical for project quality assessment, as it improves resource utilization efficiency by preventing investing in newly proposed project that have already been studied. It requires the ability to understand high-level semantics and generate constructive and valuable feedback. Existing detection methods rely on basic word- or sentence-level comparison or solely apply large language models, lacking valuable insights for experts and in-depth comprehension of project content and review criteria. To tackle this issue, we propose PD$^3$, a Project Duplication Detection framework via adapted multi-agent Debate. Inspired by real-world expert debates, it employs a fair competition format to guide multi-agent debate to retrieve relevant projects. For feedback, it incorporates both qualitative and quantitative analysis to improve its practicality. Over 800 real-world power project data spanning more than 20 specialized fields are used to evaluate the framework, demonstrating that our method outperforms existing approaches by 7.43% and 8.00% in two downstream tasks. Furthermore, we establish an online platform, Review Dingdang, to assist power experts, saving 5.73 million USD in initial detection on more than 100 newly proposed projects. 

**Abstract (ZH)**: 基于多智能体辩论的项目重复检测框架PD$^3$ 

---
# Partner Modelling Emerges in Recurrent Agents (But Only When It Matters) 

**Title (ZH)**: Recurrent 代理中的伙伴建模现象（但仅在必要时出现） 

**Authors**: Ruaridh Mon-Williams, Max Taylor-Davies, Elizabeth Mieczkowski, Natalia Velez, Neil R. Bramley, Yanwei Wang, Thomas L. Griffiths, Christopher G. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17323)  

**Abstract**: Humans are remarkably adept at collaboration, able to infer the strengths and weaknesses of new partners in order to work successfully towards shared goals. To build AI systems with this capability, we must first understand its building blocks: does such flexibility require explicit, dedicated mechanisms for modelling others -- or can it emerge spontaneously from the pressures of open-ended cooperative interaction? To investigate this question, we train simple model-free RNN agents to collaborate with a population of diverse partners. Using the `Overcooked-AI' environment, we collect data from thousands of collaborative teams, and analyse agents' internal hidden states. Despite a lack of additional architectural features, inductive biases, or auxiliary objectives, the agents nevertheless develop structured internal representations of their partners' task abilities, enabling rapid adaptation and generalisation to novel collaborators. We investigated these internal models through probing techniques, and large-scale behavioural analysis. Notably, we find that structured partner modelling emerges when agents can influence partner behaviour by controlling task allocation. Our results show that partner modelling can arise spontaneously in model-free agents -- but only under environmental conditions that impose the right kind of social pressure. 

**Abstract (ZH)**: 人类在协作方面具有出色的能力，能够推断新合作伙伴的优势和劣势，以共同实现目标。为了构建具备这种能力的AI系统，我们必须首先理解其构建块：这种灵活性是否需要显式的、专门的机制来建模他人，还是可以从开放合作交互的压力中自发产生？为了探究这一问题，我们训练简单的无模型RNN代理与多样化的合作伙伴进行合作。使用“Overcooked-AI”环境，我们收集了成千上万的合作团队的数据，并分析了代理的内部隐藏状态。尽管缺乏额外的架构特征、归纳偏置或辅助目标，代理仍然发展出结构化的内部表征，以合作伙伴的任务能力为基础，实现快速适应和对新合作伙伴的泛化。通过探测技术和大规模行为分析，我们发现，当代理能够通过控制任务分配来影响合作伙伴的行为时，结构化的合作伙伴建模会自发产生。我们的结果显示，在适当的环境条件下，无模型代理中的合作伙伴建模可以自发产生。 

---
# An Affective-Taxis Hypothesis for Alignment and Interpretability 

**Title (ZH)**: 情感趋同假设：对齐与可解释性 

**Authors**: Eli Sennesh, Maxwell Ramstead  

**Link**: [PDF](https://arxiv.org/pdf/2505.17024)  

**Abstract**: AI alignment is a field of research that aims to develop methods to ensure that agents always behave in a manner aligned with (i.e. consistently with) the goals and values of their human operators, no matter their level of capability. This paper proposes an affectivist approach to the alignment problem, re-framing the concepts of goals and values in terms of affective taxis, and explaining the emergence of affective valence by appealing to recent work in evolutionary-developmental and computational neuroscience. We review the state of the art and, building on this work, we propose a computational model of affect based on taxis navigation. We discuss evidence in a tractable model organism that our model reflects aspects of biological taxis navigation. We conclude with a discussion of the role of affective taxis in AI alignment. 

**Abstract (ZH)**: AI对齐是一个旨在开发方法以确保代理始终以与人类操作者的目标和价值观一致的方式行为的研究领域，无论代理的能力水平如何。本文提出了一种情感主义方法来解决对齐问题，重新定义了目标和价值观的概念为情感趋化性，并通过引用进化发展和计算神经科学的最新成果来解释情感价值的产生。我们回顾了相关研究，并在此基础上提出了基于趋化性导航的情感计算模型。我们讨论了在可处理的模型生物体中支持我们模型反映生物学趋化性导航特征的证据。最后，我们讨论了情感趋化性在AI对齐中的作用。 

---
# Leveraging KANs for Expedient Training of Multichannel MLPs via Preconditioning and Geometric Refinement 

**Title (ZH)**: 利用KANs加速多通道MLP训练的预条件化与几何 refinement 方法 

**Authors**: Jonas A. Actor, Graham Harper, Ben Southworth, Eric C. Cyr  

**Link**: [PDF](https://arxiv.org/pdf/2505.18131)  

**Abstract**: Multilayer perceptrons (MLPs) are a workhorse machine learning architecture, used in a variety of modern deep learning frameworks. However, recently Kolmogorov-Arnold Networks (KANs) have become increasingly popular due to their success on a range of problems, particularly for scientific machine learning tasks. In this paper, we exploit the relationship between KANs and multichannel MLPs to gain structural insight into how to train MLPs faster. We demonstrate the KAN basis (1) provides geometric localized support, and (2) acts as a preconditioned descent in the ReLU basis, overall resulting in expedited training and improved accuracy. Our results show the equivalence between free-knot spline KAN architectures, and a class of MLPs that are refined geometrically along the channel dimension of each weight tensor. We exploit this structural equivalence to define a hierarchical refinement scheme that dramatically accelerates training of the multi-channel MLP architecture. We show further accuracy improvements can be had by allowing the $1$D locations of the spline knots to be trained simultaneously with the weights. These advances are demonstrated on a range of benchmark examples for regression and scientific machine learning. 

**Abstract (ZH)**: 多层感知机（MLPs）是现代深度学习框架中的一种常用机器学习架构。然而，近年来，Kolmogorov-Arnold 网络（KANs）因在多种问题上的成功，特别是在科学机器学习任务中的表现，而变得越来越受欢迎。本文利用 KANs 和多通道 MLPs 之间的关系，揭示了如何更快地训练 MLPs 的结构洞察。我们证明 KAN 基（1）提供几何局部支持，（2）作为 ReLU 基中的预条件下降，整体上导致训练加速并提高精度。我们的结果表明，自由结节样条 KAN 架构与一类沿每个权重张量的通道维度进行几何细化的 MLPs 相等价。我们利用这种结构等价性定义了一种分层细化方案，极大地加速了多通道 MLP 架构的训练。通过同时训练样条结点的一维位置和权重，我们进一步提高了精度。这些进步在回归和科学机器学习的各种基准示例中得到了展示。 

---
# CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays 

**Title (ZH)**: CXReasonBench：用于评估胸部X光结构化诊断推理的基准 

**Authors**: Hyungyung Lee, Geon Choi, Jung-Oh Lee, Hangyul Yoon, Hyuk Gi Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18087)  

**Abstract**: Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks focus mainly on the final diagnostic answer, offering limited insight into whether models engage in clinically meaningful reasoning. To address this, we present CheXStruct and CXReasonBench, a structured pipeline and benchmark built on the publicly available MIMIC-CXR-JPG dataset. CheXStruct automatically derives a sequence of intermediate reasoning steps directly from chest X-rays, such as segmenting anatomical regions, deriving anatomical landmarks and diagnostic measurements, computing diagnostic indices, and applying clinical thresholds. CXReasonBench leverages this pipeline to evaluate whether models can perform clinically valid reasoning steps and to what extent they can learn from structured guidance, enabling fine-grained and transparent assessment of diagnostic reasoning. The benchmark comprises 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each paired with up to 4 visual inputs, and supports multi-path, multi-stage evaluation including visual grounding via anatomical region selection and diagnostic measurements. Even the strongest of 10 evaluated LVLMs struggle with structured reasoning and generalization, often failing to link abstract knowledge with anatomically grounded visual interpretation. The code is available at this https URL 

**Abstract (ZH)**: 最近在大型视觉-语言模型（LVLMs）方面的进展使其实现了在医疗任务中的多项潜在应用，如报告生成和视觉问答。然而，现有的基准主要关注最终的诊断答案，未能提供模型在临床有意义的推理方面是否涉及的洞察。为了弥补这一不足，我们提出了CheXStruct和CXReasonBench，一个基于公开的MIMIC-CXR-JPG数据集构建的结构化管道和基准。CheXStruct自动从胸部X光片中提取一系列中间推理步骤，包括分割解剖区域、提取解剖标志和诊断测量、计算诊断指标以及应用临床阈值。CXReasonBench利用这个管道评估模型是否能够执行临床有效的推理步骤，并且在多路径、多阶段评估中，包括通过解剖区域选择和诊断测量的视觉接地，评估模型从结构化指导中学到的程度，以实现细粒度和透明的诊断推理评估。基准数据集包含18,988个QA对，跨越12项诊断任务和1,200个病例，每个病例最多配有一个4个视觉输入，并支持包括视觉接地在内的多路径、多阶段评估。即使在评估的10个最强LVLM中，也有多数难以完成结构化推理和泛化，往往无法将抽象知识与解剖学相关的视觉解释联系起来。代码可在以下链接获取。 

---
# Backpropagation-Free Metropolis-Adjusted Langevin Algorithm 

**Title (ZH)**: 无回传的甲放手变形林格.GETAM（Metropolis-Adjusted Langevin Algorithm）算法 

**Authors**: Adam D. Cobb, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.18081)  

**Abstract**: Recent work on backpropagation-free learning has shown that it is possible to use forward-mode automatic differentiation (AD) to perform optimization on differentiable models. Forward-mode AD requires sampling a tangent vector for each forward pass of a model. The result is the model evaluation with the directional derivative along the tangent. In this paper, we illustrate how the sampling of this tangent vector can be incorporated into the proposal mechanism for the Metropolis-Adjusted Langevin Algorithm (MALA). As such, we are the first to introduce a backpropagation-free gradient-based Markov chain Monte Carlo (MCMC) algorithm. We also extend to a novel backpropagation-free position-specific preconditioned forward-mode MALA that leverages Hessian information. Overall, we propose four new algorithms: Forward MALA; Line Forward MALA; Pre-conditioned Forward MALA, and Pre-conditioned Line Forward MALA. We highlight the reduced computational cost of the forward-mode samplers and show that forward-mode is competitive with the original MALA, while even outperforming it depending on the probabilistic model. We include Bayesian inference results on a range of probabilistic models, including hierarchical distributions and Bayesian neural networks. 

**Abstract (ZH)**: Recent work on backpropagation-free learning has shown that it is possible to use forward-mode automatic differentiation (AD) to perform optimization on differentiable models. Forward-mode AD requires sampling a tangent vector for each forward pass of a model. The result is the model evaluation with the directional derivative along the tangent. In this paper,我们首次引入了一种无需反向传播的梯度基础马尔可夫链蒙特卡洛(MCMC)算法。我们还将这种前向模式应用到了一个新的位置特定预条件的前向模式MALA算法中，利用了海森矩阵信息。总体而言，我们提出了四种新的算法：前向MALA；线性前向MALA；预条件前向MALA，以及预条件线性前向MALA。我们突出了前向模式采样的计算成本降低，并展示了前向模式在不同概率模型下与原始MALA算法的竞争性和优越性。我们在多层次分布和贝叶斯神经网络等多种概率模型下提供了贝叶斯推理结果。 

---
# AFD-STA: Adaptive Filtering Denoising with Spatiotemporal Attention for Chaotic System Prediction 

**Title (ZH)**: 自适应滤波去噪的空间时间注意力 chaotic系统预测 

**Authors**: Chunlin Gong, Yin Wang, Jingru Li, Hanleran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18080)  

**Abstract**: This paper presents AFD-STA Net, a neural framework integrating adaptive filtering and spatiotemporal dynamics learning for predicting high-dimensional chaotic systems governed by partial differential equations. The architecture combines: 1) An adaptive exponential smoothing module with position-aware decay coefficients for robust attractor reconstruction, 2) Parallel attention mechanisms capturing cross-temporal and spatial dependencies, 3) Dynamic gated fusion of multiscale features, and 4) Deep projection networks with dimension-scaling capabilities. Numerical experiments on nonlinear PDE systems demonstrate the model's effectiveness in maintaining prediction accuracy under both smooth and strongly chaotic regimes while exhibiting noise tolerance through adaptive filtering. Component ablation studies confirm critical contributions from each module, particularly highlighting the essential role of spatiotemporal attention in learning complex dynamical interactions. The framework shows promising potential for real-world applications requiring simultaneous handling of measurement uncertainties and high-dimensional nonlinear dynamics. 

**Abstract (ZH)**: AFD-STA 网络：一种结合自适应滤波和时空动力学学习的神经框架，用于预测受偏微分方程支配的高维混沌系统 

---
# Towards Uncertainty Aware Task Delegation and Human-AI Collaborative Decision-Making 

**Title (ZH)**: 面向不确定性感知的任务委派及人机协同决策研究 

**Authors**: Min Hun Lee, Martyn Zhe Yu Tok  

**Link**: [PDF](https://arxiv.org/pdf/2505.18066)  

**Abstract**: Despite the growing promise of artificial intelligence (AI) in supporting decision-making across domains, fostering appropriate human reliance on AI remains a critical challenge. In this paper, we investigate the utility of exploring distance-based uncertainty scores for task delegation to AI and describe how these scores can be visualized through embedding representations for human-AI decision-making. After developing an AI-based system for physical stroke rehabilitation assessment, we conducted a study with 19 health professionals and 10 students in medicine/health to understand the effect of exploring distance-based uncertainty scores on users' reliance on AI. Our findings showed that distance-based uncertainty scores outperformed traditional probability-based uncertainty scores in identifying uncertain cases. In addition, after exploring confidence scores for task delegation and reviewing embedding-based visualizations of distance-based uncertainty scores, participants achieved an 8.20% higher rate of correct decisions, a 7.15% higher rate of changing their decisions to correct ones, and a 7.14% lower rate of incorrect changes after reviewing AI outputs than those reviewing probability-based uncertainty scores ($p<0.01$). Our findings highlight the potential of distance-based uncertainty scores to enhance decision accuracy and appropriate reliance on AI while discussing ongoing challenges for human-AI collaborative decision-making. 

**Abstract (ZH)**: 尽管人工智能（AI）在支持跨领域决策方面展现了 growing 的潜力，培养适当的人类依赖性仍然是一项关键挑战。本文探讨了基于距离的不确定性评分在任务委托给AI方面的效用，并描述了这些评分如何通过嵌入表示进行可视化，以辅助人类与AI的决策。在开发了一种基于AI的物理中风康复评估系统后，我们对19名医疗专业人员和10名医学院/健康科学学生进行了研究，以了解探索基于距离的不确定性评分对用户依赖AI的影响。研究结果表明，基于距离的不确定性评分在识别不确定案例方面优于基于概率的不确定性评分。此外，在探索任务委托的信心评分并审阅基于距离的不确定性评分的嵌入表示可视化后，参与者在审阅AI输出时正确决策的比例提高了8.20%，将其决策纠正的比例提高了7.15%，并且更改错误决策的比例降低了7.14%（$p<0.01$）。研究结果突显了基于距离的不确定性评分在提高决策准确性和适当依赖AI方面的潜力，同时讨论了人类与AI协作决策中的持续挑战。 

---
# AI Literacy for Legal AI Systems: A practical approach 

**Title (ZH)**: 法律AI系统中的AI素养：一种实用方法 

**Authors**: Gizem Gultekin-Varkonyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18006)  

**Abstract**: Legal AI systems are increasingly being adopted by judicial and legal system deployers and providers worldwide to support a range of applications. While they offer potential benefits such as reducing bias, increasing efficiency, and improving accountability, they also pose significant risks, requiring a careful balance between opportunities, and legal and ethical development and deployment. AI literacy, as a legal requirement under the EU AI Act and a critical enabler of ethical AI for deployers and providers, could be a tool to achieve this. The article introduces the term "legal AI systems" and then analyzes the concept of AI literacy and the benefits and risks associated with these systems. This analysis is linked to a broader AI-L concept for organizations that deal with legal AI systems. The outcome of the article, a roadmap questionnaire as a practical tool for developers and providers to assess risks, benefits, and stakeholder concerns, could be useful in meeting societal and regulatory expectations for legal AI. 

**Abstract (ZH)**: Legal AI系统在全球司法和法律系统部署者与提供商中的应用日益增多，以支持多种应用。虽然它们提供了减少偏见、提高效率和增强问责制等潜在好处，但也带来了显著的风险，需要在机会与法律和伦理的发展与部署之间谨慎平衡。作为一种法律要求，欧盟AI法案下的AI素养，以及对部署者和提供商至关重要的人工智能伦理的促进者，可能是实现这一目标的工具。本文介绍了“法律AI系统”这一术语，并分析了AI素养的概念以及这些系统关联的利益和风险。这种分析与组织处理法律AI系统时更广泛的AI-L概念相关联。本文的结果，一份路线图问卷作为开发者和提供商评估风险、利益和利益相关者关切的实际工具，有助于满足社会和监管对法律AI的期望。 

---
# An Example Safety Case for Safeguards Against Misuse 

**Title (ZH)**: 一个针对误用防护的安全案例示例 

**Authors**: Joshua Clymer, Jonah Weinbaum, Robert Kirk, Kimberly Mai, Selena Zhang, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2505.18003)  

**Abstract**: Existing evaluations of AI misuse safeguards provide a patchwork of evidence that is often difficult to connect to real-world decisions. To bridge this gap, we describe an end-to-end argument (a "safety case") that misuse safeguards reduce the risk posed by an AI assistant to low levels. We first describe how a hypothetical developer red teams safeguards, estimating the effort required to evade them. Then, the developer plugs this estimate into a quantitative "uplift model" to determine how much barriers introduced by safeguards dissuade misuse (this https URL). This procedure provides a continuous signal of risk during deployment that helps the developer rapidly respond to emerging threats. Finally, we describe how to tie these components together into a simple safety case. Our work provides one concrete path -- though not the only path -- to rigorously justifying AI misuse risks are low. 

**Abstract (ZH)**: 现有的AI滥用防护评估提供了支离破碎的证据，往往难以连接到现实世界的决策。为弥补这一差距，我们描述了一个从始至终的论证（“安全案例”），说明滥用防护如何将AI助手所带来的风险降低到低水平。我们首先描述一个假设的开发者红队测试这些防护，估计绕过它们所需的努力。然后，开发者将这一估计值输入到一个定量的“提升模型”中，以确定由防护措施引入的障碍对滥用行为的威慑程度（参见此链接：[this https URL]）。此过程提供了一个在部署期间持续的风险信号，帮助开发者迅速应对新兴威胁。最后，我们描述了如何将这些组件整合为一个简单的安全案例。我们的研究提供了一条具体的路径——尽管不是唯一路径——来严格证明AI滥用风险较低。 

---
# Outcome-based Reinforcement Learning to Predict the Future 

**Title (ZH)**: 基于 Outcome 的强化学习预测未来 

**Authors**: Benjamin Turtel, Danny Franklin, Kris Skotheim, Luke Hewitt, Philipp Schoenegger  

**Link**: [PDF](https://arxiv.org/pdf/2505.17989)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has boosted math and coding in large language models, yet there has been little effort to extend RLVR into messier, real-world domains like forecasting. One sticking point is that outcome-based reinforcement learning for forecasting must learn from binary, delayed, and noisy rewards, a regime where standard fine-tuning is brittle. We show that outcome-only online RL on a 14B model can match frontier-scale accuracy and surpass it in calibration and hypothetical prediction market betting by adapting two leading algorithms, Group-Relative Policy Optimisation (GRPO) and ReMax, to the forecasting setting. Our adaptations remove per-question variance scaling in GRPO, apply baseline-subtracted advantages in ReMax, hydrate training with 100k temporally consistent synthetic questions, and introduce lightweight guard-rails that penalise gibberish, non-English responses and missing rationales, enabling a single stable pass over 110k events. Scaling ReMax to 110k questions and ensembling seven predictions yields a 14B model that matches frontier baseline o1 on accuracy on our holdout set (Brier = 0.193, p = 0.23) while beating it in calibration (ECE = 0.042, p < 0.001). A simple trading rule turns this calibration edge into \$127 of hypothetical profit versus \$92 for o1 (p = 0.037). This demonstrates that refined RLVR methods can convert small-scale LLMs into potentially economically valuable forecasting tools, with implications for scaling this to larger models. 

**Abstract (ZH)**: 验证奖励的强化学习在大型语言模型中的数学和编码方面取得了进步，然而很少有人尝试将RLVR扩展到如预测等更复杂的现实世界领域。我们通过将两个领先算法，Group-Relative Policy Optimisation (GRPO) 和 ReMax，适应预测设置，展示了基于结果的在线RL可以在14B模型中达到前沿级别的准确性和更好的校准，以及在假设的预测市场投注中表现出色。通过这些适应，我们在GRPO中消除了答案偏差，为ReMax应用了基线减去的优势，并在训练中加入了10万个多时间一致的合成问题，同时引入了轻量级的监管措施，以惩罚无意义、非英语的回答和缺失的理由，从而使模型能够在11万多个事件上保持稳定。将ReMax扩展到11万问题并ensemble七次预测，14B模型在保留集上达到了前沿基线o1的准确度（Brier = 0.193, p = 0.23），但在校准方面超过了它（ECE = 0.042, p < 0.001）。一个简单的交易规则将这种校准优势转化为127美元的假设盈利，而o1仅为92美元（p = 0.037）。这表明改进的RLVR方法可以使小规模的LLM成为潜在具有经济价值的预测工具，并对扩大这一方法适用于更大规模的模型具有重要意义。 

---
# Federated Causal Inference from Multi-Site Observational Data via Propensity Score Aggregation 

**Title (ZH)**: 多中心观察数据通过倾向得分聚合的联邦因果推断 

**Authors**: Khellaf Rémi, Bellet Aurélien, Josse Julie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17961)  

**Abstract**: Causal inference typically assumes centralized access to individual-level data. Yet, in practice, data are often decentralized across multiple sites, making centralization infeasible due to privacy, logistical, or legal constraints. We address this by estimating the Average Treatment Effect (ATE) from decentralized observational data using federated learning, which enables inference through the exchange of aggregate statistics rather than individual-level data. We propose a novel method to estimate propensity scores in a (non-)parametric manner by computing a federated weighted average of local scores, using two theoretically grounded weighting schemes -- Membership Weights (MW) and Density Ratio Weights (DW) -- that balance communication efficiency and model flexibility. These federated scores are then used to construct two ATE estimators: the Federated Inverse Propensity Weighting estimator (Fed-IPW) and its augmented variant (Fed-AIPW). Unlike meta-analysis methods, which fail when any site violates positivity, our approach leverages heterogeneity in treatment assignment across sites to improve overlap. We show that Fed-IPW and Fed-AIPW perform well under site-level heterogeneity in sample sizes, treatment mechanisms, and covariate distributions, with theoretical analysis and experiments on simulated and real-world data highlighting their strengths and limitations relative to meta-analysis and related methods. 

**Abstract (ZH)**: 分散数据分析中的因果推断：使用联邦学习估计平均处理效应 

---
# LMask: Learn to Solve Constrained Routing Problems with Lazy Masking 

**Title (ZH)**: LMask: 基于懒惰掩码学习解决约束路由问题 

**Authors**: Tianyou Li, Haijun Zou, Jiayuan Wu, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17938)  

**Abstract**: Routing problems are canonical combinatorial optimization tasks with wide-ranging applications in logistics, transportation, and supply chain management. However, solving these problems becomes significantly more challenging when complex constraints are involved. In this paper, we propose LMask, a novel learning framework that utilizes dynamic masking to generate high-quality feasible solutions for constrained routing problems. LMask introduces the LazyMask decoding method, which lazily refines feasibility masks with the backtracking mechanism. In addition, it employs the refinement intensity embedding to encode the search trace into the model, mitigating representation ambiguities induced by backtracking. To further reduce sampling cost, LMask sets a backtracking budget during decoding, while constraint violations are penalized in the loss function during training to counteract infeasibility caused by this budget. We provide theoretical guarantees for the validity and probabilistic optimality of our approach. Extensive experiments on the traveling salesman problem with time windows (TSPTW) and TSP with draft limits (TSPDL) demonstrate that LMask achieves state-of-the-art feasibility rates and solution quality, outperforming existing neural methods. 

**Abstract (ZH)**: 约束路由问题是一种在物流、交通和供应链管理等领域广泛应用的典型组合优化任务。然而，当涉及复杂约束时，解决这些问题变得显著更具挑战性。在本文中，我们提出了一种新颖的学习框架LMask，利用动态掩码生成约束路由问题的高质量可行解。LMask引入了懒惰掩码解码方法，该方法通过回溯机制lazy地精化可行性掩码。此外，它使用精化强度嵌入将搜索轨迹编码到模型中，从而减轻由回溯引起的表示歧义。为了进一步降低采样成本，解码时LMask设置了一个回溯预算，而在训练过程中通过在损失函数中惩罚约束违反情况来弥补由此预算引起的不可行性。我们为该方法的有效性和概率最优性提供了理论保证。在带时间窗的旅行商问题（TSPTW）和带有载重限制的TSP（TSPDL）上的广泛实验表明，LMask实现了最先进的可行率和解的质量，并优于现有神经网络方法。 

---
# Towards Practical Defect-Focused Automated Code Review 

**Title (ZH)**: 面向实践的缺陷聚焦自动代码审查 

**Authors**: Junyi Lu, Lili Jiang, Xiaojia Li, Jianbing Fang, Fengjun Zhang, Li Yang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17928)  

**Abstract**: The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: 1) capturing relevant context, 2) improving key bug inclusion (KBI), 3) reducing false alarm rates (FAR), and 4) integrating human workflows. To tackle these, we propose 1) code slicing algorithms for context extraction, 2) a multi-role LLM framework for KBI, 3) a filtering mechanism for FAR reduction, and 4) a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2x improvement over standard LLMs and a 10x gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages language-agnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability. 

**Abstract (ZH)**: 代码审核的复杂性推动了自动审核评论的研究，但先前的方法通过将其简化为片段级代码到文本生成任务，并依赖如BLEU等文本相似度指标进行评估，忽略了仓库上下文、实际合并请求评估和缺陷检测，限制了其实用性。为解决这些问题，我们研究了一家拥有近4亿日活跃用户的公司在线推荐服务中的全流程自动化管道，分析包含数十万行代码的工业级C++代码库。我们识别了四个关键挑战：1) 捕获相关上下文，2) 改进关键错误包含（KBI），3) 降低误报率（FAR），4) 集成人工流程。为此，我们提出了1) 代码切片算法进行上下文提取，2) 多角色LLM框架进行KBI，3) 过滤机制以降低FAR，4) 新颖的提示设计以改善与人工的交互。我们的方法在基于历史故障报告的实际合并请求上验证，实现了标准LLM的2倍改进和之前基准的10倍提升。虽然展示的结果集中在C++上，但底层框架设计利用了语言无关的原则（如基于抽象语法树的分析），表明其有更广泛的适用潜力。 

---
# Evaluation of Few-Shot Learning Methods for Kidney Stone Type Recognition in Ureteroscopy 

**Title (ZH)**: 尿道镜下肾结石类型识别的少样本学习方法评价 

**Authors**: Carlos Salazar-Ruiz, Francisco Lopez-Tiro, Ivan Reyes-Amezcua, Clement Larose, Gilberto Ochoa-Ruiz, Christian Daul  

**Link**: [PDF](https://arxiv.org/pdf/2505.17921)  

**Abstract**: Determining the type of kidney stones is crucial for prescribing appropriate treatments to prevent recurrence. Currently, various approaches exist to identify the type of kidney stones. However, obtaining results through the reference ex vivo identification procedure can take several weeks, while in vivo visual recognition requires highly trained specialists. For this reason, deep learning models have been developed to provide urologists with an automated classification of kidney stones during ureteroscopies. Nevertheless, a common issue with these models is the lack of training data. This contribution presents a deep learning method based on few-shot learning, aimed at producing sufficiently discriminative features for identifying kidney stone types in endoscopic images, even with a very limited number of samples. This approach was specifically designed for scenarios where endoscopic images are scarce or where uncommon classes are present, enabling classification even with a limited training dataset. The results demonstrate that Prototypical Networks, using up to 25% of the training data, can achieve performance equal to or better than traditional deep learning models trained with the complete dataset. 

**Abstract (ZH)**: 基于少样本学习的深度学习方法在内窥镜图像中识别肾结石类型 

---
# DataRater: Meta-Learned Dataset Curation 

**Title (ZH)**: DataRater: 元学习的数据集策展 

**Authors**: Dan A. Calian, Gregory Farquhar, Iurii Kemaev, Luisa M. Zintgraf, Matteo Hessel, Jeremy Shar, Junhyuk Oh, András György, Tom Schaul, Jeffrey Dean, Hado van Hasselt, David Silver  

**Link**: [PDF](https://arxiv.org/pdf/2505.17895)  

**Abstract**: The quality of foundation models depends heavily on their training data. Consequently, great efforts have been put into dataset curation. Yet most approaches rely on manual tuning of coarse-grained mixtures of large buckets of data, or filtering by hand-crafted heuristics. An approach that is ultimately more scalable (let alone more satisfying) is to \emph{learn} which data is actually valuable for training. This type of meta-learning could allow more sophisticated, fine-grained, and effective curation. Our proposed \emph{DataRater} is an instance of this idea. It estimates the value of training on any particular data point. This is done by meta-learning using `meta-gradients', with the objective of improving training efficiency on held out data. In extensive experiments across a range of model scales and datasets, we find that using our DataRater to filter data is highly effective, resulting in significantly improved compute efficiency. 

**Abstract (ZH)**: 基础模型的质量高度依赖于其训练数据。因此，人们投入了大量的精力进行数据集整理。然而，大多数方法依赖于手工调整粗粒度的大数据桶混合或手工设计的启发式过滤。一种更具扩展性（更不用说更令人满意）的方法是通过学习哪些数据实际上对训练有价值。这种元学习可以使得数据整理更加复杂、精细且有效。我们提出的DataRater是这一想法的一个实例。它通过元学习方法计算任何特定数据点的训练价值，目标是提高保留数据的训练效率。在广泛实验中，我们发现使用DataRater筛选数据非常有效，显著提高了计算效率。 

---
# FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks 

**Title (ZH)**: FastCAV: 效率计算概念激活向量以解释深度神经网络 

**Authors**: Laines Schmalwasser, Niklas Penzel, Joachim Denzler, Julia Niebling  

**Link**: [PDF](https://arxiv.org/pdf/2505.17883)  

**Abstract**: Concepts such as objects, patterns, and shapes are how humans understand the world. Building on this intuition, concept-based explainability methods aim to study representations learned by deep neural networks in relation to human-understandable concepts. Here, Concept Activation Vectors (CAVs) are an important tool and can identify whether a model learned a concept or not. However, the computational cost and time requirements of existing CAV computation pose a significant challenge, particularly in large-scale, high-dimensional architectures. To address this limitation, we introduce FastCAV, a novel approach that accelerates the extraction of CAVs by up to 63.6x (on average 46.4x). We provide a theoretical foundation for our approach and give concrete assumptions under which it is equivalent to established SVM-based methods. Our empirical results demonstrate that CAVs calculated with FastCAV maintain similar performance while being more efficient and stable. In downstream applications, i.e., concept-based explanation methods, we show that FastCAV can act as a replacement leading to equivalent insights. Hence, our approach enables previously infeasible investigations of deep models, which we demonstrate by tracking the evolution of concepts during model training. 

**Abstract (ZH)**: 基于概念的解释方法旨在研究深度神经网络学习到的表示与人类可理解的概念之间的关系。概念激活向量（CAVs）是这一研究中重要的工具，能够识别模型是否学习了某一概念。然而，现有CAV计算的高计算成本和时间要求构成了一个显著挑战，特别是在大规模、高维架构中。为解决这一限制，我们提出了一种名为FastCAV的新方法，该方法在CAV提取上可加速63.6倍（平均加速46.4倍）。我们为该方法提供了理论基础，并在具体假设下，它等同于已建立的SVM基方法。我们的实验结果表明，使用FastCAV计算的CAVs在保持类似性能的同时更具高效性和稳定性。在下游应用中，即概念基解释方法中，我们证明FastCAV可以作为替代品，提供等效的洞察。因此，我们的方法使以前不可行的深度模型研究成为可能，我们通过追踪模型训练过程中概念的演变来展示这一点。 

---
# Toward Optimal ANC: Establishing Mutual Information Lower Bound 

**Title (ZH)**: Toward Optimal ANC: Establishing Mutual Information Lower Bound 

**Authors**: François Derrida, Shahar Lutati, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17877)  

**Abstract**: Active Noise Cancellation (ANC) algorithms aim to suppress unwanted acoustic disturbances by generating anti-noise signals that destructively interfere with the original noise in real time. Although recent deep learning-based ANC algorithms have set new performance benchmarks, there remains a shortage of theoretical limits to rigorously assess their improvements. To address this, we derive a unified lower bound on cancellation performance composed of two components. The first component is information-theoretic: it links residual error power to the fraction of disturbance entropy captured by the anti-noise signal, thereby quantifying limits imposed by information-processing capacity. The second component is support-based: it measures the irreducible error arising in frequency bands that the cancellation path cannot address, reflecting fundamental physical constraints. By taking the maximum of these two terms, our bound establishes a theoretical ceiling on the Normalized Mean Squared Error (NMSE) attainable by any ANC algorithm. We validate its tightness empirically on the NOISEX dataset under varying reverberation times, demonstrating robustness across diverse acoustic conditions. 

**Abstract (ZH)**: 基于信息理论和支持理论的统一下界：主动噪声取消算法的理论上限 

---
# MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback 

**Title (ZH)**: MOOSE-Chem3：基于模拟实验反馈的实验指导假设排序研究 

**Authors**: Wanhao Liu, Zonglin Yang, Jue Wang, Lidong Bing, Di Zhang, Dongzhan Zhou, Yuqiang Li, Houqiang Li, Erik Cambria, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17873)  

**Abstract**: Hypothesis ranking is a crucial component of automated scientific discovery, particularly in natural sciences where wet-lab experiments are costly and throughput-limited. Existing approaches focus on pre-experiment ranking, relying solely on large language model's internal reasoning without incorporating empirical outcomes from experiments. We introduce the task of experiment-guided ranking, which aims to prioritize candidate hypotheses based on the results of previously tested ones. However, developing such strategies is challenging due to the impracticality of repeatedly conducting real experiments in natural science domains. To address this, we propose a simulator grounded in three domain-informed assumptions, modeling hypothesis performance as a function of similarity to a known ground truth hypothesis, perturbed by noise. We curate a dataset of 124 chemistry hypotheses with experimentally reported outcomes to validate the simulator. Building on this simulator, we develop a pseudo experiment-guided ranking method that clusters hypotheses by shared functional characteristics and prioritizes candidates based on insights derived from simulated experimental feedback. Experiments show that our method outperforms pre-experiment baselines and strong ablations. 

**Abstract (ZH)**: 实验引导的假设排序在自动科学发现中的应用：基于化学假设的仿真模拟及排序方法 

---
# Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time Series Forecasting 

**Title (ZH)**: 低秩适应与部分参数共享的混合时间序列预测方法 

**Authors**: Licheng Pan, Zhichao Chen, Haoxuan Li, Guangyi Liu, Zhijian Xu, Zhaoran Liu, Hao Wang, Ying Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.17872)  

**Abstract**: Multi-task forecasting has become the standard approach for time-series forecasting (TSF). However, we show that it suffers from an Expressiveness Bottleneck, where predictions at different time steps share the same representation, leading to unavoidable errors even with optimal representations. To address this issue, we propose a two-stage framework: first, pre-train a foundation model for one-step-ahead prediction; then, adapt it using step-specific LoRA this http URL design enables the foundation model to handle any number of forecast steps while avoiding the expressiveness bottleneck. We further introduce the Mixture-of-LoRA (MoLA) model, which employs adaptively weighted LoRA experts to achieve partial parameter sharing across steps. This approach enhances both efficiency and forecasting performance by exploiting interdependencies between forecast steps. Experiments show that MoLA significantly improves model expressiveness and outperforms state-of-the-art time-series forecasting methods. Code is available at this https URL. 

**Abstract (ZH)**: 多任务预测已成为时间序列预测（TSF）的标准方法。然而，我们表明它受到表现瓶颈的影响，即不同时间步骤的预测共享相同的表示，即使使用最优表示也无法避免错误。为解决这一问题，我们提出一个两阶段框架：首先，预先训练一个基础模型进行单步预测；然后，使用特定时间步骤的LoRA进行适应。该设计理念使得基础模型能够处理任意数量的预测步骤，同时避免了表现瓶颈。我们进一步引入了混合LoRA（MoLA）模型，该模型利用自适应加权的LoRA专家在不同步骤之间实现部分参数共享。该方法通过利用预测步骤之间的相互依赖性，提高了效率和预测性能。实验证明，MoLA显著提升了模型表现并优于当前最先进的时间序列预测方法。代码可在该链接处获取。 

---
# Stochastic Weight Sharing for Bayesian Neural Networks 

**Title (ZH)**: 贝叶斯神经网络中的随机权重共享 

**Authors**: Moule Lin, Shuhao Guan, Weipeng Jing, Goetz Botterweck, Andrea Patane  

**Link**: [PDF](https://arxiv.org/pdf/2505.17856)  

**Abstract**: While offering a principled framework for uncertainty quantification in deep learning, the employment of Bayesian Neural Networks (BNNs) is still constrained by their increased computational requirements and the convergence difficulties when training very deep, state-of-the-art architectures. In this work, we reinterpret weight-sharing quantization techniques from a stochastic perspective in the context of training and inference with Bayesian Neural Networks (BNNs). Specifically, we leverage 2D adaptive Gaussian distributions, Wasserstein distance estimations, and alpha blending to encode the stochastic behaviour of a BNN in a lower dimensional, soft Gaussian representation. Through extensive empirical investigation, we demonstrate that our approach significantly reduces the computational overhead inherent in Bayesian learning by several orders of magnitude, enabling the efficient Bayesian training of large-scale models, such as ResNet-101 and Vision Transformer (VIT). On various computer vision benchmarks including CIFAR10, CIFAR100, and ImageNet1k. Our approach compresses model parameters by approximately 50x and reduces model size by 75, while achieving accuracy and uncertainty estimations comparable to the state-of-the-art. 

**Abstract (ZH)**: 基于贝叶斯神经网络的权值共享量化技术的随机重解读：提升大规模模型的高效贝叶斯训练与推理 

---
# Scaling Recurrent Neural Networks to a Billion Parameters with Zero-Order Optimization 

**Title (ZH)**: 用零阶优化将循环神经网络扩展到十亿参数 

**Authors**: Francois Chaubard, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.17852)  

**Abstract**: During inference, Recurrent Neural Networks (RNNs) scale constant in both FLOPs and GPU memory with increasing context length, as they compress all prior tokens into a fixed-size memory. In contrast, transformers scale linearly in FLOPs and, at best, linearly in memory during generation, since they must attend to all previous tokens explicitly. Despite this inference-time advantage, training large RNNs on long contexts remains impractical because standard optimization methods depend on Backpropagation Through Time (BPTT). BPTT requires retention of all intermediate activations during the forward pass, causing memory usage to scale linearly with both context length and model size. In this paper, we show that Zero-Order Optimization (ZOO) methods such as Random-vector Gradient Estimation (RGE) can successfully replace BPTT to train RNNs with convergence rates that match, or exceed BPTT by up to 19 fold, while using orders of magnitude less memory and cost, as the model remains in inference mode throughout training. We further demonstrate that Central-Difference RGE (CD-RGE) corresponds to optimizing a smoothed surrogate loss, inherently regularizing training and improving generalization. Our method matches or outperforms BPTT across three settings: (1) overfitting, (2) transduction, and (3) language modeling. Across all tasks, with sufficient perturbations, our models generalize as well as or better than those trained with BPTT, often in fewer steps. Despite the need for more forward passes per step, we can surpass BPTT wall-clock time per step using recent advancements such as FlashRNN and distributed inference. 

**Abstract (ZH)**: 基于零阶优化的递归神经网络训练方法：在保留推理优势的同时显著降低训练内存和成本 

---
# TransDF: Time-Series Forecasting Needs Transformed Label Alignment 

**Title (ZH)**: TransDF: 时间序列预测需要转换标签对齐 

**Authors**: Hao Wang, Licheng Pan, Zhichao Chen, Xu Chen, Qingyang Dai, Lei Wang, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17847)  

**Abstract**: Training time-series forecasting models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Transform-enhanced Direct Forecast (TransDF), which transforms the label sequence into decorrelated components with discriminated significance. Models are trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that TransDF achieves state-of-the-art performance and is compatible with various forecasting models. Code is available at this https URL. 

**Abstract (ZH)**: 时间序列预测模型的训练面临着设计有效学习目标的独特挑战。现有的方法主要使用时间均方误差，但面临两个关键挑战：（1）标签自相关性，导致标签序列似然性的偏差；（2）任务数量过多，会随着预测范围的增加而复杂化优化。为解决这些挑战，我们提出了Transform增强直接预测（TransDF），该方法将标签序列转换为去相关的具有鉴别显著性的成分。模型被训练以对齐这些最显著的成分，从而有效减轻标签自相关性并减少任务数量。广泛的实验表明，TransDF达到了最先进的性能，并且兼容各种预测模型。代码可在以下链接获取：this https URL。 

---
# TEDI: Trustworthy and Ethical Dataset Indicators to Analyze and Compare Dataset Documentation 

**Title (ZH)**: TEDI: 可信赖和伦理导向的数据集指标以分析和比较数据集文档 

**Authors**: Wiebke Hutiri, Mircea Cimpoi, Morgan Scheuerman, Victoria Matthews, Alice Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17841)  

**Abstract**: Dataset transparency is a key enabler of responsible AI, but insights into multimodal dataset attributes that impact trustworthy and ethical aspects of AI applications remain scarce and are difficult to compare across datasets. To address this challenge, we introduce Trustworthy and Ethical Dataset Indicators (TEDI) that facilitate the systematic, empirical analysis of dataset documentation. TEDI encompasses 143 fine-grained indicators that characterize trustworthy and ethical attributes of multimodal datasets and their collection processes. The indicators are framed to extract verifiable information from dataset documentation. Using TEDI, we manually annotated and analyzed over 100 multimodal datasets that include human voices. We further annotated data sourcing, size, and modality details to gain insights into the factors that shape trustworthy and ethical dimensions across datasets. We find that only a select few datasets have documented attributes and practices pertaining to consent, privacy, and harmful content indicators. The extent to which these and other ethical indicators are addressed varies based on the data collection method, with documentation of datasets collected via crowdsourced and direct collection approaches being more likely to mention them. Scraping dominates scale at the cost of ethical indicators, but is not the only viable collection method. Our approach and empirical insights contribute to increasing dataset transparency along trustworthy and ethical dimensions and pave the way for automating the tedious task of extracting information from dataset documentation in future. 

**Abstract (ZH)**: 可信和负责任的数据集指标 (TEDI): 促进多模态数据集文档的系统化、实证分析 

---
# Hybrid Mamba-Transformer Decoder for Error-Correcting Codes 

**Title (ZH)**: 混合Mamba-Transformer解码器用于纠错码 

**Authors**: Shy-el Cohen, Yoni Choukroun, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17834)  

**Abstract**: We introduce a novel deep learning method for decoding error correction codes based on the Mamba architecture, enhanced with Transformer layers. Our approach proposes a hybrid decoder that leverages Mamba's efficient sequential modeling while maintaining the global context capabilities of Transformers. To further improve performance, we design a novel layer-wise masking strategy applied to each Mamba layer, allowing selective attention to relevant code features at different depths. Additionally, we introduce a progressive layer-wise loss, supervising the network at intermediate stages and promoting robust feature extraction throughout the decoding process. Comprehensive experiments across a range of linear codes demonstrate that our method significantly outperforms Transformer-only decoders and standard Mamba models. 

**Abstract (ZH)**: 基于Mamba架构结合Transformer层的新型深度学习错误纠正码解码方法 

---
# An Attention Infused Deep Learning System with Grad-CAM Visualization for Early Screening of Glaucoma 

**Title (ZH)**: 一种集成注意力机制的深度学习系统及其Grad-CAM可视化在原发性青光眼早期筛查中的应用 

**Authors**: Ramanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17808)  

**Abstract**: This research work reveals the eye opening wisdom of the hybrid labyrinthine deep learning models synergy born out of combining a trailblazing convolutional neural network with a disruptive Vision Transformer, both intertwined together with a radical Cross Attention module. Here, two high yielding datasets for artificial intelligence models in detecting glaucoma, namely ACRIMA and Drishti, are utilized. 

**Abstract (ZH)**: 这项研究揭示了结合先驱卷积神经网络和颠覆性视觉变换器的混合迷宫深度学习模型的惊艳智慧，这些模型通过激进的交叉注意力模块相互交织。本文利用了两种专门用于人工智能模型检测青光眼的高产数据集，即ACRIMA和Drishti。 

---
# Hyperparameter Optimization via Interacting with Probabilistic Circuits 

**Title (ZH)**: 基于概率电路交互的超参数优化 

**Authors**: Jonas Seng, Fabrizio Ventola, Zhongjie Yu, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.17804)  

**Abstract**: Despite the growing interest in designing truly interactive hyperparameter optimization (HPO) methods, to date, only a few allow to include human feedback. Existing interactive Bayesian optimization (BO) methods incorporate human beliefs by weighting the acquisition function with a user-defined prior distribution. However, in light of the non-trivial inner optimization of the acquisition function prevalent in BO, such weighting schemes do not always accurately reflect given user beliefs. We introduce a novel BO approach leveraging tractable probabilistic models named probabilistic circuits (PCs) as a surrogate model. PCs encode a tractable joint distribution over the hybrid hyperparameter space and evaluation scores. They enable exact conditional inference and sampling. Based on conditional sampling, we construct a novel selection policy that enables an acquisition function-free generation of candidate points (thereby eliminating the need for an additional inner-loop optimization) and ensures that user beliefs are reflected accurately in the selection policy. We provide a theoretical analysis and an extensive empirical evaluation, demonstrating that our method achieves state-of-the-art performance in standard HPO and outperforms interactive BO baselines in interactive HPO. 

**Abstract (ZH)**: 尽管设计真正交互的超参数优化（HPO）方法的兴趣日益增长，至今为止，仅有少数方法允许纳入人类反馈。现有的交互贝叶斯优化（BO）方法通过用用户定义的先验分布加权获得函数来纳入人类信念。然而，在贝叶斯优化中普遍存在的获得函数的内部优化使得这种加权方案并不总是准确地反映出给定的用户信念。我们介绍了一种新颖的BO方法，利用有效的概率模型——概率电路（PCs）作为代理模型。PCs编码混合超参数空间和评估分数的可计算联合分布，并支持精确的条件推断和采样。基于条件采样，我们构建了一种新颖的选择策略，该策略允许在无需额外内部循环优化的情况下生成候选点，并确保用户信念准确反映在选择策略中。我们提供了理论分析并进行了广泛的实证评估，证明了我们的方法在标准HPO中达到最先进的性能，并在交互HPO中超越了交互BO基线方法。 

---
# Bruno: Backpropagation Running Undersampled for Novel device Optimization 

**Title (ZH)**: Bruno: 反向传播在新型设备优化中的欠采样运行 

**Authors**: Luca Fehlings, Bojian Zhang, Paolo Gibertini, Martin A. Nicholson, Erika Covi, Fernando M. Quintana  

**Link**: [PDF](https://arxiv.org/pdf/2505.17791)  

**Abstract**: Recent efforts to improve the efficiency of neuromorphic and machine learning systems have focused on the development of application-specific integrated circuits (ASICs), which provide hardware specialized for the deployment of neural networks, leading to potential gains in efficiency and performance. These systems typically feature an architecture that goes beyond the von Neumann architecture employed in general-purpose hardware such as GPUs. Neural networks developed for this specialised hardware then need to take into account the specifics of the hardware platform, which requires novel training algorithms and accurate models of the hardware, since they cannot be abstracted as a general-purpose computing platform. In this work, we present a bottom-up approach to train neural networks for hardware based on spiking neurons and synapses built on ferroelectric capacitor (FeCap) and Resistive switching non-volatile devices (RRAM) respectively. In contrast to the more common approach of designing hardware to fit existing abstract neuron or synapse models, this approach starts with compact models of the physical device to model the computational primitive of the neurons. Based on these models, a training algorithm is developed that can reliably backpropagate through these physical models, even when applying common hardware limitations, such as stochasticity, variability, and low bit precision. The training algorithm is then tested on a spatio-temporal dataset with a network composed of quantized synapses based on RRAM and ferroelectric leaky integrate-and-fire (FeLIF) neurons. The performance of the network is compared with different networks composed of LIF neurons. The results of the experiments show the potential advantage of using BRUNO to train networks with FeLIF neurons, by achieving a reduction in both time and memory for detecting spatio-temporal patterns with quantized synapses. 

**Abstract (ZH)**: 基于铁electric电容器(FeCap)和电阻切换非易失性设备(RRAM)构建的突触和.spike神经元的硬件上神经网络训练方法 

---
# MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization 

**Title (ZH)**: MetaBox-v2：元黑盒优化的统一基准平台 

**Authors**: Zeyuan Ma, Yue-Jiao Gong, Hongshu Guo, Wenjie Qiu, Sijie Ma, Hongqiao Lian, Jiajun Zhan, Kaixu Chen, Chen Wang, Zhiyang Huang, Zechuan Huang, Guojun Peng, Ran Cheng, Yining Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.17745)  

**Abstract**: Meta-Black-Box Optimization (MetaBBO) streamlines the automation of optimization algorithm design through meta-learning. It typically employs a bi-level structure: the meta-level policy undergoes meta-training to reduce the manual effort required in developing algorithms for low-level optimization tasks. The original MetaBox (2023) provided the first open-source framework for reinforcement learning-based single-objective MetaBBO. However, its relatively narrow scope no longer keep pace with the swift advancement in this field. In this paper, we introduce MetaBox-v2 (this https URL) as a milestone upgrade with four novel features: 1) a unified architecture supporting RL, evolutionary, and gradient-based approaches, by which we reproduce 23 up-to-date baselines; 2) efficient parallelization schemes, which reduce the training/testing time by 10-40x; 3) a comprehensive benchmark suite of 18 synthetic/realistic tasks (1900+ instances) spanning single-objective, multi-objective, multi-model, and multi-task optimization scenarios; 4) plentiful and extensible interfaces for custom analysis/visualization and integrating to external optimization tools/benchmarks. To show the utility of MetaBox-v2, we carry out a systematic case study that evaluates the built-in baselines in terms of the optimization performance, generalization ability and learning efficiency. Valuable insights are concluded from thorough and detailed analysis for practitioners and those new to the field. 

**Abstract (ZH)**: Meta-黑箱优化（MetaBBO）通过元学习简化了优化算法设计的自动化进程。它通常采用双层结构：元层面的策略在元训练过程中减少开发低层优化任务算法所需的手动努力。原始的MetaBox（2023）提供了首个基于强化学习的单目标元黑箱优化开源框架。然而，其相对狭窄的研究范围已跟不上该领域迅猛的发展步伐。本文介绍MetaBox-v2（https://）作为一项重要升级，具备四点新功能：1）支持基于强化学习、进化算法和梯度方法的一体化架构，重现了23个最新的基线模型；2）高效的并行化方案，将训练/测试时间缩短10-40倍；3）涵盖单目标、多目标、多模型和多任务优化场景的综合基准套件，包含18个合成/现实任务（超过1900个实例）；4）丰富的可扩展接口，方便自定义分析/可视化以及整合外部优化工具/基准。为了展示MetaBox-v2的优势，我们进行了系统性的案例研究，评估内置基线模型在优化性能、泛化能力和学习效率方面的表现。通过详尽细致的分析，本文为从业者及该领域的初学者提供了有价值的见解。 

---
# A Distributionally-Robust Framework for Nuisance in Causal Effect Estimation 

**Title (ZH)**: 分布鲁棒的中介变量在因果效应估计中的框架 

**Authors**: Akira Tanimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.17717)  

**Abstract**: Causal inference requires evaluating models on balanced distributions between treatment and control groups, while training data often exhibits imbalance due to historical decision-making policies. Most conventional statistical methods address this distribution shift through inverse probability weighting (IPW), which requires estimating propensity scores as an intermediate step. These methods face two key challenges: inaccurate propensity estimation and instability from extreme weights. We decompose the generalization error to isolate these issues--propensity ambiguity and statistical instability--and address them through an adversarial loss function. Our approach combines distributionally robust optimization for handling propensity uncertainty with weight regularization based on weighted Rademacher complexity. Experiments on synthetic and real-world datasets demonstrate consistent improvements over existing methods. 

**Abstract (ZH)**: 因果推断要求在治疗组和控制组之间评估模型的平衡分布，而训练数据由于历史上决策制定政策的原因通常表现出不平衡。大多数传统统计方法通过逆概率加权（IPW）来解决这种分布偏移问题，这需要在中间步骤估计倾向得分。这些方法面临两个关键挑战：倾向得分估计不准确和极端权重导致的不稳定性。我们通过分解泛化误差来分离这些问题——倾向得分不确定性与统计不稳定性——并通过对抗损失函数来解决这些问题。我们的方法结合了处理倾向得分不确定性的大范围优化技术，并基于加权仁科复杂性进行权重正则化。实验结果在合成和真实世界数据集上展示了相对于现有方法的一致改进。 

---
# PPO-BR: Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization 

**Title (ZH)**: PPO-BR: 双信号熵-奖励适应的可信区域策略优化 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17714)  

**Abstract**: Despite Proximal Policy Optimization (PPO) dominating policy gradient methods -- from robotic control to game AI -- its static trust region forces a brittle trade-off: aggressive clipping stifles early exploration, while late-stage updates destabilize convergence. PPO-BR establishes a new paradigm in adaptive RL by fusing exploration and convergence signals into a single bounded trust region -- a theoretically grounded innovation that outperforms five SOTA baselines with less than 2% overhead. This work bridges a critical gap in phase-aware learning, enabling real-world deployment in safety-critical systems like robotic surgery within a single adaptive mechanism. PPO-BR achieves 29.1% faster convergence by combining: (1) entropy-driven expansion (epsilon up) for exploration in high-uncertainty states, and (2) reward-guided contraction (epsilon down) for convergence stability. On six diverse benchmarks (MuJoCo, Atari, sparse-reward), PPO-BR achieves 29.1% faster convergence (p < 0.001), 2.3x lower reward variance than PPO, and less than 1.8% runtime overhead with only five lines of code change. PPO-BR's simplicity and theoretical guarantees make it ready-to-deploy in safety-critical domains -- from surgical robotics to autonomous drones. In contrast to recent methods such as Group Relative Policy Optimization (GRPO), PPO-BR offers a unified entropy-reward mechanism applicable to both language models and general reinforcement learning environments. 

**Abstract (ZH)**: 尽管最近策略优化（PPO）在机器人控制和游戏AI等领域主导了策略梯度方法——其静态信任区域强制了一种脆弱的权衡：激进的剪裁抑制了早期探索，而后期更新则破坏了收敛。PPO-BR通过将探索信号和收敛信号融合到一个单一的受限制的信任区域中，建立了自适应强化学习的新范式——这一理论上的创新比五个SOTA基线方法更具优势，同时仅增加不到2%的开销。这项工作填补了阶段感知学习中的关键空白，使安全关键系统（如机器人手术）能够在单一自适应机制内实现实际部署。PPO-BR通过结合以下两种方法实现了29.1%更快的收敛：（1）不确定性状态下基于熵的扩展（ε上升）进行探索，（2）基于奖励的收缩（ε下降）以提高收敛稳定性。在六个不同的基准测试（MuJoCo、Atari、稀疏奖励）中，PPO-BR的收敛速度快了29.1%（p < 0.001），奖励方差比PPO低2.3倍，改动仅五行代码，运行时开销不到1.8%。PPO-BR的简单性和理论保证使其在安全关键领域（从外科手术机器人到自主无人机）中随时可以部署。与最近的方法如组相对策略优化（GRPO）不同，PPO-BR提供了一种统一的熵-奖励机制，适用于语言模型和一般强化学习环境。 

---
# SynRES: Towards Referring Expression Segmentation in the Wild via Synthetic Data 

**Title (ZH)**: SynRES：通过合成数据朝无约束的引用表达分割迈进 

**Authors**: Dong-Hee Kim, Hyunjee Song, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17695)  

**Abstract**: Despite the advances in Referring Expression Segmentation (RES) benchmarks, their evaluation protocols remain constrained, primarily focusing on either single targets with short queries (containing minimal attributes) or multiple targets from distinctly different queries on a single domain. This limitation significantly hinders the assessment of more complex reasoning capabilities in RES models. We introduce WildRES, a novel benchmark that incorporates long queries with diverse attributes and non-distinctive queries for multiple targets. This benchmark spans diverse application domains, including autonomous driving environments and robotic manipulation scenarios, thus enabling more rigorous evaluation of complex reasoning capabilities in real-world settings. Our analysis reveals that current RES models demonstrate substantial performance deterioration when evaluated on WildRES. To address this challenge, we introduce SynRES, an automated pipeline generating densely paired compositional synthetic training data through three innovations: (1) a dense caption-driven synthesis for attribute-rich image-mask-expression triplets, (2) reliable semantic alignment mechanisms rectifying caption-pseudo mask inconsistencies via Image-Text Aligned Grouping, and (3) domain-aware augmentations incorporating mosaic composition and superclass replacement to emphasize generalization ability and distinguishing attributes over object categories. Experimental results demonstrate that models trained with SynRES achieve state-of-the-art performance, improving gIoU by 2.0% on WildRES-ID and 3.8% on WildRES-DS. Code and datasets are available at this https URL. 

**Abstract (ZH)**: WildRES：一种包含长查询和非独特查询的新型基准 

---
# TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments 

**Title (ZH)**: TransBench: 突破动态数字环境中可转移图形用户界面代理的障碍 

**Authors**: Yuheng Lu, Qian Yu, Hongru Wang, Zeming Liu, Wei Su, Yanping Liu, Yuhang Guo, Maocheng Liang, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17629)  

**Abstract**: Graphical User Interface (GUI) agents, which autonomously operate on digital interfaces through natural language instructions, hold transformative potential for accessibility, automation, and user experience. A critical aspect of their functionality is grounding - the ability to map linguistic intents to visual and structural interface elements. However, existing GUI agents often struggle to adapt to the dynamic and interconnected nature of real-world digital environments, where tasks frequently span multiple platforms and applications while also being impacted by version updates. To address this, we introduce TransBench, the first benchmark designed to systematically evaluate and enhance the transferability of GUI agents across three key dimensions: cross-version transferability (adapting to version updates), cross-platform transferability (generalizing across platforms like iOS, Android, and Web), and cross-application transferability (handling tasks spanning functionally distinct apps). TransBench includes 15 app categories with diverse functionalities, capturing essential pages across versions and platforms to enable robust evaluation. Our experiments demonstrate significant improvements in grounding accuracy, showcasing the practical utility of GUI agents in dynamic, real-world environments. Our code and data will be publicly available at Github. 

**Abstract (ZH)**: 图形用户界面（GUI）代理，可以通过自然语言指令自主操作数字界面，具有改变无障碍、自动化和用户体验的潜力。它们功能的一个关键方面是语义接地——即将语言意图映射到视觉和结构化的界面元件。然而，现有的GUI代理往往难以适应现实世界数字环境的动态和相互关联的性质，在这种环境中，任务经常跨越多个平台和应用程序，同时也受到版本更新的影响。为了解决这一问题，我们引入了TransBench，这是第一个旨在系统地评估和提升GUI代理跨三个关键维度的转移性能的基准：跨版本转移性能（适应版本更新）、跨平台转移性能（在iOS、Android和Web等平台之间泛化）以及跨应用转移性能（处理跨越功能不同应用程序的任务）。TransBench 包含15个具有不同功能的应用类别，涵盖了不同版本和平台的关键页面，以实现稳健的评估。我们的实验证明了在接地准确性方面的显著改进，展示了在动态现实世界环境中的GUI代理的实际用途。我们的代码和数据将在GitHub上公开。 

---
# \texttt{Range-Arithmetic}: Verifiable Deep Learning Inference on an Untrusted Party 

**Title (ZH)**: \texttt{范围算术}: 在不可信方进行可验证深度学习推断 

**Authors**: Ali Rahimi, Babak H. Khalaj, Mohammad Ali Maddah-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.17623)  

**Abstract**: Verifiable computing (VC) has gained prominence in decentralized machine learning systems, where resource-intensive tasks like deep neural network (DNN) inference are offloaded to external participants due to blockchain limitations. This creates a need to verify the correctness of outsourced computations without re-execution. We propose \texttt{Range-Arithmetic}, a novel framework for efficient and verifiable DNN inference that transforms non-arithmetic operations, such as rounding after fixed-point matrix multiplication and ReLU, into arithmetic steps verifiable using sum-check protocols and concatenated range proofs. Our approach avoids the complexity of Boolean encoding, high-degree polynomials, and large lookup tables while remaining compatible with finite-field-based proof systems. Experimental results show that our method not only matches the performance of existing approaches, but also reduces the computational cost of verifying the results, the computational effort required from the untrusted party performing the DNN inference, and the communication overhead between the two sides. 

**Abstract (ZH)**: 可验证计算（VC）在去中心化机器学习系统中日益受到重视，由于区块链的限制，资源密集型任务如深度神经网络（DNN）推理被外包给外部参与者，这产生了在不重新执行的情况下验证外包计算正确性的需求。我们提出了一种名为\texttt{Range-Arithmetic}的新框架，用于高效且可验证的DNN推理，将定点矩阵乘法后的舍入操作和ReLU等非算术操作转换为可使用和项检验协议及连接范围证明进行验证的算术步骤。我们的方法避免了布尔编码、高次多项式和大量查找表的复杂性，同时仍然与基于有限域的证明系统兼容。实验结果表明，我们的方法不仅达到了现有方法的性能水平，还降低了验证结果、不信任方进行DNN推理所需的计算成本以及双方之间的通信开销。 

---
# Model Already Knows the Best Noise: Bayesian Active Noise Selection via Attention in Video Diffusion Model 

**Title (ZH)**: 模型已经知道最好的噪声：视频扩散模型中的注意力驱动贝叶斯主动噪声选择 

**Authors**: Kwanyoung Kim, Sanghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17561)  

**Abstract**: The choice of initial noise significantly affects the quality and prompt alignment of video diffusion models, where different noise seeds for the same prompt can lead to drastically different generations. While recent methods rely on externally designed priors such as frequency filters or inter-frame smoothing, they often overlook internal model signals that indicate which noise seeds are inherently preferable. To address this, we propose ANSE (Active Noise Selection for Generation), a model-aware framework that selects high-quality noise seeds by quantifying attention-based uncertainty. At its core is BANSA (Bayesian Active Noise Selection via Attention), an acquisition function that measures entropy disagreement across multiple stochastic attention samples to estimate model confidence and consistency. For efficient inference-time deployment, we introduce a Bernoulli-masked approximation of BANSA that enables score estimation using a single diffusion step and a subset of attention layers. Experiments on CogVideoX-2B and 5B demonstrate that ANSE improves video quality and temporal coherence with only an 8% and 13% increase in inference time, respectively, providing a principled and generalizable approach to noise selection in video diffusion. See our project page: this https URL 

**Abstract (ZH)**: 初始噪声的选择显著影响视频扩散模型的质量和提示对齐，不同的噪声种子对于相同的提示可能导致截然不同的生成结果。虽然最近的方法依赖于外部设计的先验知识，如频域滤波或帧间平滑，但它们往往会忽略模型内部信号，这些信号能够指示哪些噪声种子更为优选。为了解决这一问题，我们提出了ANSE（Active Noise Selection for Generation），一种基于模型的框架，通过量化基于注意力的不确定性来选择高质量的噪声种子。其核心是BANSA（Bayesian Active Noise Selection via Attention），一种通过测量多组随机注意力抽样的熵不一致性来估计模型信心和一致性的获取函数。为了高效地在推理阶段部署，我们引入了一种针对BANSA的伯努利掩码近似，该方法仅需一个扩散步骤和部分注意力层即可估算分数。在CogVideoX-2B和5B上的实验结果显示，ANSE在仅增加8%和13%的推理时间的情况下提升了视频质量和时间连贯性，提供了一种原则性和可泛化的视频扩散中噪声选择方法。见我们的项目页面：this https URL 

---
# Universal Biological Sequence Reranking for Improved De Novo Peptide Sequencing 

**Title (ZH)**: 通用生物序列重新排序以提高从头肽测序性能 

**Authors**: Zijie Qiu, Jiaqi Wei, Xiang Zhang, Sheng Xu, Kai Zou, Zhi Jin, Zhiqiang Gao, Nanqing Dong, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17552)  

**Abstract**: De novo peptide sequencing is a critical task in proteomics. However, the performance of current deep learning-based methods is limited by the inherent complexity of mass spectrometry data and the heterogeneous distribution of noise signals, leading to data-specific biases. We present RankNovo, the first deep reranking framework that enhances de novo peptide sequencing by leveraging the complementary strengths of multiple sequencing models. RankNovo employs a list-wise reranking approach, modeling candidate peptides as multiple sequence alignments and utilizing axial attention to extract informative features across candidates. Additionally, we introduce two new metrics, PMD (Peptide Mass Deviation) and RMD (residual Mass Deviation), which offer delicate supervision by quantifying mass differences between peptides at both the sequence and residue levels. Extensive experiments demonstrate that RankNovo not only surpasses its base models used to generate training candidates for reranking pre-training, but also sets a new state-of-the-art benchmark. Moreover, RankNovo exhibits strong zero-shot generalization to unseen models whose generations were not exposed during training, highlighting its robustness and potential as a universal reranking framework for peptide sequencing. Our work presents a novel reranking strategy that fundamentally challenges existing single-model paradigms and advances the frontier of accurate de novo sequencing. Our source code is provided on GitHub. 

**Abstract (ZH)**: 新型肽序列排序框架RankNovo：多种排序模型互补增强的深度重排序方法 

---
# Learning Representational Disparities 

**Title (ZH)**: 学习表示差异 

**Authors**: Pavan Ravishankar, Rushabh Shah, Daniel B. Neill  

**Link**: [PDF](https://arxiv.org/pdf/2505.17533)  

**Abstract**: We propose a fair machine learning algorithm to model interpretable differences between observed and desired human decision-making, with the latter aimed at reducing disparity in a downstream outcome impacted by the human decision. Prior work learns fair representations without considering the outcome in the decision-making process. We model the outcome disparities as arising due to the different representations of the input seen by the observed and desired decision-maker, which we term representational disparities. Our goal is to learn interpretable representational disparities which could potentially be corrected by specific nudges to the human decision, mitigating disparities in the downstream outcome; we frame this as a multi-objective optimization problem using a neural network. Under reasonable simplifying assumptions, we prove that our neural network model of the representational disparity learns interpretable weights that fully mitigate the outcome disparity. We validate objectives and interpret results using real-world German Credit, Adult, and Heritage Health datasets. 

**Abstract (ZH)**: 我们提出一种公平的机器学习算法来建模观察到的人类决策与期望的人类决策之间的可解释差异，后者旨在减少受人类决策影响的下游结果中的不公平性。以往工作在决策过程中未考虑结果来学习公平表示。我们将结果差异建模为由于观察到的决策者和期望的决策者看到的输入表示不同所致，我们称之为表示差异。我们的目标是学习可解释的表示差异，这些差异可以通过特定的人类决策调整来潜在地纠正，从而减轻下游结果中的不公；我们将此问题表述为一个多目标优化问题，并使用神经网络来建模表示差异。在合理的简化假设下，我们证明我们的表示差异的神经网络模型学习到了完全缓解结果差异的可解释权重。我们使用真实的德国信用、成人和遗产健康数据集来验证目标并解释结果。 

---
# Multi-agent Systems for Misinformation Lifecycle : Detection, Correction And Source Identification 

**Title (ZH)**: 多 agent 系统在错误信息生命周期中的检测、纠正与源头识别 

**Authors**: Aditya Gautam  

**Link**: [PDF](https://arxiv.org/pdf/2505.17511)  

**Abstract**: The rapid proliferation of misinformation in digital media demands solutions that go beyond isolated Large Language Model(LLM) or AI Agent based detection methods. This paper introduces a novel multi-agent framework that covers the complete misinformation lifecycle: classification, detection, correction, and source verification to deliver more transparent and reliable outcomes. In contrast to single-agent or monolithic architectures, our approach employs five specialized agents: an Indexer agent for dynamically maintaining trusted repositories, a Classifier agent for labeling misinformation types, an Extractor agent for evidence based retrieval and ranking, a Corrector agent for generating fact-based correction and a Verification agent for validating outputs and tracking source credibility. Each agent can be individually evaluated and optimized, ensuring scalability and adaptability as new types of misinformation and data sources emerge. By decomposing the misinformation lifecycle into specialized agents - our framework enhances scalability, modularity, and explainability. This paper proposes a high-level system overview, agent design with emphasis on transparency, evidence-based outputs, and source provenance to support robust misinformation detection and correction at scale. 

**Abstract (ZH)**: 数字媒体中虚假信息的迅速蔓延需要超越孤立的大语言模型或AI代理检测方法的解决方案。本文介绍了一种新的多代理框架，涵盖了虚假信息的完整生命周期：分类、检测、纠正和来源验证，以提供更加透明和可靠的成果。与单代理或单一架构不同，我们的方法采用了五个专门的代理：索引代理用于动态维护可信的仓储，分类代理用于标注虚假信息类型，提取代理用于基于证据的检索和排序，纠正代理用于生成基于事实的纠正内容，验证代理用于验证输出并跟踪来源可信度。每个代理都可以单独评估和优化，确保在新类型虚假信息和数据源出现时具备可扩展性和适应性。通过将虚假信息生命周期分解为专门的代理，我们的框架增强了可扩展性、模块化和可解释性。本文提出了一种高层次的系统概述，并强调透明性、基于证据的输出和来源追溯，以支持大规模的虚假信息检测和纠正。 

---
# Managing FAIR Knowledge Graphs as Polyglot Data End Points: A Benchmark based on the rdf2pg Framework and Plant Biology Data 

**Title (ZH)**: 管理FAIR知识图谱作为多语言数据端点：基于rdf2pg框架与植物生物学数据的基准测试 

**Authors**: Marco Brandizi, Carlos Bobed, Luca Garulli, Arné de Klerk, Keywan Hassani-Pak  

**Link**: [PDF](https://arxiv.org/pdf/2505.17498)  

**Abstract**: Linked Data and labelled property graphs (LPG) are two data management approaches with complementary strengths and weaknesses, making their integration beneficial for sharing datasets and supporting software ecosystems. In this paper, we introduce rdf2pg, an extensible framework for mapping RDF data to semantically equivalent LPG formats and data-bases. Utilising this framework, we perform a comparative analysis of three popular graph databases - Virtuoso, Neo4j, and ArcadeDB - and the well-known graph query languages SPARQL, Cypher, and Gremlin. Our qualitative and quantitative as-sessments underline the strengths and limitations of these graph database technologies. Additionally, we highlight the potential of rdf2pg as a versatile tool for enabling polyglot access to knowledge graphs, aligning with established standards of Linked Data and the Semantic Web. 

**Abstract (ZH)**: Linked Data和标记属性图(LPG)是两种具有互补优势和劣势的数据管理方法，其集成对于共享数据集和支持软件生态系统有益。本文介绍了rdf2pg，一个可扩展的框架，用于将RDF数据映射到语义等价的LPG格式和数据库中。利用该框架，我们对三款流行的图数据库——Virtuoso、Neo4j和ArcadeDB，以及知名的图查询语言SPARQL、Cypher和Gremlin进行了比较分析。我们的定性和定量评估突显了这些图数据库技术的优势和局限性。此外，我们强调了rdf2pg作为多功能工具的潜力，以实现对知识图的多语言访问，符合链接数据和语义网的既定标准。 

---
# HiLAB: A Hybrid Inverse-Design Framework 

**Title (ZH)**: HiLAB：一种混合逆设计框架 

**Authors**: Reza Marzban, Hamed Abiri, Raphael Pestourie, Ali Adibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17491)  

**Abstract**: HiLAB (Hybrid inverse-design with Latent-space learning, Adjoint-based partial optimizations, and Bayesian optimization) is a new paradigm for inverse design of nanophotonic structures. Combining early-terminated topological optimization (TO) with a Vision Transformer-based variational autoencoder (VAE) and a Bayesian search, HiLAB addresses multi-functional device design by generating diverse freeform configurations at reduced simulation costs. Shortened adjoint-driven TO runs, coupled with randomized physical parameters, produce robust initial structures. These structures are compressed into a compact latent space by the VAE, enabling Bayesian optimization to co-optimize geometry and physical hyperparameters. Crucially, the trained VAE can be reused for alternative objectives or constraints by adjusting only the acquisition function. Compared to conventional TO pipelines prone to local optima, HiLAB systematically explores near-global optima with considerably fewer electromagnetic simulations. Even after accounting for training overhead, the total number of full simulations decreases by over an order of magnitude, accelerating the discovery of fabrication-friendly devices. Demonstrating its efficacy, HiLAB is used to design an achromatic beam deflector for red, green, and blue wavelengths, achieving balanced diffraction efficiencies of ~25% while mitigating chromatic aberrations-a performance surpassing existing demonstrations. Overall, HiLAB provides a flexible platform for robust, multi-parameter photonic designs and rapid adaptation to next-generation nanophotonic challenges. 

**Abstract (ZH)**: HiLAB（混合逆设计与潜在空间学习、基于雅克比的方法部分优化和贝叶斯优化）是一种纳米光子结构逆设计的新范式。 

---
# Anatomy-Guided Multitask Learning for MRI-Based Classification of Placenta Accreta Spectrum and its Subtypes 

**Title (ZH)**: 基于解剖引导多任务学习的MRI分类及其亚型分类方法的研究：植入性胎盘谱系及其亚型 

**Authors**: Hai Jiang, Qiongting Liu, Yuanpin Zhou, Jiawei Pan, Ting Song, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17484)  

**Abstract**: Placenta Accreta Spectrum Disorders (PAS) pose significant risks during pregnancy, frequently leading to postpartum hemorrhage during cesarean deliveries and other severe clinical complications, with bleeding severity correlating to the degree of placental invasion. Consequently, accurate prenatal diagnosis of PAS and its subtypes-placenta accreta (PA), placenta increta (PI), and placenta percreta (PP)-is crucial. However, existing guidelines and methodologies predominantly focus on the presence of PAS, with limited research addressing subtype recognition. Additionally, previous multi-class diagnostic efforts have primarily relied on inefficient two-stage cascaded binary classification tasks. In this study, we propose a novel convolutional neural network (CNN) architecture designed for efficient one-stage multiclass diagnosis of PAS and its subtypes, based on 4,140 magnetic resonance imaging (MRI) slices. Our model features two branches: the main classification branch utilizes a residual block architecture comprising multiple residual blocks, while the second branch integrates anatomical features of the uteroplacental area and the adjacent uterine serous layer to enhance the model's attention during classification. Furthermore, we implement a multitask learning strategy to leverage both branches effectively. Experiments conducted on a real clinical dataset demonstrate that our model achieves state-of-the-art performance. 

**Abstract (ZH)**: 胎盘植入谱系障碍（PAS）在妊娠期间造成显著风险， frequently 导致剖宫产时出现产后出血及其他严重临床并发症，出血严重程度与胎盘侵袭程度相关。因此，PAS及其亚型胎盘植入（PA）、胎盘増生（PI）和胎盘穿透（PP）的准确产前诊断至关重要。然而，现有指南和方法主要关注PAS的存在，对亚型识别的研究较少。此外，以前的多分类诊断努力主要依赖于不高效的两阶段级联二元分类任务。在本研究中，我们提出了一种新型卷积神经网络（CNN）架构，用于基于4,140张磁共振成像（MRI）切片对PAS及其亚型进行高效的一阶段多分类诊断。该模型包含两个分支：主分类分支采用包含多个残差块的残差块架构，而第二个分支整合胎盘植入区域及相邻子宫浆膜层的解剖特征，以增强模型分类过程中的注意力。此外，我们采用了多任务学习策略以有效利用两个分支。在实际临床数据集上的实验表明，我们的模型达到了最先进的性能。 

---
# Alpay Algebra II: Identity as Fixed-Point Emergence in Categorical Data 

**Title (ZH)**: Alpay代数II：身份作为范畴数据中的稳定点 emergent 现象 

**Authors**: Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2505.17480)  

**Abstract**: In this second installment of the Alpay Algebra framework, I formally define identity as a fixed point that emerges through categorical recursion. Building upon the transfinite operator $\varphi^\infty$, I characterize identity as the universal solution to a self-referential functorial equation over a small cartesian closed category. I prove the existence and uniqueness of such identity-fixed-points via ordinal-indexed iteration, and interpret their convergence through internal categorical limits. Functors, adjunctions, and morphisms are reconstructed as dynamic traces of evolving states governed by $\varphi$, reframing identity not as a static label but as a stabilized process. Through formal theorems and symbolic flows, I show how these fixed points encode symbolic memory, recursive coherence, and semantic invariance. This paper positions identity as a mathematical structure that arises from within the logic of change itself computable, convergent, and categorically intrinsic. 

**Abstract (ZH)**: 在Alpay代数框架的第二部分中，我正式定义身份作为一种通过范畴递归产生的不变点。基于超越算子$\varphi^\infty$，我将身份刻画为在小范畴闭域上自参照函子方程的普遍解。通过序数索引迭代证明了这种不变点的存在性和唯一性，并通过内部范畴极限解释其收敛性。通过重新构建函子、伴随和同态，我将身份视为由$\varphi$支配的演化状态的动态痕迹，重新定义身份不是静态标签而是稳定过程。通过形式定理和符号流，我展示了这些不变点如何编码符号记忆、递归一致性和语义不变性。本文将身份定位为一种由变化逻辑本身产生、计算、收敛且范畴内在的数学结构。 

---
# Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression 

**Title (ZH)**: 通过自回归同时 modeling 蛋白质构象和动态 

**Authors**: Yuning Shen, Lihao Wang, Huizhuo Yuan, Yan Wang, Bangji Yang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17478)  

**Abstract**: Understanding protein dynamics is critical for elucidating their biological functions. The increasing availability of molecular dynamics (MD) data enables the training of deep generative models to efficiently explore the conformational space of proteins. However, existing approaches either fail to explicitly capture the temporal dependencies between conformations or do not support direct generation of time-independent samples. To address these limitations, we introduce ConfRover, an autoregressive model that simultaneously learns protein conformation and dynamics from MD trajectories, supporting both time-dependent and time-independent sampling. At the core of our model is a modular architecture comprising: (i) an encoding layer, adapted from protein folding models, that embeds protein-specific information and conformation at each time frame into a latent space; (ii) a temporal module, a sequence model that captures conformational dynamics across frames; and (iii) an SE(3) diffusion model as the structure decoder, generating conformations in continuous space. Experiments on ATLAS, a large-scale protein MD dataset of diverse structures, demonstrate the effectiveness of our model in learning conformational dynamics and supporting a wide range of downstream tasks. ConfRover is the first model to sample both protein conformations and trajectories within a single framework, offering a novel and flexible approach for learning from protein MD data. 

**Abstract (ZH)**: 理解蛋白质动力学对于阐明其生物学功能至关重要。不断增加的分子动力学（MD）数据使得训练深度生成模型以高效探索蛋白质构象空间成为可能。然而，现有的方法要么不能明确捕捉构象之间的时序依赖性，要么不支持直接生成时间独立样本。为解决这些问题，我们引入了ConfRover，这是一种自回归模型，能够同时从MD轨迹中学习蛋白质构象和动力学，支持时间和时间独立采样。我们的模型核心由以下模块构成：（i）一个编码层，源自蛋白质折叠模型，将每时刻的蛋白质特定信息和构象嵌入到潜在空间中；（ii）一个时序模块，一种序列模型，捕捉帧间构象动力学；以及（iii）一个SE(3)扩散模型作为结构解码器，在连续空间中生成构象。在ATLAS大规模蛋白质MD数据集上进行的实验表明，我们的模型在学习构象动力学和支持广泛下游任务方面具有有效性。ConfRover是首个在同一框架中采样蛋白质构象和轨迹的模型，提供了学习蛋白质MD数据的一种新颖且灵活的方法。 

---
# Efficient compression of neural networks and datasets 

**Title (ZH)**: 神经网络和数据集的高效压缩 

**Authors**: Lukas Silvester Barth, Paulo von Petersenn  

**Link**: [PDF](https://arxiv.org/pdf/2505.17469)  

**Abstract**: We compare, improve, and contribute methods that substantially decrease the number of parameters of neural networks while maintaining high test accuracy. When applying our methods to minimize description length, we obtain very effective data compression algorithms. In particular, we develop a probabilistic reformulation of $\ell_0$ regularized optimization for nonlinear models that does not require Monte-Carlo sampling and thus improves upon previous methods. We also improve upon methods involving smooth approximations to the $\ell_0$ norm, and investigate layerwise methods. We compare the methods on different architectures and datasets, including convolutional networks trained on image datasets and transformers trained on parts of Wikipedia. We also created a synthetic teacher-student setup to investigate compression in a controlled continuous setting. Finally, we conceptually relate compression algorithms to Solomonoff's theory of inductive inference and empirically verify the prediction that regularized models can exhibit more sample-efficient convergence. 

**Abstract (ZH)**: 我们比较、改进并贡献了一类显著减少神经网络参数数量同时保持高测试准确率的方法。在应用这些方法以最小化描述长度时，我们得到了非常有效的数据压缩算法。特别是，我们开发了一种无需蒙特卡洛采样的$\ell_0$正则化优化的概率重构方法，从而改善了先前的方法。我们还改善了涉及$\ell_0$范数光滑近似的方法，并研究了逐层方法。我们在不同的架构和数据集上比较了这些方法，包括在图像数据集上训练的卷积网络和在维基百科部分数据上训练的变压器。我们还创建了一个合成的教-学设置，以在受控连续环境中研究压缩。最后，我们从概念上将压缩算法与索洛门off归纳推断理论相关联，并通过实验验证了正则化模型可以表现出更高样本效率收敛的预测。 

---
# CLIMB: Class-imbalanced Learning Benchmark on Tabular Data 

**Title (ZH)**: CLIMB: 表格数据上的类别不平衡学习基准 

**Authors**: Zhining Liu, Zihao Li, Ze Yang, Tianxin Wei, Jian Kang, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17451)  

**Abstract**: Class-imbalanced learning (CIL) on tabular data is important in many real-world applications where the minority class holds the critical but rare outcomes. In this paper, we present CLIMB, a comprehensive benchmark for class-imbalanced learning on tabular data. CLIMB includes 73 real-world datasets across diverse domains and imbalance levels, along with unified implementations of 29 representative CIL algorithms. Built on a high-quality open-source Python package with unified API designs, detailed documentation, and rigorous code quality controls, CLIMB supports easy implementation and comparison between different CIL algorithms. Through extensive experiments, we provide practical insights on method accuracy and efficiency, highlighting the limitations of naive rebalancing, the effectiveness of ensembles, and the importance of data quality. Our code, documentation, and examples are available at this https URL. 

**Abstract (ZH)**: 表格数据中的类别不平衡学习（CIL）在许多实际应用中非常重要，其中少数类包含了关键但稀有的结果。本文介绍了CLIMB，一个全面的表格数据类别不平衡学习基准。CLIMB包含来自不同领域和不同不平衡程度的73个真实数据集，以及29个代表性CIL算法的一致实现。基于高质量的开源Python包，具有统一的API设计、详细的文档和严格的代码质量控制，CLIMB支持不同CIL算法的简便实现与比较。通过广泛的实验，我们提供了关于方法准确性和效率的实用见解，强调了天真重新平衡的局限性、集成的有效性以及数据质量的重要性。我们的代码、文档和示例可在以下链接获取。 

---
# Designing an efficient and equitable humanitarian supply chain dynamically via reinforcement learning 

**Title (ZH)**: 基于强化学习的高效和公平的人道主义供应链动态设计 

**Authors**: Weijia Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17439)  

**Abstract**: This study designs an efficient and equitable humanitarian supply chain dynamically by using reinforcement learning, PPO, and compared with heuristic algorithms. This study demonstrates the model of PPO always treats average satisfaction rate as the priority. 

**Abstract (ZH)**: 本研究利用强化学习PPO设计了一个高效公平的人道主义供应链，并与启发式算法进行了比较，表明PPO模型始终以平均满意度率为优先。 

---
# Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision 

**Title (ZH)**: 从全景语义监督中学习通用和灵活的轨迹模型 

**Authors**: Yuanshao Zhu, James Jianqiao Yu, Xiangyu Zhao, Xiao Han, Qidong Liu, Xuetao Wei, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17437)  

**Abstract**: The widespread adoption of mobile devices and data collection technologies has led to an exponential increase in trajectory data, presenting significant challenges in spatio-temporal data mining, particularly for efficient and accurate trajectory retrieval. However, existing methods for trajectory retrieval face notable limitations, including inefficiencies in large-scale data, lack of support for condition-based queries, and reliance on trajectory similarity measures. To address the above challenges, we propose OmniTraj, a generalized and flexible omni-semantic trajectory retrieval framework that integrates four complementary modalities or semantics -- raw trajectories, topology, road segments, and regions -- into a unified system. Unlike traditional approaches that are limited to computing and processing trajectories as a single modality, OmniTraj designs dedicated encoders for each modality, which are embedded and fused into a shared representation space. This design enables OmniTraj to support accurate and flexible queries based on any individual modality or combination thereof, overcoming the rigidity of traditional similarity-based methods. Extensive experiments on two real-world datasets demonstrate the effectiveness of OmniTraj in handling large-scale data, providing flexible, multi-modality queries, and supporting downstream tasks and applications. 

**Abstract (ZH)**: 移动设备和数据收集技术的广泛采用导致轨迹数据呈指数级增长，为时空数据挖掘带来了重大挑战，特别是在高效和准确的轨迹检索方面。现有轨迹检索方法存在显著局限性，包括大规模数据处理效率低下、不支持基于条件的查询以及依赖于轨迹相似度度量。为应对上述挑战，我们提出了一种名为OmniTraj的通用且灵活的全域语义轨迹检索框架，将原始轨迹、拓扑结构、道路段和区域四种互补模态或语义整合到一个统一系统中。与传统方法只能将轨迹作为单一模态进行计算和处理不同，OmniTraj为每种模态设计了专用编码器，并将其嵌入到共享表示空间中。这种设计使OmniTraj能够基于任何单独模态或其组合支持准确和灵活的查询，克服了传统基于相似性的方法的僵化性。在两个真实世界数据集上的广泛实验表明，OmniTraj在处理大规模数据、提供灵活的多模态查询以及支持下游任务和应用方面具有有效性。 

---
# SEvoBench : A C++ Framework For Evolutionary Single-Objective Optimization Benchmarking 

**Title (ZH)**: SEvoBench : 一个基于C++的单一目标进化优化基准测试框架 

**Authors**: Yongkang Yang, Jian Zhao, Tengfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17430)  

**Abstract**: We present SEvoBench, a modern C++ framework for evolutionary computation (EC), specifically designed to systematically benchmark evolutionary single-objective optimization algorithms. The framework features modular implementations of Particle Swarm Optimization (PSO) and Differential Evolution (DE) algorithms, organized around three core components: (1) algorithm construction with reusable modules, (2) efficient benchmark problem suites, and (3) parallel experimental analysis. Experimental evaluations demonstrate the framework's superior performance in benchmark testing and algorithm comparison. Case studies further validate its capabilities in algorithm hybridization and parameter analysis. Compared to existing frameworks, SEvoBench demonstrates three key advantages: (i) highly efficient and reusable modular implementations of PSO and DE algorithms, (ii) accelerated benchmarking through parallel execution, and (iii) enhanced computational efficiency via SIMD (Single Instruction Multiple Data) vectorization for large-scale problems. 

**Abstract (ZH)**: SEvoBench：一种面向进化的现代C++框架，用于系统性基准测试单目标优化算法 

---
# Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition 

**Title (ZH)**: 可验证高效算法：在线主要方-代理人信息获取中最佳评标规则识别 

**Authors**: Zichen Wang, Chuanhao Li, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17379)  

**Abstract**: We investigate the problem of identifying the optimal scoring rule within the principal-agent framework for online information acquisition problem. We focus on the principal's perspective, seeking to determine the desired scoring rule through interactions with the agent. To address this challenge, we propose two algorithms: OIAFC and OIAFB, tailored for fixed confidence and fixed budget settings, respectively. Our theoretical analysis demonstrates that OIAFC can extract the desired $(\epsilon, \delta)$-scoring rule with a efficient instance-dependent sample complexity or an instance-independent sample complexity. Our analysis also shows that OIAFB matches the instance-independent performance bound of OIAFC, while both algorithms share the same complexity across fixed confidence and fixed budget settings. 

**Abstract (ZH)**: 我们研究了在主要-代理人框架下在线信息获取问题中识别最优评分规则的问题。我们从主要方的角度出发，通过与代理人的交互确定所需的评分规则。为了解决这一挑战，我们提出了两种算法：OIAFC和OIAFB，分别针对固定的置信度和固定的预算设置。我们的理论分析表明，OIAFC可以在有效的实例相关或实例无关样本复杂性下提取所需的$(\epsilon, \delta)$-评分规则。我们的分析还表明，OIAFB在实例无关性能方面与OIAFC匹配，而两个算法在固定置信度和固定预算设置下的复杂性相同。 

---
# Value-Guided Search for Efficient Chain-of-Thought Reasoning 

**Title (ZH)**: 价值导向的搜索以实现高效的心智推理 

**Authors**: Kaiwen Wang, Jin Peng Zhou, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kianté Brantley, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17373)  

**Abstract**: In this paper, we propose a simple and efficient method for value model training on long-context reasoning traces. Compared to existing process reward models (PRMs), our method does not require a fine-grained notion of "step," which is difficult to define for long-context reasoning models. By collecting a dataset of 2.5 million reasoning traces, we train a 1.5B token-level value model and apply it to DeepSeek models for improved performance with test-time compute scaling. We find that block-wise value-guided search (VGS) with a final weighted majority vote achieves better test-time scaling than standard methods such as majority voting or best-of-n. With an inference budget of 64 generations, VGS with DeepSeek-R1-Distill-1.5B achieves an average accuracy of 45.7% across four competition math benchmarks (AIME 2024 & 2025, HMMT Feb 2024 & 2025), reaching parity with o3-mini-medium. Moreover, VGS significantly reduces the inference FLOPs required to achieve the same performance of majority voting. Our dataset, model and codebase are open-sourced. 

**Abstract (ZH)**: 本文提出了一种简单高效的方法，用于在长上下文推理轨迹上进行价值模型训练。与现有的过程奖励模型相比，我们的方法不需要精细的“步骤”概念，这在长上下文推理模型中难以定义。通过收集包含250万条推理轨迹的数据集，我们训练了一种1.5B词元级别价值模型，并将其应用于DeepSeek模型，以实现测试时计算量扩展下的性能提升。我们发现，块级价值引导搜索（VGS）结合加权多数投票在测试时扩展性能上优于标准方法（如简单多数投票或Best-of-n）。在64个推理生成预算下，DeepSeek-R1-Distill-1.5B结合VGS实现了四个竞赛数学基准（AIME 2024 & 2025，HMMT Feb 2024 & 2025）的平均准确率为45.7%，达到与o3-mini-medium相当的性能。此外，VGS大幅减少了达到相同性能所需的推理FLOPs。我们的数据集、模型和代码库已开源。 

---
# FRIREN: Beyond Trajectories -- A Spectral Lens on Time 

**Title (ZH)**: FRIREN：超越轨迹——时间的频谱视角 

**Authors**: Qilin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17370)  

**Abstract**: Long-term time-series forecasting (LTSF) models are often presented as general-purpose solutions that can be applied across domains, implicitly assuming that all data is pointwise predictable. Using chaotic systems such as Lorenz-63 as a case study, we argue that geometric structure - not pointwise prediction - is the right abstraction for a dynamic-agnostic foundational model. Minimizing the Wasserstein-2 distance (W2), which captures geometric changes, and providing a spectral view of dynamics are essential for long-horizon forecasting. Our model, FRIREN (Flow-inspired Representations via Interpretable Eigen-networks), implements an augmented normalizing-flow block that embeds data into a normally distributed latent representation. It then generates a W2-efficient optimal path that can be decomposed into rotation, scaling, inverse rotation, and translation. This architecture yields locally generated, geometry-preserving predictions that are independent of the underlying dynamics, and a global spectral representation that functions as a finite Koopman operator with a small modification. This enables practitioners to identify which modes grow, decay, or oscillate, both locally and system-wide. FRIREN achieves an MSE of 11.4, MAE of 1.6, and SWD of 0.96 on Lorenz-63 in a 336-in, 336-out, dt=0.01 setting, surpassing TimeMixer (MSE 27.3, MAE 2.8, SWD 2.1). The model maintains effective prediction for 274 out of 336 steps, approximately 2.5 Lyapunov times. On Rossler (96-in, 336-out), FRIREN achieves an MSE of 0.0349, MAE of 0.0953, and SWD of 0.0170, outperforming TimeMixer's MSE of 4.3988, MAE of 0.886, and SWD of 3.2065. FRIREN is also competitive on standard LTSF datasets such as ETT and Weather. By connecting modern generative flows with classical spectral analysis, FRIREN makes long-term forecasting both accurate and interpretable, setting a new benchmark for LTSF model design. 

**Abstract (ZH)**: 基于流的代表通过可解释的本征网络（FRIREN）：混沌系统中的长期时间序列forecasting 

---
# Dual Ascent Diffusion for Inverse Problems 

**Title (ZH)**: 双重上升扩散算法求解逆问题 

**Authors**: Minseo Kim, Axel Levy, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.17353)  

**Abstract**: Ill-posed inverse problems are fundamental in many domains, ranging from astrophysics to medical imaging. Emerging diffusion models provide a powerful prior for solving these problems. Existing maximum-a-posteriori (MAP) or posterior sampling approaches, however, rely on different computational approximations, leading to inaccurate or suboptimal samples. To address this issue, we introduce a new approach to solving MAP problems with diffusion model priors using a dual ascent optimization framework. Our framework achieves better image quality as measured by various metrics for image restoration problems, it is more robust to high levels of measurement noise, it is faster, and it estimates solutions that represent the observations more faithfully than the state of the art. 

**Abstract (ZH)**: 不适定逆问题在天体物理和医学成像等领域至关重要。新兴的扩散模型提供了有效的先验方法来解决这些问题。现有的最大后验概率（MAP）或后验采样方法依赖于不同的计算近似，导致不准确或次优的结果。为了解决这一问题，我们提出了一种基于对偶上升优化框架的求解具有扩散模型先验的MAP问题的新方法。该框架在图像恢复问题中能够获得更好的图像质量，对高测量噪声更为 robust，速度快，并能更准确地估计与观测相符的解，优于现有方法。 

---
# A Multi-Head Attention Soft Random Forest for Interpretable Patient No-Show Prediction 

**Title (ZH)**: 多头注意力软随机森林可解释的患者爽约预测 

**Authors**: Ninda Nurseha Amalina, Kwadwo Boateng Ofori-Amanfo, Heungjo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.17344)  

**Abstract**: Unattended scheduled appointments, defined as patient no-shows, adversely affect both healthcare providers and patients' health, disrupting the continuity of care, operational efficiency, and the efficient allocation of medical resources. Accurate predictive modelling is needed to reduce the impact of no-shows. Although machine learning methods, such as logistic regression, random forest models, and decision trees, are widely used in predicting patient no-shows, they often rely on hard decision splits and static feature importance, limiting their adaptability to specific or complex patient behaviors. To address this limitation, we propose a new hybrid Multi-Head Attention Soft Random Forest (MHASRF) model that integrates attention mechanisms into a random forest model using probabilistic soft splitting instead of hard splitting. The MHASRF model assigns attention weights differently across the trees, enabling attention on specific patient behaviors. The model exhibited 93.56% accuracy, 93.67% precision, 93.56% recall, and a 93.59% F1 score, surpassing the performance of decision tree, logistic regression, random forest, and naive Bayes models. Furthermore, MHASRF was able to identify key predictors of patient no-shows using two levels of feature importance (tree level and attention mechanism level), offering deeper insights into patient no-show predictors. The proposed model is a robust, adaptable, and interpretable method for predicting patient no-shows that will help healthcare providers in optimizing resources. 

**Abstract (ZH)**: 未陪同预约患者的预测建模：一种新的混合多头注意力软随机森林（MHASRF）方法及其应用 

---
# FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding 

**Title (ZH)**: FS-DAG: 几乎零样本领域适应图网络在富视觉文档理解中的应用 

**Authors**: Amit Agarwal, Srikant Panda, Kulbhushan Pachauri  

**Link**: [PDF](https://arxiv.org/pdf/2505.17330)  

**Abstract**: In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : this https URL 

**Abstract (ZH)**: 在本次工作中，我们提出了一种可扩展且高效的 Few Shot Domain Adapting Graph (FS-DAG) 模型架构，用于少样本设置下的丰富视觉文档理解 (VRDU)。FS-DAG 在模块化框架中利用领域特定和视觉/语言特定的骨干网络，以少量数据适应多种文档类型。该模型对实际挑战（如处理OCR错误、拼写错误和领域偏移）具有鲁棒性，这些都是实际部署中的关键问题。FS-DAG 性能强劲，参数量少于90M，使其适用于计算资源有限的信息提取 (IE) 任务等复杂现实应用场景。通过详尽的实验展示了FS-DAG在信息提取任务上的能力，相比现有最佳方法，在收敛速度和性能上均取得了显著改进。此外，本文还强调了在不牺牲性能的情况下开发更小更高效模型的持续进展。代码：[这个链接] 

---
# Control of Renewable Energy Communities using AI and Real-World Data 

**Title (ZH)**: 基于AI和实物数据的可再生能源社区控制 

**Authors**: Tiago Fonseca, Clarisse Sousa, Ricardo Venâncio, Pedro Pires, Ricardo Severino, Paulo Rodrigues, Pedro Paiva, Luis Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2505.17321)  

**Abstract**: The electrification of transportation and the increased adoption of decentralized renewable energy generation have added complexity to managing Renewable Energy Communities (RECs). Integrating Electric Vehicle (EV) charging with building energy systems like heating, ventilation, air conditioning (HVAC), photovoltaic (PV) generation, and battery storage presents significant opportunities but also practical challenges. Reinforcement learning (RL), particularly MultiAgent Deep Deterministic Policy Gradient (MADDPG) algorithms, have shown promising results in simulation, outperforming heuristic control strategies. However, translating these successes into real-world deployments faces substantial challenges, including incomplete and noisy data, integration of heterogeneous subsystems, synchronization issues, unpredictable occupant behavior, and missing critical EV state-of-charge (SoC) information. This paper introduces a framework designed explicitly to handle these complexities and bridge the simulation to-reality gap. The framework incorporates EnergAIze, a MADDPG-based multi-agent control strategy, and specifically addresses challenges related to real-world data collection, system integration, and user behavior modeling. Preliminary results collected from a real-world operational REC with four residential buildings demonstrate the practical feasibility of our approach, achieving an average 9% reduction in daily peak demand and a 5% decrease in energy costs through optimized load scheduling and EV charging behaviors. These outcomes underscore the framework's effectiveness, advancing the practical deployment of intelligent energy management solutions in RECs. 

**Abstract (ZH)**: 交通运输的电气化和分散型可再生能源发电的增加为可再生能源社区（RECs）的管理增添了复杂性。电动汽车（EV）充电与建筑能源系统，包括供暖、通风和空调（HVAC）、光伏（PV）发电和电池储能的整合，带来了显著的机会和实际挑战。基于强化学习（RL），尤其是基于多agent深度确定性策略梯度（MADDPG）算法，已经在仿真中显示出有前途的结果，超越了启发式控制策略。然而，将这些成功转化为实际部署面临重大挑战，包括数据不完整和噪声、异构子系统整合问题、同步问题、不确定的用户行为以及缺失的关键电动汽车电量状态信息（SoC）。本文介绍了一种专为处理这些复杂性而设计的框架，旨在弥合仿真到现实的差距。该框架整合了基于MADDPG的多agent控制策略EnergAIze，并具体解决了实际数据收集、系统整合以及用户行为建模的挑战。从一个包含四栋住宅建筑的实时运行REC中收集的初步结果表明，该方法在实际应用中的可行性，通过优化负荷调度和EV充电行为，实现了每日峰荷平均9%的减少和能源成本5%的降低。这些结果证实了该框架的有效性，推动了在RECs中智能能源管理解决方案的实际部署。 

---
# LaSER: How Learning Can Guide the Evolution of Equations 

**Title (ZH)**: LaSER: 学习如何引导方程的演化 

**Authors**: Nam H. Le, Josh Bongard  

**Link**: [PDF](https://arxiv.org/pdf/2505.17309)  

**Abstract**: Evolution and learning are two distinct yet complementary forms of adaptation. While evolutionary processes operate across generations via the selection of genotypes, learning occurs within the lifetime of an individual, shaping behavior through phenotypic adjustment. The Baldwin effect describes how lifetime learning can improve evolutionary search without altering inherited structures. While this has proven effective in areas like neuroevolution, where gradient-based learning is often used to fine-tune weights or behaviors produced by evolution, it remains underexplored in systems that evolve non-differentiable symbolic structures like Genetic Programming (GP). GP evolves explicit syntax trees that represent equations, offering strong interpretability but limited generalization due to the burden of discovering both useful representations and precise mappings.
Here, we show for the first time that integrating a simple form of supervised learning, applied at the semantic or behavioral level during evaluation, can effectively guide the evolution of equations in GP. To achieve this, we propose a new GP pipeline, LaSER (Latent Semantic Evolutionary Regression), where each GP individual generates a semantic representation that is passed to a supervised learner. The quality of the learned mapping is used to assign fitness, without modifying the underlying syntax tree or evolutionary process.
Across standard symbolic regression benchmarks, in terms of generalization ability, LaSER significantly outperforms traditional GP and, in several cases, matches or exceeds popular machine learning regressors, while preserving the symbolic interpretability. By separating evolution from learning, LaSER offers a practical route to integrating GP with modern ML workflows, and opens new avenues for research at the intersection of evolutionary computation and representation learning. 

**Abstract (ZH)**: 进化与学习是两种distinct yet complementary形式的适应。进化过程通过选择表型在代际间进行，而学习则在个体的生命周期内发生，通过表型调整来塑造行为。巴迪尔效应描述了如何在不改变遗传结构的情况下通过生命周期学习提高进化的搜索效率。虽然这种方法在神经进化等领域已被证明有效，尤其是在使用基于梯度的学习来精细调整由进化产生的权重或行为时，但它在演化非可微符号结构如遗传编程（GP）的系统中仍处于未被充分探索的状态。GP生成代表方程的显式语法树，提供了很强的可解释性，但由于需要发现有用的表现形式和精确映射，其泛化能力受限。

在此，我们首次展示了通过在评估过程中应用一种简单的监督学习，特别是在语义或行为层面，可以有效指导GP中方程的进化。为此，我们提出了一种新的GP管道LaSER（潜在语义进化回归），其中每个GP个体生成一个语义表示并传递给监督学习器。学习到的映射质量用于分配适应度，而不修改底层的语法树或进化过程。

在标准符号回归基准中，无论在泛化能力方面，LaSER显著优于传统GP，而且在某些情况下，它至少与流行机器学习回归器相当，同时保持了符号的可解释性。通过分离进化与学习，LaSER提供了将GP与现代机器学习工作流集成的实用途径，并为进化计算与表示学习交叉领域的研究开辟了新的途径。 

---
# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 

**Title (ZH)**: Select2Reason: 高效指令调优数据选择用于长链推理 

**Authors**: Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Xiaojun Wu, Honghao Liu, Hui Xiong, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17266)  

**Abstract**: A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost. 

**Abstract (ZH)**: 一种实用的方法是在预训练的大语言模型中激活长链推理能力，即通过强大型推理模型（如DeepSeek-R1）合成的指令数据集进行监督微调，这是一种成本-effective的替代强化学习的方法。然而，大规模的包含超过10万样本的指令集会带来显著的训练开销，而有效的自动长链推理指令选择策略仍待探索。在此工作中，我们提出Select2Reason，一种新颖且高效的任务指令筛选框架，用于长链推理。从重新思考行为如自我纠正和回溯的出现视角出发，我们探究了能够决定长链推理指令质量的常见指标。Select2Reason利用量化器估计问题难度，并通过加权方案结合推理轨迹长度启发式方法来优先选择高效用示例。在OpenR1-Math-220K上的实验结果表明，仅使用Select2Reason选择的10%数据微调LLM的性能与其全数据微调和开源基线OpenR1-Qwen-7B在三个竞赛级别和六个全面数学基准上的性能相当或更优。进一步的实验突显了其在不同数据规模下的扩展性、推理过程中的高效性及其在较少成本下的可适应性。 

---
# Generative AI and Creativity: A Systematic Literature Review and Meta-Analysis 

**Title (ZH)**: 生成式人工智能与创造力：一项系统文献综述与元分析 

**Authors**: Niklas Holzner, Sebastian Maier, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17241)  

**Abstract**: Generative artificial intelligence (GenAI) is increasingly used to support a wide range of human tasks, yet empirical evidence on its effect on creativity remains scattered. Can GenAI generate ideas that are creative? To what extent can it support humans in generating ideas that are both creative and diverse? In this study, we conduct a meta-analysis to evaluate the effect of GenAI on the performance in creative tasks. For this, we first perform a systematic literature search, based on which we identify n = 28 relevant studies (m = 8214 participants) for inclusion in our meta-analysis. We then compute standardized effect sizes based on Hedges' g. We compare different outcomes: (i) how creative GenAI is; (ii) how creative humans augmented by GenAI are; and (iii) the diversity of ideas by humans augmented by GenAI. Our results show no significant difference in creative performance between GenAI and humans (g = -0.05), while humans collaborating with GenAI significantly outperform those working without assistance (g = 0.27). However, GenAI has a significant negative effect on the diversity of ideas for such collaborations between humans and GenAI (g = -0.86). We further analyze heterogeneity across different GenAI models (e.g., GPT-3.5, GPT-4), different tasks (e.g., creative writing, ideation, divergent thinking), and different participant populations (e.g., laypeople, business, academia). Overall, our results position GenAI as an augmentative tool that can support, rather than replace, human creativity-particularly in tasks benefiting from ideation support. 

**Abstract (ZH)**: 生成式人工智能（GenAI）日益用于支持广泛的-human任务，然而其对创造力的影响证据仍较为分散。GenAI能否产生创意的想法？它在多大程度上能辅助人类产生既有创意又多样化的想法？本研究通过元分析评估GenAI对创造力任务表现的影响。为此，我们首先进行系统的文献搜索，确定了n=28项相关研究（m=8214名参与者）纳入元分析。我们基于Hedges' g计算标准化的效应大小，比较不同结果：（i）GenAI的创造性；（ii）受GenAI辅助的人类的创造性；（iii）受GenAI辅助的人类的想法多样性。结果显示，在创造力表现上，GenAI与人类之间无显著差异（g = -0.05），但与GenAI合作的人类显著优于未得到辅助的人类（g = 0.27）。然而，GenAI对人类与GenAI合作的想法多样性有显著负面影响（g = -0.86）。进一步分析不同GenAI模型（如GPT-3.5、GPT-4）、不同任务（如创意写作、创意生成、发散思维）以及不同参与者群体（如普通人群、商业界、学术界）之间的异质性。总体而言，我们的结果将GenAI定位为一种辅助工具，能够支持而非替代人类的创造力，特别是在受益于创意支持的任务中。 

---
# ExeSQL: Self-Taught Text-to-SQL Models with Execution-Driven Bootstrapping for SQL Dialects 

**Title (ZH)**: ExeSQL：基于执行驱动-bootstrapping的自教学文本到SQL模型用于SQL方言训练 

**Authors**: Jipeng Zhang, Haolin Yang, Kehao Miao, Ruiyuan Zhang, Renjie Pi, Jiahui Gao, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17231)  

**Abstract**: Recent text-to-SQL models have achieved strong performance, but their effectiveness remains largely confined to SQLite due to dataset limitations. However, real-world applications require SQL generation across multiple dialects with varying syntax and specialized features, which remains a challenge for current models. The main obstacle in building a dialect-aware model lies in acquiring high-quality dialect-specific data. Data generated purely through static prompting - without validating SQLs via execution - tends to be noisy and unreliable. Moreover, the lack of real execution environments in the training loop prevents models from grounding their predictions in executable semantics, limiting generalization despite surface-level improvements from data filtering. This work introduces ExeSQL, a text-to-SQL framework with execution-driven, agentic bootstrapping. The method consists of iterative query generation, execution-based filtering (e.g., rejection sampling), and preference-based training, enabling the model to adapt to new SQL dialects through verifiable, feedback-guided learning. Experiments show that ExeSQL bridges the dialect gap in text-to-SQL, achieving average improvements of 15.2%, 10.38%, and 4.49% over GPT-4o on PostgreSQL, MySQL, and Oracle, respectively, across multiple datasets of varying difficulty. 

**Abstract (ZH)**: Recent text-to-SQL模型取得了强大的性能，但由于数据集的限制，其有效性主要局限于SQLite。然而，实际应用要求在具有不同语法和专业化功能的多种方言下生成SQL，这仍然是现有模型的挑战。构建具有方言意识模型的主要障碍在于获取高质量的方言特定数据。仅通过静态提示生成的数据——未经执行验证——往往会变得嘈杂和不可靠。此外，训练循环中缺乏真实的执行环境限制了模型将预测锚定在可执行语义上，尽管通过数据过滤在表面上有所改进。本项工作引入了ExeSQL，这是一种通过执行驱动、有代理性的训练框架。该方法包括迭代的查询生成、基于执行的过滤（例如，拒绝采样）和基于偏好的训练，使模型通过可验证的、基于反馈的指导学习来适应新的SQL方言。实验表明，ExeSQL在不同难度级别的多个数据集上分别在PostgreSQL、MySQL和Oracle上分别实现了15.2%、10.38%和4.49%的平均改进，填补了SQL方言之间的差距。 

---
# LengthLogD: A Length-Stratified Ensemble Framework for Enhanced Peptide Lipophilicity Prediction via Multi-Scale Feature Integration 

**Title (ZH)**: LengthLogD：一种基于长度分层的集成框架，通过多尺度特征集成增强肽的疏水性预测 

**Authors**: Shuang Wu, Meijie Wang, Lun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17198)  

**Abstract**: Peptide compounds demonstrate considerable potential as therapeutic agents due to their high target affinity and low toxicity, yet their drug development is constrained by their low membrane permeability. Molecular weight and peptide length have significant effects on the logD of peptides, which in turn influences their ability to cross biological membranes. However, accurate prediction of peptide logD remains challenging due to the complex interplay between sequence, structure, and ionization states. This study introduces LengthLogD, a predictive framework that establishes specialized models through molecular length stratification while innovatively integrating multi-scale molecular representations. We constructed feature spaces across three hierarchical levels: atomic (10 molecular descriptors), structural (1024-bit Morgan fingerprints), and topological (3 graph-based features including Wiener index), optimized through stratified ensemble learning. An adaptive weight allocation mechanism specifically developed for long peptides significantly enhances model generalizability. Experimental results demonstrate superior performance across all categories: short peptides (R^2=0.855), medium peptides (R^2=0.816), and long peptides (R^2=0.882), with a 34.7% reduction in prediction error for long peptides compared to conventional single-model approaches. Ablation studies confirm: 1) The length-stratified strategy contributes 41.2% to performance improvement; 2) Topological features account for 28.5% of predictive importance. Compared to state-of-the-art models, our method maintains short peptide prediction accuracy while achieving a 25.7% increase in the coefficient of determination (R^2) for long peptides. This research provides a precise logD prediction tool for peptide drug development, particularly demonstrating unique value in optimizing long peptide lead compounds. 

**Abstract (ZH)**: 肽类化合物由于其高靶点亲和力和低毒性展现出作为治疗剂的巨大潜力，但其药物开发受限于低膜通透性。肽的分子量和长度对其logD值有显著影响，进而影响其穿过生物膜的能力。然而，由于序列、结构和电离状态之间的复杂相互作用，准确预测肽的logD依然具有挑战性。本研究引入了LengthLogD，这是一种通过分子长度分层建立专用模型并创新性集成多层次分子表示的预测框架。我们构建了三个层次的特征空间：原子级（10个分子描述符）、结构级（1024位默尔指纹图谱）、拓扑级（3个图基特征包括Wiener指数），并通过分层集成学习进行优化。专为长肽开发的自适应权重分配机制显著提高了模型的泛化能力。实验结果显示，在所有类别中均表现出优异的性能：短肽（R²=0.855）、中等长度肽（R²=0.816）和长肽（R²=0.882），长肽的预测误差降低了34.7%。消融研究表明：1）长度分层策略贡献了性能改进的41.2%；2）拓扑特征占预测重要性的28.5%。与最先进的模型相比，我们的方法在保持短肽预测准确性的同时，长肽的决定系数（R²）提高了25.7%。本研究提供了一种精确的logD预测工具，特别在于优化长肽先导化合物方面展现出独特价值。 

---
# A Toolkit for Compliance, a Toolkit for Justice: Drawing on Cross-sectoral Expertise to Develop a Pro-justice EU AI Act Toolkit 

**Title (ZH)**: 合规工具箱，公正工具箱：整合跨界专家经验以制定有利于公正的欧盟AI法案工具箱 

**Authors**: Tomasz Hollanek, Yulu Pi, Cosimo Fiorini, Virginia Vignali, Dorian Peters, Eleanor Drage  

**Link**: [PDF](https://arxiv.org/pdf/2505.17165)  

**Abstract**: The introduction of the AI Act in the European Union presents the AI research and practice community with a set of new challenges related to compliance. While it is certain that AI practitioners will require additional guidance and tools to meet these requirements, previous research on toolkits that aim to translate the theory of AI ethics into development and deployment practice suggests that such resources suffer from multiple limitations. These limitations stem, in part, from the fact that the toolkits are either produced by industry-based teams or by academics whose work tends to be abstract and divorced from the realities of industry. In this paper, we discuss the challenge of developing an AI ethics toolkit for practitioners that helps them comply with new AI-focused regulation, but that also moves beyond mere compliance to consider broader socio-ethical questions throughout development and deployment. The toolkit was created through a cross-sectoral collaboration between an academic team based in the UK and an industry team in Italy. We outline the background and rationale for creating a pro-justice AI Act compliance toolkit, detail the process undertaken to develop it, and describe the collaboration and negotiation efforts that shaped its creation. We aim for the described process to serve as a blueprint for other teams navigating the challenges of academia-industry partnerships and aspiring to produce usable and meaningful AI ethics resources. 

**Abstract (ZH)**: 《欧洲联盟AI法案的引入为AI研究与实践社区带来了合规方面的全新挑战：开发兼顾合规与社会伦理的AI伦理工具箱及其跨领域合作过程》 

---
# Efficient Training of Neural SDEs Using Stochastic Optimal Control 

**Title (ZH)**: 高效训练神经SDE模型的随机最优控制方法 

**Authors**: Rembert Daems, Manfred Opper, Guillaume Crevecoeur, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2505.17150)  

**Abstract**: We present a hierarchical, control theory inspired method for variational inference (VI) for neural stochastic differential equations (SDEs). While VI for neural SDEs is a promising avenue for uncertainty-aware reasoning in time-series, it is computationally challenging due to the iterative nature of maximizing the ELBO. In this work, we propose to decompose the control term into linear and residual non-linear components and derive an optimal control term for linear SDEs, using stochastic optimal control. Modeling the non-linear component by a neural network, we show how to efficiently train neural SDEs without sacrificing their expressive power. Since the linear part of the control term is optimal and does not need to be learned, the training is initialized at a lower cost and we observe faster convergence. 

**Abstract (ZH)**: 基于控制理论的层次化变分推断方法用于神经随机微分方程 

---
# Evaluating the Performance of Nigerian Lecturers using Multilayer Perceptron 

**Title (ZH)**: 评价尼日利亚 lecturer 性能的多层感知机方法 

**Authors**: I.E. Ezeibe, S.O. Okide, D.C. Asogwa  

**Link**: [PDF](https://arxiv.org/pdf/2505.17143)  

**Abstract**: Evaluating the performance of a lecturer has been essential for enhancing teaching quality, improving student learning outcomes, and strengthening the institution's reputation. The absence of such a system brings about lecturer performance evaluation which was neither comprehensive nor holistic. This system was designed using a web-based platform, created a secure database, and by using a custom dataset, captured some performance metrics which included student evaluation scores, Research Publications, Years of Experience, and Administrative Duties. Multilayer Perceptron (MLP) algorithm was utilized due to its ability to process complex data patterns and generates accurate predictions in a lecturer's performance based on historical data. This research focused on designing multiple performance metrics beyond the standard ones, incorporating student participation, and integrating analytical tools to deliver a comprehensive and holistic evaluation of lecturers' performance and was developed using Object-Oriented Analysis and Design (OOAD) methodology. Lecturers' performance is evaluated by the model, and the evaluation accuracy is about 91% compared with actual performance. Finally, by evaluating the performance of the MLP model, it is concluded that MLP enhanced lecturer performance evaluation by providing accurate predictions, reducing bias, and supporting data-driven decisions, ultimately improving the fairness and efficiency of the evaluation process. The MLP model's performance was evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE), achieved a test loss (MSE) of 256.99 and a MAE of 13.76, and reflected a high level of prediction accuracy. The model also demonstrated an estimated accuracy rate of approximately 96%, validated its effectiveness in predicting lecturer performance. 

**Abstract (ZH)**: 评价讲师绩效对于提升教学质量和增强机构声誉至关重要。缺乏这种系统的评价导致绩效评价既不全面也不完整。该系统利用基于Web的平台设计，创建了安全的数据库，并使用自定义数据集捕捉了包括学生评价分数、研究出版物、工作经验年限和行政职责在内的绩效指标。多层感知器（MLP）算法被采用，因其能够处理复杂的数据模式并根据历史数据生成准确的预测。该研究重点在于设计超越标准绩效指标的多个指标，融入学生参与度，并集成分析工具，以提供全面和整体的讲师绩效评价，并使用面向对象分析与设计（OOAD）方法论进行开发。模型对讲师绩效的评价准确率为约91%，最终评价结果显示，MLP通过提供精确预测、减少偏见和支持基于数据的决策，提升了评价过程的公平性和效率。MLP模型的性能通过均方误差（MSE）和平均绝对误差（MAE）进行评估，测试损失（MSE）为256.99，MAE为13.76，显示出高度的预测准确性，并验证了其在预测讲师绩效方面有效性，模型估计准确率为约96%。 

---
# MetaSTH-Sleep: Towards Effective Few-Shot Sleep Stage Classification with Spatial-Temporal Hypergraph Enhanced Meta-Learning 

**Title (ZH)**: MetaSTH-Sleep：基于空间-时间超图增强元学习的高效少量样本睡眠阶段分类 

**Authors**: Jingyu Li, Tiehua Zhang, Jinze Wang, Yi Zhang, Yuhuan Li, Yifan Zhao, Zhishu Shen, Jiannan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17142)  

**Abstract**: Accurate classification of sleep stages based on bio-signals is fundamental for automatic sleep stage annotation. Traditionally, this task relies on experienced clinicians to manually annotate data, a process that is both time-consuming and labor-intensive. In recent years, deep learning methods have shown promise in automating this task. However, three major challenges remain: (1) deep learning models typically require large-scale labeled datasets, making them less effective in real-world settings where annotated data is limited; (2) significant inter-individual variability in bio-signals often results in inconsistent model performance when applied to new subjects, limiting generalization; and (3) existing approaches often overlook the high-order relationships among bio-signals, failing to simultaneously capture signal heterogeneity and spatial-temporal dependencies. To address these issues, we propose MetaSTH-Sleep, a few-shot sleep stage classification framework based on spatial-temporal hypergraph enhanced meta-learning. Our approach enables rapid adaptation to new subjects using only a few labeled samples, while the hypergraph structure effectively models complex spatial interconnections and temporal dynamics simultaneously in EEG signals. Experimental results demonstrate that MetaSTH-Sleep achieves substantial performance improvements across diverse subjects, offering valuable insights to support clinicians in sleep stage annotation. 

**Abstract (ZH)**: 基于时空超图增强元学习的少量样本睡眠分期分类方法（MetaSTH-Sleep） 

---
# Fashion Industry in the Age of Generative Artificial Intelligence and Metaverse: A systematic Review 

**Title (ZH)**: 时尚行业在生成式人工智能和元宇宙时代的现状：一项系统性综述 

**Authors**: Rania Ahmed, Eman Ahmed, Ahmed Elbarbary, Ashraf Darwish, Aboul Ella Hassanien  

**Link**: [PDF](https://arxiv.org/pdf/2505.17141)  

**Abstract**: The fashion industry is an extremely profitable market that generates trillions of dollars in revenue by producing and distributing apparel, footwear, and accessories. This systematic literature review (SLR) seeks to systematically review and analyze the research landscape about the Generative Artificial Intelligence (GAI) and metaverse in the fashion industry. Thus, investigating the impact of integrating both technologies to enhance the fashion industry. This systematic review uses the Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) methodology, including three essential phases: identification, evaluation, and reporting. In the identification phase, the target search problems are determined by selecting appropriate keywords and alternative synonyms. After that 578 documents from 2014 to the end of 2023 are retrieved. The evaluation phase applies three screening steps to assess papers and choose 118 eligible papers for full-text reading. Finally, the reporting phase thoroughly examines and synthesizes the 118 eligible papers to identify key themes associated with GAI and Metaverse in the fashion industry. Based on Strengths, Weaknesses, Opportunities, and Threats (SWOT) analyses performed for both GAI and metaverse for the fashion industry, it is concluded that the integration of GAI and the metaverse holds the capacity to profoundly revolutionize the fashion sector, presenting chances for improved manufacturing, design, sales, and client experiences. Accordingly, the research proposes a new framework to integrate GAI and metaverse to enhance the fashion industry. The framework presents different use cases to promote the fashion industry using the integration. Future research points for achieving a successful integration are demonstrated. 

**Abstract (ZH)**: 时尚行业是一个极其有利可图的市场，通过生产和分配服装、鞋类和配饰产生万亿美元的收入。本文系统文献综述（SLR）旨在系统地回顾和分析有关生成式人工智能（GAI）和元宇宙在时尚行业的研究 landscape，从而探讨将两者技术整合以增强时尚行业的影响。本系统综述采用系统评价和元分析报告项目（PRISMA）方法，包括三个关键阶段：识别、评估和报告。在识别阶段，通过选择适当的关键词和同义词来确定目标搜索问题。随后，从2014年至2023年底检索到578篇文献。评估阶段应用了三个筛选步骤来评估论文并选择118篇合格论文进行全文阅读。最后，报告阶段详细检查并综合了118篇合格论文，以识别与GAI和元宇宙在时尚行业相关的关键主题。根据对GAI和元宇宙在时尚行业的优势、劣势、机会和威胁（SWOT）分析，得出结论认为将GAI与元宇宙整合有望深刻变革时尚产业，为制造、设计、销售和客户体验的提升提供机会。因此，本文提出了一种新框架以整合GAI和元宇宙来增强时尚行业。该框架提出了不同的应用场景以通过整合推动时尚产业的发展。展示了实现成功整合的未来研究方向。 

---
# Learning Probabilities of Causation from Finite Population Data 

**Title (ZH)**: 从有限总体数据中学习因果概率 

**Authors**: Shuai Wang, Song Jiang, Yizhou Sun, Judea Pearl, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17133)  

**Abstract**: Probabilities of causation play a crucial role in modern decision-making. This paper addresses the challenge of predicting probabilities of causation for subpopulations with \textbf{insufficient} data using machine learning models. Tian and Pearl first defined and derived tight bounds for three fundamental probabilities of causation: the probability of necessity and sufficiency (PNS), the probability of sufficiency (PS), and the probability of necessity (PN). However, estimating these probabilities requires both experimental and observational distributions specific to each subpopulation, which are often unavailable or impractical to obtain with limited population-level data. Therefore, for most subgroups, the amount of data they have is not enough to guarantee the accuracy of their probabilities. Hence, to estimate these probabilities for subpopulations with \textbf{insufficient} data, we propose using machine learning models that draw insights from subpopulations with sufficient data. Our evaluation of multiple machine learning models indicates that, given the population-level data and an appropriate choice of machine learning model and activation function, PNS can be effectively predicted. Through simulation studies on multiple Structured Causal Models (SCMs), we show that our multilayer perceptron (MLP) model with the Mish activation function achieves a mean absolute error (MAE) of approximately $0.02$ in predicting PNS for $32,768$ subpopulations across most SCMs using data from only $2,000$ subpopulations with known PNS values. 

**Abstract (ZH)**: 因果概率在现代决策中起着关键作用。本文探讨了使用机器学习模型预测数据不足子群体的因果概率的挑战。田和佩尔首先定义并推导了三种基本的因果概率：必要性和充分性概率（PNS）、充分性概率（PS）和必要性概率（PN）的确切界。然而，估计这些概率需要针对每个子群体的具体实验分布和观察分布，这些分布在限于总体数据时往往 unavailable 或难以获得。因此，对于大多数子群体，它们的数据量不足以保证这些概率的准确性。因此，为了预测数据不足子群体的因果概率，我们提出了利用数据充足子群体的机器学习模型来获取见解的方法。通过对多种机器学习模型的评估表明，在给定总体数据和适当的机器学习模型及激活函数选择下，PNS 可以有效预测。通过对多个结构化因果模型（SCM）进行模拟研究，我们展示了使用仅来自 2,000 个包含 PNS 值的子群体的数据，我们的具有 Mish 激活函数的多层感知机（MLP）模型在预测 32,768 个子群体的 PNS 时的平均绝对误差（MAE）约为 0.02。 

---
# NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation 

**Title (ZH)**: NeSyGeo: 一种多模态几何推理的神经符号框架数据生成 

**Authors**: Weiming Wu, Zi-kang Wang, Jin Ye, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17121)  

**Abstract**: Obtaining large-scale, high-quality data with reasoning paths is crucial for improving the geometric reasoning capabilities of multi-modal large language models (MLLMs). However, existing data generation methods, whether based on predefined templates or constrained symbolic provers, inevitably face diversity and numerical generalization limitations. To address these limitations, we propose NeSyGeo, a novel neuro-symbolic framework for generating geometric reasoning data. First, we propose a domain-specific language grounded in the entity-relation-constraint paradigm to comprehensively represent all components of plane geometry, along with generative actions defined within this symbolic space. We then design a symbolic-visual-text pipeline that synthesizes symbolic sequences, maps them to corresponding visual and textual representations, and generates diverse question-answer (Q&A) pairs using large language models (LLMs). To the best of our knowledge, we are the first to propose a neuro-symbolic approach in generating multimodal reasoning data. Based on this framework, we construct NeSyGeo-CoT and NeSyGeo-Caption datasets, containing 100k samples, and release a new benchmark NeSyGeo-Test for evaluating geometric reasoning abilities in MLLMs. Experiments demonstrate that the proposal significantly and consistently improves the performance of multiple MLLMs under both reinforcement and supervised fine-tuning. With only 4k samples and two epochs of reinforcement fine-tuning, base models achieve improvements of up to +15.8% on MathVision, +8.4% on MathVerse, and +7.3% on GeoQA. Notably, a 4B model can be improved to outperform an 8B model from the same series on geometric reasoning tasks. 

**Abstract (ZH)**: 基于推理路径的大规模高质数据获取对于提高多模态大规模语言模型的几何推理能力至关重要。然而，现有的数据生成方法，无论是基于预定义模板还是受限符号证明器，不可避免地面临着多样性和数值泛化的限制。为了解决这些限制，我们提出了NeSyGeo，一种新型的神经-符号框架以生成几何推理数据。首先，我们提出了一种基于实体-关系-约束范式的领域特定语言，全面表示平面几何的所有组件，并在该符号空间内定义生成动作。随后，我们设计了一个符号-视觉-文本流水线，生成符号序列，将其映射到相应的视觉和文本表示，并使用大规模语言模型生成多样化的问答（Q&A）对。据我们所知，我们是首次提出在生成多模态推理数据中使用神经-符号方法。基于该框架，我们构建了NeSyGeo-CoT和NeSyGeo-Caption数据集，包含100,000个样本，并发布了一个新的基准NeSyGeo-Test以评估多模态大规模语言模型的几何推理能力。实验表明，该提案在强化和监督微调下显著且一致地提高了多个多模态大规模语言模型的性能。仅用4,000个样本和两轮强化微调，基模型在MathVision上提高了15.8%，在MathVerse上提高了8.4%，在GeoQA上提高了7.3%。值得一提的是，一个4B模型可以提高到在其系列中8B模型在几何推理任务上表现出色。 

---
# REMS: a unified solution representation, problem modeling and metaheuristic algorithm design for general combinatorial optimization problems 

**Title (ZH)**: REMS：通用组合优化问题的一体化解决方案表示、问题建模与元启发式算法设计 

**Authors**: Aijuan Song, Guohua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17108)  

**Abstract**: Combinatorial optimization problems (COPs) with discrete variables and finite search space are critical across numerous fields, and solving them in metaheuristic algorithms is popular. However, addressing a specific COP typically requires developing a tailored and handcrafted algorithm. Even minor adjustments, such as constraint changes, may necessitate algorithm redevelopment. Therefore, establishing a framework for formulating diverse COPs into a unified paradigm and designing reusable metaheuristic algorithms is valuable. A COP can be typically viewed as the process of giving resources to perform specific tasks, subjecting to given constraints. Motivated by this, a resource-centered modeling and solving framework (REMS) is introduced for the first time. We first extract and define resources and tasks from a COP. Subsequently, given predetermined resources, the solution structure is unified as assigning tasks to resources, from which variables, objectives, and constraints can be derived and a problem model is constructed. To solve the modeled COPs, several fundamental operators are designed based on the unified solution structure, including the initial solution, neighborhood structure, destruction and repair, crossover, and ranking. These operators enable the development of various metaheuristic algorithms. Specially, 4 single-point-based algorithms and 1 population-based algorithm are configured herein. Experiments on 10 COPs, covering routing, location, loading, assignment, scheduling, and graph coloring problems, show that REMS can model these COPs within the unified paradigm and effectively solve them with the designed metaheuristic algorithms. Furthermore, REMS is more competitive than GUROBI and SCIP in tackling large-scale instances and complex COPs, and outperforms OR-TOOLS on several challenging COPs. 

**Abstract (ZH)**: 资源中心导向的组合优化问题建模与求解框架（REMS） 

---
# Transparency in Healthcare AI: Testing European Regulatory Provisions against Users' Transparency Needs 

**Title (ZH)**: 医疗保健AI中的透明度：测试欧盟监管规定以满足用户透明度需求 

**Authors**: Anna Spagnolli, Cecilia Tolomini, Elisa Beretta, Claudio Sarra  

**Link**: [PDF](https://arxiv.org/pdf/2505.17105)  

**Abstract**: Artificial Intelligence (AI) plays an essential role in healthcare and is pervasively incorporated into medical software and equipment. In the European Union, healthcare is a high-risk application domain for AI, and providers must prepare Instructions for Use (IFU) according to the European regulation 2024/1689 (AI Act). To this regulation, the principle of transparency is cardinal and requires the IFU to be clear and relevant to the users. This study tests whether these latter requirements are satisfied by the IFU structure. A survey was administered online via the Qualtrics platform to four types of direct stakeholders, i.e., managers (N = 238), healthcare professionals (N = 115), patients (N = 229), and Information Technology experts (N = 230). The participants rated the relevance of a set of transparency needs and indicated the IFU section addressing them. The results reveal differentiated priorities across stakeholders and a troubled mapping of transparency needs onto the IFU structure. Recommendations to build a locally meaningful IFU are derived. 

**Abstract (ZH)**: 人工智能（AI）在医疗健康领域扮演着重要角色，并广泛应用于医疗软件和设备中。在欧盟，医疗健康是AI的高风险应用领域，提供者必须根据欧盟2024/1689号条例（AI法案）准备产品使用说明书（IFU）。根据该条例，透明性原则至关重要，要求IFU对用户来说必须清晰且相关。本研究测试这些要求是否被IFU结构所满足。通过Qualtrics平台在线发放问卷，调查了四种类型的利益相关方：管理人员（N=238）、医疗专业人员（N=115）、患者（N=229）和信息技术专家（N=230）。参与者对一套透明性需求的相关性进行了评级，并指出了IFU中相应的部分。研究结果揭示了不同利益相关方的优先事项差异，并指出透明性需求与IFU结构之间的匹配存在困难。研究还获得了构建具有地方意义的IFU的建议。 

---
# Informatics for Food Processing 

**Title (ZH)**: 食品加工中的信息学 

**Authors**: Gordana Ispirova, Michael Sebek, Giulia Menichetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.17087)  

**Abstract**: This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research. 

**Abstract (ZH)**: 本章探讨了食品加工的演变、分类及其健康影响，并强调了机器学习、人工智能（AI）和数据科学在食品信息学领域中的变革性作用。它从历史概述出发，对传统的分类框架（如NOVA、Nutri-Score和SIGA）进行了批判性回顾，指出了这些框架的优势与局限性，特别是其在流行病学研究和公共政策制定中遇到的主观性和可重复性难题。为解决这些问题，本章提出了新颖的计算方法，包括使用营养成分数据训练的随机森林模型FoodProX，以推断加工水平并生成连续的FPro评分。此外，本章还探讨了如何利用如BERT和BioBERT这样的大型语言模型通过语义嵌入食品描述和成分列表来进行预测任务，即使在数据缺失的情况下也是如此。本章的一个主要贡献是使用Open Food Facts数据库进行的一项新颖案例研究，展示了多模态AI模型如何整合结构化和非结构化数据以大规模分类食品，为公共健康和研究中的食品加工评估提供了新的范式。 

---
# GSDFuse: Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis 

**Title (ZH)**: GSDFuse: 从社交媒体隐写分析中多维度弱信号捕获认知不一致性 

**Authors**: Kaibo Huang, Zipei Zhang, Yukun Wei, TianXin Zhang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17085)  

**Abstract**: The ubiquity of social media platforms facilitates malicious linguistic steganography, posing significant security risks. Steganalysis is profoundly hindered by the challenge of identifying subtle cognitive inconsistencies arising from textual fragmentation and complex dialogue structures, and the difficulty in achieving robust aggregation of multi-dimensional weak signals, especially given extreme steganographic sparsity and sophisticated steganography. These core detection difficulties are compounded by significant data imbalance. This paper introduces GSDFuse, a novel method designed to systematically overcome these obstacles. GSDFuse employs a holistic approach, synergistically integrating hierarchical multi-modal feature engineering to capture diverse signals, strategic data augmentation to address sparsity, adaptive evidence fusion to intelligently aggregate weak signals, and discriminative embedding learning to enhance sensitivity to subtle inconsistencies. Experiments on social media datasets demonstrate GSDFuse's state-of-the-art (SOTA) performance in identifying sophisticated steganography within complex dialogue environments. The source code for GSDFuse is available at this https URL. 

**Abstract (ZH)**: 社交媒体平台的普遍性促进了恶意语言隐写术的应用，引发了显著的安全风险。传统的文本隐写分析受到识别由文本碎片化和复杂对话结构产生的微妙认知不一致性挑战的影响，并且在聚合多维度弱信号方面存在困难，尤其是在极端隐写稀疏性和复杂的隐写术条件下。这些核心检测困难进一步受到严重数据不平衡的影响。本文提出了一种新技术GSDFuse，以系统地克服这些障碍。GSDFuse采用整体方法，结合层次多模态特征工程捕捉不同信号、战略数据增强处理稀疏性问题、自适应证据融合智能聚合弱信号以及判别嵌入学习提高对微妙不一致性的敏感度。在社交媒体数据集上的实验表明，GSDFuse在复杂对话环境中识别复杂隐写术方面达到了当前最佳性能（SOTA）。GSDFuse的源代码可通过此链接获取。 

---
# Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English 

**Title (ZH)**: 帧率对语音分词器的影响：以普通话和英语为例的研究 

**Authors**: Haoyang Zhang, Hexin Liu, Xiangyu Zhang, Qiquan Zhang, Yuchen Hu, Junqi Zhao, Fei Tian, Xuerui Yang, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17076)  

**Abstract**: The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications. 

**Abstract (ZH)**: 语音切分器在近期的语音任务中扮演着至关重要的角色，通常作为语音信号与语言模型之间的桥梁。尽管低帧率编码器广泛用于语音切分，但帧率对语音切分的影响仍未充分探讨。本研究通过分析两种类型不同的语言（普通话和英语）来探讨不同帧率如何影响语音切分，并在语音识别任务中评估由此产生的语义切分。研究发现，帧率的变化以不同的方式影响每种语言的语音切分，突显了帧率、音位密度和语言特定声学特征之间的相互作用。这些结果为优化语音切分器的帧率选择提供了见解，对自动语音识别、文本到语音以及其他相关语音应用具有重要意义。 

---
# Development and Validation of Engagement and Rapport Scales for Evaluating User Experience in Multimodal Dialogue Systems 

**Title (ZH)**: 多模态对话系统中用户体验的参与度和关系量表的开发与验证 

**Authors**: Fuma Kurata, Mao Saeki, Masaki Eguchi, Shungo Suzuki, Hiroaki Takatsu, Yoichi Matsuyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.17075)  

**Abstract**: This study aimed to develop and validate two scales of engagement and rapport to evaluate the user experience quality with multimodal dialogue systems in the context of foreign language learning. The scales were designed based on theories of engagement in educational psychology, social psychology, and second language this http URL-four Japanese learners of English completed roleplay and discussion tasks with trained human tutors and a dialog agent. After each dialogic task was completed, they responded to the scales of engagement and rapport. The validity and reliability of the scales were investigated through two analyses. We first conducted analysis of Cronbach's alpha coefficient and a series of confirmatory factor analyses to test the structural validity of the scales and the reliability of our designed items. We then compared the scores of engagement and rapport between the dialogue with human tutors and the one with a dialogue agent. The results revealed that our scales succeeded in capturing the difference in the dialogue experience quality between the human interlocutors and the dialogue agent from multiple perspectives. 

**Abstract (ZH)**: 本研究旨在基于教育心理学、社会心理学及外语学习理论，开发和完善两个用于评估外语学习中多模态对话系统用户体验质量的效度和可靠性的参与度和关系度量表。Japanese英语学习者与经过培训的人类导师和对话代理进行了角色扮演和讨论任务，并在每个对话任务完成后对参与度和关系度量表做出了回应。通过两种分析验证了度量表的有效性和可靠性。首先，我们计算Cronbach’s α系数并进行了系列确证性因子分析，以测试量表的结构效度和项目可靠性。然后，我们将人类导师对话和与对话代理对话的参与度和关系得分进行了比较。研究结果表明，我们的量表能够在多个视角上成功捕捉人类对话伙伴与对话代理之间的对话体验质量差异。 

---
# Improving endpoint detection in end-to-end streaming ASR for conversational speech 

**Title (ZH)**: 改进端到端流式ASR在对话语音中的端点检测 

**Authors**: Anandh C, Karthik Pandia Durai, Jeena Prakash, Manickavela Arumugam, Kadri Hacioglu, S.Pavankumar Dubagunta, Andreas Stolcke, Shankar Venkatesan, Aravind Ganapathiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.17070)  

**Abstract**: ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method. 

**Abstract (ZH)**: ASR 末端检测（EP）在支持人类或人工代理的人与人/机器对话产品中提供良好用户体验中扮演重要角色。基于转导的ASR（T-ASR）是一种优选的端到端（E2E）ASR建模技术，适用于流式处理。T-ASR的主要限制是ASR输出的延迟发出，这可能会导致末端检测（EP）错误或延迟。不准确的末端检测会中断用户的讲话，返回不完整的记录，而延迟的末端检测会增加感知的延迟，降低用户体验。我们通过解决延迟排放和末端检测错误来改进末端检测。为了应对延迟排放问题，我们在每个单词结束处引入一个词尾标记，并加入延迟惩罚。通过使用辅助网络获得可靠的帧级语音活动检测来解决末端检测延迟问题。我们将在Switchboard对话语音语料库上应用所提出的方法，并将其与延迟惩罚方法进行比较评估。 

---
# Unveil Multi-Picture Descriptions for Multilingual Mild Cognitive Impairment Detection via Contrastive Learning 

**Title (ZH)**: 多 picture 描述的对比学习在多语言轻度认知障碍检测中的应用 

**Authors**: Kristin Qi, Jiali Cheng, Youxiang Zhu, Hadi Amiri, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17067)  

**Abstract**: Detecting Mild Cognitive Impairment from picture descriptions is critical yet challenging, especially in multilingual and multiple picture settings. Prior work has primarily focused on English speakers describing a single picture (e.g., the 'Cookie Theft'). The TAUKDIAL-2024 challenge expands this scope by introducing multilingual speakers and multiple pictures, which presents new challenges in analyzing picture-dependent content. To address these challenges, we propose a framework with three components: (1) enhancing discriminative representation learning via supervised contrastive learning, (2) involving image modality rather than relying solely on speech and text modalities, and (3) applying a Product of Experts (PoE) strategy to mitigate spurious correlations and overfitting. Our framework improves MCI detection performance, achieving a +7.1% increase in Unweighted Average Recall (UAR) (from 68.1% to 75.2%) and a +2.9% increase in F1 score (from 80.6% to 83.5%) compared to the text unimodal baseline. Notably, the contrastive learning component yields greater gains for the text modality compared to speech. These results highlight our framework's effectiveness in multilingual and multi-picture MCI detection. 

**Abstract (ZH)**: 从图片描述中检测轻度认知 impairment 在多语言和多图片设置下是关键但具有挑战性：TAUKDIAL-2024 挑战赛扩展了这一范围 

---
# DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation 

**Title (ZH)**: DO-RAG：一种基于知识图谱增强检索生成的领域特定问答框架 

**Authors**: David Osei Opoku, Ming Sheng, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17058)  

**Abstract**: Domain-specific QA systems require not just generative fluency but high factual accuracy grounded in structured expert knowledge. While recent Retrieval-Augmented Generation (RAG) frameworks improve context recall, they struggle with integrating heterogeneous data and maintaining reasoning consistency. To address these challenges, we propose DO-RAG, a scalable and customizable hybrid QA framework that integrates multi-level knowledge graph construction with semantic vector retrieval. Our system employs a novel agentic chain-of-thought architecture to extract structured relationships from unstructured, multimodal documents, constructing dynamic knowledge graphs that enhance retrieval precision. At query time, DO-RAG fuses graph and vector retrieval results to generate context-aware responses, followed by hallucination mitigation via grounded refinement. Experimental evaluations in the database and electrical domains show near-perfect recall and over 94% answer relevancy, with DO-RAG outperforming baseline frameworks by up to 33.38%. By combining traceability, adaptability, and performance efficiency, DO-RAG offers a reliable foundation for multi-domain, high-precision QA at scale. 

**Abstract (ZH)**: 领域特定的问答系统不仅需要生成流畅性，还需要基于结构化专家知识的高度事实准确性。虽然最近的检索增强生成（RAG）框架在上下文召回方面有所改善，但它们在整合异构数据和保持推理一致性方面存在困难。为应对这些挑战，我们提出了一种可扩展且可定制的混合问答框架DO-RAG，该框架结合了多级知识图构建与语义向量检索。该系统采用了一种新颖的行动者思维链架构，从非结构化、多模态文档中提取结构化关系，构建动态知识图，从而提高检索精度。在查询时，DO-RAG 将图检索和向量检索结果融合生成上下文相关的回答，并通过基于事实的改进减轻虚构信息。在数据库和电气领域的实验评估显示，DO-RAG 的召回率接近完美，回答相关性超过 94%，且相比基准框架性能提升高达 33.38%。通过结合可追溯性、适应性和性能效率，DO-RAG 为多领域、高精度的大规模问答提供了可靠的基石。 

---
# METHOD: Modular Efficient Transformer for Health Outcome Discovery 

**Title (ZH)**: 模块化高效变压器在健康结果发现中的应用 

**Authors**: Linglong Qian, Zina Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17054)  

**Abstract**: Recent advances in transformer architectures have revolutionised natural language processing, but their application to healthcare domains presents unique challenges. Patient timelines are characterised by irregular sampling, variable temporal dependencies, and complex contextual relationships that differ substantially from traditional language tasks. This paper introduces \METHOD~(Modular Efficient Transformer for Health Outcome Discovery), a novel transformer architecture specifically designed to address the challenges of clinical sequence modelling in electronic health records. \METHOD~integrates three key innovations: (1) a patient-aware attention mechanism that prevents information leakage whilst enabling efficient batch processing; (2) an adaptive sliding window attention scheme that captures multi-scale temporal dependencies; and (3) a U-Net inspired architecture with dynamic skip connections for effective long sequence processing. Evaluations on the MIMIC-IV database demonstrate that \METHOD~consistently outperforms the state-of-the-art \ETHOS~model, particularly in predicting high-severity cases that require urgent clinical intervention. \METHOD~exhibits stable performance across varying inference lengths, a crucial feature for clinical deployment where patient histories vary significantly in length. Analysis of learned embeddings reveals that \METHOD~better preserves clinical hierarchies and relationships between medical concepts. These results suggest that \METHOD~represents a significant advancement in transformer architectures optimised for healthcare applications, providing more accurate and clinically relevant predictions whilst maintaining computational efficiency. 

**Abstract (ZH)**: Recent Advances in Transformer Architectures for Healthcare Domain Applications: Introducing \METHOD~(Modular Efficient Transformer for Health Outcome Discovery) 

---
# Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally 

**Title (ZH)**: Worlds的语言纽带：全球央行政策沟通解码的统一框架 

**Authors**: Agam Shah, Siddhant Sukhani, Huzaifa Pardawala, Saketh Budideti, Riya Bhadani, Rudra Gopal, Siddhartha Somani, Michael Galarnyk, Soungmin Lee, Arnav Hiray, Akshar Ravichandran, Eric Kim, Pranav Aluru, Joshua Zhang, Sebastian Jaskowski, Veer Guda, Meghaj Tarte, Liqin Ye, Spencer Gosden, Rutwik Routu, Rachel Yuh, Sloka Chava, Sahasra Chava, Dylan Patrick Kelly, Aiden Chiang, Harsit Mittal, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.17048)  

**Abstract**: Central banks around the world play a crucial role in maintaining economic stability. Deciphering policy implications in their communications is essential, especially as misinterpretations can disproportionately impact vulnerable populations. To address this, we introduce the World Central Banks (WCB) dataset, the most comprehensive monetary policy corpus to date, comprising over 380k sentences from 25 central banks across diverse geographic regions, spanning 28 years of historical data. After uniformly sampling 1k sentences per bank (25k total) across all available years, we annotate and review each sentence using dual annotators, disagreement resolutions, and secondary expert reviews. We define three tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation, with each sentence annotated for all three. We benchmark seven Pretrained Language Models (PLMs) and nine Large Language Models (LLMs) (Zero-Shot, Few-Shot, and with annotation guide) on these tasks, running 15,075 benchmarking experiments. We find that a model trained on aggregated data across banks significantly surpasses a model trained on an individual bank's data, confirming the principle "the whole is greater than the sum of its parts." Additionally, rigorous human evaluations, error analyses, and predictive tasks validate our framework's economic utility. Our artifacts are accessible through the HuggingFace and GitHub under the CC-BY-NC-SA 4.0 license. 

**Abstract (ZH)**: 全球中央银行（WCB）数据集：最全面的货币政策语料库及其应用 

---
# QRA++: Quantified Reproducibility Assessment for Common Types of Results in Natural Language Processing 

**Title (ZH)**: QRA++: 量化自然语言处理中常见类型结果的可再现性评估 

**Authors**: Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17043)  

**Abstract**: Reproduction studies reported in NLP provide individual data points which in combination indicate worryingly low levels of reproducibility in the field. Because each reproduction study reports quantitative conclusions based on its own, often not explicitly stated, criteria for reproduction success/failure, the conclusions drawn are hard to interpret, compare, and learn from. In this paper, we present QRA++, a quantitative approach to reproducibility assessment that (i) produces continuous-valued degree of reproducibility assessments at three levels of granularity; (ii) utilises reproducibility measures that are directly comparable across different studies; and (iii) grounds expectations about degree of reproducibility in degree of similarity between experiments. QRA++ enables more informative reproducibility assessments to be conducted, and conclusions to be drawn about what causes reproducibility to be better/poorer. We illustrate this by applying QRA++ to three example sets of comparable experiments, revealing clear evidence that degree of reproducibility depends on similarity of experiment properties, but also system type and evaluation method. 

**Abstract (ZH)**: NLP领域中报道的再现研究提供了单一的数据点，这些数据点结合在一起显示出令人担忧的低再现性水平。由于每项再现研究基于其自身（常常未明确陈述）的再现成功/失败标准报告定量结论，因此得出的结论难以解释、比较和学习。在本文中，我们提出QRA++，这是一种定量的再现性评估方法，它能够（i）在三个粒度级别上产生连续值的再现性评估；（ii）利用可以直接在不同研究之间进行比较的再现性度量；（iii）将对再现性的期望建立在实验相似性程度的基础上。QRA++能够进行更具信息量的再现性评估，并从中得出再现性更好或较差的原因。我们通过将QRA++应用于三个可比实验集的例子，展示了这一点，结果显示再现性程度取决于实验属性的相似性，同时也受系统类型和评估方法的影响。 

---
# ReMi: A Random Recurrent Neural Network Approach to Music Production 

**Title (ZH)**: ReMi: 一种用于音乐生产的目的随机循环神经网络方法 

**Authors**: Hugo Chateau-Laurent, Tara Vanhatalo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17023)  

**Abstract**: Generative artificial intelligence raises concerns related to energy consumption, copyright infringement and creative atrophy. We show that randomly initialized recurrent neural networks can produce arpeggios and low-frequency oscillations that are rich and configurable. In contrast to end-to-end music generation that aims to replace musicians, our approach expands their creativity while requiring no data and much less computational power. More information can be found at: this https URL 

**Abstract (ZH)**: 生成型人工智能引发了与能耗、版权侵犯和创造力萎缩相关的问题。我们展示了随机初始化的循环神经网络可以产生丰富且可配置的和弦进行和低频振荡。与旨在替代 musicians 的端到端音乐生成方法不同，我们的方法能够扩展 musicians 的创造力，且无需数据和较少的计算资源。更多详细信息请参见：this https URL 

---
