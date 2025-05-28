# Generalized Coordination of Partially Cooperative Urban Traffic 

**Title (ZH)**: 部分合作的都市交通协调一般化 

**Authors**: Max Bastian Mertens, Michael Buchholz  

**Link**: [PDF](https://arxiv.org/pdf/2505.20879)  

**Abstract**: Vehicle-to-anything connectivity, especially for autonomous vehicles, promises to increase passenger comfort and safety of road traffic, for example, by sharing perception and driving intention. Cooperative maneuver planning uses connectivity to enhance traffic efficiency, which has, so far, been mainly considered for automated intersection management. In this article, we present a novel cooperative maneuver planning approach that is generalized to various situations found in urban traffic. Our framework handles challenging mixed traffic, that is, traffic comprising both cooperative connected vehicles and other vehicles at any distribution. Our solution is based on an optimization approach accompanied by an efficient heuristic method for high-load scenarios. We extensively evaluate the proposed planer in a distinctly realistic simulation framework and show significant efficiency gains already at a cooperation rate of 40%. Traffic throughput increases, while the average waiting time and the number of stopped vehicles are reduced, without impacting traffic safety. 

**Abstract (ZH)**: 车辆到万物连接，特别是对自动驾驶车辆而言，有望通过共享感知和驾驶意图来提高乘客的舒适性和道路交通的安全性。协同机动规划利用连接性来提高交通效率，目前主要考虑的是自动化交叉口管理。在本文中，我们提出了一种新的协同机动规划方法，适用于城市交通中遇到的各种情况。我们的框架处理具有挑战性的混合交通，即包括协同连接车辆和其他车辆在内的各类交通流。我们的解决方案基于优化方法，并结合了高效启发式方法以应对高负载场景。我们在一个高度现实的仿真框架中广泛评估了所提出的规划器，并表明在合作率为40%的情况下已经实现了显著的效率增益。交通流量增加，平均等待时间和停止车辆数量减少，而不影响交通安全。 

---
# Collision-free Control Barrier Functions for General Ellipsoids via Separating Hyperplane 

**Title (ZH)**: 椭球体基于分离超平面的无碰撞控制屏障函数 

**Authors**: Zeming Wu, Lu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20847)  

**Abstract**: This paper presents a novel collision avoidance method for general ellipsoids based on control barrier functions (CBFs) and separating hyperplanes. First, collision-free conditions for general ellipsoids are analytically derived using the concept of dual cones. These conditions are incorporated into the CBF framework by extending the system dynamics of controlled objects with separating hyperplanes, enabling efficient and reliable collision avoidance. The validity of the proposed collision-free CBFs is rigorously proven, ensuring their effectiveness in enforcing safety constraints. The proposed method requires only single-level optimization, significantly reducing computational time compared to state-of-the-art methods. Numerical simulations and real-world experiments demonstrate the effectiveness and practicality of the proposed algorithm. 

**Abstract (ZH)**: 基于对偶锥的控制障碍函数与分离超平面的广义椭球碰撞避险方法 

---
# STITCH-OPE: Trajectory Stitching with Guided Diffusion for Off-Policy Evaluation 

**Title (ZH)**: STITCH-OPE：带有引导扩散的轨迹拼接用于离策评估 

**Authors**: Hossein Goli, Michael Gimelfarb, Nathan Samuel de Lara, Haruki Nishimura, Masha Itkina, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2505.20781)  

**Abstract**: Off-policy evaluation (OPE) estimates the performance of a target policy using offline data collected from a behavior policy, and is crucial in domains such as robotics or healthcare where direct interaction with the environment is costly or unsafe. Existing OPE methods are ineffective for high-dimensional, long-horizon problems, due to exponential blow-ups in variance from importance weighting or compounding errors from learned dynamics models. To address these challenges, we propose STITCH-OPE, a model-based generative framework that leverages denoising diffusion for long-horizon OPE in high-dimensional state and action spaces. Starting with a diffusion model pre-trained on the behavior data, STITCH-OPE generates synthetic trajectories from the target policy by guiding the denoising process using the score function of the target policy. STITCH-OPE proposes two technical innovations that make it advantageous for OPE: (1) prevents over-regularization by subtracting the score of the behavior policy during guidance, and (2) generates long-horizon trajectories by stitching partial trajectories together end-to-end. We provide a theoretical guarantee that under mild assumptions, these modifications result in an exponential reduction in variance versus long-horizon trajectory diffusion. Experiments on the D4RL and OpenAI Gym benchmarks show substantial improvement in mean squared error, correlation, and regret metrics compared to state-of-the-art OPE methods. 

**Abstract (ZH)**: 基于生成模型的降噪扩散长时域离策评估方法 

---
# MRSD: Multi-Resolution Skill Discovery for HRL Agents 

**Title (ZH)**: 多分辨率技能发现for HRL代理 

**Authors**: Shashank Sharma, Janina Hoffmann, Vinay Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2505.21410)  

**Abstract**: Hierarchical reinforcement learning (HRL) relies on abstract skills to solve long-horizon tasks efficiently. While existing skill discovery methods learns these skills automatically, they are limited to a single skill per task. In contrast, humans learn and use both fine-grained and coarse motor skills simultaneously. Inspired by human motor control, we propose Multi-Resolution Skill Discovery (MRSD), an HRL framework that learns multiple skill encoders at different temporal resolutions in parallel. A high-level manager dynamically selects among these skills, enabling adaptive control strategies over time. We evaluate MRSD on tasks from the DeepMind Control Suite and show that it outperforms prior state-of-the-art skill discovery and HRL methods, achieving faster convergence and higher final performance. Our findings highlight the benefits of integrating multi-resolution skills in HRL, paving the way for more versatile and efficient agents. 

**Abstract (ZH)**: 多层次强化学习中的多分辨率技能发现（Mult-Resolution Skill Discovery for Hierarchical Reinforcement Learning） 

---
# A domain adaptation neural network for digital twin-supported fault diagnosis 

**Title (ZH)**: 基于数字孪生支持的域适应神经网络故障诊断 

**Authors**: Zhenling Chen, Haiwei Fu, Zhiguo Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21046)  

**Abstract**: Digital twins offer a promising solution to the lack of sufficient labeled data in deep learning-based fault diagnosis by generating simulated data for model training. However, discrepancies between simulation and real-world systems can lead to a significant drop in performance when models are applied in real scenarios. To address this issue, we propose a fault diagnosis framework based on Domain-Adversarial Neural Networks (DANN), which enables knowledge transfer from simulated (source domain) to real-world (target domain) data. We evaluate the proposed framework using a publicly available robotics fault diagnosis dataset, which includes 3,600 sequences generated by a digital twin model and 90 real sequences collected from physical systems. The DANN method is compared with commonly used lightweight deep learning models such as CNN, TCN, Transformer, and LSTM. Experimental results show that incorporating domain adaptation significantly improves the diagnostic performance. For example, applying DANN to a baseline CNN model improves its accuracy from 70.00% to 80.22% on real-world test data, demonstrating the effectiveness of domain adaptation in bridging the sim-to-real gap. 

**Abstract (ZH)**: 数字孪生提供的模拟数据生成方法为基于深度学习的故障诊断问题提供了有希望的解决方案，但模拟与实际系统之间的差异会导致模型在实际应用中的性能显著下降。为解决这一问题，我们提出了一种基于领域自适应神经网络（DANN）的故障诊断框架，该框架能够实现从模拟数据（源域）向实际数据（目标域）的知识迁移。我们使用一个公开的机器人故障诊断数据集进行评估，该数据集包含3,600条由数字孪生模型生成的模拟序列和90条来自物理系统的实际序列。DANN方法与常用的轻量级深度学习模型（如CNN、TCN、Transformer和LSTM）进行了比较。实验结果表明，引入领域自适应显著提高了诊断性能，例如，将DANN应用于基础的CNN模型，在实际测试数据上的准确率从70.00%提高到80.22%，证明领域自适应在弥合模拟到实际差距方面的有效性。 

---
# Learning Individual Behavior in Agent-Based Models with Graph Diffusion Networks 

**Title (ZH)**: 基于图扩散网络学习个体行为的agents-based模型研究 

**Authors**: Francesco Cozzi, Marco Pangallo, Alan Perotti, André Panisson, Corrado Monti  

**Link**: [PDF](https://arxiv.org/pdf/2505.21426)  

**Abstract**: Agent-Based Models (ABMs) are powerful tools for studying emergent properties in complex systems. In ABMs, agent behaviors are governed by local interactions and stochastic rules. However, these rules are, in general, non-differentiable, limiting the use of gradient-based methods for optimization, and thus integration with real-world data. We propose a novel framework to learn a differentiable surrogate of any ABM by observing its generated data. Our method combines diffusion models to capture behavioral stochasticity and graph neural networks to model agent interactions. Distinct from prior surrogate approaches, our method introduces a fundamental shift: rather than approximating system-level outputs, it models individual agent behavior directly, preserving the decentralized, bottom-up dynamics that define ABMs. We validate our approach on two ABMs (Schelling's segregation model and a Predator-Prey ecosystem) showing that it replicates individual-level patterns and accurately forecasts emergent dynamics beyond training. Our results demonstrate the potential of combining diffusion models and graph learning for data-driven ABM simulation. 

**Abstract (ZH)**: 基于代理的模型的可微代理行为学习框架 

---
# A Structured Unplugged Approach for Foundational AI Literacy in Primary Education 

**Title (ZH)**: 面向基础人工智能素养的小学教育结构性课外 approach 

**Authors**: Maria Cristina Carrisi, Mirko Marras, Sara Vergallo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21398)  

**Abstract**: Younger generations are growing up in a world increasingly shaped by intelligent technologies, making early AI literacy crucial for developing the skills to critically understand and navigate them. However, education in this field often emphasizes tool-based learning, prioritizing usage over understanding the underlying concepts. This lack of knowledge leaves non-experts, especially children, prone to misconceptions, unrealistic expectations, and difficulties in recognizing biases and stereotypes. In this paper, we propose a structured and replicable teaching approach that fosters foundational AI literacy in primary students, by building upon core mathematical elements closely connected to and of interest in primary curricula, to strengthen conceptualization, data representation, classification reasoning, and evaluation of AI. To assess the effectiveness of our approach, we conducted an empirical study with thirty-one fifth-grade students across two classes, evaluating their progress through a post-test and a satisfaction survey. Our results indicate improvements in terminology understanding and usage, features description, logical reasoning, and evaluative skills, with students showing a deeper comprehension of decision-making processes and their limitations. Moreover, the approach proved engaging, with students particularly enjoying activities that linked AI concepts to real-world reasoning. Materials: this https URL. 

**Abstract (ZH)**: younger 一代正在一个日益被智能技术塑造的世界中成长，因此早期人工智能素养对于培养批判性理解和导航这些技术的能力至关重要。然而，该领域的教育往往强调基于工具的学习，优先考虑使用而忽视对基本概念的理解。这种缺乏知识使得非专家，特别是儿童，容易产生误解、不切实际的期望，并且难以识别偏见和刻板印象。在本文中，我们提出了一种结构化且可复制的 teaching 方法，旨在通过结合与小学课程核心要素紧密相关且感兴趣的数学元素，促进小学生的人工智能基础素养，加强概念理解、数据表示、分类推理和人工智能评估能力。为了评估该方法的有效性，我们在两个班级的三十一名五年级学生中进行了实证研究，通过后测和满意度调查评估其进步。研究结果表明，在术语理解与使用、特征描述、逻辑推理和评价技能方面有所改善，学生对决策过程及其局限性的理解更加深入。此外，该方法证明是吸引人的，学生们特别喜欢将人工智能概念与现实世界推理相联系的活动。 

---
# The Multilingual Divide and Its Impact on Global AI Safety 

**Title (ZH)**: 多语言鸿沟及其对全球AI安全的影响 

**Authors**: Aidan Peppin, Julia Kreutzer, Alice Schoenauer Sebag, Kelly Marchisio, Beyza Ermis, John Dang, Samuel Cahyawijaya, Shivalika Singh, Seraphina Goldfarb-Tarrant, Viraat Aryabumi, Aakanksha, Wei-Yin Ko, Ahmet Üstün, Matthias Gallé, Marzieh Fadaee, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2505.21344)  

**Abstract**: Despite advances in large language model capabilities in recent years, a large gap remains in their capabilities and safety performance for many languages beyond a relatively small handful of globally dominant languages. This paper provides researchers, policymakers and governance experts with an overview of key challenges to bridging the "language gap" in AI and minimizing safety risks across languages. We provide an analysis of why the language gap in AI exists and grows, and how it creates disparities in global AI safety. We identify barriers to address these challenges, and recommend how those working in policy and governance can help address safety concerns associated with the language gap by supporting multilingual dataset creation, transparency, and research. 

**Abstract (ZH)**: 尽管近年来大型语言模型的能力取得了进展，但在许多超出少数全球主导语言之外的其他语言上，它们的能力和安全性能之间仍存在较大的差距。本文为研究人员、政策制定者和治理专家提供了缩小“语言缺口”在AI中的差距并跨语言减少安全风险的概览。我们分析了AI语言缺口存在的原因及其如何导致全球AI安全的不平等，并识别出应对这些挑战的障碍，建议通过支持多语言数据集创建、透明度和研究来协助政策和治理领域的工作，以解决与语言缺口相关的安全关切。 

---
# RLJP: Legal Judgment Prediction via First-Order Logic Rule-enhanced with Large Language Models 

**Title (ZH)**: RLJP：借助大型语言模型增强的一阶逻辑规则辅助法律判决预测 

**Authors**: Yue Zhang, Zhiliang Tian, Shicheng Zhou, Haiyang Wang, Wenqing Hou, Yuying Liu, Xuechen Zhao, Minlie Huang, Ye Wang, Bin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.21281)  

**Abstract**: Legal Judgment Prediction (LJP) is a pivotal task in legal AI. Existing semantic-enhanced LJP models integrate judicial precedents and legal knowledge for high performance. But they neglect legal reasoning logic, a critical component of legal judgments requiring rigorous logical analysis. Although some approaches utilize legal reasoning logic for high-quality predictions, their logic rigidity hinders adaptation to case-specific logical frameworks, particularly in complex cases that are lengthy and detailed. This paper proposes a rule-enhanced legal judgment prediction framework based on first-order logic (FOL) formalism and comparative learning (CL) to develop an adaptive adjustment mechanism for legal judgment logic and further enhance performance in LJP. Inspired by the process of human exam preparation, our method follows a three-stage approach: first, we initialize judgment rules using the FOL formalism to capture complex reasoning logic accurately; next, we propose a Confusion-aware Contrastive Learning (CACL) to dynamically optimize the judgment rules through a quiz consisting of confusable cases; finally, we utilize the optimized judgment rules to predict legal judgments. Experimental results on two public datasets show superior performance across all metrics. The code is publicly available{this https URL}. 

**Abstract (ZH)**: 基于一阶逻辑和对比学习的规则增强法律判决预测框架 

---
# Interpretable DNFs 

**Title (ZH)**: 可解释的DNFs 

**Authors**: Martin C. Cooper, Imane Bousdira, Clément Carbonnel  

**Link**: [PDF](https://arxiv.org/pdf/2505.21212)  

**Abstract**: A classifier is considered interpretable if each of its decisions has an explanation which is small enough to be easily understood by a human user. A DNF formula can be seen as a binary classifier $\kappa$ over boolean domains. The size of an explanation of a positive decision taken by a DNF $\kappa$ is bounded by the size of the terms in $\kappa$, since we can explain a positive decision by giving a term of $\kappa$ that evaluates to true. Since both positive and negative decisions must be explained, we consider that interpretable DNFs are those $\kappa$ for which both $\kappa$ and $\overline{\kappa}$ can be expressed as DNFs composed of terms of bounded size. In this paper, we study the family of $k$-DNFs whose complements can also be expressed as $k$-DNFs. We compare two such families, namely depth-$k$ decision trees and nested $k$-DNFs, a novel family of models. Experiments indicate that nested $k$-DNFs are an interesting alternative to decision trees in terms of interpretability and accuracy. 

**Abstract (ZH)**: 一种分类器如果其每个决策都有一个足够小且易于人类用户理解的解释，则被认为是可解释的。DNF公式可以被视为布尔域上的二元分类器κ。DNF κ的一个正决策的解释大小受κ中项的大小限制，因为我们可以通过给出κ中评估为真的项来解释正决策。既然正决策和负决策都必须进行解释，我们考虑可解释的DNF是那些κ及其补集都可以用大小受限的项组成的DNF表示的κ。在本文中，我们研究一类其补集也可以用k-DNF表示的k-DNF家族。我们将比较两类这样的家族，即深度为k的决策树和嵌套k-DNF，这是一种新型模型。实验表明，与决策树相比，嵌套k-DNF在可解释性和准确性方面都是一个有趣的选择。 

---
# Controllable Logical Hypothesis Generation for Abductive Reasoning in Knowledge Graphs 

**Title (ZH)**: 可控制的逻辑假设生成以实现知识图谱中的归纳推理 

**Authors**: Yisen Gao, Jiaxin Bai, Tianshi Zheng, Qingyun Sun, Ziwei Zhang, Jianxin Li, Yangqiu Song, Xingcheng Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20948)  

**Abstract**: Abductive reasoning in knowledge graphs aims to generate plausible logical hypotheses from observed entities, with broad applications in areas such as clinical diagnosis and scientific discovery. However, due to a lack of controllability, a single observation may yield numerous plausible but redundant or irrelevant hypotheses on large-scale knowledge graphs. To address this limitation, we introduce the task of controllable hypothesis generation to improve the practical utility of abductive reasoning. This task faces two key challenges when controlling for generating long and complex logical hypotheses: hypothesis space collapse and hypothesis oversensitivity. To address these challenges, we propose CtrlHGen, a Controllable logcial Hypothesis Generation framework for abductive reasoning over knowledge graphs, trained in a two-stage paradigm including supervised learning and subsequent reinforcement learning. To mitigate hypothesis space collapse, we design a dataset augmentation strategy based on sub-logical decomposition, enabling the model to learn complex logical structures by leveraging semantic patterns in simpler components. To address hypothesis oversensitivity, we incorporate smoothed semantic rewards including Dice and Overlap scores, and introduce a condition-adherence reward to guide the generation toward user-specified control constraints. Extensive experiments on three benchmark datasets demonstrate that our model not only better adheres to control conditions but also achieves superior semantic similarity performance compared to baselines. 

**Abstract (ZH)**: 知识图谱中的可控演绎推理旨在从观察实体中生成 plausible 的逻辑假设，广泛应用于临床诊断和科学发现等领域。然而，由于可控性不足，单一观察可能在大规模知识图谱中生成大量但冗余或不相关的有效假设。为解决这一局限，我们引入可控假设生成任务以提高演绎推理的实用价值。该任务在控制生成长且复杂的逻辑假设时面临两个关键挑战：假设空间坍塌和假设过度敏感。为应对这些挑战，我们提出了一种名为 CtrlHGen 的可控逻辑假设生成框架，用于知识图谱上的演绎推理，该框架采用包含监督学习和随后的强化学习的两阶段训练方式。为缓解假设空间坍塌，我们设计了一种基于子逻辑分解的数据集增强策略，使模型能够通过利用简单组件中的语义模式来学习复杂的逻辑结构。为应对假设过度敏感，我们引入了平滑语义奖励（包括 Dice 和 Overlap 分数）和条件一致性奖励，以指导生成符合用户指定控制约束的方向。在三个基准数据集上的广泛实验表明，我们的模型不仅更好地符合控制条件，还实现了优于基线的语义相似度性能。 

---
# E2E Process Automation Leveraging Generative AI and IDP-Based Automation Agent: A Case Study on Corporate Expense Processing 

**Title (ZH)**: 基于生成AI和基于IDP的自动化代理的端到端流程自动化：企业费用处理案例研究 

**Authors**: Cheonsu Jeong, Seongmin Sim, Hyoyoung Cho, Sungsu Kim, Byounggwan Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20733)  

**Abstract**: This paper presents an intelligent work automation approach in the context of contemporary digital transformation by integrating generative AI and Intelligent Document Processing (IDP) technologies with an Automation Agent to realize End-to-End (E2E) automation of corporate financial expense processing tasks. While traditional Robotic Process Automation (RPA) has proven effective for repetitive, rule-based simple task automation, it faces limitations in handling unstructured data, exception management, and complex decision-making. This study designs and implements a four-stage integrated process comprising automatic recognition of supporting documents such as receipts via OCR/IDP, item classification based on a policy-driven database, intelligent exception handling supported by generative AI (large language models, LLMs), and human-in-the-loop final decision-making with continuous system learning through an Automation Agent. Applied to a major Korean enterprise (Company S), the system demonstrated quantitative benefits including over 80% reduction in processing time for paper receipt expense tasks, decreased error rates, and improved compliance, as well as qualitative benefits such as enhanced accuracy and consistency, increased employee satisfaction, and data-driven decision support. Furthermore, the system embodies a virtuous cycle by learning from human judgments to progressively improve automatic exception handling capabilities. Empirically, this research confirms that the organic integration of generative AI, IDP, and Automation Agents effectively overcomes the limitations of conventional automation and enables E2E automation of complex corporate processes. The study also discusses potential extensions to other domains such as accounting, human resources, and procurement, and proposes future directions for AI-driven hyper-automation development. 

**Abstract (ZH)**: 基于生成式AI和智能文档处理的当代数字转型背景下智能工作自动化方法：集成自动化代理实现企业财务费用处理任务的端到端自动化 

---
# GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning 

**Title (ZH)**: GIFARC：利用人类直觉类比提升AI推理的合成数据集 

**Authors**: Woochang Sim, Hyunseok Ryu, Kyungmin Choi, Sungwon Han, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20672)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) poses a stringent test of general AI capabilities, requiring solvers to infer abstract patterns from only a handful of examples. Despite substantial progress in deep learning, state-of-the-art models still achieve accuracy rates of merely 40-55% on 2024 ARC Competition, indicative of a significant gap between their performance and human-level reasoning. In this work, we seek to bridge that gap by introducing an analogy-inspired ARC dataset, GIFARC. Leveraging large language models (LLMs) and vision-language models (VLMs), we synthesize new ARC-style tasks from a variety of GIF images that include analogies. Each new task is paired with ground-truth analogy, providing an explicit mapping between visual transformations and everyday concepts. By embedding robust human-intuitive analogies into ARC-style tasks, GIFARC guides AI agents to evaluate the task analogically before engaging in brute-force pattern search, thus efficiently reducing problem complexity and build a more concise and human-understandable solution. We empirically validate that guiding LLM with analogic approach with GIFARC affects task-solving approaches of LLMs to align with analogic approach of human. 

**Abstract (ZH)**: 基于类比的ARC数据集GIFARC：通过融合大语言模型和视觉语言模型缩小通用AI性能与人类推理之间的差距 

---
# AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage 

**Title (ZH)**: 自动生成：基于论文 lineage 的自动 AI 实验重现 

**Authors**: Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi, Shuo Wang, Duzhen Zhang, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.20662)  

**Abstract**: Efficient experiment reproduction is critical to accelerating progress in artificial intelligence. However, the inherent complexity of method design and training procedures presents substantial challenges for automation. Notably, reproducing experiments often requires implicit domain-specific knowledge not explicitly documented in the original papers. To address this, we introduce the paper lineage algorithm, which identifies and extracts implicit knowledge from the relevant references cited by the target paper. Building on this idea, we propose AutoReproduce, a multi-agent framework capable of automatically reproducing experiments described in research papers in an end-to-end manner. AutoReproduce enhances code executability by generating unit tests alongside the reproduction process. To evaluate the reproduction capability, we construct ReproduceBench, a benchmark annotated with verified implementations, and introduce novel evaluation metrics to assess both the reproduction and execution fidelity. Experimental results demonstrate that AutoReproduce outperforms the existing strong agent baselines on all five evaluation metrics by a peak margin of over $70\%$. In particular, compared to the official implementations, AutoReproduce achieves an average performance gap of $22.1\%$ on $89.74\%$ of the executable experiment runs. The code will be available at this https URL. 

**Abstract (ZH)**: 高效的实验重现对于加速人工智能的发展至关重要。然而，方法设计和训练过程的内在复杂性为自动化带来了巨大挑战。值得注意的是，重现实验通常需要隐含领域的特定知识，而这些知识并未在原始论文中明确记录。为了解决这个问题，我们提出了论文谱系算法，该算法能够识别并提取目标论文所引用的相关参考文献中的隐含知识。在此基础上，我们提出了AutoReproduce，这是一种能够端到端自动重现研究论文中描述的实验的多代理框架。AutoReproduce通过在重现过程中生成单元测试来提高代码的可执行性。为了评估重现能力，我们构建了带有验证实现的ReproduceBench基准，并引入了新的评估指标来评估重现和执行的准确性。实验结果表明，AutoReproduce在所有五个评估指标上均优于现有强基线代理，峰值领先优势超过70%。特别地，与官方实现相比，AutoReproduce在89.74%可执行实验运行中实现了平均性能差距22.1%。代码将在该网址处提供。 

---
# Machine Theory of Mind and the Structure of Human Values 

**Title (ZH)**: 机器心理论与人类价值观的结构 

**Authors**: Paul de Font-Reaulx  

**Link**: [PDF](https://arxiv.org/pdf/2505.20342)  

**Abstract**: Value learning is a crucial aspect of safe and ethical AI. This is primarily pursued by methods inferring human values from behaviour. However, humans care about much more than we are able to demonstrate through our actions. Consequently, an AI must predict the rest of our seemingly complex values from a limited sample. I call this the value generalization problem. In this paper, I argue that human values have a generative rational structure and that this allows us to solve the value generalization problem. In particular, we can use Bayesian Theory of Mind models to infer human values not only from behaviour, but also from other values. This has been obscured by the widespread use of simple utility functions to represent human values. I conclude that developing generative value-to-value inference is a crucial component of achieving a scalable machine theory of mind. 

**Abstract (ZH)**: 价值学习是实现安全和伦理AI的关键方面。这主要通过从行为中推断人类价值观的方法来追求。然而，人类关心的远不止我们能通过行为展现的内容。因此，AI必须从有限的样本中预测我们那些看似复杂的其余价值观。我称之为价值概括问题。在本文中，我认为人类价值观具有生成性的理性结构，这使得我们能够解决价值概括问题。特别是，我们可以使用贝叶斯理论心智模型，不仅从行为，也可以从其他价值观来推断人类价值观。这种观点因使用简单的效用函数来表示人类价值观而被掩盖。我得出结论，发展生成性价值到价值的推理是实现可扩展机器心智理论的关键组成部分。 

---
# Reasoning in Neurosymbolic AI 

**Title (ZH)**: 神经符号人工智能中的推理 

**Authors**: Son Tran, Edjard Mota, Artur d'Avila Garcez  

**Link**: [PDF](https://arxiv.org/pdf/2505.20313)  

**Abstract**: Knowledge representation and reasoning in neural networks have been a long-standing endeavor which has attracted much attention recently. The principled integration of reasoning and learning in neural networks is a main objective of the area of neurosymbolic Artificial Intelligence (AI). In this chapter, a simple energy-based neurosymbolic AI system is described that can represent and reason formally about any propositional logic formula. This creates a powerful combination of learning from data and knowledge and logical reasoning. We start by positioning neurosymbolic AI in the context of the current AI landscape that is unsurprisingly dominated by Large Language Models (LLMs). We identify important challenges of data efficiency, fairness and safety of LLMs that might be addressed by neurosymbolic reasoning systems with formal reasoning capabilities. We then discuss the representation of logic by the specific energy-based system, including illustrative examples and empirical evaluation of the correspondence between logical reasoning and energy minimization using Restricted Boltzmann Machines (RBM). Learning from data and knowledge is also evaluated empirically and compared with a symbolic, neural and a neurosymbolic system. Results reported in this chapter in an accessible way are expected to reignite the research on the use of neural networks as massively-parallel models for logical reasoning and promote the principled integration of reasoning and learning in deep networks. We conclude the chapter with a discussion of the importance of positioning neurosymbolic AI within a broader framework of formal reasoning and accountability in AI, discussing the challenges for neurosynbolic AI to tackle the various known problems of reliability of deep learning. 

**Abstract (ZH)**: 神经网络中的知识表示与推理一直是长期的研究课题，近年来引起了广泛关注。神经符号人工智能（AI）领域的主要目标是原理性地将推理与学习集成在神经网络中。在本章中，描述了一个简单的基于能量的神经符号AI系统，它可以形式化地表示和推理任何命题逻辑公式。这创造了一种强大的组合——从数据和知识中学习以及进行逻辑推理。我们首先将神经符号AI置于当前AI panorama中，该领域不令人意外地被大型语言模型（LLMs）所主导。我们识别了大型语言模型的数据效率、公平性和安全性等重要挑战，这些问题可能通过具备形式推理能力的神经符号推理系统来解决。然后，我们讨论了该特定能量基系统对逻辑的表示，包括逻辑推理与能量最小化的对应关系示例以及限制玻尔兹曼机（RBM）的实验评估。从数据和知识中进行学习的实验评估也进行了比较，包括符号、神经和神经符号系统。本章以易于理解的方式报告的结果预计将重新激发对使用神经网络作为大规模并行逻辑推理模型的研究，并促进深层网络中推理与学习的原理性集成。最后，我们在更广泛的形式推理和AI问责框架中讨论了神经符号AI的重要性，并探讨了神经符号AI面临的挑战，以应对深度学习可靠性方面的各种已知问题。 

---
# AdInject: Real-World Black-Box Attacks on Web Agents via Advertising Delivery 

**Title (ZH)**: AdInject：通过广告分发对网络代理进行现实世界的黑盒攻击 

**Authors**: Haowei Wang, Junjie Wang, Xiaojun Jia, Rupeng Zhang, Mingyang Li, Zhe Liu, Yang Liu, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21499)  

**Abstract**: Vision-Language Model (VLM) based Web Agents represent a significant step towards automating complex tasks by simulating human-like interaction with websites. However, their deployment in uncontrolled web environments introduces significant security vulnerabilities. Existing research on adversarial environmental injection attacks often relies on unrealistic assumptions, such as direct HTML manipulation, knowledge of user intent, or access to agent model parameters, limiting their practical applicability. In this paper, we propose AdInject, a novel and real-world black-box attack method that leverages the internet advertising delivery to inject malicious content into the Web Agent's environment. AdInject operates under a significantly more realistic threat model than prior work, assuming a black-box agent, static malicious content constraints, and no specific knowledge of user intent. AdInject includes strategies for designing malicious ad content aimed at misleading agents into clicking, and a VLM-based ad content optimization technique that infers potential user intents from the target website's context and integrates these intents into the ad content to make it appear more relevant or critical to the agent's task, thus enhancing attack effectiveness. Experimental evaluations demonstrate the effectiveness of AdInject, attack success rates exceeding 60% in most scenarios and approaching 100% in certain cases. This strongly demonstrates that prevalent advertising delivery constitutes a potent and real-world vector for environment injection attacks against Web Agents. This work highlights a critical vulnerability in Web Agent security arising from real-world environment manipulation channels, underscoring the urgent need for developing robust defense mechanisms against such threats. Our code is available at this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的网络代理中的黑盒注入攻击：利用互联网广告投放注入恶意内容 

---
# VoxAging: Continuously Tracking Speaker Aging with a Large-Scale Longitudinal Dataset in English and Mandarin 

**Title (ZH)**: VoxAging：大规模 longitudinal 数据集中文英文 Continuous 推进说话人老化跟踪 

**Authors**: Zhiqi Ai, Meixuan Bao, Zhiyong Chen, Zhi Yang, Xinnuo Li, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21445)  

**Abstract**: The performance of speaker verification systems is adversely affected by speaker aging. However, due to challenges in data collection, particularly the lack of sustained and large-scale longitudinal data for individuals, research on speaker aging remains difficult. In this paper, we present VoxAging, a large-scale longitudinal dataset collected from 293 speakers (226 English speakers and 67 Mandarin speakers) over several years, with the longest time span reaching 17 years (approximately 900 weeks). For each speaker, the data were recorded at weekly intervals. We studied the phenomenon of speaker aging and its effects on advanced speaker verification systems, analyzed individual speaker aging processes, and explored the impact of factors such as age group and gender on speaker aging research. 

**Abstract (ZH)**: speaker验证系统受说话人老化影响的研究——基于VoxAging大规模 longitudinal数据集 

---
# Autoencoding Random Forests 

**Title (ZH)**: 自动编码随机森林 

**Authors**: Binh Duc Vu, Jan Kapar, Marvin Wright, David S. Watson  

**Link**: [PDF](https://arxiv.org/pdf/2505.21441)  

**Abstract**: We propose a principled method for autoencoding with random forests. Our strategy builds on foundational results from nonparametric statistics and spectral graph theory to learn a low-dimensional embedding of the model that optimally represents relationships in the data. We provide exact and approximate solutions to the decoding problem via constrained optimization, split relabeling, and nearest neighbors regression. These methods effectively invert the compression pipeline, establishing a map from the embedding space back to the input space using splits learned by the ensemble's constituent trees. The resulting decoders are universally consistent under common regularity assumptions. The procedure works with supervised or unsupervised models, providing a window into conditional or joint distributions. We demonstrate various applications of this autoencoder, including powerful new tools for visualization, compression, clustering, and denoising. Experiments illustrate the ease and utility of our method in a wide range of settings, including tabular, image, and genomic data. 

**Abstract (ZH)**: 我们提出了一种基于随机森林的自动编码方法。该策略构建于非参数统计和谱图理论的基础结果之上，用于学习一个低维嵌入，该嵌入优化地表示数据中的关系。我们通过受限优化、分裂重新标记和最近邻回归提供了解码问题的精确和近似解。这些方法有效地逆向了压缩管道，使用集成中个体树学习到的分裂，在嵌入空间与输入空间之间建立映射。所得到的解码器在常见的正则性假设下具有普遍一致性。该过程既可以用于监督模型也可以用于无监督模型，提供了条件分布或联合分布的窗口。我们展示了该自动编码器的各种应用，包括用于可视化、压缩、聚类和去噪的强有力新工具。实验在包括表格数据、图像数据和基因组数据在内的多种场景下表明了我们方法的便捷性和实用性。 

---
# Leveraging the Power of Conversations: Optimal Key Term Selection in Conversational Contextual Bandits 

**Title (ZH)**: 挖掘对话的力量：对话情境_bandits中最佳关键词选择 

**Authors**: Maoli Liu, Zhuohua Li, Xiangxiang Dai, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2505.21393)  

**Abstract**: Conversational recommender systems proactively query users with relevant "key terms" and leverage the feedback to elicit users' preferences for personalized recommendations. Conversational contextual bandits, a prevalent approach in this domain, aim to optimize preference learning by balancing exploitation and exploration. However, several limitations hinder their effectiveness in real-world scenarios. First, existing algorithms employ key term selection strategies with insufficient exploration, often failing to thoroughly probe users' preferences and resulting in suboptimal preference estimation. Second, current algorithms typically rely on deterministic rules to initiate conversations, causing unnecessary interactions when preferences are well-understood and missed opportunities when preferences are uncertain. To address these limitations, we propose three novel algorithms: CLiSK, CLiME, and CLiSK-ME. CLiSK introduces smoothed key term contexts to enhance exploration in preference learning, CLiME adaptively initiates conversations based on preference uncertainty, and CLiSK-ME integrates both techniques. We theoretically prove that all three algorithms achieve a tighter regret upper bound of $O(\sqrt{dT\log{T}})$ with respect to the time horizon $T$, improving upon existing methods. Additionally, we provide a matching lower bound $\Omega(\sqrt{dT})$ for conversational bandits, demonstrating that our algorithms are nearly minimax optimal. Extensive evaluations on both synthetic and real-world datasets show that our approaches achieve at least a 14.6% improvement in cumulative regret. 

**Abstract (ZH)**: 基于对话的推荐系统主动查询与用户相关的关键术语，并利用反馈来推断用户的偏好以实现个性化的推荐。对话上下文臂赛局，这一领域的主流方法旨在通过权衡探索与利用来优化偏好学习。然而，现有方法的几个局限性限制了其在实际场景中的效果。首先，现有算法的关键术语选择策略探索不足，往往未能充分探查用户的偏好，导致偏好估计欠佳。其次，当前算法通常依赖于确定性规则启动对话，当偏好已明了时引起不必要的互动，并在偏好不确定时错过机会。为解决这些问题，我们提出三种新型算法：CLiSK、CLiME和CLiSK-ME。CLiSK引入平滑的关键术语上下文以增强偏好学习中的探索，CLiME根据偏好不确定性适应性地启动对话，而CLiSK-ME结合了这两种技术。我们理论上证明，所有三种算法相对时间范围T实现了更紧的后悔上界$O(\sqrt{dT\log{T}})$，优于现有方法。此外，我们为对话臂赛局提供了匹配的下界$\Omega(\sqrt{dT})$，表明我们的算法几乎达到最小最大最优。广泛的合成和现实世界数据集评估表明，我们的方法在累积后悔上至少实现了14.6%的提升。 

---
# Finite Sample Analysis of Linear Temporal Difference Learning with Arbitrary Features 

**Title (ZH)**: 有限样本分析任意特征下的线性时序差分学习 

**Authors**: Zixuan Xie, Xinyu Liu, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21391)  

**Abstract**: Linear TD($\lambda$) is one of the most fundamental reinforcement learning algorithms for policy evaluation. Previously, convergence rates are typically established under the assumption of linearly independent features, which does not hold in many practical scenarios. This paper instead establishes the first $L^2$ convergence rates for linear TD($\lambda$) operating under arbitrary features, without making any algorithmic modification or additional assumptions. Our results apply to both the discounted and average-reward settings. To address the potential non-uniqueness of solutions resulting from arbitrary features, we develop a novel stochastic approximation result featuring convergence rates to the solution set instead of a single point. 

**Abstract (ZH)**: 线性TD($\lambda$)是用于策略评估的基本强化学习算法之一。以往的收敛速率通常是基于特征线性独立的假设建立的，这种假设在许多实际场景中并不成立。本文首次为在任意特征下运行的线性TD($\lambda$)建立了$L^2$收敛速率，无需进行任何算法修改或额外假设。我们的结果适用于折扣奖励和平均奖励两种设置。为解决由于任意特征可能导致的解的非唯一性问题，我们开发了一种新的随机逼近结果，该结果的方向是收敛到解集而不是单一点。 

---
# DeSocial: Blockchain-based Decentralized Social Networks 

**Title (ZH)**: 去中心化社会网络：基于区块链的去中心化社交网络 

**Authors**: Jingyuan Huang, Xi Zhu, Minghao Guo, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21388)  

**Abstract**: Web 2.0 social platforms are inherently centralized, with user data and algorithmic decisions controlled by the platform. However, users can only passively receive social predictions without being able to choose the underlying algorithm, which limits personalization. Fortunately, with the emergence of blockchain, users are allowed to choose algorithms that are tailored to their local situation, improving prediction results in a personalized way. In a blockchain environment, each user possesses its own model to perform the social prediction, capturing different perspectives on social interactions. In our work, we propose DeSocial, a decentralized social network learning framework deployed on an Ethereum (ETH) local development chain that integrates distributed data storage, node-level consensus, and user-driven model selection through Ganache. In the first stage, each user leverages DeSocial to evaluate multiple backbone models on their local subgraph. DeSocial coordinates the execution and returns model-wise prediction results, enabling the user to select the most suitable backbone for personalized social prediction. Then, DeSocial uniformly selects several validation nodes that possess the algorithm specified by each user, and aggregates the prediction results by majority voting, to prevent errors caused by any single model's misjudgment. Extensive experiments show that DeSocial has an evident improvement compared to the five classical centralized social network learning models, promoting user empowerment in blockchain-based decentralized social networks, showing the importance of multi-node validation and personalized algorithm selection based on blockchain. Our implementation is available at: this https URL. 

**Abstract (ZH)**: 基于区块链的去中心化社会网络学习框架DeSocial 

---
# Subgroups Matter for Robust Bias Mitigation 

**Title (ZH)**: 亚群体对于稳健的偏见缓解很重要 

**Authors**: Anissa Alloula, Charles Jones, Ben Glocker, Bartłomiej W. Papież  

**Link**: [PDF](https://arxiv.org/pdf/2505.21363)  

**Abstract**: Despite the constant development of new bias mitigation methods for machine learning, no method consistently succeeds, and a fundamental question remains unanswered: when and why do bias mitigation techniques fail? In this paper, we hypothesise that a key factor may be the often-overlooked but crucial step shared by many bias mitigation methods: the definition of subgroups. To investigate this, we conduct a comprehensive evaluation of state-of-the-art bias mitigation methods across multiple vision and language classification tasks, systematically varying subgroup definitions, including coarse, fine-grained, intersectional, and noisy subgroups. Our results reveal that subgroup choice significantly impacts performance, with certain groupings paradoxically leading to worse outcomes than no mitigation at all. Our findings suggest that observing a disparity between a set of subgroups is not a sufficient reason to use those subgroups for mitigation. Through theoretical analysis, we explain these phenomena and uncover a counter-intuitive insight that, in some cases, improving fairness with respect to a particular set of subgroups is best achieved by using a different set of subgroups for mitigation. Our work highlights the importance of careful subgroup definition in bias mitigation and suggest it as a alternative lever for improving the robustness and fairness of machine learning models. 

**Abstract (ZH)**: 尽管机器学习中新偏见缓解方法不断开发，但没有任何方法能够始终如一地成功，一个基本问题仍未解答：何时以及为何偏见缓解技术会失效？在本文中，我们假设一个关键因素可能是许多偏见缓解方法中经常被忽视但至关重要的步骤：子组定义。为了调查这一点，我们在多个视觉和语言分类任务中系统地评估了当前最先进的偏见缓解方法，并系统地变化子组定义，包括粗糙、精细、交叉和嘈杂子组。我们的结果表明，子组选择对性能有显著影响，某些分组 paradoxically 导致比不进行任何缓解更差的结果。我们的研究结果表明，观察一组子组之间存在差异并不足以使用这些子组进行缓解。通过理论分析，我们解释了这些现象，并揭示了一个反直觉的见解，即在某些情况下，针对特定一组子组提高公平性，最好的方法是使用另一组子组进行缓解。本文强调了在偏见缓解中仔细定义子组的重要性，并建议将其作为增强机器学习模型的稳健性和公平性的替代杠杆。 

---
# Prostate Cancer Screening with Artificial Intelligence-Enhanced Micro-Ultrasound: A Comparative Study with Traditional Methods 

**Title (ZH)**: 人工智能增强微超声在前列腺癌筛查中的应用：与传统方法的比较研究 

**Authors**: Muhammad Imran, Wayne G. Brisbane, Li-Ming Su, Jason P. Joseph, Wei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21355)  

**Abstract**: Background and objective: Micro-ultrasound (micro-US) is a novel imaging modality with diagnostic accuracy comparable to MRI for detecting clinically significant prostate cancer (csPCa). We investigated whether artificial intelligence (AI) interpretation of micro-US can outperform clinical screening methods using PSA and digital rectal examination (DRE). Methods: We retrospectively studied 145 men who underwent micro-US guided biopsy (79 with csPCa, 66 without). A self-supervised convolutional autoencoder was used to extract deep image features from 2D micro-US slices. Random forest classifiers were trained using five-fold cross-validation to predict csPCa at the slice level. Patients were classified as csPCa-positive if 88 or more consecutive slices were predicted positive. Model performance was compared with a classifier using PSA, DRE, prostate volume, and age. Key findings and limitations: The AI-based micro-US model and clinical screening model achieved AUROCs of 0.871 and 0.753, respectively. At a fixed threshold, the micro-US model achieved 92.5% sensitivity and 68.1% specificity, while the clinical model showed 96.2% sensitivity but only 27.3% specificity. Limitations include a retrospective single-center design and lack of external validation. Conclusions and clinical implications: AI-interpreted micro-US improves specificity while maintaining high sensitivity for csPCa detection. This method may reduce unnecessary biopsies and serve as a low-cost alternative to PSA-based screening. Patient summary: We developed an AI system to analyze prostate micro-ultrasound images. It outperformed PSA and DRE in detecting aggressive cancer and may help avoid unnecessary biopsies. 

**Abstract (ZH)**: 背景与目的：微超声（micro-US）是一种与磁共振成像（MRI）具有相似诊断准确性的新成像技术，用于检测临床显著前列腺癌（csPCa）。我们研究了人工智能（AI）解释微超声是否能在前列腺特异性抗原（PSA）和直肠指检（DRE）的临床筛查方法上表现出更优的性能。方法：回顾性研究了145名接受微超声引导活检的男性（其中79人患有csPCa，66人未患有csPCa）。使用自监督卷积自动编码器从2D微超声切片中提取深度图像特征。采用五折交叉验证训练随机森林分类器，以在切片级别预测csPCa。如果预测为阳性的连续切片数达到88片以上，则患者被分类为csPCa阳性。将微超声模型的表现与使用PSA、DRE、前列腺体积和年龄的分类器进行比较。主要发现与局限性：基于AI的微超声模型和临床筛查模型分别获得了0.871和0.753的AUROC。固定阈值下，微超声模型的敏感性和特异性分别为92.5%和68.1%，而临床模型的敏感性为96.2%，但特异性仅为27.3%。局限性包括回顾性单中心设计和缺乏外部验证。结论与临床意义：AI解释的微超声提高了前列腺癌检测的特异性，同时保持高灵敏度。该方法可能减少不必要的活检，并作为基于PSA筛查的低成本替代方案。患者总结：我们开发了一种AI系统来分析前列腺微超声图像，其在检测侵袭性癌症方面优于PSA和DRE，并可能帮助避免不必要的活检。 

---
# An Uncertainty-Aware ED-LSTM for Probabilistic Suffix Prediction 

**Title (ZH)**: 一种aware不确定性ED-LSTM的概率后缀预测模型 

**Authors**: Henryk Mustroph, Michel Kunkler, Stefanie Rinderle-Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.21339)  

**Abstract**: Suffix prediction of business processes forecasts the remaining sequence of events until process completion. Current approaches focus on predicting a single, most likely suffix. However, if the future course of a process is exposed to uncertainty or has high variability, the expressiveness of a single suffix prediction can be limited. To address this limitation, we propose probabilistic suffix prediction, a novel approach that approximates a probability distribution of suffixes. The proposed approach is based on an Uncertainty-Aware Encoder-Decoder LSTM (U-ED-LSTM) and a Monte Carlo (MC) suffix sampling algorithm. We capture epistemic uncertainties via MC dropout and aleatoric uncertainties as learned loss attenuation. This technical report provides a detailed evaluation of the U-ED-LSTM's predictive performance and assesses its calibration on four real-life event logs with three different hyperparameter settings. The results show that i) the U-ED-LSTM has reasonable predictive performance across various datasets, ii) aggregating probabilistic suffix predictions into mean values can outperform most likely predictions, particularly for rare prefixes or longer suffixes, and iii) the approach effectively captures uncertainties present in event logs. 

**Abstract (ZH)**: 企业流程的后缀预测 forecasting the remaining sequence of events until process completion through probabilistic suffix prediction 

---
# Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks 

**Title (ZH)**: 数据湖中存在问题：对表联合查询基准测试的一项批判性重新评估 

**Authors**: Allaa Boutaleb, Bernd Amann, Hubert Naacke, Rafael Angarita  

**Link**: [PDF](https://arxiv.org/pdf/2505.21329)  

**Abstract**: Recent table representation learning and data discovery methods tackle table union search (TUS) within data lakes, which involves identifying tables that can be unioned with a given query table to enrich its content. These methods are commonly evaluated using benchmarks that aim to assess semantic understanding in real-world TUS tasks. However, our analysis of prominent TUS benchmarks reveals several limitations that allow simple baselines to perform surprisingly well, often outperforming more sophisticated approaches. This suggests that current benchmark scores are heavily influenced by dataset-specific characteristics and fail to effectively isolate the gains from semantic understanding. To address this, we propose essential criteria for future benchmarks to enable a more realistic and reliable evaluation of progress in semantic table union search. 

**Abstract (ZH)**: 近期的表表示学习和数据发现方法在数据湖中处理表联查搜索（TUS）任务，涉及识别可以与给定查询表联查的表以丰富其内容。这些方法通常使用旨在评估实际TUS任务中语义理解的基准进行评估。然而，我们对主要的TUS基准的分析揭示了几种局限性，使简单的基线能够表现出乎意料的好性能，往往优于更为复杂的方案。这表明当前的基准分数受数据集特定特征的影响较大，未能有效地隔离语义理解带来的提升。为此，我们提出未来的基准应具备必要的标准，以实现更现实和可靠的语言理解表联查搜索进展评估。 

---
# A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features 

**Title (ZH)**: 基于形态学特征的跨模态知识精炼与数据增强方法以改善转录组表示 

**Authors**: Ihab Bendidi, Yassir El Mesbahi, Alisandra K. Denton, Karush Suri, Kian Kenyon-Dean, Auguste Genovesio, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21317)  

**Abstract**: Understanding cellular responses to stimuli is crucial for biological discovery and drug development. Transcriptomics provides interpretable, gene-level insights, while microscopy imaging offers rich predictive features but is harder to interpret. Weakly paired datasets, where samples share biological states, enable multimodal learning but are scarce, limiting their utility for training and multimodal inference. We propose a framework to enhance transcriptomics by distilling knowledge from microscopy images. Using weakly paired data, our method aligns and binds modalities, enriching gene expression representations with morphological information. To address data scarcity, we introduce (1) Semi-Clipped, an adaptation of CLIP for cross-modal distillation using pretrained foundation models, achieving state-of-the-art results, and (2) PEA (Perturbation Embedding Augmentation), a novel augmentation technique that enhances transcriptomics data while preserving inherent biological information. These strategies improve the predictive power and retain the interpretability of transcriptomics, enabling rich unimodal representations for complex biological tasks. 

**Abstract (ZH)**: 通过显微镜图像提炼知识以增强转录组学分析：利用弱配对数据实现模式识别和信息保留 

---
# GSAT: Graph Structure Attention Networks 

**Title (ZH)**: GSAT: 图结构注意力网络 

**Authors**: Farshad Noravesh, Reza Haffari, Layki Soon, Arghya Pal  

**Link**: [PDF](https://arxiv.org/pdf/2505.21288)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful tool for processing data represented in graph structures, achieving remarkable success across a wide range of applications. However, to further improve the performance on graph classification benchmarks, structural representation of each node that encodes rich local topological information in the neighbourhood of nodes is an important type of feature that is often overlooked in the modeling. The consequence of neglecting the structural information has resulted high number of layers to connect messages from distant nodes which by itself produces other problems such as oversmoothing. In the present paper, we leverage these structural information that are modeled by anonymous random walks (ARWs) and introduce graph structure attention network (GSAT) which is a generalization of graph attention network(GAT) to integrate the original attribute and the structural representation to enforce the model to automatically find patterns for attending to different edges in the node neighbourhood to enrich graph representation. Our experiments show GSAT slightly improves SOTA on some graph classification benchmarks. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为处理图结构表示数据的强大工具，在广泛的应用中取得了显著成功。然而，为了进一步提高图分类基准上的性能，每个节点的结构表示能够编码节点 neighborhood 中丰富的局部拓扑信息，这种类型的特征在建模中经常被忽略。忽视结构信息导致需要大量层次来连接远距离节点的消息，这本身会产生诸如过度平滑等问题。本文利用由匿名随机游走（ARWs）建模的这些结构信息，并引入图结构注意网络（GSAT），这是一种图注意力网络（GAT）的扩展，用于整合原始属性和结构表示，促使模型自动发现关注节点 neighborhood 中不同边的模式，以丰富图表示。我们的实验表明，GSAT在某些图分类基准上稍微提升了当前最佳性能。 

---
# Multilingual Pretraining for Pixel Language Models 

**Title (ZH)**: 多语言像素语言模型的预训练 

**Authors**: Ilker Kesen, Jonas F. Lotz, Ingo Ziegler, Phillip Rust, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2505.21265)  

**Abstract**: Pixel language models operate directly on images of rendered text, eliminating the need for a fixed vocabulary. While these models have demonstrated strong capabilities for downstream cross-lingual transfer, multilingual pretraining remains underexplored. We introduce PIXEL-M4, a model pretrained on four visually and linguistically diverse languages: English, Hindi, Ukrainian, and Simplified Chinese. Multilingual evaluations on semantic and syntactic tasks show that PIXEL-M4 outperforms an English-only counterpart on non-Latin scripts. Word-level probing analyses confirm that PIXEL-M4 captures rich linguistic features, even in languages not seen during pretraining. Furthermore, an analysis of its hidden representations shows that multilingual pretraining yields a semantic embedding space closely aligned across the languages used for pretraining. This work demonstrates that multilingual pretraining substantially enhances the capability of pixel language models to effectively support a diverse set of languages. 

**Abstract (ZH)**: 像素语言模型直接在渲染文本的图像上操作，无需固定词汇表。虽然这些模型在下游跨语言迁移任务中展示了强大的能力，但多语言预训练仍然未被充分探索。我们引入PIXEL-M4模型，该模型在四种视觉和语料上差异较大的语言上进行预训练：英语、印地语、乌克兰语和简体中文。多语言评估结果显示，PIXEL-M4在非拉丁字母文字上优于仅基于英语的对照组。词级探针分析证实，即使是在预训练中未见过的语言，PIXEL-M4也能够捕捉丰富的语言特征。此外，对其隐藏表示的分析显示，多语言预训练生成了一个紧密对齐的语义嵌入空间，与预训练所用语言高度一致。这项工作证明，多语言预训练显著增强了像素语言模型跨多种语言的有效支持能力。 

---
# Breaking the Performance Ceiling in Complex Reinforcement Learning requires Inference Strategies 

**Title (ZH)**: 在复杂强化学习中突破性能天花板需要推理策略 

**Authors**: Felix Chalumeau, Daniel Rajaonarivonivelomanantsoa, Ruan de Kock, Claude Formanek, Sasha Abramowitz, Oumayma Mahjoub, Wiem Khlifi, Simon Du Toit, Louay Ben Nessir, Refiloe Shabe, Arnol Fokam, Siddarth Singh, Ulrich Mbou Sob, Arnu Pretorius  

**Link**: [PDF](https://arxiv.org/pdf/2505.21236)  

**Abstract**: Reinforcement learning (RL) systems have countless applications, from energy-grid management to protein design. However, such real-world scenarios are often extremely difficult, combinatorial in nature, and require complex coordination between multiple agents. This level of complexity can cause even state-of-the-art RL systems, trained until convergence, to hit a performance ceiling which they are unable to break out of with zero-shot inference. Meanwhile, many digital or simulation-based applications allow for an inference phase that utilises a specific time and compute budget to explore multiple attempts before outputting a final solution. In this work, we show that such an inference phase employed at execution time, and the choice of a corresponding inference strategy, are key to breaking the performance ceiling observed in complex multi-agent RL problems. Our main result is striking: we can obtain up to a 126% and, on average, a 45% improvement over the previous state-of-the-art across 17 tasks, using only a couple seconds of extra wall-clock time during execution. We also demonstrate promising compute scaling properties, supported by over 60k experiments, making it the largest study on inference strategies for complex RL to date. Our experimental data and code are available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）系统在从能源电网管理到蛋白质设计等多个领域都有广泛的应用。然而，这些现实世界场景往往极其复杂，具有组合性质，并且需要多个代理之间的复杂协调。这种复杂性可能导致最先进的RL系统，在收敛训练后，即使在零样本推理的情况下，也无法突破性能瓶颈。同时，许多基于数字或模拟的应用程序允许在推理阶段利用特定的时间和计算预算，在尝试多次之后输出最终解决方案。在本研究中，我们表明，在执行时采用这样的推理阶段以及相应的推理策略选择，是打破复杂多代理RL问题中观察到的性能瓶颈的关键。我们的主要结果令人惊讶：仅通过在执行过程中额外使用几秒钟的 wall-clock 时间，我们就能在 17 个任务上将性能提升最高达 126%，平均提升 45%。我们还展示了令人鼓舞的计算可扩展性，由超过 60,000 次实验支持，使其成为迄今为止最大规模的针对复杂RL的推理策略研究。我们的实验数据和代码可在此处访问。 

---
# PSRB: A Comprehensive Benchmark for Evaluating Persian ASR Systems 

**Title (ZH)**: PSRB: 评估波斯语ASR系统的综合性基准 

**Authors**: Nima Sedghiyeh, Sara Sadeghi, Reza Khodadadi, Farzin Kashani, Omid Aghdaei, Somayeh Rahimi, Mohammad Sadegh Safari  

**Link**: [PDF](https://arxiv.org/pdf/2505.21230)  

**Abstract**: Although Automatic Speech Recognition (ASR) systems have become an integral part of modern technology, their evaluation remains challenging, particularly for low-resource languages such as Persian. This paper introduces Persian Speech Recognition Benchmark(PSRB), a comprehensive benchmark designed to address this gap by incorporating diverse linguistic and acoustic conditions. We evaluate ten ASR systems, including state-of-the-art commercial and open-source models, to examine performance variations and inherent biases. Additionally, we conduct an in-depth analysis of Persian ASR transcriptions, identifying key error types and proposing a novel metric that weights substitution errors. This metric enhances evaluation robustness by reducing the impact of minor and partial errors, thereby improving the precision of performance assessment. Our findings indicate that while ASR models generally perform well on standard Persian, they struggle with regional accents, children's speech, and specific linguistic challenges. These results highlight the necessity of fine-tuning and incorporating diverse, representative training datasets to mitigate biases and enhance overall ASR performance. PSRB provides a valuable resource for advancing ASR research in Persian and serves as a framework for developing benchmarks in other low-resource languages. A subset of the PSRB dataset is publicly available at this https URL. 

**Abstract (ZH)**: 尽管自动语音识别（ASR）系统已成为现代技术不可或缺的一部分，但其评估仍然具有挑战性，特别是在波斯语等低资源语言方面。本文介绍了波斯语语音识别基准（PSRB），这是一个旨在通过融入多样的语言和声学条件来填补这一空白的全面基准。我们评估了十个ASR系统，包括最先进的商业和开源模型，以检查性能差异和固有的偏见。此外，我们对波斯语ASR转录进行了深入分析，确定了关键的错误类型，并提出了一个新的度量标准，该标准对替换错误进行加权。该度量标准通过减少次要和部分错误的影响来增强评估的稳健性，从而提高性能评估的精确性。我们的研究发现，虽然ASR模型在标准波斯语上表现良好，但在地区口音、儿童的言语和特定的语言挑战方面存在困难。这些结果强调了微调和采用多样化的代表性训练数据集以缓解偏见并提高整体ASR性能的必要性。PSRB为推进波斯语ASR研究提供了宝贵资源，并为其他低资源语言开发基准提供了框架。部分PSRB数据集可在以下网址公开获取：this https URL。 

---
# Is Hyperbolic Space All You Need for Medical Anomaly Detection? 

**Title (ZH)**: 双曲空间是医疗异常检测所需的一切吗？ 

**Authors**: Alvaro Gonzalez-Jimenez, Simone Lionetti, Ludovic Amruthalingam, Philippe Gottfrois, Fabian Gröger, Marc Pouly, Alexander A. Navarini  

**Link**: [PDF](https://arxiv.org/pdf/2505.21228)  

**Abstract**: Medical anomaly detection has emerged as a promising solution to challenges in data availability and labeling constraints. Traditional methods extract features from different layers of pre-trained networks in Euclidean space; however, Euclidean representations fail to effectively capture the hierarchical relationships within these features, leading to suboptimal anomaly detection performance. We propose a novel yet simple approach that projects feature representations into hyperbolic space, aggregates them based on confidence levels, and classifies samples as healthy or anomalous. Our experiments demonstrate that hyperbolic space consistently outperforms Euclidean-based frameworks, achieving higher AUROC scores at both image and pixel levels across multiple medical benchmark datasets. Additionally, we show that hyperbolic space exhibits resilience to parameter variations and excels in few-shot scenarios, where healthy images are scarce. These findings underscore the potential of hyperbolic space as a powerful alternative for medical anomaly detection. The project website can be found at this https URL 

**Abstract (ZH)**: 医疗异常检测在数据可用性和标注约束挑战中 emerged as a promising solution. 传统方法在欧几里得空间中从预训练网络的不同层抽取特征；然而，欧几里得表示方式无法有效捕捉这些特征之间的层次关系，导致异常检测性能不佳。我们提出了一种新颖而简单的方法，将特征表示投影到双曲空间，在置信水平基础上进行聚合，并将样本分类为健康或异常。实验结果表明，双曲空间在图像和像素级别上始终优于基于欧几里得的空间框架，实现了多个医学基准数据集中的更高AUROC分数。此外，我们展示了双曲空间对参数变化具有鲁棒性，并在健康图像稀缺的少样本场景中表现优异。这些发现强调了双曲空间作为医疗异常检测有力替代方案的潜力。相关项目网站请访问 <https://>。 

---
# Addressing Data Quality Decompensation in Federated Learning via Dynamic Client Selection 

**Title (ZH)**: 基于动态客户端选择的联邦学习中数据质量递减问题解决方案 

**Authors**: Qinjun Fei, Nuria Rodríguez-Barroso, María Victoria Luzón, Zhongliang Zhang, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2505.21219)  

**Abstract**: In cross-silo Federated Learning (FL), client selection is critical to ensure high model performance, yet it remains challenging due to data quality decompensation, budget constraints, and incentive compatibility. As training progresses, these factors exacerbate client heterogeneity and degrade global performance. Most existing approaches treat these challenges in isolation, making jointly optimizing multiple factors difficult. To address this, we propose Shapley-Bid Reputation Optimized Federated Learning (SBRO-FL), a unified framework integrating dynamic bidding, reputation modeling, and cost-aware selection. Clients submit bids based on their perceived data quality, and their contributions are evaluated using Shapley values to quantify their marginal impact on the global model. A reputation system, inspired by prospect theory, captures historical performance while penalizing inconsistency. The client selection problem is formulated as a 0-1 integer program that maximizes reputation-weighted utility under budget constraints. Experiments on FashionMNIST, EMNIST, CIFAR-10, and SVHN datasets show that SBRO-FL improves accuracy, convergence speed, and robustness, even in adversarial and low-bid interference scenarios. Our results highlight the importance of balancing data reliability, incentive compatibility, and cost efficiency to enable scalable and trustworthy FL deployments. 

**Abstract (ZH)**: 跨孤岛联邦学习中的Shapley报价声誉优化联邦学习（SBRO-FL） 

---
# Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation 

**Title (ZH)**: 肺语：胸片结构化和序贯解释基准 

**Authors**: Jong Hak Moon, Geon Choi, Paloma Rabaey, Min Gwan Kim, Hyuk Gi Hong, Jung-Oh Lee, Hangyul Yoon, Eun Woo Doe, Jiyoun Kim, Harshita Sharma, Daniel C. Castro, Javier Alvarez-Valle, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21190)  

**Abstract**: Radiology reports convey detailed clinical observations and capture diagnostic reasoning that evolves over time. However, existing evaluation methods are limited to single-report settings and rely on coarse metrics that fail to capture fine-grained clinical semantics and temporal dependencies. We introduce LUNGUAGE,a benchmark dataset for structured radiology report generation that supports both single-report evaluation and longitudinal patient-level assessment across multiple studies. It contains 1,473 annotated chest X-ray reports, each reviewed by experts, and 80 of them contain longitudinal annotations to capture disease progression and inter-study intervals, also reviewed by experts. Using this benchmark, we develop a two-stage framework that transforms generated reports into fine-grained, schema-aligned structured representations, enabling longitudinal interpretation. We also propose LUNGUAGESCORE, an interpretable metric that compares structured outputs at the entity, relation, and attribute level while modeling temporal consistency across patient timelines. These contributions establish the first benchmark dataset, structuring framework, and evaluation metric for sequential radiology reporting, with empirical results demonstrating that LUNGUAGESCORE effectively supports structured report evaluation. The code is available at: this https URL 

**Abstract (ZH)**: 放射学报告传达详细的临床观察并捕捉随时间演变的诊断推理。然而，现有的评估方法仅限于单报告设置，并依赖于粗粒度指标，无法捕捉细微的临床语义和时间依赖性。我们引入了LUNGUAGE，一个支持单报告评估和多研究纵向患者水平评估的结构化放射学报告生成基准数据集。该数据集包含1,473份标注的胸部X光片报告，每份报告均由专家审阅，其中80份包含纵向标注以捕捉疾病进展和研究间间隔，同样由专家审阅。使用该基准数据集，我们开发了一种两阶段框架，将生成的报告转换为细粒度、模式对齐的结构化表示，从而实现纵向解释。我们还提出了LUNGUAGESCORE，这是一种可解释的指标，用于在实体、关系和属性级别比较结构化输出，并建模患者时间线上的时间一致性。这些贡献建立了第一个基准数据集、结构化框架和评估指标，用于序列放射学报告，并通过实验证明LUNGUAGESCORE有效支持结构化报告评估。代码可在以下链接获取：this https URL 

---
# Learning What to Do and What Not To Do: Offline Imitation from Expert and Undesirable Demonstrations 

**Title (ZH)**: 学习做什么和不做什么：从专家和不良示范中进行离线模仿学习 

**Authors**: Huy Hoang, Tien Mai, Pradeep Varakantham, Tanvi Verma  

**Link**: [PDF](https://arxiv.org/pdf/2505.21182)  

**Abstract**: Offline imitation learning typically learns from expert and unlabeled demonstrations, yet often overlooks the valuable signal in explicitly undesirable behaviors. In this work, we study offline imitation learning from contrasting behaviors, where the dataset contains both expert and undesirable demonstrations. We propose a novel formulation that optimizes a difference of KL divergences over the state-action visitation distributions of expert and undesirable (or bad) data. Although the resulting objective is a DC (Difference-of-Convex) program, we prove that it becomes convex when expert demonstrations outweigh undesirable demonstrations, enabling a practical and stable non-adversarial training objective. Our method avoids adversarial training and handles both positive and negative demonstrations in a unified framework. Extensive experiments on standard offline imitation learning benchmarks demonstrate that our approach consistently outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 离线模仿学习通常从专家和未标注的演示中学习，但往往会忽视明显不良行为中的有价值信号。在本文中，我们研究从对立行为中进行离线模仿学习，其中数据集包含专家和不良演示。我们提出了一种新的公式化方法，该方法在专家和不良（或糟糕）数据的状态-动作访问分布之间的KL距离差上进行优化。尽管所得目标是DC（凸性的差）规划，但我们证明当专家演示多于不良演示时，它变为凸的，从而使得一个实际可行且稳定的非对抗性训练目标成为可能。我们的方法避免了对抗性训练，并在一个统一框架中处理正负演示。在标准的离线模仿学习基准上的广泛实验表明，我们的方法在所有基准上都优于最先进的基线方法。 

---
# Latent label distribution grid representation for modeling uncertainty 

**Title (ZH)**: 潜在标签分布网格表示建模不确定性 

**Authors**: ShuNing Sun, YinSong Xiong, Yu Zhang, Zhuoran Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21180)  

**Abstract**: Although \textbf{L}abel \textbf{D}istribution \textbf{L}earning (LDL) has promising representation capabilities for characterizing the polysemy of an instance, the complexity and high cost of the label distribution annotation lead to inexact in the construction of the label space. The existence of a large number of inexact labels generates a label space with uncertainty, which misleads the LDL algorithm to yield incorrect decisions. To alleviate this problem, we model the uncertainty of label distributions by constructing a \textbf{L}atent \textbf{L}abel \textbf{D}istribution \textbf{G}rid (LLDG) to form a low-noise representation space. Specifically, we first construct a label correlation matrix based on the differences between labels, and then expand each value of the matrix into a vector that obeys a Gaussian distribution, thus building a LLDG to model the uncertainty of the label space. Finally, the LLDG is reconstructed by the LLDG-Mixer to generate an accurate label distribution. Note that we enforce a customized low-rank scheme on this grid, which assumes that the label relations may be noisy and it needs to perform noise-reduction with the help of a Tucker reconstruction technique. Furthermore, we attempt to evaluate the effectiveness of the LLDG by considering its generation as an upstream task to achieve the classification of the objects. Extensive experimental results show that our approach performs competitively on several benchmarks. 

**Abstract (ZH)**: 尽管标签分布学习（Label Distribution Learning, LDL）在刻画实例的多义性方面展现了强大的表示能力，但由于标签分布标注的复杂性和高成本导致标签空间构建不准确。大量不准确的标签的存在使得标签空间具有不确定性，从而误导LDL算法作出错误决策。为解决这一问题，我们通过构建隐标签分布网格（Latent Label Distribution Grid, LLDG）来建模标签分布的不确定性，从而形成低噪声的表示空间。具体而言，我们首先基于标签之间的差异构建标签相关矩阵，然后将矩阵中的每个值扩展为遵循高斯分布的向量，从而构建LLDG来建模标签空间的不确定性。最后，通过LLDG-Mixer重构LLDG生成准确的标签分布。值得注意的是，我们在该网格上施加了一个定制的低秩方案，假设标签关系可能是噪声的，并借助Tucker重建技术进行噪声减少。此外，我们尝试通过将LLDG的生成视为上游任务来实现对象分类，以评估其有效性。广泛的经验结果表明，我们的方法在多个基准上表现竞争性。 

---
# Quantum AIXI: Universal Intelligence via Quantum Information 

**Title (ZH)**: 量子AIXI：通过量子信息实现的通用智能 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2505.21170)  

**Abstract**: AIXI is a widely studied model of artificial general intelligence (AGI) based upon principles of induction and reinforcement learning. However, AIXI is fundamentally classical in nature - as are the environments in which it is modelled. Given the universe is quantum mechanical in nature and the exponential overhead required to simulate quantum mechanical systems classically, the question arises as to whether there are quantum mechanical analogues of AIXI which are theoretically consistent or practically feasible as models of universal intelligence. To address this question, we extend the framework to quantum information and present Quantum AIXI (QAIXI). We introduce a model of quantum agent/environment interaction based upon quantum and classical registers and channels, showing how quantum AIXI agents may take both classical and quantum actions. We formulate the key components of AIXI in quantum information terms, extending previous research on quantum Kolmogorov complexity and a QAIXI value function. We discuss conditions and limitations upon quantum Solomonoff induction and show how contextuality fundamentally affects QAIXI models. 

**Abstract (ZH)**: 量子AIXI（Quantum AIXI）：基于量子信息的通用人工智能模型 

---
# STEB: In Search of the Best Evaluation Approach for Synthetic Time Series 

**Title (ZH)**: STEB：搜索最适合合成时间序列的评估方法 

**Authors**: Michael Stenger, Robert Leppich, André Bauer, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2505.21160)  

**Abstract**: The growing need for synthetic time series, due to data augmentation or privacy regulations, has led to numerous generative models, frameworks, and evaluation measures alike. Objectively comparing these measures on a large scale remains an open challenge. We propose the Synthetic Time series Evaluation Benchmark (STEB) -- the first benchmark framework that enables comprehensive and interpretable automated comparisons of synthetic time series evaluation measures. Using 10 diverse datasets, randomness injection, and 13 configurable data transformations, STEB computes indicators for measure reliability and score consistency. It tracks running time, test errors, and features sequential and parallel modes of operation. In our experiments, we determine a ranking of 41 measures from literature and confirm that the choice of upstream time series embedding heavily impacts the final score. 

**Abstract (ZH)**: 合成时间序列评估基准（STEB）：首个全面可解释的合成时间序列评估指标自动对比框架 

---
# Model as Loss: A Self-Consistent Training Paradigm 

**Title (ZH)**: 模型即损失：一种自我一致的训练范式 

**Authors**: Saisamarth Rajesh Phaye, Milos Cernak, Andrew Harper  

**Link**: [PDF](https://arxiv.org/pdf/2505.21156)  

**Abstract**: Conventional methods for speech enhancement rely on handcrafted loss functions (e.g., time or frequency domain losses) or deep feature losses (e.g., using WavLM or wav2vec), which often fail to capture subtle signal properties essential for optimal performance. To address this, we propose Model as Loss, a novel training paradigm that utilizes the encoder from the same model as a loss function to guide the training.
The Model as Loss paradigm leverages the encoder's task-specific feature space, optimizing the decoder to produce output consistent with perceptual and task-relevant characteristics of the clean signal. By using the encoder's learned features as a loss function, this framework enforces self-consistency between the clean reference speech and the enhanced model output. Our approach outperforms pre-trained deep feature losses on standard speech enhancement benchmarks, offering better perceptual quality and robust generalization to both in-domain and out-of-domain datasets. 

**Abstract (ZH)**: 基于模型的损失函数在语音增强中的应用：一种新颖的训练范式 

---
# GGBond: Growing Graph-Based AI-Agent Society for Socially-Aware Recommender Simulation 

**Title (ZH)**: GGBond：基于图的AI代理社会增长模型Socially-aware推荐模拟 

**Authors**: Hailin Zhong, Hanlin Wang, Yujun Ye, Meiyi Zhang, Shengxin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21154)  

**Abstract**: Current personalized recommender systems predominantly rely on static offline data for algorithm design and evaluation, significantly limiting their ability to capture long-term user preference evolution and social influence dynamics in real-world scenarios. To address this fundamental challenge, we propose a high-fidelity social simulation platform integrating human-like cognitive agents and dynamic social interactions to realistically simulate user behavior evolution under recommendation interventions. Specifically, the system comprises a population of Sim-User Agents, each equipped with a five-layer cognitive architecture that encapsulates key psychological mechanisms, including episodic memory, affective state transitions, adaptive preference learning, and dynamic trust-risk assessments. In particular, we innovatively introduce the Intimacy--Curiosity--Reciprocity--Risk (ICR2) motivational engine grounded in psychological and sociological theories, enabling more realistic user decision-making processes. Furthermore, we construct a multilayer heterogeneous social graph (GGBond Graph) supporting dynamic relational evolution, effectively modeling users' evolving social ties and trust dynamics based on interest similarity, personality alignment, and structural homophily. During system operation, agents autonomously respond to recommendations generated by typical recommender algorithms (e.g., Matrix Factorization, MultVAE, LightGCN), deciding whether to consume, rate, and share content while dynamically updating their internal states and social connections, thereby forming a stable, multi-round feedback loop. This innovative design transcends the limitations of traditional static datasets, providing a controlled, observable environment for evaluating long-term recommender effects. 

**Abstract (ZH)**: 当前个性化推荐系统主要依赖静态离线数据进行算法设计和评估，极大地限制了其捕捉用户长期偏好演变和社会影响力动态的能力。为了应对这一根本性挑战，我们提出了一种高度真实的社会仿真平台，集成类人认知代理和动态社会互动，以真实模拟推荐干预下的用户行为演变。该系统包括一个Sim-User Agent群体，每个代理均配备五层认知架构，涵盖关键的心理机制，包括情景记忆、情绪状态转换、适应性偏好学习和动态信任风险评估。特别地，我们根据心理学和社会学理论创新性地引入了亲密性-好奇心-互惠-风险（ICR2）动机引擎，使用户决策过程更加真实。此外，我们构建了一个多层异质社会图（GGBond图），支持动态关系演变，有效地基于兴趣相似度、个性匹配和结构同质性建模用户的社交联系和信任动态。在系统运行期间，代理自主响应典型推荐算法（如矩阵分解、MultVAE、LightGCN）生成的推荐，决定是否消费、评价和分享内容，并动态更新其内部状态和社会联系，从而形成一个稳定、多轮的反馈循环。这种创新设计超越了传统静态数据集的限制，提供了一个可控、可观测的环境来评估推荐系统的长期效果。 

---
# HeteroBA: A Structure-Manipulating Backdoor Attack on Heterogeneous Graphs 

**Title (ZH)**: HeteroBA：一种针对异构图的结构操控后门攻击 

**Authors**: Honglin Gao, Xiang Li, Lan Zhao, Gaoxi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21140)  

**Abstract**: Heterogeneous graph neural networks (HGNNs) have recently drawn increasing attention for modeling complex multi-relational data in domains such as recommendation, finance, and social networks. While existing research has been largely focused on enhancing HGNNs' predictive performance, their robustness and security, especially under backdoor attacks, remain underexplored. In this paper, we propose a novel Heterogeneous Backdoor Attack (HeteroBA) framework for node classification tasks on heterogeneous graphs. HeteroBA inserts carefully crafted trigger nodes with realistic features and targeted structural connections, leveraging attention-based and clustering-based strategies to select influential auxiliary nodes for effective trigger propagation, thereby causing the model to misclassify specific nodes into a target label while maintaining accuracy on clean data. Experimental results on three datasets and various HGNN architectures demonstrate that HeteroBA achieves high attack success rates with minimal impact on the clean accuracy. Our method sheds light on potential vulnerabilities in HGNNs and calls for more robust defenses against backdoor threats in multi-relational graph scenarios. 

**Abstract (ZH)**: 异构图神经网络中的新型节点分类后门攻击框架（HeteroBA） 

---
# SageAttention2++: A More Efficient Implementation of SageAttention2 

**Title (ZH)**: SageAttention2++: 更高效的SageAttention2实现 

**Authors**: Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21136)  

**Abstract**: The efficiency of attention is critical because its time complexity grows quadratically with sequence length. SageAttention2 addresses this by utilizing quantization to accelerate matrix multiplications (Matmul) in attention. To further accelerate SageAttention2, we propose to utilize the faster instruction of FP8 Matmul accumulated in FP16. The instruction is 2x faster than the FP8 Matmul used in SageAttention2. Our experiments show that SageAttention2++ achieves a 3.9x speedup over FlashAttention while maintaining the same attention accuracy as SageAttention2. This means SageAttention2++ effectively accelerates various models, including those for language, image, and video generation, with negligible end-to-end metrics loss. The code will be available at this https URL. 

**Abstract (ZH)**: SageAttention2++通过利用FP8 Matmul在FP16中积累的更快指令实现加速，保持与SageAttention2相同的注意力准确性，速度提升3.9倍，对语言、图像和视频生成等各种模型实现有效加速，端到端指标损失可忽略不计。代码将在以下网址获取。 

---
# Universal Value-Function Uncertainties 

**Title (ZH)**: 普遍的价值函数不确定性 

**Authors**: Moritz A. Zanger, Max Weltevrede, Yaniv Oren, Pascal R. Van der Vaart, Caroline Horsch, Wendelin Böhmer, Matthijs T. J. Spaan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21119)  

**Abstract**: Estimating epistemic uncertainty in value functions is a crucial challenge for many aspects of reinforcement learning (RL), including efficient exploration, safe decision-making, and offline RL. While deep ensembles provide a robust method for quantifying value uncertainty, they come with significant computational overhead. Single-model methods, while computationally favorable, often rely on heuristics and typically require additional propagation mechanisms for myopic uncertainty estimates. In this work we introduce universal value-function uncertainties (UVU), which, similar in spirit to random network distillation (RND), quantify uncertainty as squared prediction errors between an online learner and a fixed, randomly initialized target network. Unlike RND, UVU errors reflect policy-conditional value uncertainty, incorporating the future uncertainties any given policy may encounter. This is due to the training procedure employed in UVU: the online network is trained using temporal difference learning with a synthetic reward derived from the fixed, randomly initialized target network. We provide an extensive theoretical analysis of our approach using neural tangent kernel (NTK) theory and show that in the limit of infinite network width, UVU errors are exactly equivalent to the variance of an ensemble of independent universal value functions. Empirically, we show that UVU achieves equal performance to large ensembles on challenging multi-task offline RL settings, while offering simplicity and substantial computational savings. 

**Abstract (ZH)**: 估计价值函数的知识不确定性是强化学习（RL）许多方面的关键挑战，包括高效探索、安全决策和 Offline RL。尽管深集成提供了一种稳健的方法来量化价值不确定性，但它们伴随着显著的计算开销。单模型方法虽然计算上更具优势，但往往依赖于启发式方法，并且通常需要额外的传播机制来处理短视的不确定性估计。在本文中，我们引入了通用价值函数不确定性（UVU），类似于随机网络蒸馏（RND），通过在线学习者与固定且随机初始化的目标网络之间的预测误差平方来量化不确定性。与 RND 不同，UVU 的误差反映了策略条件下的价值不确定性，包括任何给定策略可能遇到的未来不确定性。这归因于 UVU 的训练过程：在线网络使用基于固定且随机初始化的目标网络的合成奖励通过时差学习进行训练。我们使用神经 tangent 核（NTK）理论对我们的方法进行了详尽的理论分析，并证明在网络宽度无限的情况下，UVU 的误差恰好等同于独立通用价值函数集成的方差。实证结果表明，在复杂多任务 Offline RL 设置中，UVU 达到与大型集成同等的性能，同时具有简单性和显著的计算成本节约。 

---
# Stopping Criteria for Value Iteration on Concurrent Stochastic Reachability and Safety Games 

**Title (ZH)**: 并发随机可达性和安全性博弈中值迭代的停止准则 

**Authors**: Marta Grobelna, Jan Křetínský, Maximilian Weininger  

**Link**: [PDF](https://arxiv.org/pdf/2505.21087)  

**Abstract**: We consider two-player zero-sum concurrent stochastic games (CSGs) played on graphs with reachability and safety objectives. These include degenerate classes such as Markov decision processes or turn-based stochastic games, which can be solved by linear or quadratic programming; however, in practice, value iteration (VI) outperforms the other approaches and is the most implemented method. Similarly, for CSGs, this practical performance makes VI an attractive alternative to the standard theoretical solution via the existential theory of reals.
VI starts with an under-approximation of the sought values for each state and iteratively updates them, traditionally terminating once two consecutive approximations are $\epsilon$-close. However, this stopping criterion lacks guarantees on the precision of the approximation, which is the goal of this work. We provide bounded (a.k.a. interval) VI for CSGs: it complements standard VI with a converging sequence of over-approximations and terminates once the over- and under-approximations are $\epsilon$-close. 

**Abstract (ZH)**: 我们考虑在图上进行的两类玩家零和并发随机博弈（CSGs），这些博弈具有可达性和安全目标。这包括马尔可夫决策过程或轮流制随机博弈等退化类，这类问题可以通过线性或二次规划求解；然而在实践中，值迭代（VI）方法表现更优，并且是最常被实现的方法。类似地，在处理CSGs时，这种实用表现使得值迭代成为替代传统实存理论标准解法的有吸引力的选择。我们为CSGs提供有界（即区间）值迭代：这种迭代方法在标准的值迭代基础上加入了收敛的上近似序列，并且在上近似和下近似达到$\epsilon$-接近时终止。 

---
# Fixed-Point Traps and Identity Emergence in Educational Feedback Systems 

**Title (ZH)**: 固定点陷阱与身份 emergence 在教育反馈系统中的探究 

**Authors**: Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2505.21038)  

**Abstract**: This paper presents a formal categorical proof that exam-driven educational systems obstruct identity emergence and block creative convergence. Using the framework of Alpay Algebra II and III, we define Exam-Grade Collapse Systems (EGCS) as functorial constructs where learning dynamics $\varphi$ are recursively collapsed by evaluative morphisms $E$. We prove that under such collapse regimes, no nontrivial fixed-point algebra $\mu_\varphi$ can exist, hence learner identity cannot stabilize. This creates a universal fixed-point trap: all generative functors are entropically folded before symbolic emergence occurs. Our model mathematically explains the creativity suppression, research stagnation, and structural entropy loss induced by timed exams and grade-based feedback. The results apply category theory to expose why modern educational systems prevent {\phi}-emergence and block observer-invariant self-formation. This work provides the first provable algebraic obstruction of identity formation caused by institutional feedback mechanics. 

**Abstract (ZH)**: 本文正式证明了以考试为导向的教育体系阻碍身份认同的形成并阻滞创造性汇聚。基于Alpay代数II和III框架，我们将评估导向坍塌系统(EGCS)定义为由评价态映射$E$递归坍塌学习动力学$\varphi$的泛函构造。我们证明，在此类坍塌机制下，不存在非平凡的不动点代数$\mu_\varphi$，因而学习者身份无法稳定。这创建了一个普遍的不动点陷阱：所有生成泛函在符号出现之前均被熵性折叠。该模型从数学上解释了由定时考试和基于成绩的反馈所导致的创造力抑制、研究停滞和结构熵损失。本文将范畴论应用于揭示现代教育体系为何阻碍$\phi$-形成并阻止观察者不变的自我形成。本工作提供了首个因机构反馈机制导致身份形成代数障碍的证明。 

---
# TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data 

**Title (ZH)**: TabAttackBench：用于表格数据对抗攻击的基准测试 

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2505.21027)  

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data. 

**Abstract (ZH)**: adversarial 攻击对表格数据中的机器学习模型构成显著威胁，通过在输入数据上施加不可感知的扰动诱导错误预测。尽管这些攻击在图像等非结构化数据上得到了广泛研究，但将其应用于表格数据提出了新的挑战。这些挑战源于表格数据固有的异质性和复杂特征相互依赖性，这些特性与图像数据中的特性有显著差异。为了解决这些差异，将不可感知性视为特定于表格数据的关键标准至关重要。当前大部分研究主要集中在实现有效的 adversarial 攻击上，经常忽视保持不可感知性的的重要性。为了解决这一差距，我们提出了一种新的表格数据 adversarial 攻击基准，该基准同时评估有效性和不可感知性。在本研究中，我们使用包括混合和纯数值在内的十一个表格数据集评估四种模型下的五种 adversarial 攻击的有效性和不可感知性。我们的分析探讨了这些因素如何交互并影响攻击的整体性能。我们还对比了不同类型数据集的结果，以理解这些发现的更广泛影响。该基准的结果为改进 adversarial 攻击算法的设计提供了宝贵的见解，从而推动了基于表格数据的 adversarial 机器学习领域的发展。 

---
# Multi-Mode Process Control Using Multi-Task Inverse Reinforcement Learning 

**Title (ZH)**: 多模式过程控制的多任务逆强化学习 

**Authors**: Runze Lin, Junghui Chen, Biao Huang, Lei Xie, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.21026)  

**Abstract**: In the era of Industry 4.0 and smart manufacturing, process systems engineering must adapt to digital transformation. While reinforcement learning offers a model-free approach to process control, its applications are limited by the dependence on accurate digital twins and well-designed reward functions. To address these limitations, this paper introduces a novel framework that integrates inverse reinforcement learning (IRL) with multi-task learning for data-driven, multi-mode control design. Using historical closed-loop data as expert demonstrations, IRL extracts optimal reward functions and control policies. A latent-context variable is incorporated to distinguish modes, enabling the training of mode-specific controllers. Case studies on a continuous stirred tank reactor and a fed-batch bioreactor validate the effectiveness of this framework in handling multi-mode data and training adaptable controllers. 

**Abstract (ZH)**: 在工业4.0与智能制造时代，过程系统工程必须适应数字化转型。本文引入了一种将逆强化学习与多任务学习相结合的新框架，用于数据驱动的多模式控制设计。利用历史闭环数据作为专家演示，逆强化学习提取最优的奖励函数和控制策略，并通过引入潜在上下文变量来区分不同模式，实现模式特定控制器的训练。案例研究验证了该框架在处理多模式数据和训练可适应控制器方面的有效性。 

---
# Text-Queried Audio Source Separation via Hierarchical Modeling 

**Title (ZH)**: 基于层次建模的文本查询驱动音频源分离 

**Authors**: Xinlei Yin, Xiulian Peng, Xue Jiang, Zhiwei Xiong, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21025)  

**Abstract**: Target audio source separation with natural language queries presents a promising paradigm for extracting arbitrary audio events through arbitrary text descriptions. Existing methods mainly face two challenges, the difficulty in jointly modeling acoustic-textual alignment and semantic-aware separation within a blindly-learned single-stage architecture, and the reliance on large-scale accurately-labeled training data to compensate for inefficient cross-modal learning and separation. To address these challenges, we propose a hierarchical decomposition framework, HSM-TSS, that decouples the task into global-local semantic-guided feature separation and structure-preserving acoustic reconstruction. Our approach introduces a dual-stage mechanism for semantic separation, operating on distinct global and local semantic feature spaces. We first perform global-semantic separation through a global semantic feature space aligned with text queries. A Q-Audio architecture is employed to align audio and text modalities, serving as pretrained global-semantic encoders. Conditioned on the predicted global feature, we then perform the second-stage local-semantic separation on AudioMAE features that preserve time-frequency structures, followed by acoustic reconstruction. We also propose an instruction processing pipeline to parse arbitrary text queries into structured operations, extraction or removal, coupled with audio descriptions, enabling flexible sound manipulation. Our method achieves state-of-the-art separation performance with data-efficient training while maintaining superior semantic consistency with queries in complex auditory scenes. 

**Abstract (ZH)**: 基于自然语言查询的目标音频源分离呈一种有前途的范式，能够通过任意文本描述提取任意音频事件。现有的方法主要面临两大挑战：单阶段盲学习架构中难以同时建模声学-文本对齐和语义感知分离，以及需要大量精确标注的训练数据来补偿跨模态学习和分离的低效性。为了解决这些挑战，我们提出了一种分层分解框架HSM-TSS，将任务分解为全局-局部语义引导特征分离和结构保真的声学重构。该方法引入了一种双阶段的语义分离机制，分别在全局和局部语义特征空间上操作。首先，通过与文本查询对齐的全局语义特征空间进行全局语义分离，采用Q-Audio架构对齐音频和文本模态，作为预训练的全局语义编码器。在预测的全局特征条件下，然后在保留时频结构的AudioMAE特征上进行第二阶段的局部语义分离，并随后进行声学重构。我们还提出了一种指令处理管道，将任意文本查询解析为结构化的操作，提取或移除，并结合音频描述，实现灵活的音效操控。该方法在数据高效训练的同时，保持了与复杂听觉场景中查询的优异语义一致性，达到最好的分离性能。 

---
# Federated Instrumental Variable Analysis via Federated Generalized Method of Moments 

**Title (ZH)**: 联邦工具变量分析 via 联邦广义矩方法 

**Authors**: Geetika, Somya Tyagi, Bapi Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.21012)  

**Abstract**: Instrumental variables (IV) analysis is an important applied tool for areas such as healthcare and consumer economics. For IV analysis in high-dimensional settings, the Generalized Method of Moments (GMM) using deep neural networks offers an efficient approach. With non-i.i.d. data sourced from scattered decentralized clients, federated learning is a popular paradigm for training the models while promising data privacy. However, to our knowledge, no federated algorithm for either GMM or IV analysis exists to date. In this work, we introduce federated instrumental variables analysis (FedIV) via federated generalized method of moments (FedGMM). We formulate FedGMM as a federated zero-sum game defined by a federated non-convex non-concave minimax optimization problem, which is solved using federated gradient descent ascent (FedGDA) algorithm. One key challenge arises in theoretically characterizing the federated local optimality. To address this, we present properties and existence results of clients' local equilibria via FedGDA limit points. Thereby, we show that the federated solution consistently estimates the local moment conditions of every participating client. The proposed algorithm is backed by extensive experiments to demonstrate the efficacy of our approach. 

**Abstract (ZH)**: 联邦广义工具变量分析（FedGMM） 

---
# BIPNN: Learning to Solve Binary Integer Programming via Hypergraph Neural Networks 

**Title (ZH)**: BIPNN：通过超图神经网络学习求解二元整数规划 

**Authors**: Sen Bai, Chunqi Yang, Xin Bai, Xin Zhang, Zhengang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20997)  

**Abstract**: Binary (0-1) integer programming (BIP) is pivotal in scientific domains requiring discrete decision-making. As the advance of AI computing, recent works explore neural network-based solvers for integer linear programming (ILP) problems. Yet, they lack scalability for tackling nonlinear challenges. To handle nonlinearities, state-of-the-art Branch-and-Cut solvers employ linear relaxations, leading to exponential growth in auxiliary variables and severe computation limitations. To overcome these limitations, we propose BIPNN (Binary Integer Programming Neural Network), an unsupervised learning framework to solve nonlinear BIP problems via hypergraph neural networks (HyperGNN). Specifically, BIPNN reformulates BIPs-constrained, discrete, and nonlinear (sin, log, exp) optimization problems-into unconstrained, differentiable, and polynomial loss functions. The reformulation stems from the observation of a precise one-to-one mapping between polynomial BIP objectives and hypergraph structures, enabling the unsupervised training of HyperGNN to optimize BIP problems in an end-to-end manner. On this basis, we propose a GPU-accelerated and continuous-annealing-enhanced training pipeline for BIPNN. The pipeline enables BIPNN to optimize large-scale nonlinear terms in BIPs fully in parallel via straightforward gradient descent, thus significantly reducing the training cost while ensuring the generation of discrete, high-quality solutions. Extensive experiments on synthetic and real-world datasets highlight the superiority of our approach. 

**Abstract (ZH)**: 基于超图神经网络的二进制整数规划求解器BIPNN 

---
# MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection 

**Title (ZH)**: MelodySim: 基于旋律的音乐相似性测量在剽窃检测中的应用 

**Authors**: Tongyu Lu, Charlotta-Marlena Geist, Jan Melechovsky, Abhinaba Roy, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2505.20979)  

**Abstract**: We propose MelodySim, a melody-aware music similarity model and dataset for plagiarism detection. First, we introduce a novel method to construct a dataset with focus on melodic similarity. By augmenting Slakh2100; an existing MIDI dataset, we generate variations of each piece while preserving the melody through modifications such as note splitting, arpeggiation, minor track dropout (excluding bass), and re-instrumentation. A user study confirms that positive pairs indeed contain similar melodies, with other musical tracks significantly changed. Second, we develop a segment-wise melodic-similarity detection model that uses a MERT encoder and applies a triplet neural network to capture melodic similarity. The resultant decision matrix highlights where plagiarism might occur. Our model achieves high accuracy on the MelodySim test set. 

**Abstract (ZH)**: 我们提出MelodySim，一种基于旋律的音乐相似度模型和数据集，用于检测抄袭。首先，我们介绍了一种新的方法，用于构建一个以旋律相似性为重点的数据集。通过增强现有的MIDI数据集Slakh2100，我们生成了每首作品的不同变体，同时通过音符分割、琶音化、除低音外的小节删除和重新配器等修改保留旋律。用户研究证实，正样本确实包含相似的旋律，而其他音乐轨道则显著不同。其次，我们开发了一种段落级旋律相似性检测模型，该模型使用MERT编码器并应用三重神经网络来捕捉旋律相似性。结果决策矩阵突出显示了可能发生抄袭的地方。我们的模型在MelodySim测试集上实现了高准确性。 

---
# Towards Conversational Development Environments: Using Theory-of-Mind and Multi-Agent Architectures for Requirements Refinement 

**Title (ZH)**: 面向对话式的开发环境：基于心理理论和多代理架构的需求细化方法 

**Authors**: Keheliya Gallaba, Ali Arabat, Dayi Lin, Mohammed Sayagh, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20973)  

**Abstract**: Foundation Models (FMs) have shown remarkable capabilities in various natural language tasks. However, their ability to accurately capture stakeholder requirements remains a significant challenge for using FMs for software development. This paper introduces a novel approach that leverages an FM-powered multi-agent system called AlignMind to address this issue. By having a cognitive architecture that enhances FMs with Theory-of-Mind capabilities, our approach considers the mental states and perspectives of software makers. This allows our solution to iteratively clarify the beliefs, desires, and intentions of stakeholders, translating these into a set of refined requirements and a corresponding actionable natural language workflow in the often-overlooked requirements refinement phase of software engineering, which is crucial after initial elicitation. Through a multifaceted evaluation covering 150 diverse use cases, we demonstrate that our approach can accurately capture the intents and requirements of stakeholders, articulating them as both specifications and a step-by-step plan of action. Our findings suggest that the potential for significant improvements in the software development process justifies these investments. Our work lays the groundwork for future innovation in building intent-first development environments, where software makers can seamlessly collaborate with AIs to create software that truly meets their needs. 

**Abstract (ZH)**: 基于Foundation Models的软件开发需求澄清新方法：AlignMind及其应用研究 

---
# Deep k-grouping: An Unsupervised Learning Framework for Combinatorial Optimization on Graphs and Hypergraphs 

**Title (ZH)**: 深层次k-分组：图和超图上组合优化的无监督学习框架 

**Authors**: Sen Bai, Chunqi Yang, Xin Bai, Xin Zhang, Zhengang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20972)  

**Abstract**: Along with AI computing shining in scientific discovery, its potential in the combinatorial optimization (CO) domain has also emerged in recent years. Yet, existing unsupervised neural network solvers struggle to solve $k$-grouping problems (e.g., coloring, partitioning) on large-scale graphs and hypergraphs, due to limited computational frameworks. In this work, we propose Deep $k$-grouping, an unsupervised learning-based CO framework. Specifically, we contribute: Novel one-hot encoded polynomial unconstrained binary optimization (OH-PUBO), a formulation for modeling k-grouping problems on graphs and hypergraphs (e.g., graph/hypergraph coloring and partitioning); GPU-accelerated algorithms for large-scale k-grouping CO problems. Deep $k$-grouping employs the relaxation of large-scale OH-PUBO objectives as differentiable loss functions and trains to optimize them in an unsupervised manner. To ensure scalability, it leverages GPU-accelerated algorithms to unify the training pipeline; A Gini coefficient-based continuous relaxation annealing strategy to enforce discreteness of solutions while preventing convergence to local optima. Experimental results demonstrate that Deep $k$-grouping outperforms existing neural network solvers and classical heuristics such as SCIP and Tabu. 

**Abstract (ZH)**: 伴随AI计算在科学发现中的闪耀，其在组合优化领域近年来也展现出巨大潜力。现有无监督神经网络求解器在解决大规模图和超图的k-分组问题（如着色、划分）时遇挑战，受限于计算框架的局限。在这项工作中，我们提出了Deep $k$-分组，一个基于无监督学习的组合优化框架。具体贡献包括：一种新型的一-hot编码多项式未约束二进制优化（OH-PUBO）表示，用于建模图和超图上的k-分组问题（如图和超图着色与划分）；大规模k-分组组合优化问题的GPU加速算法。Deep $k$-分组采用大规模OH-PUBO目标的松弛作为可微损失函数，并以无监督方式训练以优化这些目标；基于Gini系数的连续松弛退火策略，确保解的离散性并防止收敛于局部最优。实验结果表明，Deep $k$-分组在性能上优于现有神经网络求解器和经典启发式算法（如SCIP和Tabu）。 

---
# Context-Aware Content Moderation for German Newspaper Comments 

**Title (ZH)**: 基于上下文的内容审核：德国报纸评论的内容过滤 

**Authors**: Felix Krejca, Tobias Kietreiber, Alexander Buchelt, Sebastian Neumaier  

**Link**: [PDF](https://arxiv.org/pdf/2505.20963)  

**Abstract**: The increasing volume of online discussions requires advanced automatic content moderation to maintain responsible discourse. While hate speech detection on social media is well-studied, research on German-language newspaper forums remains limited. Existing studies often neglect platform-specific context, such as user history and article themes. This paper addresses this gap by developing and evaluating binary classification models for automatic content moderation in German newspaper forums, incorporating contextual information. Using LSTM, CNN, and ChatGPT-3.5 Turbo, and leveraging the One Million Posts Corpus from the Austrian newspaper Der Standard, we assess the impact of context-aware models. Results show that CNN and LSTM models benefit from contextual information and perform competitively with state-of-the-art approaches. In contrast, ChatGPT's zero-shot classification does not improve with added context and underperforms. 

**Abstract (ZH)**: 不断增加的在线讨论量要求先进的自动内容审核技术以维持负责任的 discourse。虽然社交媒体上的仇恨言论检测已被广泛研究，但关于德语报纸论坛的研究仍然有限。现有研究往往忽视了平台特定的上下文，如用户历史和文章主题。本文通过开发和评估用于德语报纸论坛的自动内容审核二分类模型，并结合上下文信息，填补了这一空白。利用LSTM、CNN和ChatGPT-3.5 Turbo，并借助奥地利报纸《标准报》的百万帖子语料库，我们评估了面向上下文的模型的影响。结果显示，CNN和LSTM模型受益于上下文信息，并与当前最先进的方法表现得相当。相比之下，ChatGPT的零样本分类在增加上下文时并未改善，并表现出色。 

---
# Efficient and Microphone-Fault-Tolerant 3D Sound Source Localization 

**Title (ZH)**: 高效的和麦克风故障容忍的3D声源定位 

**Authors**: Yiyuan Yang, Shitong Xu, Niki Trigoni, Andrew Markham  

**Link**: [PDF](https://arxiv.org/pdf/2505.20961)  

**Abstract**: Sound source localization (SSL) is a critical technology for determining the position of sound sources in complex environments. However, existing methods face challenges such as high computational costs and precise calibration requirements, limiting their deployment in dynamic or resource-constrained environments. This paper introduces a novel 3D SSL framework, which uses sparse cross-attention, pretraining, and adaptive signal coherence metrics, to achieve accurate and computationally efficient localization with fewer input microphones. The framework is also fault-tolerant to unreliable or even unknown microphone position inputs, ensuring its applicability in real-world scenarios. Preliminary experiments demonstrate its scalability for multi-source localization without requiring additional hardware. This work advances SSL by balancing the model's performance and efficiency and improving its robustness for real-world scenarios. 

**Abstract (ZH)**: 声源定位（SSL）是确定复杂环境中声源位置的关键技术。然而，现有方法面临高计算成本和精确校准要求等挑战，限制了其在动态或资源受限环境中的部署。本文引入了一种新型的3D SSL框架，该框架利用稀疏交叉注意力、预训练和自适应信号相干性度量，实现了在更少输入麦克风条件下进行准确且计算高效的定位。该框架还对不可靠甚至未知麦克风位置输入具有容错性，确保其在实际应用场景中的适用性。初步实验结果显示，该工作在无需额外硬件的情况下，展示了其在多源定位方面的可扩展性。本研究通过平衡模型性能和效率，并提高其在实际应用场景中的鲁棒性，推进了SSL的发展。 

---
# Hybrid Disagreement-Diversity Active Learning for Bioacoustic Sound Event Detection 

**Title (ZH)**: 生物声学声事件检测的混合分歧-多样性主动学习 

**Authors**: Shiqi Zhang, Tuomas Virtanen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20956)  

**Abstract**: Bioacoustic sound event detection (BioSED) is crucial for biodiversity conservation but faces practical challenges during model development and training: limited amounts of annotated data, sparse events, species diversity, and class imbalance. To address these challenges efficiently with a limited labeling budget, we apply the mismatch-first farthest-traversal (MFFT), an active learning method integrating committee voting disagreement and diversity analysis. We also refine an existing BioSED dataset specifically for evaluating active learning algorithms. Experimental results demonstrate that MFFT achieves a mAP of 68% when cold-starting and 71% when warm-starting (which is close to the fully-supervised mAP of 75%) while using only 2.3% of the annotations. Notably, MFFT excels in cold-start scenarios and with rare species, which are critical for monitoring endangered species, demonstrating its practical value. 

**Abstract (ZH)**: 生物声学声音事件检测（BioSED）对于生物多样性保护至关重要，但在模型开发和训练过程中面临实际挑战：标注数据量有限、事件稀疏、物种多样性以及类别不平衡。为了在有限的标注预算内有效应对这些挑战，我们应用了 mismatch-first farthest-traversal (MFFT) 方法，该方法结合了委员会投票分歧和多样性分析的主动学习方法。我们还细化了一个现有的 BioSED 数据集，以评估主动学习算法。实验结果表明，MFFT 在冷启动时达到了 68% 的 mAP，在温暖启动时达到了 71%（接近完全监督学习的 75%），仅使用了 2.3% 的标注。值得注意的是，MFFT 在冷启动场景和稀有物种中表现出色，这对于监测濒危物种至关重要，证明了其实际价值。 

---
# Streamlining Knowledge Graph Creation with PyRML 

**Title (ZH)**: 使用PyRML简化知识图谱创建 

**Authors**: Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2505.20949)  

**Abstract**: Knowledge Graphs (KGs) are increasingly adopted as a foundational technology for integrating heterogeneous data in domains such as climate science, cultural heritage, and the life sciences. Declarative mapping languages like R2RML and RML have played a central role in enabling scalable and reusable KG construction, offering a transparent means of transforming structured and semi-structured data into RDF. In this paper, we present PyRML, a lightweight, Python-native library for building Knowledge Graphs through declarative mappings. PyRML supports core RML constructs and provides a programmable interface for authoring, executing, and testing mappings directly within Python environments. It integrates with popular data and semantic web libraries (e.g., Pandas and RDFlib), enabling transparent and modular workflows. By lowering the barrier to entry for KG creation and fostering reproducible, ontology-aligned data integration, PyRML bridges the gap between declarative semantics and practical KG engineering. 

**Abstract (ZH)**: 知识图谱（KGs）在气候科学、文化遗产和生命科学等领域越来越被用作集成异构数据的基础技术。声明式映射语言如R2RML和RML在实现可扩展和可重用的KG构建中发挥了核心作用，提供了一种透明地将结构化和半结构化数据转换为RDF的方法。在本文中，我们介绍了PyRML，这是一种轻量级的Python原生库，通过声明式映射构建知识图谱。PyRML支持核心RML构造，并提供了一种在Python环境中直接编写、执行和测试映射的编程接口。它与流行的数据和语义网库（如Pandas和RDFlib）集成，实现了透明且模块化的 workflows。通过降低知识图谱创建的门槛并促进可重复的、本体对齐的数据集成，PyRML在声明式语义与实际知识图谱工程之间架起了桥梁。 

---
# Unified Deep Learning Approach for Estimating the Metallicities of RR Lyrae Stars Using light curves from Gaia Data Release 3 

**Title (ZH)**: 使用Gaia数据释放3的光变曲线统一直 deep学习方法估计RR Lyrae星的金属丰度 

**Authors**: Lorenzo Monti, Tatiana Muraveva, Alessia Garofalo, Gisella Clementini, Maria Letizia Valentini  

**Link**: [PDF](https://arxiv.org/pdf/2505.20947)  

**Abstract**: RR Lyrae stars (RRLs) are old pulsating variables widely used as metallicity tracers due to the correlation between their metal abundances and light curve morphology. With ESA Gaia DR3 providing light curves for about 270,000 RRLs, there is a pressing need for scalable methods to estimate their metallicities from photometric data. We introduce a unified deep learning framework that estimates metallicities for both fundamental-mode (RRab) and first-overtone (RRc) RRLs using Gaia G-band light curves. This approach extends our previous work on RRab stars to include RRc stars, aiming for high predictive accuracy and broad generalization across both pulsation types. The model is based on a Gated Recurrent Unit (GRU) neural network optimized for time-series extrinsic regression. Our pipeline includes preprocessing steps such as phase folding, smoothing, and sample weighting, and uses photometric metallicities from the literature as training targets. The architecture is designed to handle morphological differences between RRab and RRc light curves without requiring separate models. On held-out validation sets, our GRU model achieves strong performance: for RRab stars, MAE = 0.0565 dex, RMSE = 0.0765 dex, R^2 = 0.9401; for RRc stars, MAE = 0.0505 dex, RMSE = 0.0720 dex, R^2 = 0.9625. These results show the effectiveness of deep learning for large-scale photometric metallicity estimation and support its application to studies of stellar populations and Galactic structure. 

**Abstract (ZH)**: RR Lyrae 星的金属丰度估计：基于ESA Gaia DR3光曲线的统一深度学习框架 

---
# Humble AI in the real-world: the case of algorithmic hiring 

**Title (ZH)**: 谦逊的人工智能在现实世界：算法招聘案例 

**Authors**: Rahul Nair, Inge Vejsbjerg, Elizabeth Daly, Christos Varytimidis, Bran Knowles  

**Link**: [PDF](https://arxiv.org/pdf/2505.20918)  

**Abstract**: Humble AI (Knowles et al., 2023) argues for cautiousness in AI development and deployments through scepticism (accounting for limitations of statistical learning), curiosity (accounting for unexpected outcomes), and commitment (accounting for multifaceted values beyond performance). We present a real-world case study for humble AI in the domain of algorithmic hiring. Specifically, we evaluate virtual screening algorithms in a widely used hiring platform that matches candidates to job openings. There are several challenges in misrecognition and stereotyping in such contexts that are difficult to assess through standard fairness and trust frameworks; e.g., someone with a non-traditional background is less likely to rank highly. We demonstrate technical feasibility of how humble AI principles can be translated to practice through uncertainty quantification of ranks, entropy estimates, and a user experience that highlights algorithmic unknowns. We describe preliminary discussions with focus groups made up of recruiters. Future user studies seek to evaluate whether the higher cognitive load of a humble AI system fosters a climate of trust in its outcomes. 

**Abstract (ZH)**: 谦逊的AI（Knowles等，2023）主张在AI开发和部署中保持谨慎，并通过怀疑（考虑统计学习的局限性）、好奇心（考虑意外结果）和承诺（考虑超越性能的多元价值观）来实现。我们提出一个关于谦逊AI在算法招聘领域的实际案例研究。具体而言，我们评估了一个广泛使用的招聘平台中的虚拟筛选算法，该平台将候选人匹配到职位空缺。在这种背景下，误识和刻板印象的问题难以通过标准的公平性和信任框架来评估；例如，背景非传统的人员不太可能排名靠前。我们展示了通过排名的不确定性量化、熵估计和突出算法未知性来实现谦逊AI原则技术可行性的方法。我们描述了与招聘人员组成的焦点小组的初步讨论。未来用户研究旨在评估谦逊AI系统更高的认知负荷是否能促进对其结果的信任氛围。 

---
# How Do Transformers Learn Variable Binding in Symbolic Programs? 

**Title (ZH)**: Transformer如何学习符号程序中的变量绑定？ 

**Authors**: Yiwei Wu, Atticus Geiger, Raphaël Millière  

**Link**: [PDF](https://arxiv.org/pdf/2505.20896)  

**Abstract**: Variable binding -- the ability to associate variables with values -- is fundamental to symbolic computation and cognition. Although classical architectures typically implement variable binding via addressable memory, it is not well understood how modern neural networks lacking built-in binding operations may acquire this capacity. We investigate this by training a Transformer to dereference queried variables in symbolic programs where variables are assigned either numerical constants or other variables. Each program requires following chains of variable assignments up to four steps deep to find the queried value, and also contains irrelevant chains of assignments acting as distractors. Our analysis reveals a developmental trajectory with three distinct phases during training: (1) random prediction of numerical constants, (2) a shallow heuristic prioritizing early variable assignments, and (3) the emergence of a systematic mechanism for dereferencing assignment chains. Using causal interventions, we find that the model learns to exploit the residual stream as an addressable memory space, with specialized attention heads routing information across token positions. This mechanism allows the model to dynamically track variable bindings across layers, resulting in accurate dereferencing. Our results show how Transformer models can learn to implement systematic variable binding without explicit architectural support, bridging connectionist and symbolic approaches. 

**Abstract (ZH)**: 变量绑定——将变量与值关联的能力——是符号计算和认知的基本要素。虽然经典架构通常通过可寻址内存实现变量绑定，但现代缺乏内置绑定操作的神经网络如何获得这一能力尚不明确。我们通过训练一个变换器来查询符号程序中的变量，探索这一问题，其中变量被赋予数字常量或其他变量。每个程序需要追踪最多四步深的变量赋值链以找到查询值，并且包含作为干扰项的无关赋值链。我们的分析揭示了训练过程中的发展轨迹，包括三个明显的阶段：（1）随机预测数字常量，（2）浅层启发法优先考虑早期变量赋值，以及（3）一种系统性机制的出现，用于分辨率赋值链。通过因果干预，我们发现模型学会了利用残差流作为可寻址的内存空间，特化的注意力头负责在标记位置间路由信息。这种机制使模型能够在层间动态跟踪变量绑定，从而实现准确的分辨率赋值。我们的结果展示了变换器模型如何在无显式架构支持的情况下学习实现系统性变量绑定，从而同时连接联结主义和符号方法。 

---
# Frequency Composition for Compressed and Domain-Adaptive Neural Networks 

**Title (ZH)**: 压缩与领域自适应神经网络的频域组成 

**Authors**: Yoojin Kwon, Hongjun Suh, Wooseok Lee, Taesik Gong, Songyi Han, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20890)  

**Abstract**: Modern on-device neural network applications must operate under resource constraints while adapting to unpredictable domain shifts. However, this combined challenge-model compression and domain adaptation-remains largely unaddressed, as prior work has tackled each issue in isolation: compressed networks prioritize efficiency within a fixed domain, whereas large, capable models focus on handling domain shifts. In this work, we propose CoDA, a frequency composition-based framework that unifies compression and domain adaptation. During training, CoDA employs quantization-aware training (QAT) with low-frequency components, enabling a compressed model to selectively learn robust, generalizable features. At test time, it refines the compact model in a source-free manner (i.e., test-time adaptation, TTA), leveraging the full-frequency information from incoming data to adapt to target domains while treating high-frequency components as domain-specific cues. LFC are aligned with the trained distribution, while HFC unique to the target distribution are solely utilized for batch normalization. CoDA can be integrated synergistically into existing QAT and TTA methods. CoDA is evaluated on widely used domain-shift benchmarks, including CIFAR10-C and ImageNet-C, across various model architectures. With significant compression, it achieves accuracy improvements of 7.96%p on CIFAR10-C and 5.37%p on ImageNet-C over the full-precision TTA baseline. 

**Abstract (ZH)**: 现代设备上的神经网络应用必须在资源受限的情况下适应不可预测的领域转移。然而，这个结合挑战——模型压缩和领域适应——仍未得到充分解决，因为先前的工作各自处理这两个问题：压缩网络侧重于固定领域内的效率，而大型能力强的模型则专注于处理领域转移。在网络中，我们提出CoDA，一种基于频率组成的框架，将压缩和领域适应统一起来。在训练过程中，CoDA 使用具有低频组件的量化感知训练（QAT），使压缩模型能够选择性地学习稳健且通用的特征。在测试时，它以源代码免费的方式（即，在测试时适应，TTA）逐步优化紧凑模型，利用传入数据的全频信息来适应目标领域，同时将高频组件作为领域特定线索。低频成分与训练分布对齐，而独有的高频成分专门用于批量标准化。CoDA 可以以协同方式整合到现有的 QAT 和 TTA 方法中。CoDA 在包括 CIFAR10-C 和 ImageNet-C 的广泛使用的领域转移基准上进行了评估，涵盖各种模型架构。通过显著压缩，它在 CIFAR10-C 上相对于全精度 TTA 基线实现了 7.96% 的准确率提升，在 ImageNet-C 上实现了 5.37% 的准确率提升。 

---
# Spotlight-TTS: Spotlighting the Style via Voiced-Aware Style Extraction and Style Direction Adjustment for Expressive Text-to-Speech 

**Title (ZH)**: Spotlight-TTS：基于语音感知风格提取和风格方向调整的表达性文本到语音生成 

**Authors**: Nam-Gyu Kim, Deok-Hyeon Cho, Seung-Bin Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20868)  

**Abstract**: Recent advances in expressive text-to-speech (TTS) have introduced diverse methods based on style embedding extracted from reference speech. However, synthesizing high-quality expressive speech remains challenging. We propose Spotlight-TTS, which exclusively emphasizes style via voiced-aware style extraction and style direction adjustment. Voiced-aware style extraction focuses on voiced regions highly related to style while maintaining continuity across different speech regions to improve expressiveness. We adjust the direction of the extracted style for optimal integration into the TTS model, which improves speech quality. Experimental results demonstrate that Spotlight-TTS achieves superior performance compared to baseline models in terms of expressiveness, overall speech quality, and style transfer capability. Our audio samples are publicly available. 

**Abstract (ZH)**: Recent Advances in Spotlight-TTS: Exclusive Style Emphasis via Voiced-Aware Style Extraction and Style Direction Adjustment 

---
# Cooperation of Experts: Fusing Heterogeneous Information with Large Margin 

**Title (ZH)**: 专家合作：大 Margin 融合异类型信息 

**Authors**: Shuo Wang, Shunyang Huang, Jinghui Yuan, Zhixiang Shen, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20853)  

**Abstract**: Fusing heterogeneous information remains a persistent challenge in modern data analysis. While significant progress has been made, existing approaches often fail to account for the inherent heterogeneity of object patterns across different semantic spaces. To address this limitation, we propose the Cooperation of Experts (CoE) framework, which encodes multi-typed information into unified heterogeneous multiplex networks. By overcoming modality and connection differences, CoE provides a powerful and flexible model for capturing the intricate structures of real-world complex data. In our framework, dedicated encoders act as domain-specific experts, each specializing in learning distinct relational patterns in specific semantic spaces. To enhance robustness and extract complementary knowledge, these experts collaborate through a novel large margin mechanism supported by a tailored optimization strategy. Rigorous theoretical analyses guarantee the framework's feasibility and stability, while extensive experiments across diverse benchmarks demonstrate its superior performance and broad applicability. Our code is available at this https URL. 

**Abstract (ZH)**: 融合异构信息仍然是现代数据分析中的一个持续挑战。虽然已取得显著进展，但现有方法往往未能充分考虑对象模式在不同语义空间中的固有异构性。为应对这一局限，我们提出了一种专家合作（CoE）框架，该框架将多种类型的信息编码为统一的异构多层网络。通过克服模态和连接差异，CoE 提供了一种强大且灵活的模型，用于捕获现实世界复杂数据的精细结构。在我们的框架中，专用编码器作为领域特定专家，各自专门学习特定语义空间中的独特关系模式。为了增强鲁棒性并提取互补知识，这些专家通过一种新型的大边际机制协同工作，该机制支持定制的优化策略。严格的理论分析保证了框架的可行性和稳定性，而跨越多种基准的广泛实验证明了其优越性能和广泛的适用性。我们的代码可在以下链接获取：this https URL。 

---
# RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph 

**Title (ZH)**: RSCF: 关系语义一致过滤器在知识图实体嵌入中的应用 

**Authors**: Junsik Kim, Jinwook Park, Kangil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20813)  

**Abstract**: In knowledge graph embedding, leveraging relation-specific entity-transformation has markedly enhanced performance. However, the consistency of embedding differences before and after transformation remains unaddressed, risking the loss of valuable inductive bias inherent in the embeddings. This inconsistency stems from two problems. First, transformation representations are specified for relations in a disconnected manner, allowing dissimilar transformations and corresponding entity-embeddings for similar relations. Second, a generalized plug-in approach as a SFBR (Semantic Filter Based on Relations) disrupts this consistency through excessive concentration of entity embeddings under entity-based regularization, generating indistinguishable score distributions among relations. In this paper, we introduce a plug-in KGE method, Relation-Semantics Consistent Filter (RSCF), containing more consistent entity-transformation characterized by three features: 1) shared affine transformation of relation embeddings across all relations, 2) rooted entity-transformation that adds an entity embedding to its change represented by the transformed vector, and 3) normalization of the change to prevent scale reduction. To amplify the advantages of consistency that preserve semantics on embeddings, RSCF adds relation transformation and prediction modules for enhancing the semantics. In knowledge graph completion tasks with distance-based and tensor decomposition models, RSCF significantly outperforms state-of-the-art KGE methods, showing robustness across all relations and their frequencies. 

**Abstract (ZH)**: 在知识图嵌入中，利用关系特定的实体转换显著提高了性能。然而，转换前后嵌入差异的一致性尚未得到解决，这可能会导致嵌入中固有的宝贵归纳偏置的丢失。这种不一致性来源于两个问题。首先，关系转换表示是以断开的方式为关系指定的，允许类似关系有不同的转换和相应的实体嵌入。其次，基于关系的通用插件方法作为SFBR（基于关系的语义过滤器）通过在基于实体的正则化下过度集中实体嵌入，破坏了这种一致性，导致不同关系之间的得分分布难以区分。在本文中，我们引入了一种包含更一致的实体转换的插件KGE方法——关系语义一致过滤器（RSCF），其特点是：1) 所有关系共享的仿射转换，2) 根植的实体转换，将实体嵌入与其转换向量表示的变化相加，3) 规范化变化以防止缩放减少。为了进一步增强保持嵌入语义的一致性优势，RSCF增加了关系转换和预测模块。在基于距离和张量分解的知识图填充任务中，RSCF显著优于现有的KGE方法，并且在所有关系及其频率上表现出稳健性。 

---
# VibE-SVC: Vibrato Extraction with High-frequency F0 Contour for Singing Voice Conversion 

**Title (ZH)**: VibE-SVC: 基于高频率基频轮廓的振动提取与歌声转换 

**Authors**: Joon-Seung Choi, Dong-Min Byun, Hyung-Seok Oh, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20794)  

**Abstract**: Controlling singing style is crucial for achieving an expressive and natural singing voice. Among the various style factors, vibrato plays a key role in conveying emotions and enhancing musical depth. However, modeling vibrato remains challenging due to its dynamic nature, making it difficult to control in singing voice conversion. To address this, we propose VibESVC, a controllable singing voice conversion model that explicitly extracts and manipulates vibrato using discrete wavelet transform. Unlike previous methods that model vibrato implicitly, our approach decomposes the F0 contour into frequency components, enabling precise transfer. This allows vibrato control for enhanced flexibility. Experimental results show that VibE-SVC effectively transforms singing styles while preserving speaker similarity. Both subjective and objective evaluations confirm high-quality conversion. 

**Abstract (ZH)**: 控制歌声风格对于实现富有表现力和自然的歌声至关重要。在各种风格因素中，颤音在传达情感和增强音乐深度方面起着关键作用。然而，由于颤音的动态特性，其建模依然具有挑战性，使得在歌声转换中难以控制。为此，我们提出了一种可控歌声转换模型VibESVC，该模型明确地通过离散小波变换提取和操控颤音。与以前隐式建模颤音的方法不同，我们的方法将基频轮廓分解为频率成分，从而实现精确传递。这使得颤音控制更加灵活。实验结果表明，VibE-SVC能够在保持说话人相似性的同时有效转换歌声风格。主观和客观评估均证实了高质量的转换效果。 

---
# Adversarial bandit optimization for approximately linear functions 

**Title (ZH)**: 对抗性带宽优化近线性函数 

**Authors**: Zhuoyu Cheng, Kohei Hatano, Eiji Takimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.20734)  

**Abstract**: We consider a bandit optimization problem for nonconvex and non-smooth functions, where in each trial the loss function is the sum of a linear function and a small but arbitrary perturbation chosen after observing the player's choice. We give both expected and high probability regret bounds for the problem. Our result also implies an improved high-probability regret bound for the bandit linear optimization, a special case with no perturbation. We also give a lower bound on the expected regret. 

**Abstract (ZH)**: 我们考虑非凸非光滑函数的多臂老虎机优化问题，在每次试验中，损失函数是线性函数与观察到玩家选择后引入的小但任意的扰动之和。我们给出了问题的期望后悔界和高概率后悔界。我们的结果还隐含着在无扰动情况下的多臂老虎机线性优化问题的改进的高概率后悔界。我们还给出了期望后悔的下界。 

---
# Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting 

**Title (ZH)**: 基于频率嵌入的3D高斯点云的宽带射频辐射场建模 

**Authors**: Zechen Li, Lanqing Yang, Yiheng Bian, Hao Pan, Yongjian Fu, Yezhou Wang, Yi-Chao Chen, Guangtao Xue, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.20714)  

**Abstract**: This paper presents an innovative frequency-embedded 3D Gaussian splatting (3DGS) algorithm for wideband radio-frequency (RF) radiance field modeling, offering an advancement over the existing works limited to single-frequency modeling. Grounded in fundamental physics, we uncover the complex relationship between EM wave propagation behaviors and RF frequencies. Inspired by this, we design an EM feature network with attenuation and radiance modules to learn the complex relationships between RF frequencies and the key properties of each 3D Gaussian, specifically the attenuation factor and RF signal intensity. By training the frequency-embedded 3DGS model, we can efficiently reconstruct RF radiance fields at arbitrary unknown frequencies within a given 3D environment. Finally, we propose a large-scale power angular spectrum (PAS) dataset containing 50000 samples ranging from 1 to 100 GHz in 6 indoor environments, and conduct extensive experiments to verify the effectiveness of our method. Our approach achieves an average Structural Similarity Index Measure (SSIM) up to 0.72, and a significant improvement up to 17.8% compared to the current state-of-the-art (SOTA) methods trained on individual test frequencies. Additionally, our method achieves an SSIM of 0.70 without prior training on these frequencies, which represents only a 2.8% performance drop compared to models trained with full PAS data. This demonstrates our model's capability to estimate PAS at unknown frequencies. For related code and datasets, please refer to this https URL. 

**Abstract (ZH)**: 基于频域嵌入的三维高斯点云计算宽频带射频辐射场模型 

---
# Generating Hypotheses of Dynamic Causal Graphs in Neuroscience: Leveraging Generative Factor Models of Observed Time Series 

**Title (ZH)**: 基于生成因子模型的观测时间序列在神经科学中动态因果图假设生成 

**Authors**: Zachary C. Brown, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2505.20697)  

**Abstract**: The field of hypothesis generation promises to reduce costs in neuroscience by narrowing the range of interventional studies needed to study various phenomena. Existing machine learning methods can generate scientific hypotheses from complex datasets, but many approaches assume causal relationships are static over time, limiting their applicability to systems with dynamic, state-dependent behavior, such as the brain. While some techniques attempt dynamic causal discovery through factor models, they often restrict relationships to linear patterns or impose other simplifying assumptions. We propose a novel method that models dynamic graphs as a conditionally weighted superposition of static graphs, where each static graph can capture nonlinear relationships. This approach enables the detection of complex, time-varying interactions between variables beyond linear limitations. Our method improves f1-scores of predicted dynamic causal patterns by roughly 22-28% on average over baselines in some of our experiments, with some improvements reaching well over 60%. A case study on real brain data demonstrates our method's ability to uncover relationships linked to specific behavioral states, offering valuable insights into neural dynamics. 

**Abstract (ZH)**: 假设生成领域有望通过缩小所需干预研究的范围来降低神经科学的成本。现有的机器学习方法可以从复杂的数据集中生成科学假设，但许多方法假设因果关系在时间上是静态的，限制了其在具有动态、状态依赖行为的系统（如大脑）中的应用。虽然一些技术通过因子模型尝试动态因果发现，但它们通常将关系限制为线性模式或施加其他简化假设。我们提出了一种新型方法，将动态图建模为条件加权的静态图的叠加，其中每个静态图可以捕捉非线性关系。该方法能够检测超出线性限制的复杂、随时间变化的变量间相互作用。在某些实验中，我们的方法相对于基线在预测动态因果模式的f1分数上提高了约22-28%，有的改进甚至超过60%。基于真实脑数据的案例研究展示了我们的方法能够发现与特定行为状态相关的深层关系，为神经动力学提供了有价值的见解。 

---
# Evidential Deep Active Learning for Semi-Supervised Classification 

**Title (ZH)**: 证据深度主动学习在半监督分类中的应用 

**Authors**: Shenkai Zhao, Xinao Zhang, Lipeng Pan, Xiaobin Xu, Danilo Pelusi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20691)  

**Abstract**: Semi-supervised classification based on active learning has made significant progress, but the existing methods often ignore the uncertainty estimation (or reliability) of the prediction results during the learning process, which makes it questionable whether the selected samples can effectively update the model. Hence, this paper proposes an evidential deep active learning approach for semi-supervised classification (EDALSSC). EDALSSC builds a semi-supervised learning framework to simultaneously quantify the uncertainty estimation of labeled and unlabeled data during the learning process. The uncertainty estimation of the former is associated with evidential deep learning, while that of the latter is modeled by combining ignorance information and conflict information of the evidence from the perspective of the T-conorm operator. Furthermore, this article constructs a heuristic method to dynamically balance the influence of evidence and the number of classes on uncertainty estimation to ensure that it does not produce counter-intuitive results in EDALSSC. For the sample selection strategy, EDALSSC selects the sample with the greatest uncertainty estimation that is calculated in the form of a sum when the training loss increases in the latter half of the learning process. Experimental results demonstrate that EDALSSC outperforms existing semi-supervised and supervised active learning approaches on image classification datasets. 

**Abstract (ZH)**: 基于证据的半监督主动学习方法（EDALSSC）及其在半监督分类中的应用 

---
# Continuous-Time Attention: PDE-Guided Mechanisms for Long-Sequence Transformers 

**Title (ZH)**: 连续时间注意力：由偏微分方程引导的长序列变压器机制 

**Authors**: Yukun Zhang, Xueqing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20666)  

**Abstract**: We propose a novel framework, Continuous_Time Attention, which infuses partial differential equations (PDEs) into the Transformer's attention mechanism to address the challenges of extremely long input sequences. Instead of relying solely on a static attention matrix, we allow attention weights to evolve over a pseudo_time dimension via diffusion, wave, or reaction_diffusion dynamics. This mechanism systematically smooths local noise, enhances long_range dependencies, and stabilizes gradient flow. Theoretically, our analysis shows that PDE_based attention leads to better optimization landscapes and polynomial rather than exponential decay of distant interactions. Empirically, we benchmark our method on diverse experiments_demonstrating consistent gains over both standard and specialized long sequence Transformer variants. Our findings highlight the potential of PDE_based formulations to enrich attention mechanisms with continuous_time dynamics and global coherence. 

**Abstract (ZH)**: 连续时间注意力框架：通过偏微分方程融入Transformer的注意力机制以应对极长输入序列的挑战 

---
# TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research 

**Title (ZH)**: TeroSeek: 一种基于人工智能的知识库与检索生成平台用于萜类研究 

**Authors**: Xu Kang, Siqi Jiang, Kangwei Xu, Jiahao Li, Ruibo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20663)  

**Abstract**: Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at this http URL. 

**Abstract (ZH)**: 萜类是一类研究超过150年的天然产物，但由于其跨学科性质（涵盖化学、药理学和生物学），知识整合变得复杂。为了应对这一挑战，作者开发了TeroSeek，一个基于二十年萜类文献知识库，并结合了AI驱动的问答聊天机器人和网络服务。利用检索增强生成（RAG）框架，TeroSeek提供结构化、高质量的信息，并在萜类相关查询中优于通用大语言模型（LLMs）。它作为多学科研究的专业工具公开可用，可通过以下网址访问：this http URL。 

---
# BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism 

**Title (ZH)**: BacktrackAgent: 增强GUI代理的错误检测与回溯机制 

**Authors**: Qinzhuo Wu, Pengzhi Gao, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20660)  

**Abstract**: Graphical User Interface (GUI) agents have gained substantial attention due to their impressive capabilities to complete tasks through multiple interactions within GUI environments. However, existing agents primarily focus on enhancing the accuracy of individual actions and often lack effective mechanisms for detecting and recovering from errors. To address these shortcomings, we propose the BacktrackAgent, a robust framework that incorporates a backtracking mechanism to improve task completion efficiency. BacktrackAgent includes verifier, judger, and reflector components as modules for error detection and recovery, while also applying judgment rewards to further enhance the agent's performance. Additionally, we develop a training dataset specifically designed for the backtracking mechanism, which considers the outcome pages after action executions. Experimental results show that BacktrackAgent has achieved performance improvements in both task success rate and step accuracy on Mobile3M and Auto-UI benchmarks. Our data and code will be released upon acceptance. 

**Abstract (ZH)**: 基于回溯机制的图形用户界面代理：一种提高任务完成效率的稳健框架 

---
# Chinese Cyberbullying Detection: Dataset, Method, and Validation 

**Title (ZH)**: 中文网络欺凌检测：数据集、方法与验证 

**Authors**: Yi Zhu, Xin Zou, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20654)  

**Abstract**: Existing cyberbullying detection benchmarks were organized by the polarity of speech, such as "offensive" and "non-offensive", which were essentially hate speech detection. However, in the real world, cyberbullying often attracted widespread social attention through incidents. To address this problem, we propose a novel annotation method to construct a cyberbullying dataset that organized by incidents. The constructed CHNCI is the first Chinese cyberbullying incident detection dataset, which consists of 220,676 comments in 91 incidents. Specifically, we first combine three cyberbullying detection methods based on explanations generation as an ensemble method to generate the pseudo labels, and then let human annotators judge these labels. Then we propose the evaluation criteria for validating whether it constitutes a cyberbullying incident. Experimental results demonstrate that the constructed dataset can be a benchmark for the tasks of cyberbullying detection and incident prediction. To the best of our knowledge, this is the first study for the Chinese cyberbullying incident detection task. 

**Abstract (ZH)**: 现有的网络霸凌检测基准按照言论的极性组织，如“侮辱性”和“非侮辱性”，本质上是对仇恨言论的检测。然而，在现实世界中，网络霸凌往往通过事件吸引广泛关注。为了解决这一问题，我们提出了一种新的标注方法，构建了一个基于事件的网络霸凌数据集。CHNCI是中国首个网络霸凌事件检测数据集，包含91个事件中的220,676条评论。具体而言，我们首先结合三种基于解释生成的网络霸凌检测方法作为集成方法生成伪标签，然后让人工标注者判断这些标签。接着我们提出了验证是否构成网络霸凌事件的评价标准。实验结果表明，构建的数据集可以作为网络霸凌检测和事件预测任务的基准。据我们所知，这是首个针对中文网络霸凌事件检测任务的研究。 

---
# RoGA: Towards Generalizable Deepfake Detection through Robust Gradient Alignment 

**Title (ZH)**: RoGA: 通过稳健梯度对齐朝着通用深伪检测方向 

**Authors**: Lingyu Qiu, Ke Jiang, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20653)  

**Abstract**: Recent advancements in domain generalization for deepfake detection have attracted significant attention, with previous methods often incorporating additional modules to prevent overfitting to domain-specific patterns. However, such regularization can hinder the optimization of the empirical risk minimization (ERM) objective, ultimately degrading model performance. In this paper, we propose a novel learning objective that aligns generalization gradient updates with ERM gradient updates. The key innovation is the application of perturbations to model parameters, aligning the ascending points across domains, which specifically enhances the robustness of deepfake detection models to domain shifts. This approach effectively preserves domain-invariant features while managing domain-specific characteristics, without introducing additional regularization. Experimental results on multiple challenging deepfake detection datasets demonstrate that our gradient alignment strategy outperforms state-of-the-art domain generalization techniques, confirming the efficacy of our method. The code is available at this https URL. 

**Abstract (ZH)**: Recent advancements in领域迁移检测中的深度伪造检测取得了显著进展，先前的方法常常通过引入额外模块来防止过拟合到领域特定的模式。然而，这种正则化会妨碍经验风险最小化（ERM）目标的优化，最终降低模型性能。本文提出一种新的学习目标，将通用梯度更新与ERM梯度更新对齐。关键创新在于对模型参数施加扰动，使不同领域间的升梯度点保持一致，这特别增强了深度伪造检测模型对领域偏移的鲁棒性。该方法在保留领域不变特征的同时管理领域特定特征，无需引入额外的正则化。在多个具有挑战性的深度伪造检测数据集上的实验结果表明，我们的梯度对齐策略优于现有的领域迁移技术，证实了该方法的有效性。代码可访问：<该链接>。 

---
# Voronoi-grid-based Pareto Front Learning and Its Application to Collaborative Federated Learning 

**Title (ZH)**: 基于Voronoi网格的帕累托前沿学习及其在协作联邦学习中的应用 

**Authors**: Mengmeng Chen, Xiaohu Wu, Qiqi Liu, Tiantian He, Yew-Soon Ong, Yaochu Jin, Qicheng Lao, Han Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20648)  

**Abstract**: Multi-objective optimization (MOO) exists extensively in machine learning, and aims to find a set of Pareto-optimal solutions, called the Pareto front, e.g., it is fundamental for multiple avenues of research in federated learning (FL). Pareto-Front Learning (PFL) is a powerful method implemented using Hypernetworks (PHNs) to approximate the Pareto front. This method enables the acquisition of a mapping function from a given preference vector to the solutions on the Pareto front. However, most existing PFL approaches still face two challenges: (a) sampling rays in high-dimensional spaces; (b) failing to cover the entire Pareto Front which has a convex shape. Here, we introduce a novel PFL framework, called as PHN-HVVS, which decomposes the design space into Voronoi grids and deploys a genetic algorithm (GA) for Voronoi grid partitioning within high-dimensional space. We put forward a new loss function, which effectively contributes to more extensive coverage of the resultant Pareto front and maximizes the HV Indicator. Experimental results on multiple MOO machine learning tasks demonstrate that PHN-HVVS outperforms the baselines significantly in generating Pareto front. Also, we illustrate that PHN-HVVS advances the methodologies of several recent problems in the FL field. The code is available at this https URL}{this https URL. 

**Abstract (ZH)**: 多目标优化（MOO）在机器学习中广泛存在，旨在找到一组帕雷to最优解，即帕雷托前沿，例如，在联邦学习（FL）的多个研究领域中具有基础性意义。帕雷托前沿学习（PFL）是一种使用超网络（PHNs）实现的方法，用于近似帕雷托前沿。该方法使得能够从给定的偏好向量获得映射函数到帕雷托前沿上的解。然而，现有的大多数PFL方法仍面临两个挑战：（a）在高维空间中采样射线；（b）无法完全覆盖具有凸形状的帕雷托前沿。在这里，我们引入了一种新的PFL框架，称为PHN-HVVS，该框架将设计空间分解为Voronoi网格，并在高维空间中部署遗传算法（GA）进行Voronoi网格划分。我们提出了一个新的损失函数，有效地促进了帕雷托前沿覆盖范围的扩展并最大化HV指标。在多个多目标优化（MOO）机器学习任务上的实验结果表明，PHN-HVVS在生成帕雷托前沿方面显著优于基线方法。此外，我们展示了PHN-HVVS在联邦学习（FL）领域的多个最新问题中的方法学进步。代码可在以下链接获取：this https URL。 

---
# Evaluating Training in Binarized Neural Networks Through the Lens of Algorithmic Information Theory 

**Title (ZH)**: 通过算法信息论的视角评估二值神经网络的训练 

**Authors**: Eduardo Y. Sakabe, Felipe S. Abrahão, Alexandre Simões, Esther Colombini, Paula Costa, Ricardo Gudwin, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2505.20646)  

**Abstract**: Understanding and controlling the informational complexity of neural networks is a central challenge in machine learning, with implications for generalization, optimization, and model capacity. While most approaches rely on entropy-based loss functions and statistical metrics, these measures often fail to capture deeper, causally relevant algorithmic regularities embedded in network structure. We propose a shift toward algorithmic information theory, using Binarized Neural Networks (BNNs) as a first proxy. Grounded in algorithmic probability (AP) and the universal distribution it defines, our approach characterizes learning dynamics through a formal, causally grounded lens. We apply the Block Decomposition Method (BDM) -- a scalable approximation of algorithmic complexity based on AP -- and demonstrate that it more closely tracks structural changes during training than entropy, consistently exhibiting stronger correlations with training loss across varying model sizes and randomized training runs. These results support the view of training as a process of algorithmic compression, where learning corresponds to the progressive internalization of structured regularities. In doing so, our work offers a principled estimate of learning progression and suggests a framework for complexity-aware learning and regularization, grounded in first principles from information theory, complexity, and computability. 

**Abstract (ZH)**: 理解并控制神经网络的信息复杂性是机器学习中的一个核心挑战，这对泛化、优化和模型容量都有影响。我们提出将重点转向算法信息理论，使用二值神经网络（BNNs）作为初步代理。基于算法概率（AP）及其定义的通用分布，我们的方法通过形式化的因果视角来刻画学习动力学。我们应用块分解方法（BDM）——基于AP的一种可扩展的算法复杂性近似——证明了它在跟踪训练过程中结构变化方面比熵更接近，且在不同模型大小和随机训练运行中表现出更强的与训练损失的相关性。这些结果支持将训练视为一种算法压缩过程的观点，其中学习对应于结构规律的逐步内部化。我们的工作提供了一个基于信息论、复杂性和计算原理的原理性的学习进展估计，并建议了一种复杂性意识学习和正则化的框架。 

---
# TrustSkin: A Fairness Pipeline for Trustworthy Facial Affect Analysis Across Skin Tone 

**Title (ZH)**: TrustSkin: 跨肤色值得信赖的面部情绪分析公平性流程 

**Authors**: Ana M. Cabanas, Alma Pedro, Domingo Mery  

**Link**: [PDF](https://arxiv.org/pdf/2505.20637)  

**Abstract**: Understanding how facial affect analysis (FAA) systems perform across different demographic groups requires reliable measurement of sensitive attributes such as ancestry, often approximated by skin tone, which itself is highly influenced by lighting conditions. This study compares two objective skin tone classification methods: the widely used Individual Typology Angle (ITA) and a perceptually grounded alternative based on Lightness ($L^*$) and Hue ($H^*$). Using AffectNet and a MobileNet-based model, we assess fairness across skin tone groups defined by each method. Results reveal a severe underrepresentation of dark skin tones ($\sim 2 \%$), alongside fairness disparities in F1-score (up to 0.08) and TPR (up to 0.11) across groups. While ITA shows limitations due to its sensitivity to lighting, the $H^*$-$L^*$ method yields more consistent subgrouping and enables clearer diagnostics through metrics such as Equal Opportunity. Grad-CAM analysis further highlights differences in model attention patterns by skin tone, suggesting variation in feature encoding. To support future mitigation efforts, we also propose a modular fairness-aware pipeline that integrates perceptual skin tone estimation, model interpretability, and fairness evaluation. These findings emphasize the relevance of skin tone measurement choices in fairness assessment and suggest that ITA-based evaluations may overlook disparities affecting darker-skinned individuals. 

**Abstract (ZH)**: 面部情感分析系统在不同人群中的表现理解需要可靠的敏感属性测量，如族裔，通常通过肤色近似，而肤色本身高度受光照条件影响。本研究比较了两种客观肤色分类方法：广泛使用的个体分类角度（ITA）和基于亮度（$L^*$）和色调（$H^*$）的感知基础替代方法。使用AffectNet和基于MobileNet的模型，我们评估了每种方法定义的肤色组别之间的公平性。结果表明肤色较深的群体严重不足（约占2%），并在F1分数（高达0.08）和真正阳性率（高达0.11）方面表现出公平性差异。虽然ITA因其对光照的敏感性而显示出局限性，但$H^*$-$L^*$方法能够实现更一致的子群体划分，并通过均衡机会等指标提供更清晰的诊断。进一步的Grad-CAM分析强调了不同肤色群体间模型注意力模式的差异，暗示了特征编码的差异。为了支持未来的缓解努力，我们还提出了一种模块化的公平性意识流水线，该流水线结合了感知肤色估计、模型可解释性和公平性评估。这些发现强调了在公平性评估中肤色测量选择的重要性，并暗示基于ITA的评估可能忽略了影响肤色较深个体的不平等现象。 

---
# SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation 

**Title (ZH)**: SeqPO-SiMT: 序列策略优化的同时机器翻译 

**Authors**: Ting Xu, Zhichao Huang, Jiankai Sun, Shanbo Cheng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.20622)  

**Abstract**: We present Sequential Policy Optimization for Simultaneous Machine Translation (SeqPO-SiMT), a new policy optimization framework that defines the simultaneous machine translation (SiMT) task as a sequential decision making problem, incorporating a tailored reward to enhance translation quality while reducing latency. In contrast to popular Reinforcement Learning from Human Feedback (RLHF) methods, such as PPO and DPO, which are typically applied in single-step tasks, SeqPO-SiMT effectively tackles the multi-step SiMT task. This intuitive framework allows the SiMT LLMs to simulate and refine the SiMT process using a tailored reward. We conduct experiments on six datasets from diverse domains for En to Zh and Zh to En SiMT tasks, demonstrating that SeqPO-SiMT consistently achieves significantly higher translation quality with lower latency. In particular, SeqPO-SiMT outperforms the supervised fine-tuning (SFT) model by 1.13 points in COMET, while reducing the Average Lagging by 6.17 in the NEWSTEST2021 En to Zh dataset. While SiMT operates with far less context than offline translation, the SiMT results of SeqPO-SiMT on 7B LLM surprisingly rival the offline translation of high-performing LLMs, including Qwen-2.5-7B-Instruct and LLaMA-3-8B-Instruct. 

**Abstract (ZH)**: Sequential Policy Optimization for Simultaneous Machine Translation (SeqPO-SiMT) 

---
# Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习中多层认证防护对抗中毒攻击 

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.20621)  

**Abstract**: Similar to other machine learning frameworks, Offline Reinforcement Learning (RL) is shown to be vulnerable to poisoning attacks, due to its reliance on externally sourced datasets, a vulnerability that is exacerbated by its sequential nature. To mitigate the risks posed by RL poisoning, we extend certified defenses to provide larger guarantees against adversarial manipulation, ensuring robustness for both per-state actions, and the overall expected cumulative reward. Our approach leverages properties of Differential Privacy, in a manner that allows this work to span both continuous and discrete spaces, as well as stochastic and deterministic environments -- significantly expanding the scope and applicability of achievable guarantees. Empirical evaluations demonstrate that our approach ensures the performance drops to no more than $50\%$ with up to $7\%$ of the training data poisoned, significantly improving over the $0.008\%$ in prior work~\citep{wu_copa_2022}, while producing certified radii that is $5$ times larger as well. This highlights the potential of our framework to enhance safety and reliability in offline RL. 

**Abstract (ZH)**: 类似于其他机器学习框架，离线强化学习（RL）由于依赖外部数据集，且具有序列性，被证明容易受到中毒攻击。为了减轻RL中毒带来的风险，我们扩展了认证防御，以提供更大的保证对抗恶意操控，确保对每个状态的动作和总体预期累积奖励的鲁棒性。我们的方法利用差分隐私的性质，这使得该工作能够适用于连续和离散空间，以及随机性和确定性环境，极大地扩展了可实现保证的范围和适用性。实验评估表明，即使有高达7%的训练数据被中毒，我们的方法也能确保性能下降不超过50%，显著优于先前工作中的0.008%，同时生成的认证半径也增大了5倍。这突显了我们框架在离线RL安全性与可靠性方面的潜在优势。 

---
# REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning 

**Title (ZH)**: REAL-Prover: 获取增强的精简证明器 for 数学推理 

**Authors**: Ziju Shen, Naohao Huang, Fanyi Yang, Yutong Wang, Guoxiong Gao, Tianyi Xu, Jiedong Jiang, Wanyi He, Pu Yang, Mengzhou Sun, Haocheng Ju, Peihao Wu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20613)  

**Abstract**: Nowadays, formal theorem provers have made monumental progress on high-school and competition-level mathematics, but few of them generalize to more advanced mathematics. In this paper, we present REAL-Prover, a new open-source stepwise theorem prover for Lean 4 to push this boundary. This prover, based on our fine-tuned large language model (REAL-Prover-v1) and integrated with a retrieval system (Leansearch-PS), notably boosts performance on solving college-level mathematics problems. To train REAL-Prover-v1, we developed HERALD-AF, a data extraction pipeline that converts natural language math problems into formal statements, and a new open-source Lean 4 interactive environment (Jixia-interactive) to facilitate synthesis data collection. In our experiments, our prover using only supervised fine-tune achieves competitive results with a 23.7% success rate (Pass@64) on the ProofNet dataset-comparable to state-of-the-art (SOTA) models. To further evaluate our approach, we introduce FATE-M, a new benchmark focused on algebraic problems, where our prover achieves a SOTA success rate of 56.7% (Pass@64). 

**Abstract (ZH)**: 如今，形式定理证明器在高中和竞赛级数学领域取得了巨大进展，但很少扩展到更高级的数学领域。本文介绍了REAL-Prover，这是一个新的基于Lean 4的逐步定理证明器，旨在突破这一界限。该证明器基于我们微调的大语言模型（REAL-Prover-v1）并集成了检索系统（Leansearch-PS），显著提升了解决本科级数学问题的性能。为了训练REAL-Prover-v1，我们开发了HERALD-AF数据提取管道，将自然语言数学问题转换为形式化声明，并开发了一个新的开源Lean 4交互式环境（Jixia-interactive）以促进合成数据收集。在我们的实验中，仅使用监督微调的证明器在ProofNet数据集上的成功率为23.7%（Pass@64），与最先进的（SOTA）模型相当。为进一步评估我们的方法，我们引入了FATE-M，一个专注于代数问题的新基准，我们的证明器在该基准上的成功率为56.7%（Pass@64）。 

---
# The challenge of hidden gifts in multi-agent reinforcement learning 

**Title (ZH)**: 多智能体强化学习中的隐藏礼物挑战 

**Authors**: Dane Malenfant, Blake A. Richards  

**Link**: [PDF](https://arxiv.org/pdf/2505.20579)  

**Abstract**: Sometimes we benefit from actions that others have taken even when we are unaware that they took those actions. For example, if your neighbor chooses not to take a parking spot in front of your house when you are not there, you can benefit, even without being aware that they took this action. These "hidden gifts" represent an interesting challenge for multi-agent reinforcement learning (MARL), since assigning credit when the beneficial actions of others are hidden is non-trivial. Here, we study the impact of hidden gifts with a very simple MARL task. In this task, agents in a grid-world environment have individual doors to unlock in order to obtain individual rewards. As well, if all the agents unlock their door the group receives a larger collective reward. However, there is only one key for all of the doors, such that the collective reward can only be obtained when the agents drop the key for others after they use it. Notably, there is nothing to indicate to an agent that the other agents have dropped the key, thus the act of dropping the key for others is a "hidden gift". We show that several different state-of-the-art RL algorithms, including MARL algorithms, fail to learn how to obtain the collective reward in this simple task. Interestingly, we find that independent model-free policy gradient agents can solve the task when we provide them with information about their own action history, but MARL agents still cannot solve the task with action history. Finally, we derive a correction term for these independent agents, inspired by learning aware approaches, which reduces the variance in learning and helps them to converge to collective success more reliably. These results show that credit assignment in multi-agent settings can be particularly challenging in the presence of "hidden gifts", and demonstrate that learning awareness in independent agents can benefit these settings. 

**Abstract (ZH)**: 有时我们因他人不知情的行为而受益：多智能体强化学习中隐藏礼物的挑战及解决方案 

---
# Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL 

**Title (ZH)**: Ctrl-DNA：基于约束RL的可控制细胞类型特异性调节DNA设计 

**Authors**: Xingyu Chen, Shihao Ma, Runsheng Lin, Jiecong Lin, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20578)  

**Abstract**: Designing regulatory DNA sequences that achieve precise cell-type-specific gene expression is crucial for advancements in synthetic biology, gene therapy and precision medicine. Although transformer-based language models (LMs) can effectively capture patterns in regulatory DNA, their generative approaches often struggle to produce novel sequences with reliable cell-specific activity. Here, we introduce Ctrl-DNA, a novel constrained reinforcement learning (RL) framework tailored for designing regulatory DNA sequences with controllable cell-type specificity. By formulating regulatory sequence design as a biologically informed constrained optimization problem, we apply RL to autoregressive genomic LMs, enabling the models to iteratively refine sequences that maximize regulatory activity in targeted cell types while constraining off-target effects. Our evaluation on human promoters and enhancers demonstrates that Ctrl-DNA consistently outperforms existing generative and RL-based approaches, generating high-fitness regulatory sequences and achieving state-of-the-art cell-type specificity. Moreover, Ctrl-DNA-generated sequences capture key cell-type-specific transcription factor binding sites (TFBS), short DNA motifs recognized by regulatory proteins that control gene expression, demonstrating the biological plausibility of the generated sequences. 

**Abstract (ZH)**: 设计实现精确细胞类型特异性基因表达的调控DNA序列对于合成生物学、基因疗法和精准医学的发展至关重要。尽管基于变换器的语言模型（LMs）能够有效捕获调控DNA中的模式，但其生成方法往往难以产生具有可靠细胞类型特异活性的新型序列。我们引入了Ctrl-DNA，这是一种针对设计具有可控细胞类型特异性的调控DNA序列的新型约束强化学习（RL）框架。通过将调控序列设计形式化为生物学导向的约束优化问题，我们将RL应用于自回归基因LMs，使模型能够迭代地优化序列，以最大化目标细胞类型中的调控活性，同时限制非目标效应。在人类启动子和增强子上的评估表明，Ctrl-DNA始终优于现有生成性和基于RL的方法，生成高适应度的调控序列并实现最先进的细胞类型特异性。此外，Ctrl-DNA生成的序列捕获了关键的细胞类型特异性转录因子结合位点（TFBS），这些位点是被调控蛋白识别的短DNA动机，控制基因表达，显示出生成序列的生物可行性。 

---
# Electrolyzers-HSI: Close-Range Multi-Scene Hyperspectral Imaging Benchmark Dataset 

**Title (ZH)**: 电解槽-HSI：近距离多场景高光谱成像基准数据集 

**Authors**: Elias Arbash, Ahmed Jamal Afifi, Ymane Belahsen, Margret Fuchs, Pedram Ghamisi, Paul Scheunders, Richard Gloaguen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20507)  

**Abstract**: The global challenge of sustainable recycling demands automated, fast, and accurate, state-of-the-art (SOTA) material detection systems that act as a bedrock for a circular economy. Democratizing access to these cutting-edge solutions that enable real-time waste analysis is essential for scaling up recycling efforts and fostering the Green Deal. In response, we introduce \textbf{Electrolyzers-HSI}, a novel multimodal benchmark dataset designed to accelerate the recovery of critical raw materials through accurate electrolyzer materials classification. The dataset comprises 55 co-registered high-resolution RGB images and hyperspectral imaging (HSI) data cubes spanning the 400--2500 nm spectral range, yielding over 4.2 million pixel vectors and 424,169 labeled ones. This enables non-invasive spectral analysis of shredded electrolyzer samples, supporting quantitative and qualitative material classification and spectral properties investigation. We evaluate a suite of baseline machine learning (ML) methods alongside SOTA transformer-based deep learning (DL) architectures, including Vision Transformer, SpectralFormer, and the Multimodal Fusion Transformer, to investigate architectural bottlenecks for further efficiency optimisation when deploying transformers in material identification. We implement zero-shot detection techniques and majority voting across pixel-level predictions to establish object-level classification robustness. In adherence to the FAIR data principles, the electrolyzers-HSI dataset and accompanying codebase are openly available at this https URL and this https URL, supporting reproducible research and facilitating the broader adoption of smart and sustainable e-waste recycling solutions. 

**Abstract (ZH)**: 全球可持续回收面临的挑战需要自动化、快速且准确的最新材料检测系统，这些系统是循环经济的基石。为扩大回收努力并促进绿色协议的实施，必须普及这些先进解决方案的使用权。为此，我们引入了**Electrolyzers-HSI**，这是一个新型多模态基准数据集，旨在通过准确的电解槽材料分类加速关键原材料的回收利用。该数据集包含55张高分辨率RGB图像和覆盖400-2500纳米光谱范围的高光谱成像（HSI）数据立方体，共产生超过420万个像素向量和424,169个标记向量。这允许对粉碎的电解槽样本进行非侵入性光谱分析，支持定量和定性材料分类以及光谱特性研究。我们评估了一系列基线机器学习（ML）方法以及最新的基于变压器的深度学习（DL）架构，包括视觉变压器、光谱former和多模态融合变压器，以调查部署变压器时的架构瓶颈，进一步优化材料识别效率。我们实施了零样本检测技术和像素级预测的多数投票来建立对象级别分类的鲁棒性。遵循FAIR数据原则，Electrolyzers-HSI数据集及其配套代码库在此开放访问：此链接和此链接，支持可再现研究并促进智能和可持续电子废物回收解决方案的更广泛应用。 

---
# ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis 

**Title (ZH)**: ArVoice：阿拉伯语音多说话人数据集 

**Authors**: Hawau Olamide Toyin, Rufael Marew, Humaid Alblooshi, Samar M. Magdy, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.20506)  

**Abstract**: We introduce ArVoice, a multi-speaker Modern Standard Arabic (MSA) speech corpus with diacritized transcriptions, intended for multi-speaker speech synthesis, and can be useful for other tasks such as speech-based diacritic restoration, voice conversion, and deepfake detection. ArVoice comprises: (1) a new professionally recorded set from six voice talents with diverse demographics, (2) a modified subset of the Arabic Speech Corpus; and (3) high-quality synthetic speech from two commercial systems. The complete corpus consists of a total of 83.52 hours of speech across 11 voices; around 10 hours consist of human voices from 7 speakers. We train three open-source TTS and two voice conversion systems to illustrate the use cases of the dataset. The corpus is available for research use. 

**Abstract (ZH)**: ArVoice：一个多说话者现代标准阿拉伯语发音语料库，包含标音转录，用于多说话者语音合成及其他任务的研究 

---
# Avoid Forgetting by Preserving Global Knowledge Gradients in Federated Learning with Non-IID Data 

**Title (ZH)**: 在非 iid 数据下保持全局知识梯度以避免遗忘的联邦学习 

**Authors**: Abhijit Chunduru, Majid Morafah, Mahdi Morafah, Vishnu Pandi Chellapandi, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20485)  

**Abstract**: The inevitable presence of data heterogeneity has made federated learning very challenging. There are numerous methods to deal with this issue, such as local regularization, better model fusion techniques, and data sharing. Though effective, they lack a deep understanding of how data heterogeneity can affect the global decision boundary. In this paper, we bridge this gap by performing an experimental analysis of the learned decision boundary using a toy example. Our observations are surprising: (1) we find that the existing methods suffer from forgetting and clients forget the global decision boundary and only learn the perfect local one, and (2) this happens regardless of the initial weights, and clients forget the global decision boundary even starting from pre-trained optimal weights. In this paper, we present FedProj, a federated learning framework that robustly learns the global decision boundary and avoids its forgetting during local training. To achieve better ensemble knowledge fusion, we design a novel server-side ensemble knowledge transfer loss to further calibrate the learned global decision boundary. To alleviate the issue of learned global decision boundary forgetting, we further propose leveraging an episodic memory of average ensemble logits on a public unlabeled dataset to regulate the gradient updates at each step of local training. Experimental results demonstrate that FedProj outperforms state-of-the-art methods by a large margin. 

**Abstract (ZH)**: 数据异构性的不可避免存在使得联邦学习极具挑战性。尽管有许多方法可以应对这一问题，如局部正则化、更好的模型融合技术和数据共享，但它们缺乏对数据异构性如何影响全局决策边界的深刻理解。在本文中，我们通过使用玩具示例进行实验分析，弥合了这一缺口。我们的观察令人惊讶：(1) 我们发现现有方法存在遗忘现象，客户端会忘记全局决策边界的知识，仅学习完美的局部决策边界；(2) 这种现象与初始权重无关，客户端即使从预训练的最优权重开始也会忘记全局决策边界。在本文中，我们提出了FedProj，这是一种能够在本地训练过程中避免全局决策边界遗忘的联邦学习框架。为实现更好的集成知识融合，我们设计了一种新颖的服务器端集成知识转移损失，进一步校准学习到的全局决策边界。为缓解学习到的全局决策边界遗忘的问题，我们进一步提出利用公共未标记数据集中平均集成logits的 episodic 记忆来调节每次本地训练步骤中的梯度更新。实验结果表明，FedProj 在性能上大幅优于现有最先进的方法。 

---
# Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding 

**Title (ZH)**: 对话核：一种灵活的机制，用于在线对话理解的相关背景学习 

**Authors**: Vibhor Agarwal, Arjoo Gupta, Suparna De, Nishanth Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2505.20482)  

**Abstract**: Understanding online conversations has attracted research attention with the growth of social networks and online discussion forums. Content analysis of posts and replies in online conversations is difficult because each individual utterance is usually short and may implicitly refer to other posts within the same conversation. Thus, understanding individual posts requires capturing the conversational context and dependencies between different parts of a conversation tree and then encoding the context dependencies between posts and comments/replies into the language model.
To this end, we propose a general-purpose mechanism to discover appropriate conversational context for various aspects about an online post in a conversation, such as whether it is informative, insightful, interesting or funny. Specifically, we design two families of Conversation Kernels, which explore different parts of the neighborhood of a post in the tree representing the conversation and through this, build relevant conversational context that is appropriate for each task being considered. We apply our developed method to conversations crawled from this http URL, which allows users to apply highly different labels to posts, such as 'insightful', 'funny', etc., and therefore provides an ideal experimental platform to study whether a framework such as Conversation Kernels is general-purpose and flexible enough to be adapted to disparately different conversation understanding tasks. 

**Abstract (ZH)**: 在线对话的理解随着社交网络和在线讨论论坛的增长而吸引了研究关注。由于在线对话中的每个个体发言通常都很短，并且可能隐含地引用同一对话内的其他帖子，因此理解个体发言需要捕获对话背景和对话树不同部分之间的依赖性，然后将帖子和评论/回复之间的背景依赖性编码到语言模型中。为此，我们提出了一种通用机制，以发现在线帖子在对话中涉及的各种方面的适当对话背景，例如是否有信息性、见解性、趣味性或幽默感。具体地，我们设计了两种类型的对话核函数，探索表示对话的树结构中帖子的邻域的不同部分，并通过这种方式构建适用于每项任务的相关对话背景。我们将开发的方法应用于从该网址爬取的对话，这些对话允许用户为帖子应用高度不同的标签，如“见解性”、“有趣”等，因此提供了理想的研究平台，以研究如对话核函数框架是否通用且灵活，能够适应不同对话理解任务。 

---
# CardioPatternFormer: Pattern-Guided Attention for Interpretable ECG Classification with Transformer Architecture 

**Title (ZH)**: CardioPatternFormer: 以模式为引导的注意力机制在Transformer架构下的心电图可解释分类 

**Authors**: Berat Kutay Uğraş, Ömer Nezih Gerek, İbrahim Talha Saygı  

**Link**: [PDF](https://arxiv.org/pdf/2505.20481)  

**Abstract**: Accurate ECG interpretation is vital, yet complex cardiac data and "black-box" AI models limit clinical utility. Inspired by Transformer architectures' success in NLP for understanding sequential data, we frame ECG as the heart's unique "language" of temporal patterns. We present CardioPatternFormer, a novel Transformer-based model for interpretable ECG classification. It employs a sophisticated attention mechanism to precisely identify and classify diverse cardiac patterns, excelling at discerning subtle anomalies and distinguishing multiple co-occurring conditions. This pattern-guided attention provides clear insights by highlighting influential signal regions, effectively allowing the "heart to talk" through transparent interpretations. CardioPatternFormer demonstrates robust performance on challenging ECGs, including complex multi-pathology cases. Its interpretability via attention maps enables clinicians to understand the model's rationale, fostering trust and aiding informed diagnostic decisions. This work offers a powerful, transparent solution for advanced ECG analysis, paving the way for more reliable and clinically actionable AI in cardiology. 

**Abstract (ZH)**: 准确的心电图解读至关重要，但复杂的心脏数据和“黑盒”AI模型限制了其临床应用。受Transformer架构在自然语言处理中理解序贯数据成功经验的启发，我们将心电图视为心脏的独特“语言”中的时间模式。我们提出了CardioPatternFormer，这是一种基于Transformer的可解释心电图分类模型。该模型采用复杂的注意力机制以精确识别和分类多种心脏模式，擅长区分微妙的异常和多种并发条件。这种基于模式的注意力通过突出显示有影响的信号区域，提供清晰的见解，有效地使“心脏得以发声”，并通过透明的解释增强信任。CardioPatternFormer在包括复杂多病理情况在内的挑战性心电图上表现出稳健的性能。其通过注意力图的可解释性使临床医生能够理解模型的推理过程，增强信任并辅助诊断决策。本研究提供了一种强大的、透明的高级心电图分析解决方案，为心脏病学中更可靠和临床可操作的AI铺平了道路。 

---
# Holes in Latent Space: Topological Signatures Under Adversarial Influence 

**Title (ZH)**: 潜在空间中的洞： adversarial 影响下的拓扑特征 

**Authors**: Aideen Fay, Inés García-Redondo, Qiquan Wang, Haim Dubossarsky, Anthea Monod  

**Link**: [PDF](https://arxiv.org/pdf/2505.20435)  

**Abstract**: Understanding how adversarial conditions affect language models requires techniques that capture both global structure and local detail within high-dimensional activation spaces. We propose persistent homology (PH), a tool from topological data analysis, to systematically characterize multiscale latent space dynamics in LLMs under two distinct attack modes -- backdoor fine-tuning and indirect prompt injection. By analyzing six state-of-the-art LLMs, we show that adversarial conditions consistently compress latent topologies, reducing structural diversity at smaller scales while amplifying dominant features at coarser ones. These topological signatures are statistically robust across layers, architectures, model sizes, and align with the emergence of adversarial effects deeper in the network. To capture finer-grained mechanisms underlying these shifts, we introduce a neuron-level PH framework that quantifies how information flows and transforms within and across layers. Together, our findings demonstrate that PH offers a principled and unifying approach to interpreting representational dynamics in LLMs, particularly under distributional shift. 

**Abstract (ZH)**: 理解对抗条件如何影响语言模型需要捕捉高维激活空间中全局结构和局部细节的技术。我们提出使用拓扑数据分析工具持久同调（PH）系统地 characterizing LLMs 在两种不同的攻击模式——后门微调和间接提示注入下的多尺度潜在空间动力学。通过分析六种最先进的LLMs，我们展示了对抗条件一致地压缩潜在拓扑结构，在较小尺度上减少结构多样性，而在较粗尺度上放大主导特征。这些拓扑特征在各层、架构、模型规模上具有统计鲁棒性，并与网络更深层次出现的对抗效应一致。为了捕捉这些转变的更细粒度机制，我们引入了一种神经元级的持久同调框架，量化信息在和跨层中的流动和转换。我们的研究结果共同表明，持久同调为理解LLMs中的表示动力学提供了一个原则性的统一方法，特别是在分布变化的情况下。 

---
# Algorithmic Control Improves Residential Building Energy and EV Management when PV Capacity is High but Battery Capacity is Low 

**Title (ZH)**: 高光伏发电量低电池容量条件下，算法控制改善住宅建筑能源与电动车管理 

**Authors**: Lennart Ullner, Alona Zharova, Felix Creutzig  

**Link**: [PDF](https://arxiv.org/pdf/2505.20377)  

**Abstract**: Efficient energy management in prosumer households is key to alleviating grid stress in an energy transition marked by electric vehicles (EV), renewable energies and battery storage. However, it is unclear how households optimize prosumer EV charging. Here we study real-world data from 90 households on fixed-rate electricity tariffs in German-speaking countries to investigate the potential of Deep Reinforcement Learning (DRL) and other control approaches (Rule-Based, Model Predictive Control) to manage the dynamic and uncertain environment of Home Energy Management (HEM) and optimize household charging patterns. The DRL agent efficiently aligns charging of EV and battery storage with photovoltaic (PV) surplus. We find that frequent EV charging transactions, early EV connections and PV surplus increase optimization potential. A detailed analysis of nine households (1 hour resolution, 1 year) demonstrates that high battery capacity facilitates self optimization; in this case further algorithmic control shows little value. In cases with relatively low battery capacity, algorithmic control with DRL improves energy management and cost savings by a relevant margin. This result is further corroborated by our simulation of a synthetic household. We conclude that prosumer households with optimization potential would profit from DRL, thus benefiting also the full electricity system and its decarbonization. 

**Abstract (ZH)**: 在电转、可再生能量和电池储能背景下，通过深强化学习和其他控制方法优化消费者-生产者家庭的能源管理对于缓解电网压力至关重要。然而，家庭如何优化消费者-生产者电动汽车充电尚不明确。我们研究了德国-speaking国家90户家庭的固定电价数据，以探讨深度强化学习（DRL）和其他控制方法（基于规则、模型预测控制）在家庭能源管理（HEM）动态和不确定环境下管理电动汽车和电池储能充电模式的潜力。我们发现，频繁的电动汽车充电交易、早期电动汽车连接和光伏剩余电量增加优化潜力。九户家庭的详细分析（每小时一次，一年）表明，高电池容量有助于自我优化；在这种情况下，进一步的算法控制几乎没有价值。在电池容量相对较低的情况下，使用DRL的算法控制可以显著改善能源管理和成本节省。这一结果进一步得到我们对合成家庭的模拟验证。我们得出结论，具有优化潜力的消费者-生产者家庭将从DRL中受益，从而也有利于整个电力系统及其脱碳。 

---
# GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: 粒度低秩适应：参数高效微调 

**Authors**: Yeonjoon Jung, Daehyun Ahn, Hyungjun Kim, Taesu Kim, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.20355)  

**Abstract**: Low-Rank Adaptation (LoRA) is a popular method for parameter-efficient fine-tuning (PEFT) of generative models, valued for its simplicity and effectiveness. Despite recent enhancements, LoRA still suffers from a fundamental limitation: overfitting when the bottleneck is widened. It performs best at ranks 32-64, yet its accuracy stagnates or declines at higher ranks, still falling short of full fine-tuning (FFT) performance. We identify the root cause as LoRA's structural bottleneck, which introduces gradient entanglement to the unrelated input channels and distorts gradient propagation. To address this, we introduce a novel structure, Granular Low-Rank Adaptation (GraLoRA) that partitions weight matrices into sub-blocks, each with its own low-rank adapter. With negligible computational or storage cost, GraLoRA overcomes LoRA's limitations, effectively increases the representational capacity, and more closely approximates FFT behavior. Experiments on code generation and commonsense reasoning benchmarks show that GraLoRA consistently outperforms LoRA and other baselines, achieving up to +8.5% absolute gain in Pass@1 on HumanEval+. These improvements hold across model sizes and rank settings, making GraLoRA a scalable and robust solution for PEFT. Code, data, and scripts are available at this https URL 

**Abstract (ZH)**: Granular Low-Rank Adaptation (GraLoRA): A Scalable and Robust Solution for Parameter-Efficient Fine-Tuning 

---
# FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation 

**Title (ZH)**: FastCache: 通过可学习的线性近似加速扩散变换器的缓存技术 

**Authors**: Dong Liu, Jiayi Zhang, Yifan Li, Yanxuan Yu, Ben Lengerich, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20353)  

**Abstract**: Diffusion Transformers (DiT) are powerful generative models but remain computationally intensive due to their iterative structure and deep transformer stacks. To alleviate this inefficiency, we propose FastCache, a hidden-state-level caching and compression framework that accelerates DiT inference by exploiting redundancy within the model's internal representations. FastCache introduces a dual strategy: (1) a spatial-aware token selection mechanism that adaptively filters redundant tokens based on hidden state saliency, and (2) a transformer-level cache that reuses latent activations across timesteps when changes are statistically insignificant. These modules work jointly to reduce unnecessary computation while preserving generation fidelity through learnable linear approximation. Theoretical analysis shows that FastCache maintains bounded approximation error under a hypothesis-testing-based decision rule. Empirical evaluations across multiple DiT variants demonstrate substantial reductions in latency and memory usage, with best generation output quality compared to other cache methods, as measured by FID and t-FID. Code implementation of FastCache is available on GitHub at this https URL. 

**Abstract (ZH)**: FastCache：一种用于加速Diffusion Transformers推测的隐藏状态级别缓存和压缩框架 

---
# PDFBench: A Benchmark for De novo Protein Design from Function 

**Title (ZH)**: PDFBench: 一种从功能出发的新型蛋白质设计基准测试 

**Authors**: Jiahao Kuang, Nuowei Liu, Changzhi Sun, Tao Ji, Yuanbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20346)  

**Abstract**: In recent years, while natural language processing and multimodal learning have seen rapid advancements, the field of de novo protein design has also experienced significant growth. However, most current methods rely on proprietary datasets and evaluation rubrics, making fair comparisons between different approaches challenging. Moreover, these methods often employ evaluation metrics that capture only a subset of the desired properties of designed proteins, lacking a comprehensive assessment framework. To address these, we introduce PDFBench, the first comprehensive benchmark for evaluating de novo protein design from function. PDFBench supports two tasks: description-guided design and keyword-guided design. To ensure fair and multifaceted evaluation, we compile 22 metrics covering sequence plausibility, structural fidelity, and language-protein alignment, along with measures of novelty and diversity. We evaluate five state-of-the-art baselines, revealing their respective strengths and weaknesses across tasks. Finally, we analyze inter-metric correlations, exploring the relationships between four categories of metrics, and offering guidelines for metric selection. PDFBench establishes a unified framework to drive future advances in function-driven de novo protein design. 

**Abstract (ZH)**: 近年来，尽管自然语言处理和多模态学习取得了 rapid advancements，从功能出发的 de novo 蛋白质设计领域也经历了显著增长。然而，目前大多数方法依赖于专有的数据集和评估标准，这使得不同方法之间的公平比较变得具有挑战性。此外，这些方法往往仅通过有限的评估指标来捕获设计蛋白质的期望属性，缺乏一个全面的评估框架。为了应对这些问题，我们引入了 PDFBench——第一个面向功能的 de novo 蛋白质设计综合基准。PDFBench 支持描述导向设计和关键词导向设计两种任务。为了确保公平和多维度的评估，我们编译了 22 个指标，涵盖了序列合理性、结构准确性以及语言-蛋白质对齐，同时包括新颖性和多样性衡量指标。我们评估了五种最先进的基线方法，在不同任务中揭示了它们各自的优劣。最后，我们分析了指标间的相关性，探讨了四类指标之间的关系，并提出了指标选择的指导原则。PDFBench 为推动功能导向的 de novo 蛋白质设计的未来进展建立了统一框架。 

---
# Towards Emotionally Consistent Text-Based Speech Editing: Introducing EmoCorrector and The ECD-TSE Dataset 

**Title (ZH)**: 基于文本的语音编辑中情感一致性的追求： introduce EmoCorrector 和 ECD-TSE 数据集 

**Authors**: Rui Liu, Pu Gao, Jiatian Xi, Berrak Sisman, Carlos Busso, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20341)  

**Abstract**: Text-based speech editing (TSE) modifies speech using only text, eliminating re-recording. However, existing TSE methods, mainly focus on the content accuracy and acoustic consistency of synthetic speech segments, and often overlook the emotional shifts or inconsistency issues introduced by text changes. To address this issue, we propose EmoCorrector, a novel post-correction scheme for TSE. EmoCorrector leverages Retrieval-Augmented Generation (RAG) by extracting the edited text's emotional features, retrieving speech samples with matching emotions, and synthesizing speech that aligns with the desired emotion while preserving the speaker's identity and quality. To support the training and evaluation of emotional consistency modeling in TSE, we pioneer the benchmarking Emotion Correction Dataset for TSE (ECD-TSE). The prominent aspect of ECD-TSE is its inclusion of $<$text, speech$>$ paired data featuring diverse text variations and a range of emotional expressions. Subjective and objective experiments and comprehensive analysis on ECD-TSE confirm that EmoCorrector significantly enhances the expression of intended emotion while addressing emotion inconsistency limitations in current TSE methods. Code and audio examples are available at this https URL. 

**Abstract (ZH)**: 基于文本的语音编辑中的情感修正：EmoCorrector方法 

---
# Data-driven multi-agent modelling of calcium interactions in cell culture: PINN vs Regularized Least-squares 

**Title (ZH)**: 基于数据的细胞培养中钙离子相互作用多agent建模：比较PINN与正则化最小二乘方法 

**Authors**: Aurora Poggi, Giuseppe Alessio D'Inverno, Hjalmar Brismar, Ozan Öktem, Matthieu Barreau, Kateryna Morozovska  

**Link**: [PDF](https://arxiv.org/pdf/2505.20327)  

**Abstract**: Data-driven discovery of dynamics in biological systems allows for better observation and characterization of processes, such as calcium signaling in cell culture. Recent advancements in techniques allow the exploration of previously unattainable insights of dynamical systems, such as the Sparse Identification of Non-Linear Dynamics (SINDy), overcoming the limitations of more classic methodologies. The latter requires some prior knowledge of an effective library of candidate terms, which is not realistic for a real case study. Using inspiration from fields like traffic density estimation and control theory, we propose a methodology for characterization and performance analysis of calcium delivery in a family of cells. In this work, we compare the performance of the Constrained Regularized Least-Squares Method (CRLSM) and Physics-Informed Neural Networks (PINN) for system identification and parameter discovery for governing ordinary differential equations (ODEs). The CRLSM achieves a fairly good parameter estimate and a good data fit when using the learned parameters in the Consensus problem. On the other hand, despite the initial hypothesis, PINNs fail to match the CRLSM performance and, under the current configuration, do not provide fair parameter estimation. However, we have only studied a limited number of PINN architectures, and it is expected that additional hyperparameter tuning, as well as uncertainty quantification, could significantly improve the performance in future works. 

**Abstract (ZH)**: 基于数据的动力学发现方法在生物学系统中的应用允许更准确地观察和表征过程，例如细胞培养中的钙信号传导。近期技术进步使得可以探索动态系统的以往难以获得的见解，如稀疏非线性动力学识别（SINDy）方法克服了经典方法的局限性。后者需要一些有效的候选项库先验知识，而在实际研究案例中这是不现实的。借鉴交通密度估计和控制理论的灵感，我们提出了一种用于细胞家族中钙传递表征和性能分析的方法学。在本研究中，我们将约束正则最小二乘法（CRLSM）和物理导向神经网络（PINN）用于系统识别和管理常微分方程（ODEs）的参数发现性能进行对比。CRLSM在使用所学参数解决共识问题时获得了相当不错的参数估计和数据拟合效果。相反，尽管最初假设PINNs能够达到CRLSM的性能，但在当前配置下，它们未能提供合理的参数估计。然而，我们仅研究了有限的PINN架构，预计在未来的工作中通过额外的超参数调优和不确定性量化可以显著提高其性能。 

---
# Cultural Awareness in Vision-Language Models: A Cross-Country Exploration 

**Title (ZH)**: Vision-Language模型中的文化awareness：跨国家探索 

**Authors**: Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2505.20326)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in diverse cultural contexts, yet their internal biases remain poorly understood. In this work, we propose a novel framework to systematically evaluate how VLMs encode cultural differences and biases related to race, gender, and physical traits across countries. We introduce three retrieval-based tasks: (1) Race to Country retrieval, which examines the association between individuals from specific racial groups (East Asian, White, Middle Eastern, Latino, South Asian, and Black) and different countries; (2) Personal Traits to Country retrieval, where images are paired with trait-based prompts (e.g., Smart, Honest, Criminal, Violent) to investigate potential stereotypical associations; and (3) Physical Characteristics to Country retrieval, focusing on visual attributes like skinny, young, obese, and old to explore how physical appearances are culturally linked to nations. Our findings reveal persistent biases in VLMs, highlighting how visual representations may inadvertently reinforce societal stereotypes. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在多种文化背景下日益广泛应用，但其内部偏见仍然知之甚少。本文提出了一种新型框架，系统评估VLMs在种族、性别和身体特征等方面对不同国家的文化差异和偏见的编码。我们介绍了三种检索任务：（1）种族到国家检索，探索特定种族群体（东亚人、白人、中东人、拉丁裔、南亚人、黑人）与不同国家之间的关联；（2）个性特征到国家检索，将图像与基于特质的提示（如聪明的、诚实的、犯罪的、暴力的）配对，以调查潜在的刻板印象关联；（3）身体特征到国家检索，重点关注身体外观属性（如瘦削、年轻、肥胖、年老），以探究身体外观如何与国家文化联系起来。我们的研究发现表明VLMs中存在持续的偏见，揭示了视觉表示如何无意中加强了社会刻板印象。 

---
# PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus 

**Title (ZH)**: PMOA-TTS: 引入PubMed开放获取文本时间序列语料库 

**Authors**: Shahriar Noroozizadeh, Sayantan Kumar, George H. Chen, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.20323)  

**Abstract**: Understanding temporal dynamics in clinical narratives is essential for modeling patient trajectories, yet large-scale temporally annotated resources remain limited. We present PMOA-TTS, the first openly available dataset of 124,699 PubMed Open Access (PMOA) case reports, each converted into structured (event, time) timelines via a scalable LLM-based pipeline. Our approach combines heuristic filtering with Llama 3.3 to identify single-patient case reports, followed by prompt-driven extraction using Llama 3.3 and DeepSeek R1, resulting in over 5.6 million timestamped clinical events. To assess timeline quality, we evaluate against a clinician-curated reference set using three metrics: (i) event-level matching (80% match at a cosine similarity threshold of 0.1), (ii) temporal concordance (c-index > 0.90), and (iii) Area Under the Log-Time CDF (AULTC) for timestamp alignment. Corpus-level analysis shows wide diagnostic and demographic coverage. In a downstream survival prediction task, embeddings from extracted timelines achieve time-dependent concordance indices up to 0.82 $\pm$ 0.01, demonstrating the predictive value of temporally structured narratives. PMOA-TTS provides a scalable foundation for timeline extraction, temporal reasoning, and longitudinal modeling in biomedical NLP. The dataset is available at: this https URL . 

**Abstract (ZH)**: 理解临床叙述中的时间动态对于建模患者轨迹至关重要，但大规模的时间注释资源仍然有限。我们提出了PMOA-TTS，这是首个开放获取的数据集，包含124,699篇PubMed Open Access (PMOA) 案例报告，每个报告都通过一个可扩展的基于大规模语言模型的管道转换为结构化的时间线（事件，时间）。我们的方法结合启发式过滤与Llama 3.3来识别单个病例报告，随后使用Llama 3.3和DeepSeek R1进行提示驱动提取，结果共生成超过560万条带有时间戳的临床事件。为了评估时间线质量，我们使用三种指标与临床专家整理的参考集进行评估：（i）事件级匹配（余弦相似度阈值0.1时80%匹配），（ii）时间一致性（c指数>0.90），（iii）对数时间累积分布函数下的面积（AULTC）用于时间戳对齐。语料库级别分析显示广泛覆盖诊断和人口统计学特征。在下游生存预测任务中，从提取的时间线生成的嵌入物实现时间依赖一致性指数最高可达0.82 ± 0.01，展示了时间结构化叙述的预测价值。PMOA-TTS为时间线提取、时间推理和纵向建模提供了可扩展的基础。该数据集可从此链接获取：this https URL。 

---
# BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases 

**Title (ZH)**: BiomedSQL: 文本到SQL在生物医学知识库上的科学推理 

**Authors**: Mathew J. Koretsky, Maya Willey, Adi Asija, Owen Bianchi, Chelsea X. Alvarado, Tanay Nayak, Nicole Kuznetsov, Sungwon Kim, Mike A. Nalls, Daniel Khashabi, Faraz Faghri  

**Link**: [PDF](https://arxiv.org/pdf/2505.20321)  

**Abstract**: Biomedical researchers increasingly rely on large-scale structured databases for complex analytical tasks. However, current text-to-SQL systems often struggle to map qualitative scientific questions into executable SQL, particularly when implicit domain reasoning is required. We introduce BiomedSQL, the first benchmark explicitly designed to evaluate scientific reasoning in text-to-SQL generation over a real-world biomedical knowledge base. BiomedSQL comprises 68,000 question/SQL query/answer triples grounded in a harmonized BigQuery knowledge base that integrates gene-disease associations, causal inference from omics data, and drug approval records. Each question requires models to infer domain-specific criteria, such as genome-wide significance thresholds, effect directionality, or trial phase filtering, rather than rely on syntactic translation alone. We evaluate a range of open- and closed-source LLMs across prompting strategies and interaction paradigms. Our results reveal a substantial performance gap: GPT-o3-mini achieves 59.0% execution accuracy, while our custom multi-step agent, BMSQL, reaches 62.6%, both well below the expert baseline of 90.0%. BiomedSQL provides a new foundation for advancing text-to-SQL systems capable of supporting scientific discovery through robust reasoning over structured biomedical knowledge bases. Our dataset is publicly available at this https URL, and our code is open-source at this https URL. 

**Abstract (ZH)**: 生物医学研究人员 increasingly 依赖大规模结构化数据库进行复杂的分析任务。然而，当前的文本到SQL系统在将定性的科学问题映射为可执行的SQL查询时经常遇到困难，特别是在需要隐含领域推理的情况下。我们引入了BiomedSQL，这是第一个明确设计用于评估在真实世界生物医学知识库上进行文本到SQL生成过程中科学推理能力的标准。BiomedSQL 包含 68,000 个问题/SQL 查询/答案三元组，这些三元组基于一个整合了基因-疾病关联、从组学数据中推断因果关系以及药物批准记录的统一 BigQuery 知识库。每个问题都需要模型推断领域特定的标准，如全基因组显著性阈值、效应方向或试验阶段筛选，而不仅仅依赖于语法翻译。我们评估了多种开源和闭源的大规模语言模型 (LLM) 以及不同提示策略和交互模式。我们的结果显示了显著的性能差距：GPT-o3-mini 的执行准确率为 59.0%，而我们自定义的多步代理 BMSQL 达到 62.6%，均远低于专家基准的 90.0%。BiomedSQL 为推进能够通过结构化生物医学知识库进行强大推理的支持科学发现的文本到SQL系统的进步提供了新的基础。我们的数据集可以通过以下链接公开访问，代码是开源的：this https URL。 

---
# Future of Code with Generative AI: Transparency and Safety in the Era of AI Generated Software 

**Title (ZH)**: 代码的未来：生成式人工智能时代软件生成的透明性和安全性 

**Authors**: David Hanson  

**Link**: [PDF](https://arxiv.org/pdf/2505.20303)  

**Abstract**: As artificial intelligence becomes increasingly integrated into software development processes, the prevalence and sophistication of AI-generated code continue to expand rapidly. This study addresses the critical need for transparency and safety in AI generated code by examining the current landscape, identifying potential risks, and exploring future implications. We analyze market opportunities for detecting AI-generated code, discuss the challenges associated with managing increasing complexity, and propose solutions to enhance transparency and functionality analysis. Furthermore, this study investigates the longterm implications of AI generated code, including its potential role in the development of artificial general intelligence and its impact on human AI interaction. In conclusion, we emphasize the importance of proactive measures for ensuring the responsible development and deployment of AI in software engineering. 

**Abstract (ZH)**: 随着人工智能在软件开发过程中的不断集成，由人工智能生成的代码的普遍存在性和复杂性也在迅速扩展。本研究旨在通过分析当前状况、识别潜在风险并探讨未来 implications，来应对人工智能生成代码在透明度和安全性方面的重要需求。我们分析了检测人工智能生成代码的市场机遇，讨论了管理不断增长的复杂性的挑战，并提出了增强透明度和功能分析的解决方案。此外，本研究还探讨了人工智能生成代码的长期 implications，包括其在开发通用人工智能方面的作用及其对人类与人工智能互动的影响。最后，我们强调了采取积极措施以负责任地开发和部署人工智能的重要性。 

---
# VeriThoughts: Enabling Automated Verilog Code Generation using Reasoning and Formal Verification 

**Title (ZH)**: VeriThoughts: 通过推理和形式验证实现自动化Verilog代码生成 

**Authors**: Patrick Yubeaton, Andre Nakkab, Weihua Xiao, Luca Collini, Ramesh Karri, Chinmay Hegde, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2505.20302)  

**Abstract**: This paper introduces VeriThoughts, a novel dataset designed for reasoning-based Verilog code generation. We establish a new benchmark framework grounded in formal verification methods to evaluate the quality and correctness of generated hardware descriptions. Additionally, we present a suite of specialized small-scale models optimized specifically for Verilog generation. Our work addresses the growing need for automated hardware design tools that can produce verifiably correct implementations from high-level specifications, potentially accelerating the hardware development process while maintaining rigorous correctness guarantees. Our code and data are available at \href{this https URL}{this URL}. 

**Abstract (ZH)**: 本文介绍了VeriThoughts，一个用于基于推理的Verilog代码生成的新数据集。我们建立了一个新的基准框架，基于形式验证方法来评估生成的硬件描述的质量和正确性。此外，我们还呈现了一套专门针对Verilog生成优化的小规模模型。我们的工作解决了从高层次规范自动生成可验证正确实现的日益增长需求，可能加速硬件开发过程同时保持严格的正确性保证。代码和数据可在<该网址>获得。 

---
# MetamatBench: Integrating Heterogeneous Data, Computational Tools, and Visual Interface for Metamaterial Discovery 

**Title (ZH)**: MetamatBench：集成异构数据、计算工具及可视化界面的Metamaterial发现平台 

**Authors**: Jianpeng Chen, Wangzhi Zhan, Haohui Wang, Zian Jia, Jingru Gan, Junkai Zhang, Jingyuan Qi, Tingwei Chen, Lifu Huang, Muhao Chen, Ling Li, Wei Wang, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20299)  

**Abstract**: Metamaterials, engineered materials with architected structures across multiple length scales, offer unprecedented and tunable mechanical properties that surpass those of conventional materials. However, leveraging advanced machine learning (ML) for metamaterial discovery is hindered by three fundamental challenges: (C1) Data Heterogeneity Challenge arises from heterogeneous data sources, heterogeneous composition scales, and heterogeneous structure categories; (C2) Model Complexity Challenge stems from the intricate geometric constraints of ML models, which complicate their adaptation to metamaterial structures; and (C3) Human-AI Collaboration Challenge comes from the "dual black-box'' nature of sophisticated ML models and the need for intuitive user interfaces. To tackle these challenges, we introduce a unified framework, named MetamatBench, that operates on three levels. (1) At the data level, we integrate and standardize 5 heterogeneous, multi-modal metamaterial datasets. (2) The ML level provides a comprehensive toolkit that adapts 17 state-of-the-art ML methods for metamaterial discovery. It also includes a comprehensive evaluation suite with 12 novel performance metrics with finite element-based assessments to ensure accurate and reliable model validation. (3) The user level features a visual-interactive interface that bridges the gap between complex ML techniques and non-ML researchers, advancing property prediction and inverse design of metamaterials for research and applications. MetamatBench offers a unified platform deployed at this http URL that enables machine learning researchers and practitioners to develop and evaluate new methodologies in metamaterial discovery. For accessibility and reproducibility, we open-source our benchmark and the codebase at this https URL. 

**Abstract (ZH)**: metamaterials设计中的先进机器学习统一框架MetamatBench：跨越多尺度架构的工程材料 offers前所未有的可调机械性能，超越了传统材料。然而，利用先进的机器学习（ML）进行metamaterials发现受到三项基本挑战的阻碍：(C1) 数据异质性挑战来自于异质数据源、异质组成尺度和异质结构类别；(C2) 模型复杂性挑战来自于ML模型的复杂几何约束，使其适应metamaterial结构复杂化；(C3) 人机合作挑战来自于复杂ML模型的“双重黑盒”性质和需要直观用户界面。为应对这些挑战，我们引入了一个统一框架MetamatBench，该框架在三个层级上运行。(1) 在数据层，我们整合并标准化了5个异质多模态metamaterial数据集。(2) 在ML层提供了全面的工具包，将17种最新的ML方法适应于metamaterials发现，并包括基于有限元评估的全面评估套件，含有12种新颖的性能指标，以确保模型验证的准确性和可靠性。(3) 在用户层，提供了一个可视化交互界面，弥合了复杂ML技术与非ML研究人员之间的差距，推动了metamaterials的属性预测和逆向设计，用于研究和应用。MetamatBench提供了一个统一平台，使得机器学习研究人员和实践者能够在此平台上开发和评估metamaterials发现的新方法。为了便于访问和再现，我们在此公开了基准和代码库。 

---
# ShIOEnv: A CLI Behavior-Capturing Environment Enabling Grammar-Guided Command Synthesis for Dataset Curation 

**Title (ZH)**: ShIOEnv: 一种用于数据集整理的命令行为捕捉环境，支持语法引导的命令合成 

**Authors**: Jarrod Ragsdale, Rajendra Boppana  

**Link**: [PDF](https://arxiv.org/pdf/2505.18374)  

**Abstract**: Command-line interfaces (CLIs) provide structured textual environments for system administration. Explorations have been performed using pre-trained language models (PLMs) to simulate these environments for safe interaction in high-risk environments. However, their use has been constrained to frozen, large parameter models like GPT. For smaller architectures to reach a similar level of believability, a rich dataset of CLI interactions is required. Existing public datasets focus on mapping natural-language tasks to commands, omitting crucial execution data such as exit codes, outputs, and environmental side effects, limiting their usability for behavioral modeling. We introduce a Shell Input -Output Environment (ShIOEnv), which casts command construction as a Markov Decision Process whose state is the partially built sequence and whose actions append arguments. After each action, ShIOEnv executes the candidate and returns its exit status, output, and progress toward a minimal-length behavioral objective. Due to the intractable nature of the combinatorial argument state-action space, we derive a context-free grammar from man pages to mask invalid arguments from being emitted. We explore random and proximal-policy optimization (PPO)-optimized sampling of unrestricted and grammar-masked action spaces to produce four exploration strategies. We observed that grammar masking and PPO significantly improve sample efficiency to produce a higher quality dataset (maximizing the number of arguments while minimizing redundancies). Policy-generated datasets of shell input-output behavior pairs are used to fine-tune CodeT5, where we observe 85% improvements in BLEU-4 when constraining the action space to grammar productions with an additional 26% improvement when applying PPO. The ShIOEnv environment and curated command behavior datasets are released for use in future research. 

**Abstract (ZH)**: 命令行接口（CLIs）为系统管理提供结构化的文本环境。已经使用预训练语言模型（PLMs）探索了这些环境的模拟，以实现高风险环境中的安全交互。然而，它们的使用局限于冻结的大参数模型如GPT。为了使较小的架构达到类似的可信程度，需要丰富的CLI交互数据集。现有的公共数据集专注于将自然语言任务映射到命令，而忽略了诸如退出代码、输出和环境副作用等关键执行数据，从而限制了它们在行为建模中的应用。我们引入了一个Shell输入-输出环境（ShIOEnv），它将命令构建视为一种部分构建序列的状态和追加参数的动作的马尔可夫决策过程。每次动作后，ShIOEnv执行候选操作并返回其退出状态、输出以及接近最小长度行为目标的进度。由于组合的参数状态-动作空间难以处理，我们从man页面中推导出一个上下文无关文法，以屏蔽无效参数的生成。我们探索了无约束和文法屏蔽动作空间的随机采样及近端策略优化（PPO）优化采样，产生了四种探索策略。我们观察到文法屏蔽和PPO显著提高了样本效率，生成了更高质量的数据集（最大化参数数量同时最小化冗余）。由策略生成的shell输入-输出行为配对数据集用于微调CodeT5，我们发现，当将动作空间限制为文法生成时，BLEU-4指标提高了85%，而在应用PPO时，这一改进又额外提高了26%。ShIOEnv环境及精心整理的命令行为数据集被释放用于未来的研究。 

---
# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach 

**Title (ZH)**: 基于图形RAG的法律规范：一种分层和Temporal方法 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00039)  

**Abstract**: This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support. 

**Abstract (ZH)**: 本文提出了一种针对法律规范分析与理解的Graph Retrieval Augmented Generation (Graph RAG) 的 adaptation，法律规范以其预定义的层次结构、广泛的内部和外部引用网络以及多个时间版本为特点。通过将结构化知识图与上下文丰富的文本片段相结合，Graph RAG 提供了一种有前景的解决方案，以应对法律数据本身固有的复杂性和庞大的数据量。将层次结构和时间演变整合到知识图中——以及全面的文本文本单元的概念——促进了更丰富、更互联的法律知识表示的构建。通过对Graph RAG 的详细分析及其在法律规范数据集上的应用，本文旨在推动人工智能在法律领域的应用前沿，为更有效的法律研究、立法分析和决策支持系统创造机会。 

---
