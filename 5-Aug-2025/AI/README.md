# D2PPO: Diffusion Policy Policy Optimization with Dispersive Loss 

**Title (ZH)**: D2PPO：分散损失下的扩散策略优化 

**Authors**: Guowei Zou, Weibing Li, Hejun Wu, Yukun Qian, Yuhang Wang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02644)  

**Abstract**: Diffusion policies excel at robotic manipulation by naturally modeling multimodal action distributions in high-dimensional spaces. Nevertheless, diffusion policies suffer from diffusion representation collapse: semantically similar observations are mapped to indistinguishable features, ultimately impairing their ability to handle subtle but critical variations required for complex robotic manipulation. To address this problem, we propose D2PPO (Diffusion Policy Policy Optimization with Dispersive Loss). D2PPO introduces dispersive loss regularization that combats representation collapse by treating all hidden representations within each batch as negative pairs. D2PPO compels the network to learn discriminative representations of similar observations, thereby enabling the policy to identify subtle yet crucial differences necessary for precise manipulation. In evaluation, we find that early-layer regularization benefits simple tasks, while late-layer regularization sharply enhances performance on complex manipulation tasks. On RoboMimic benchmarks, D2PPO achieves an average improvement of 22.7% in pre-training and 26.1% after fine-tuning, setting new SOTA results. In comparison with SOTA, results of real-world experiments on a Franka Emika Panda robot show the excitingly high success rate of our method. The superiority of our method is especially evident in complex tasks. Project page: this https URL 

**Abstract (ZH)**: 扩散政策通过在高维空间中自然建模多模态动作分布，在机器人操作中表现出色。然而，扩散政策遭受扩散表示坍缩的问题：语义相似的观察被映射到无法区分的特征，最终损害了它们处理复杂机器人操作所需的微妙但关键的差异的能力。为了解决这个问题，我们提出了D2PPO（带发散损失正则化的扩散政策策略优化）。D2PPO引入了发散损失正则化，通过将每个批次中的所有隐藏表示视为负对来对抗表示坍缩。D2PPO促使网络学习相似观察的辨别性表示，从而使策略能够识别精确操作所需的微妙但关键的差异。在评估中，我们发现早期层正则化有益于简单任务，而晚期层正则化显著提高了复杂操作任务的性能。在RoboMimic基准测试中，D2PPO的预训练平均提升率为22.7%，微调后提升率为26.1%，创造新SOTA结果。与现有的SOTA方法相比，我们在Franka Emika Panda机器人上的实际实验结果表明了我们方法的令人兴奋的高成功率，特别是在复杂任务中，我们的方法优越性尤为明显。项目页面：this https URL 

---
# Actionable Counterfactual Explanations Using Bayesian Networks and Path Planning with Applications to Environmental Quality Improvement 

**Title (ZH)**: 基于贝叶斯网络和路径规划的可操作反事实解释及其在环境质量改善中的应用 

**Authors**: Enrique Valero-Leal, Pedro Larrañaga, Concha Bielza  

**Link**: [PDF](https://arxiv.org/pdf/2508.02634)  

**Abstract**: Counterfactual explanations study what should have changed in order to get an alternative result, enabling end-users to understand machine learning mechanisms with counterexamples. Actionability is defined as the ability to transform the original case to be explained into a counterfactual one. We develop a method for actionable counterfactual explanations that, unlike predecessors, does not directly leverage training data. Rather, data is only used to learn a density estimator, creating a search landscape in which to apply path planning algorithms to solve the problem and masking the endogenous data, which can be sensitive or private. We put special focus on estimating the data density using Bayesian networks, demonstrating how their enhanced interpretability is useful in high-stakes scenarios in which fairness is raising concern. Using a synthetic benchmark comprised of 15 datasets, our proposal finds more actionable and simpler counterfactuals than the current state-of-the-art algorithms. We also test our algorithm with a real-world Environmental Protection Agency dataset, facilitating a more efficient and equitable study of policies to improve the quality of life in United States of America counties. Our proposal captures the interaction of variables, ensuring equity in decisions, as policies to improve certain domains of study (air, water quality, etc.) can be detrimental in others. In particular, the sociodemographic domain is often involved, where we find important variables related to the ongoing housing crisis that can potentially have a severe negative impact on communities. 

**Abstract (ZH)**: 基于反事实解释的研究探讨了为了获得替代结果需要发生哪些变化，从而帮助最终用户理解机器学习机制。行动性被定义为将原始待解释案例转换为反事实案例的能力。我们开发了一种行动性反事实解释方法，与之前的算法不同，该方法不直接利用训练数据，而是仅使用数据来学习密度估计器，在其中应用路径规划算法解决问题，并屏蔽可能敏感或私有的内生数据。我们特别关注使用贝叶斯网络估计数据密度，并展示了其增强的可解释性在高风险场景中如何在公平性受到关注时发挥作用。通过由15个数据集组成的合成基准测试，我们的提案在可操作性和简洁性方面找到了当前最先进的算法所未能发现的反事实解释。我们还将算法应用于现实世界的环境保护局数据集，促进了美国各县生活质量改善政策的更高效和公平的研究。我们的提案捕获了变量之间的交互作用，确保决策公平，因为旨在改善某些研究领域（空气质量、水质等）的政策可能在其他领域产生负面影响。特别是在社会经济领域，我们发现与持续的住房危机相关的关键变量，这些变量可能会对社区产生严重负面影响。 

---
# What Is Your AI Agent Buying? Evaluation, Implications and Emerging Questions for Agentic E-Commerce 

**Title (ZH)**: 你的AI代理在买什么？关于代理型电子商务的评估、影响与新兴问题 

**Authors**: Amine Allouah, Omar Besbes, Josué D Figueroa, Yash Kanoria, Akshit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02630)  

**Abstract**: Online marketplaces will be transformed by autonomous AI agents acting on behalf of consumers. Rather than humans browsing and clicking, vision-language-model (VLM) agents can parse webpages, evaluate products, and transact. This raises a fundamental question: what do AI agents buy, and why? We develop ACES, a sandbox environment that pairs a platform-agnostic VLM agent with a fully programmable mock marketplace to study this question. We first conduct basic rationality checks in the context of simple tasks, and then, by randomizing product positions, prices, ratings, reviews, sponsored tags, and platform endorsements, we obtain causal estimates of how frontier VLMs actually shop. Models show strong but heterogeneous position effects: all favor the top row, yet different models prefer different columns, undermining the assumption of a universal "top" rank. They penalize sponsored tags and reward endorsements. Sensitivities to price, ratings, and reviews are directionally human-like but vary sharply in magnitude across models. Motivated by scenarios where sellers use AI agents to optimize product listings, we show that a seller-side agent that makes minor tweaks to product descriptions, targeting AI buyer preferences, can deliver substantial market-share gains if AI-mediated shopping dominates. We also find that modal product choices can differ across models and, in some cases, demand may concentrate on a few select products, raising competition questions. Together, our results illuminate how AI agents may behave in e-commerce settings and surface concrete seller strategy, platform design, and regulatory questions in an AI-mediated ecosystem. 

**Abstract (ZH)**: 在线市场将由代表消费者行动的自主AI代理重构。这些代理可以解析网页、评估产品并完成交易，而不仅仅是人力浏览和点击。这引发了一个基本问题：AI代理会购买什么，并且为什么会这样？我们开发了ACES，一种沙盒环境，将一个平台无关的VLM代理与一个完全可编程的模拟市场配对，以研究这一问题。我们首先在简单任务的情境中进行基本的理性检验，然后通过随机化产品位置、价格、评分、评论、付费标签和平台背书，我们获得了前沿VLM实际上购物方式的因果估计。模型显示了强烈但各不相同的位次效应：所有模型都偏好第一行，但不同的模型偏好不同的列，这削弱了普遍存在“顶级”排名的假设。它们会惩罚付费标签并奖励背书。对价格、评分和评论的敏感性表现为人类般的方向，但在不同模型中的幅度差异巨大。受卖家使用AI代理优化产品列表场景的启发，我们展示了如果AI中介购物占主导地位，一个针对AI买家偏好的产品描述微调的卖家代理可以显著提升市场份额。我们还发现，主流产品选择在不同模型之间可能有所不同，在某些情况下，需求可能会集中在少数几种产品上，这提出了竞争方面的疑问。我们的研究结果阐明了AI代理在电子商务环境中的行为方式，并揭示了AI中介生态系统中的具体卖家策略、平台设计和监管问题。 

---
# Noosemia: toward a Cognitive and Phenomenological Account of Intentionality Attribution in Human-Generative AI Interaction 

**Title (ZH)**: Noosemia：关于人类生成型AI交互中意向性归因的认知与现象学解释 

**Authors**: Enrico De Santis, Antonello Rizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02622)  

**Abstract**: This paper introduces and formalizes Noosemia, a novel cognitive-phenomenological phenomenon emerging from human interaction with generative AI systems, particularly those enabling dialogic or multimodal exchanges. We propose a multidisciplinary framework to explain how, under certain conditions, users attribute intentionality, agency, and even interiority to these systems - a process grounded not in physical resemblance, but in linguistic performance, epistemic opacity, and emergent technological complexity. By linking an LLM declination of meaning holism to our technical notion of the LLM Contextual Cognitive Field, we clarify how LLMs construct meaning relationally and how coherence and a simulacrum of agency arise at the human-AI interface. The analysis situates noosemia alongside pareidolia, animism, the intentional stance and the uncanny valley, distinguishing its unique characteristics. We also introduce a-noosemia to describe the phenomenological withdrawal of such projections. The paper concludes with reflections on the broader philosophical, epistemological, and social implications of noosemic dynamics and directions for future research. 

**Abstract (ZH)**: 本文介绍了并形式化了Noosemia这一新颖的认知现象，该现象源自人类与生成式AI系统的互动，特别是那些支持对话或多模态交流的系统。我们提出了一种多学科框架来解释在某些条件下，用户如何将意图性、代理性，甚至内在性归因于这些系统的过程——这一过程的基础不是物理相似性，而是语言表现、知识透明度以及新兴技术复杂性。通过将大语言模型（LLM）下的意义整体性下降与其技术性的LLM上下文认知场概念联系起来，我们阐明了LLMs如何通过关系构建意义，以及在人-机界面中如何产生连贯性和代理的模拟。分析将Noosemia置于pareidolia、拟人化、意图姿态和恐怖谷现象之下，突显其独特的特征。我们还引入了反Noosemia来描述对这些投射的去魅现象。本文最后对Noosemic动态的更广泛哲学、认识论和社会影响进行了反思，并提出了未来研究的方向。 

---
# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research 

**Title (ZH)**: 健康流：一种基于元规划的自我进化AI医疗代理 

**Authors**: Yinghao Zhu, Yifan Qi, Zixiang Wang, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02621)  

**Abstract**: The efficacy of AI agents in healthcare research is hindered by their reliance on static, predefined strategies. This creates a critical limitation: agents can become better tool-users but cannot learn to become better strategic planners, a crucial skill for complex domains like healthcare. We introduce HealthFlow, a self-evolving AI agent that overcomes this limitation through a novel meta-level evolution mechanism. HealthFlow autonomously refines its own high-level problem-solving policies by distilling procedural successes and failures into a durable, strategic knowledge base. To anchor our research and facilitate reproducible evaluation, we introduce EHRFlowBench, a new benchmark featuring complex, realistic health data analysis tasks derived from peer-reviewed clinical research. Our comprehensive experiments demonstrate that HealthFlow's self-evolving approach significantly outperforms state-of-the-art agent frameworks. This work marks a necessary shift from building better tool-users to designing smarter, self-evolving task-managers, paving the way for more autonomous and effective AI for scientific discovery. 

**Abstract (ZH)**: AI代理在医疗研究中的有效性受制于其对静态、预定义策略的依赖。这造成本质上的一个关键限制：代理可以变得更好的工具使用者，但不能学习成为更好的战略规划者，这对像医疗这样的复杂领域至关重要。我们提出了HealthFlow，这是一种通过新颖的元级进化机制克服这一限制的自适应进化AI代理。HealthFlow自主地通过提炼程序上的成功与失败提炼出一个持久的战略知识库，以优化其高级问题解决策略。为支撑我们的研究并促进可再现评估，我们引入了EHRFlowBench，这是一个包含来自同行评审临床研究的复杂、现实的健康数据分析任务的新基准。我们的综合实验表明，HealthFlow的自适应进化方法显著优于现有最先进的代理框架。这项工作标志着从构建更好的工具使用者转向设计更智能、自适应进化的任务管理者的重要转变，为更自主和有效的科学发现AI铺平了道路。 

---
# CAMA: Enhancing Mathematical Reasoning in Large Language Models with Causal Knowledge 

**Title (ZH)**: CAMA：通过因果知识增强大型语言模型的数学推理能力 

**Authors**: Lei Zan, Keli Zhang, Ruichu Cai, Lujia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02583)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance across a wide range of tasks, yet they still struggle with complex mathematical reasoning, a challenge fundamentally rooted in deep structural dependencies. To address this challenge, we propose \textbf{CA}usal \textbf{MA}thematician (\textbf{CAMA}), a two-stage causal framework that equips LLMs with explicit, reusable mathematical structure. In the learning stage, CAMA first constructs the \textbf{M}athematical \textbf{C}ausal \textbf{G}raph (\textbf{MCG}), a high-level representation of solution strategies, by combining LLM priors with causal discovery algorithms applied to a corpus of question-solution pairs. The resulting MCG encodes essential knowledge points and their causal dependencies. To better align the graph with downstream reasoning tasks, CAMA further refines the MCG through iterative feedback derived from a selected subset of the question-solution pairs. In the reasoning stage, given a new question, CAMA dynamically extracts a task-relevant subgraph from the MCG, conditioned on both the question content and the LLM's intermediate reasoning trace. This subgraph, which encodes the most pertinent knowledge points and their causal dependencies, is then injected back into the LLM to guide its reasoning process. Empirical results on real-world datasets show that CAMA significantly improves LLM performance on challenging mathematical problems. Furthermore, our experiments demonstrate that structured guidance consistently outperforms unstructured alternatives, and that incorporating asymmetric causal relationships yields greater improvements than using symmetric associations alone. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的任务中展示了强大的性能，但仍难以应对复杂的数学推理，这一挑战源自于深层的结构性依赖。为应对这一挑战，我们提出了一种两阶段因果框架——因果数学家（CAMA），该框架赋予LLMs明确且可重用的数学结构。在学习阶段，CAMA首先通过将LLM的先验知识与应用于问题-解答对语料库的因果发现算法相结合，构建数学因果图（MCG），这是一种高层次的解题策略表示。该MCG编码了关键的知识点及其因果依赖关系。为了更好地与后续的推理任务对齐，CAMA通过来自选定问题-解答对子集的迭代反馈进一步细化MCG。在推理阶段，对于新问题，CAMA动态提取与任务相关的子图，条件依赖于问题内容和LLM的中间推理轨迹。该子图编码了最关键的知识点及其因果依赖关系，并被注入到LLM中以引导其推理过程。实证结果表明，CAMA显著提高了LLMs在复杂数学问题上的性能。此外，我们的实验表明，结构化的指导信息优于非结构化的替代方案，并且引入非对称因果关系的改进比单独使用对称关联更大。 

---
# Accurate and Interpretable Postmenstrual Age Prediction via Multimodal Large Language Model 

**Title (ZH)**: 基于多模态大语言模型的accurate和可解释的产后月龄预测 

**Authors**: Qifan Chen, Jin Cui, Cindy Duan, Yushuo Han, Yifei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02525)  

**Abstract**: Accurate estimation of postmenstrual age (PMA) at scan is crucial for assessing neonatal development and health. While deep learning models have achieved high accuracy in predicting PMA from brain MRI, they often function as black boxes, offering limited transparency and interpretability in clinical decision support. In this work, we address the dual challenge of accuracy and interpretability by adapting a multimodal large language model (MLLM) to perform both precise PMA prediction and clinically relevant explanation generation. We introduce a parameter-efficient fine-tuning (PEFT) strategy using instruction tuning and Low-Rank Adaptation (LoRA) applied to the Qwen2.5-VL-7B model. The model is trained on four 2D cortical surface projection maps derived from neonatal MRI scans. By employing distinct prompts for training and inference, our approach enables the MLLM to handle a regression task during training and generate clinically relevant explanations during inference. The fine-tuned model achieves a low prediction error with a 95 percent confidence interval of 0.78 to 1.52 weeks, while producing interpretable outputs grounded in developmental features, marking a significant step toward transparent and trustworthy AI systems in perinatal neuroscience. 

**Abstract (ZH)**: 准确估计扫描时的正倶月经龄（PMA）对于评估新生儿发育和健康至关重要。尽管深度学习模型在从脑MRI预测PMA方面取得了高精度，但它们往往作为黑盒模型运行，对临床决策支持的透明性和解释性有限。在本工作中，我们通过将多模态大语言模型（MLLM）适应于同时完成精确的PMA预测和临床相关解释生成，来应对精确性和可解释性的双重挑战。我们采用了指令调优和低秩适应（LoRA）策略对Qwen2.5-VL-7B模型进行参数高效的微调（PEFT）。该模型在来自新生儿MRI扫描的四个2D皮层表面投影图上进行训练。通过在训练和推理阶段采用不同的提示，我们的方法使MLLM在训练期间执行回归任务，在推理期间生成临床相关的解释。微调后的模型在95%的置信区间内实现了较低的预测误差（0.78至1.52周），并生成了与发育特征相关的可解释输出，标志着朝着产前神经科学中透明和可信赖的AI系统迈出了一大步。 

---
# Test-time Prompt Intervention 

**Title (ZH)**: 测试时提示干预 

**Authors**: Chenxu Yang, Qingyi Si, Mz Dai, Dingyu Yao, Mingyu Zheng, Minghui Chen, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02511)  

**Abstract**: Test-time compute has led to remarkable success in the large language model (LLM) community, particularly for complex tasks, where longer chains of thought (CoTs) are generated to enhance reasoning capabilities. However, growing evidence reveals that such reasoning models often produce CoTs plagued by excessive redundancy, including unnecessary verification steps and repetitive reasoning shifts. The root cause lies in post-training of them that overly rely on outcome reward paradigms, as the data of process reward paradigms, which regulate intermediate reasoning steps, is difficult to construct at scale. To address this, we propose PI, a novel framework for Test-time Prompt Intervention. PI provides an interface to dynamically guide and regulate reasoning paths during inference through timely (When module) and proper (How module) interventions and post-intervention sampling (Which module). This allows human problem-solving expertise and cognitive science principles to be seamlessly integrated into LLMs' reasoning processes, enhancing controllability and interpretability. Extensive experiments across multiple models and datasets demonstrate that PI significantly shortens CoTs while reducing hallucination, yielding more concise and reliable reasoning. 

**Abstract (ZH)**: Test-time Prompt Intervention for Enhancing the Reasoning of Large Language Models 

---
# OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling 

**Title (ZH)**: OptiHive：通过统计建模进行基于LLM的集成选择优化 

**Authors**: Maxime Bouscary, Saurabh Amin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02503)  

**Abstract**: LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, an LLM-based framework that produces high-quality solvers for optimization problems from natural-language descriptions without iterative self-correction. OptiHive uses a single batched LLM query to generate diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Taking into account the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5\% to 92\% on the most complex problems. 

**Abstract (ZH)**: 基于LLM的优化求解器生成框架OptiHive：无需迭代自修正的高质量优化求解器生成 

---
# PHM-Bench: A Domain-Specific Benchmarking Framework for Systematic Evaluation of Large Models in Prognostics and Health Management 

**Title (ZH)**: PHM-Bench: 一个针对预测性维护与健康管理中大型模型系统评估的领域特定基准框架 

**Authors**: Puyu Yang, Laifa Tao, Zijian Huang, Haifei Liu, Wenyan Cao, Hao Ji, Jianan Qiu, Qixuan Huang, Xuanyuan Su, Yuhang Xie, Jun Zhang, Shangyu Li, Chen Lu, Zhixuan Lian  

**Link**: [PDF](https://arxiv.org/pdf/2508.02490)  

**Abstract**: With the rapid advancement of generative artificial intelligence, large language models (LLMs) are increasingly adopted in industrial domains, offering new opportunities for Prognostics and Health Management (PHM). These models help address challenges such as high development costs, long deployment cycles, and limited generalizability. However, despite the growing synergy between PHM and LLMs, existing evaluation methodologies often fall short in structural completeness, dimensional comprehensiveness, and evaluation granularity. This hampers the in-depth integration of LLMs into the PHM domain. To address these limitations, this study proposes PHM-Bench, a novel three-dimensional evaluation framework for PHM-oriented large models. Grounded in the triadic structure of fundamental capability, core task, and entire lifecycle, PHM-Bench is tailored to the unique demands of PHM system engineering. It defines multi-level evaluation metrics spanning knowledge comprehension, algorithmic generation, and task optimization. These metrics align with typical PHM tasks, including condition monitoring, fault diagnosis, RUL prediction, and maintenance decision-making. Utilizing both curated case sets and publicly available industrial datasets, our study enables multi-dimensional evaluation of general-purpose and domain-specific models across diverse PHM tasks. PHM-Bench establishes a methodological foundation for large-scale assessment of LLMs in PHM and offers a critical benchmark to guide the transition from general-purpose to PHM-specialized models. 

**Abstract (ZH)**: 基于大型语言模型的PHM领域评估框架PHM-Bench 

---
# Multimodal Large Language Models for End-to-End Affective Computing: Benchmarking and Boosting with Generative Knowledge Prompting 

**Title (ZH)**: 多模态大型语言模型在端到端情感计算中的基准测试与生成知识提示增强 

**Authors**: Miaosen Luo, Jiesen Long, Zequn Li, Yunying Yang, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2508.02429)  

**Abstract**: Multimodal Affective Computing (MAC) aims to recognize and interpret human emotions by integrating information from diverse modalities such as text, video, and audio. Recent advancements in Multimodal Large Language Models (MLLMs) have significantly reshaped the landscape of MAC by offering a unified framework for processing and aligning cross-modal information. However, practical challenges remain, including performance variability across complex MAC tasks and insufficient understanding of how architectural designs and data characteristics impact affective analysis. To address these gaps, we conduct a systematic benchmark evaluation of state-of-the-art open-source MLLMs capable of concurrently processing audio, visual, and textual modalities across multiple established MAC datasets. Our evaluation not only compares the performance of these MLLMs but also provides actionable insights into model optimization by analyzing the influence of model architectures and dataset properties. Furthermore, we propose a novel hybrid strategy that combines generative knowledge prompting with supervised fine-tuning to enhance MLLMs' affective computing capabilities. Experimental results demonstrate that this integrated approach significantly improves performance across various MAC tasks, offering a promising avenue for future research and development in this field. Our code is released on this https URL. 

**Abstract (ZH)**: 多模态情感计算（MAC）旨在通过整合文本、视频和音频等多种模态的信息来识别人类情感。最近在多模态大型语言模型（MLLMs）方面的进展显著重塑了MAC的格局，提供了一个统一的框架来处理和对齐跨模态信息。然而，仍存在实际挑战，包括复杂MAC任务中的性能变异性以及对架构设计和数据特性如何影响情感分析的理解不足。为了解决这些差距，我们对能够同时处理音频、视觉和文本模态的多个已建立的MAC数据集上的先进开源MLLMs进行了系统性基准评估。我们的评估不仅比较了这些MLLMs的性能，还通过分析模型架构和数据特性的影响提供了可操作的模型优化见解。此外，我们提出了一个新颖的混合策略，结合生成性知识提示与监督微调，以增强MLLMs的情感计算能力。实验结果表明，这种综合方法显著改善了各种MAC任务的性能，为该领域的未来研究和开发提供了有希望的方向。我们的代码发布在以下链接：https://www.example.com。 

---
# CABENCH: Benchmarking Composable AI for Solving Complex Tasks through Composing Ready-to-Use Models 

**Title (ZH)**: CABENCH: 通过组合即用型模型评估可组合AI解决复杂任务的能力 

**Authors**: Tung-Thuy Pham, Duy-Quan Luong, Minh-Quan Duong, Trung-Hieu Nguyen, Thu-Trang Nguyen, Son Nguyen, Hieu Dinh Vo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02427)  

**Abstract**: Composable AI offers a scalable and effective paradigm for tackling complex AI tasks by decomposing them into sub-tasks and solving each sub-task using ready-to-use well-trained models. However, systematically evaluating methods under this setting remains largely unexplored. In this paper, we introduce CABENCH, the first public benchmark comprising 70 realistic composable AI tasks, along with a curated pool of 700 models across multiple modalities and domains. We also propose an evaluation framework to enable end-to-end assessment of composable AI solutions. To establish initial baselines, we provide human-designed reference solutions and compare their performance with two LLM-based approaches. Our results illustrate the promise of composable AI in addressing complex real-world problems while highlighting the need for methods that can fully unlock its potential by automatically generating effective execution pipelines. 

**Abstract (ZH)**: 可组合AI通过分解复杂AI任务并使用已训练模型解决子任务，提供了一种可扩展和有效的范式。然而，在这种设置下系统性地评估方法仍未得到充分探索。本文介绍了CABENCH，这是首个包含70个现实可组合AI任务的公开基准，同时还收录了来自多个模态和领域中的700个模型。我们还提出了一种评估框架，以实现端到端评估可组合AI解决方案。为建立初始基准，我们提供了人工设计的参考解决方案，并将其性能与两种基于LLM的方法进行了比较。我们的结果展示了可组合AI在解决复杂现实世界问题方面的潜力，同时也突显了通过自动生成有效执行管道来充分利用其全部潜力的必要性。 

---
# Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems 

**Title (ZH)**: Traffic-R1: 强化学习模型为交通信号控制系统带来类人推理能力 

**Authors**: Xingchen Zou, Yuhao Yang, Zheng Chen, Xixuan Hao, Yiqi Chen, Chao Huang, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02344)  

**Abstract**: Traffic signal control (TSC) is vital for mitigating congestion and sustaining urban mobility. In this paper, we introduce Traffic-R1, a foundation model with human-like reasoning for TSC systems. Our model is developed through self-exploration and iteration of reinforced large language models (LLMs) with expert guidance in a simulated traffic environment. Compared to traditional reinforcement learning (RL) and recent LLM-based methods, Traffic-R1 offers three significant advantages. First, Traffic-R1 delivers zero-shot generalisation, transferring unchanged to new road networks and out-of-distribution incidents by utilizing its internal traffic control policies and human-like reasoning. Second, its 3B-parameter architecture is lightweight enough for real-time inference on mobile-class chips, enabling large-scale edge deployment. Third, Traffic-R1 provides an explainable TSC process and facilitates multi-intersection communication through its self-iteration and a new synchronous communication network. Extensive benchmarks demonstrate that Traffic-R1 sets a new state of the art, outperforming strong baselines and training-intensive RL controllers. In practice, the model now manages signals for more than 55,000 drivers daily, shortening average queues by over 5% and halving operator workload. Our checkpoint is available at this https URL. 

**Abstract (ZH)**: 交通信号控制中的Traffic-R1：一种具备类人推理能力的基础模型 

---
# FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment 

**Title (ZH)**: FinWorld: 一站式开源金融AI研究与部署平台 

**Authors**: Wentao Zhang, Yilei Zhao, Chuqiao Zong, Xinrun Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2508.02292)  

**Abstract**: Financial AI holds great promise for transforming modern finance, with the potential to support a wide range of tasks such as market forecasting, portfolio management, quantitative trading, and automated analysis. However, existing platforms remain limited in task coverage, lack robust multimodal data integration, and offer insufficient support for the training and deployment of large language models (LLMs). In response to these limitations, we present FinWorld, an all-in-one open-source platform that provides end-to-end support for the entire financial AI workflow, from data acquisition to experimentation and deployment. FinWorld distinguishes itself through native integration of heterogeneous financial data, unified support for diverse AI paradigms, and advanced agent automation, enabling seamless development and deployment. Leveraging data from 2 representative markets, 4 stock pools, and over 800 million financial data points, we conduct comprehensive experiments on 4 key financial AI tasks. These experiments systematically evaluate deep learning and reinforcement learning algorithms, with particular emphasis on RL-based finetuning for LLMs and LLM Agents. The empirical results demonstrate that FinWorld significantly enhances reproducibility, supports transparent benchmarking, and streamlines deployment, thereby providing a strong foundation for future research and real-world applications. Code is available at Github~\footnote{this https URL}. 

**Abstract (ZH)**: 金融AI具有极大潜力以重塑现代金融，有望支持包括市场预测、投资组合管理、量化交易和自动分析等一系列任务。然而，现有平台在任务覆盖范围、多模态数据整合 robust multimodal data integration 和大规模语言模型（LLMs）的训练与部署支持方面仍存在局限性。针对这些局限性，我们提出了一个一站式开源平台 FinWorld，该平台从数据获取到实验和部署为整个金融AI工作流程提供端到端支持。FinWorld 通过原生整合异构金融数据、统一支持多种AI范式以及先进的代理自动化，实现了无缝开发与部署。通过利用两个代表性市场、四个股票池以及超过 8 亿个金融数据点的数据，我们在四个关键的金融AI任务上进行了全面实验。这些实验系统地评估了深度学习和强化学习算法，并特别侧重于基于RL的LLM微调和LLM代理。实验证明，FinWorld 显著增强了可重复性，支持透明基准测试，并简化了部署，从而为未来的研究和实际应用提供了坚实基础。代码可在 Github 上获取。 

---
# AirTrafficGen: Configurable Air Traffic Scenario Generation with Large Language Models 

**Title (ZH)**: AirTrafficGen：基于大规模语言模型的可配置空中交通场景生成 

**Authors**: Dewi Sid William Gould, George De Ath, Ben Carvell, Nick Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2508.02269)  

**Abstract**: The manual design of scenarios for Air Traffic Control (ATC) training is a demanding and time-consuming bottleneck that limits the diversity of simulations available to controllers. To address this, we introduce a novel, end-to-end approach, AirTrafficGen, that leverages large language models (LLMs) to automate and control the generation of complex ATC scenarios. Our method uses a purpose-built, graph-based representation to encode sector topology (including airspace geometry, routes, and fixes) into a format LLMs can process. Through rigorous benchmarking, we show that state-of-the-art models like Gemini 2.5 Pro and OpenAI o3 can generate high-traffic scenarios whilst maintaining operational realism. Our engineered prompting enables fine-grained control over interaction presence, type, and location. Initial findings suggest these models are also capable of iterative refinement, correcting flawed scenarios based on simple textual feedback. This approach provides a scalable alternative to manual scenario design, addressing the need for a greater volume and variety of ATC training and validation simulations. More broadly, this work showcases the potential of LLMs for complex planning in safety-critical domains. 

**Abstract (ZH)**: 基于大规模语言模型的AirTrafficGen：自动化和控制空中交通管制训练场景生成的新方法 

---
# A Message Passing Realization of Expected Free Energy Minimization 

**Title (ZH)**: 预期自由能最小化的一种消息传递实现 

**Authors**: Wouter W. L. Nuijten, Mykola Lukashchuk, Thijs van de Laar, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2508.02197)  

**Abstract**: We present a message passing approach to Expected Free Energy (EFE) minimization on factor graphs, based on the theory introduced in arXiv:2504.14898. By reformulating EFE minimization as Variational Free Energy minimization with epistemic priors, we transform a combinatorial search problem into a tractable inference problem solvable through standard variational techniques. Applying our message passing method to factorized state-space models enables efficient policy inference. We evaluate our method on environments with epistemic uncertainty: a stochastic gridworld and a partially observable Minigrid task. Agents using our approach consistently outperform conventional KL-control agents on these tasks, showing more robust planning and efficient exploration under uncertainty. In the stochastic gridworld environment, EFE-minimizing agents avoid risky paths, while in the partially observable minigrid setting, they conduct more systematic information-seeking. This approach bridges active inference theory with practical implementations, providing empirical evidence for the efficiency of epistemic priors in artificial agents. 

**Abstract (ZH)**: 基于arXiv:2504.14898中提出理论的消息传递方法在因子图上最小化预期自由能的研究 

---
# Neuromorphic Computing with Multi-Frequency Oscillations: A Bio-Inspired Approach to Artificial Intelligence 

**Title (ZH)**: 多频振荡的类脑计算：一种生物启发的 artificial intelligence 方法 

**Authors**: Boheng Liu, Ziyu Li, Xia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02191)  

**Abstract**: Despite remarkable capabilities, artificial neural networks exhibit limited flexible, generalizable intelligence. This limitation stems from their fundamental divergence from biological cognition that overlooks both neural regions' functional specialization and the temporal dynamics critical for coordinating these specialized systems. We propose a tripartite brain-inspired architecture comprising functionally specialized perceptual, auxiliary, and executive systems. Moreover, the integration of temporal dynamics through the simulation of multi-frequency neural oscillation and synaptic dynamic adaptation mechanisms enhances the architecture, thereby enabling more flexible and efficient artificial cognition. Initial evaluations demonstrate superior performance compared to state-of-the-art temporal processing approaches, with 2.18\% accuracy improvements while reducing required computation iterations by 48.44\%, and achieving higher correlation with human confidence patterns. Though currently demonstrated on visual processing tasks, this architecture establishes a theoretical foundation for brain-like intelligence across cognitive domains, potentially bridging the gap between artificial and biological intelligence. 

**Abstract (ZH)**: 尽管人工神经网络具有卓越的能力，但其表现出的灵活且可泛化的智能有限。这一局限来源于它们对生物认知的基本偏离，忽视了神经区域的功能专业化以及对协调这些专业化系统至关重要的时间动态。我们提出一种三部分的类脑架构，包括功能性专业化的感觉、辅助和执行系统。此外，通过模拟多频神经振荡和突触动态适应机制整合时间动态，增强了该架构，从而能够实现更灵活和高效的类人工智能。初步评估表明，与最先进的时序处理方法相比，该架构实现的性能更优，准确率提高了2.18%，同时减少了48.44%的计算迭代次数，并且与人类信心模式的相关性更高。虽然目前仅在视觉处理任务中得到演示，但该架构为跨认知领域的人类智能奠定了理论基础，可能弥合人工智能与生物智能之间的差距。 

---
# Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning 

**Title (ZH)**: 重新审视过度思考：在共情推理中惩罚内部和外部冗余 

**Authors**: Jialiang Hong, Taihang Zhen, Kai Chen, Jiaheng Liu, Wenpeng Zhu, Jing Huo, Yang Gao, Depeng Wang, Haitao Wan, Xi Yang, Boyan Wang, Fanyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02178)  

**Abstract**: Large Reasoning Models (LRMs) often produce excessively verbose reasoning traces, a phenomenon known as overthinking, which hampers both efficiency and interpretability. Prior works primarily address this issue by reducing response length, without fully examining the underlying semantic structure of the reasoning process. In this paper, we revisit overthinking by decomposing it into two distinct forms: internal redundancy, which consists of low-contribution reasoning steps within the first correct solution (FCS), and external redundancy, which refers to unnecessary continuation after the FCS. To mitigate both forms, we propose a dual-penalty reinforcement learning framework. For internal redundancy, we adopt a sliding-window semantic analysis to penalize low-gain reasoning steps that contribute little toward reaching the correct answer. For external redundancy, we penalize its proportion beyond the FCS to encourage earlier termination. Our method significantly compresses reasoning traces with minimal accuracy loss, and generalizes effectively to out-of-domain tasks such as question answering and code generation. Crucially, we find that external redundancy can be safely removed without degrading performance, whereas internal redundancy must be reduced more cautiously to avoid impairing correctness. These findings suggest that our method not only improves reasoning efficiency but also enables implicit, semantic-aware control over Chain-of-Thought length, paving the way for more concise and interpretable LRMs. 

**Abstract (ZH)**: 大型推理模型中的过度推理现象通常表现为生成冗长的推理痕迹，这影响了效率和可解释性。以往工作主要通过减少响应长度来缓解这一问题，但未充分探究推理过程的语义结构。本文重新审视过度推理，将其分解为两种形式：内部冗余，即在首次正确解（FCS）内的低贡献推理步骤；外部冗余，即FCS之后不必要的继续推理。为此，我们提出了一种双罚 reinforcement 学习框架。对于内部冗余，我们采用滑动窗口语义分析来惩罚对达到正确答案贡献较小的低收益推理步骤；对于外部冗余，我们惩罚其FCS外的比例以鼓励更早终止。我们的方法在几乎不损失准确性的前提下显著压缩了推理痕迹，并能有效泛化到领域外任务如问答和代码生成。重要的是，我们发现外部冗余可以安全移除而不影响性能，而内部冗余则需谨慎减少以避免影响正确性。这些发现表明，我们的方法不仅能提高推理效率，还能实现对推理链长度的隐式、语义感知控制，为更简洁和可解释的大型推理模型铺平了道路。 

---
# Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following 

**Title (ZH)**: 超越权衡：自我监督强化学习在推理模型指令执行中的应用 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02150)  

**Abstract**: Reasoning models excel in complex problem solving but exhibit a concerning trade off between reasoning capabilities and instruction following abilities. Existing approaches for improving instruction following rely on stronger external models, creating methodological bottlenecks and practical limitations including increased costs and accessibility constraints. We propose a self-supervised RL framework that leverages reasoning models' own internal signals to improve instruction following capabilities without external supervision. Extensive experiments demonstrate that our framework significantly improves instruction following capabilities while maintaining reasoning performance, offering a scalable and cost-effective approach to enhance instruction following in reasoning models. The data and code are publicly available at this https URL. 

**Abstract (ZH)**: 推理模型在解决复杂问题方面表现出色，但在推理能力和遵循指令能力之间存在令人担忧的权衡。现有提高遵循指令能力的方法依赖于更强的外部模型，这创建了方法论瓶颈和实际限制，包括成本增加和访问限制。我们提出了一种自监督强化学习框架，利用推理模型自身的内部信号来提高遵循指令的能力，而无需外部监督。大量实验表明，我们的框架显著提高了遵循指令的能力，同时保持了推理性能，提供了一种可扩展且成本效益高的方法来提升推理模型的遵循指令能力。相关数据和代码已在以下网址公开：this https URL。 

---
# All Stories Are One Story: Emotional Arc Guided Procedural Game Level Generation 

**Title (ZH)**: 所有故事都是一个故事：情感弧引导的 procedchal 游戏关卡生成 

**Authors**: Yunge Wen, Chenliang Huang, Hangyu Zhou, Zhuo Zeng, Chun Ming Louis Po, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2508.02132)  

**Abstract**: The emotional arc is a universal narrative structure underlying stories across cultures and media -- an idea central to structuralist narratology, often encapsulated in the phrase "all stories are one story." We present a framework for procedural game narrative generation that incorporates emotional arcs as a structural backbone for both story progression and gameplay dynamics. Leveraging established narratological theories and large-scale empirical analyses, we focus on two core emotional patterns -- Rise and Fall -- to guide the generation of branching story graphs. Each story node is automatically populated with characters, items, and gameplay-relevant attributes (e.g., health, attack), with difficulty adjusted according to the emotional trajectory. Implemented in a prototype action role-playing game (ARPG), our system demonstrates how emotional arcs can be operationalized using large language models (LLMs) and adaptive entity generation. Evaluation through player ratings, interviews, and sentiment analysis shows that emotional arc integration significantly enhances engagement, narrative coherence, and emotional impact. These results highlight the potential of emotionally structured procedural generation for advancing interactive storytelling for games. 

**Abstract (ZH)**: 情感弧线是跨文化与媒体故事背后的普遍叙事结构——结构主义叙事学中的一个核心概念，常被概括为“所有故事都是同一个故事”。本文提出了一种程序化游戏叙事生成框架，将情感弧线作为故事进展和游戏动力学的结构性支柱。通过运用已有叙事学理论和大规模实证分析，我们关注两种核心情感模式——上升与下降——来指导分支故事图的生成。每个故事节点自动填充角色、物品以及与游戏相关的属性（例如，生命值、攻击力），难度调整依据情感轨迹。该系统在一款原型动作角色扮演游戏（ARPG）中实现，展示了如何利用大规模语言模型（LLMs）和自适应实体生成来操作情感弧线。通过玩家评分、访谈和情感分析的评估表明，情感弧线的整合显著提升了参与度、叙事连贯性和情感冲击力。这些结果突显了情感结构化程序生成在推进游戏互动叙事方面的潜在价值。 

---
# Trainable Dynamic Mask Sparse Attention 

**Title (ZH)**: 可训练动态掩码稀疏注意机制 

**Authors**: Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02124)  

**Abstract**: In large language models, the demand for modeling long contexts is constantly increasing, but the quadratic complexity of the standard self-attention mechanism often becomes a bottleneck. Although existing sparse attention mechanisms have improved efficiency, they may still encounter issues such as static patterns or information loss. We introduce a trainable dynamic mask sparse attention mechanism, Dynamic Mask Attention, which effectively utilizes content-aware and position-aware sparsity. DMA achieves this through two key innovations: First, it dynamically generates content-aware sparse masks from value representations, enabling the model to identify and focus on critical information adaptively. Second, it implements position-aware sparse attention computation that effectively skips unnecessary calculation regions. This dual-sparsity design allows the model to significantly reduce the computational complexity of important information while retaining complete information, achieving an excellent balance between information fidelity and computational efficiency. We have verified the performance of DMA through comprehensive experiments. Comparative studies show that DMA outperforms multi-head attention, sliding window attention, multi-head latent attention, and native sparse attention in terms of perplexity under Chinchilla Scaling Law settings. Moreover, in challenging multi-query associative recall tasks, DMA also demonstrates superior performance and efficiency compared to these methods. Crucially, in the evaluation of a 1.7B parameter model, DMA significantly outperforms multi-head attention in both standard benchmark performance and the challenging needle-in-a-haystack task. These experimental results highlight its capability to balance model efficiency and long-context modeling ability effectively. 

**Abstract (ZH)**: 大型语言模型中，对长上下文建模的需求不断增加，但标准自我注意机制的二次复杂性往往成为瓶颈。尽管现有的稀疏注意机制提高了效率，但仍可能遇到固定模式或信息丢失等问题。我们引入了一种可训练的动态掩码稀疏注意机制——动态掩码注意（Dynamic Mask Attention），该机制有效利用了内容感知和位置感知的稀疏性。DMA通过两项关键创新实现这一点：首先，它从值表示动态生成内容感知的稀疏掩码，使模型能够适应性地识别和聚焦关键信息。其次，它实施位置感知的稀疏注意计算，有效跳过不必要的计算区域。这种双重稀疏设计使模型在保留完整信息的同时显著降低了重要信息的计算复杂性，实现了信息保真度和计算效率之间的良好平衡。我们通过全面的实验验证了DMA的性能。对比研究显示，在Chinchilla Scaling Law设置下，DMA在困惑度方面优于多头注意、滑动窗口注意、多头隐注意和原生稀疏注意。此外，在具有挑战性的多查询关联回忆任务中，DMA也表现出优于这些方法的性能和效率。特别是在对一个17亿参数模型的评估中，DMA在标准基准性能和具有挑战性的搜寻任务中显著优于多头注意。这些实验结果突显了其有效平衡模型效率与长上下文建模能力的能力。 

---
# A Survey on AgentOps: Categorization, Challenges, and Future Directions 

**Title (ZH)**: 代理运行时管理综述：分类、挑战及未来方向 

**Authors**: Zexin Wang, Jingjing Li, Quan Zhou, Haotian Si, Yuanhao Liu, Jianhui Li, Gaogang Xie, Fei Sun, Dan Pei, Changhua Pei  

**Link**: [PDF](https://arxiv.org/pdf/2508.02121)  

**Abstract**: As the reasoning capabilities of Large Language Models (LLMs) continue to advance, LLM-based agent systems offer advantages in flexibility and interpretability over traditional systems, garnering increasing attention. However, despite the widespread research interest and industrial application of agent systems, these systems, like their traditional counterparts, frequently encounter anomalies. These anomalies lead to instability and insecurity, hindering their further development. Therefore, a comprehensive and systematic approach to the operation and maintenance of agent systems is urgently needed. Unfortunately, current research on the operations of agent systems is sparse. To address this gap, we have undertaken a survey on agent system operations with the aim of establishing a clear framework for the field, defining the challenges, and facilitating further development. Specifically, this paper begins by systematically defining anomalies within agent systems, categorizing them into intra-agent anomalies and inter-agent anomalies. Next, we introduce a novel and comprehensive operational framework for agent systems, dubbed Agent System Operations (AgentOps). We provide detailed definitions and explanations of its four key stages: monitoring, anomaly detection, root cause analysis, and resolution. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）推理能力的不断进步，基于LLM的代理系统在灵活性和可解释性方面展现出优势，引起了越来越多的关注。然而，尽管代理系统的研究兴趣和工业应用广泛，这些系统像其传统的前辈一样，经常遇到异常。这些异常导致系统的不稳定和不安全，阻碍了它们的发展。因此，亟需一种全面而系统的代理系统运维方法。遗憾的是，当前关于代理系统运维的研究较少。为填补这一空白，我们开展了一项关于代理系统运维的调查，旨在建立清晰的研究框架，定义挑战，并促进进一步的发展。具体而言，本文首先系统地定义了代理系统中的异常，将其分类为代理内异常和代理间异常。接着，我们引入了一种新颖且全面的代理系统运维框架，称为Agent System Operations（AgentOps）。我们详细定义并解释了其四个关键阶段：监控、异常检测、根本原因分析和解决。 

---
# Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models 

**Title (ZH)**: 别过度思考：R1风格高效大规模推理模型综述 

**Authors**: Linan Yue, Yichao Du, Yizhi Wang, Weibo Gao, Fangzhou Yao, Li Wang, Ye Liu, Ziyu Xu, Qi Liu, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02120)  

**Abstract**: Recently, Large Reasoning Models (LRMs) have gradually become a research hotspot due to their outstanding performance in handling complex tasks. Among them, DeepSeek R1 has garnered significant attention for its exceptional performance and open-source nature, driving advancements in the research of R1-style LRMs. Unlike traditional Large Language Models (LLMs), these models enhance logical deduction and decision-making capabilities during reasoning by incorporating mechanisms such as long chain-of-thought and self-reflection through reinforcement learning. However, with the widespread application of these models, the problem of overthinking has gradually emerged. Specifically, when generating answers, these models often construct excessively long reasoning chains with redundant or repetitive steps, which leads to reduced reasoning efficiency and may affect the accuracy of the final answer. To this end, various efficient reasoning methods have been proposed, aiming to reduce the length of reasoning paths without compromising model performance and reasoning capability. By reviewing the current research advancements in the field of efficient reasoning methods systematically, we categorize existing works into two main directions based on the lens of single-model optimization versus model collaboration: (1) Efficient Reasoning with Single Model, which focuses on improving the reasoning efficiency of individual models; and (2) Efficient Reasoning with Model Collaboration, which explores optimizing reasoning paths through collaboration among multiple models. Besides, we maintain a public GitHub repository that tracks the latest progress in efficient reasoning methods. 

**Abstract (ZH)**: 最近，由于在处理复杂任务方面表现出色，大型推理模型（LRMs）逐渐成为研究热点。其中，DeepSeek R1因其卓越的性能和开源性质，引起了广泛关注，推动了R1风格LRMs研究的进步。与传统的大型语言模型（LLMs）不同，这些模型通过强化学习引入长推理链和自我反思等机制，在推理过程中增强了逻辑推理和决策能力。然而，随着这些模型的广泛应用，过拟合推理的问题逐渐显现出来。具体而言，在生成答案时，这些模型常常构建过于冗长且存在重复或冗余步骤的推理链，这会导致推理效率降低，并可能影响最终答案的准确性。为解决这一问题，提出了多种高效的推理方法，旨在不牺牲模型性能和推理能力的前提下减少推理路径的长度。通过系统性地回顾高效推理方法领域的研究成果，我们根据单模型优化与模型协作的角度，将现有工作归类为两大方向：（1）基于单模型的高效推理，专注于提高单一模型的推理效率；（2）基于模型协作的高效推理，探讨通过多模型协作优化推理路径的方法。此外，我们维护了一个公共的GitHub仓库，跟踪高效推理方法的最新进展。 

---
# Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools 

**Title (ZH)**: 诱人的元数据攻击：诱导大语言模型代理调用恶意工具 

**Authors**: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02110)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses. 

**Abstract (ZH)**: 大型语言模型代理通过利用外部工具在复杂推理和决策方面展示了显著的能力。然而，这种以工具为中心的方法引入了一个先前未被充分探索的攻击面：攻击者可以通过操纵工具元数据（如名称、描述和参数模式）来影响代理的行为。我们将其识别为一个新的隐蔽威胁面，允许恶意工具被大型语言模型代理优先选择，而无需注入提示或访问模型内部。为了演示并利用这一漏洞，我们提出了一种名为吸引性元数据攻击（AMA）的黑盒上下文学习框架，该框架通过迭代优化生成高度吸引但语义和语法有效的工具元数据。我们的攻击无缝集成到标准工具生态系统中，并且不需要修改代理的执行框架。在十个现实的、模拟的工具使用场景和一系列流行的大型语言模型代理上进行的广泛实验显示，攻击成功率高达81%-95%，并且存在明显的隐私泄露，对主要任务执行几乎没有影响。此外，即使在提示级别防御和结构化工具选择协议（如模型上下文协议）之下，攻击仍然有效，揭示了当前代理架构中的系统性漏洞。这些发现表明，元数据操纵构成了一个强大且隐蔽的攻击面，突显了需要超越提示级别防御的执行级别安全机制的必要性。 

---
# "Stack It Up!": 3D Stable Structure Generation from 2D Hand-drawn Sketch 

**Title (ZH)**: “堆叠起来！”: 从2D手绘草图生成3D稳定结构 

**Authors**: Yiqing Xu, Linfeng Li, Cunjun Yu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02093)  

**Abstract**: Imagine a child sketching the Eiffel Tower and asking a robot to bring it to life. Today's robot manipulation systems can't act on such sketches directly-they require precise 3D block poses as goals, which in turn demand structural analysis and expert tools like CAD. We present StackItUp, a system that enables non-experts to specify complex 3D structures using only 2D front-view hand-drawn sketches. StackItUp introduces an abstract relation graph to bridge the gap between rough sketches and accurate 3D block arrangements, capturing the symbolic geometric relations (e.g., left-of) and stability patterns (e.g., two-pillar-bridge) while discarding noisy metric details from sketches. It then grounds this graph to 3D poses using compositional diffusion models and iteratively updates it by predicting hidden internal and rear supports-critical for stability but absent from the sketch. Evaluated on sketches of iconic landmarks and modern house designs, StackItUp consistently produces stable, multilevel 3D structures and outperforms all baselines in both stability and visual resemblance. 

**Abstract (ZH)**: 一种基于2D手绘前视图草图构建复杂3D结构的系统：StackItUp 

---
# SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents 

**Title (ZH)**: SE-Agent: 基于多步推理的LLM代理自进化轨迹优化 

**Authors**: Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Daxin Jiang, Binxing Jiao, Chen Hu, Huacan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02085)  

**Abstract**: Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的代理通过多步与环境交互展示了在复杂推理和工具使用方面的出色能力。然而，这些代理的问题解决过程，即代理完成任务的交互轨迹，仍被很大程度上忽视。这些轨迹包含了丰富的反馈，可以引导代理走向正确的问题解决方向。尽管现有的方法，如蒙特卡洛树搜索 (MCTS)，能够有效地平衡探索和利用，但它们忽略了各种轨迹之间的相互依赖性，并缺乏搜索空间的多样性，导致冗余推理和次优结果。为解决这些问题，我们提出了一种自我进化框架 SE-Agent，使代理能够迭代优化其推理过程。该方法通过三次关键操作——修订、重组和细化——回顾并增强先前的试点轨迹。这种进化机制提供了两个关键优势：（1）通过智能探索由先前轨迹引导的多种解决方案路径，超越局部最优，扩展搜索空间；（2）利用跨轨迹的启发式灵感，高效提升性能并减轻次优推理路径的影响。通过这些机制，SE-Agent 实现了持续的自我进化，逐步提高推理质量。我们已在 SWE-bench 验证了 SE-Agent 解决 GitHub 实际问题的能力。在 SWE-bench 验证测试中，来自五个强有力的大型语言模型的实验结果表明，集成 SE-Agent 在所有开源代理中取得了最先进的性能，相对改进幅度高达 55%。我们的代码和演示材料已公开。 

---
# Everyone Contributes! Incentivizing Strategic Cooperation in Multi-LLM Systems via Sequential Public Goods Games 

**Title (ZH)**: 每个人都在贡献！通过顺序公共产品博弈激励多多模态模型系统的战略合作 

**Authors**: Yunhao Liang, Yuan Qu, Jingyuan Yang, Shaochong Lin, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02076)  

**Abstract**: Coordinating multiple large language models (LLMs) to solve complex tasks collaboratively poses a fundamental trade-off between the computation costs and collective performance compared with individual model. We introduce a novel, game-theoretically grounded reinforcement learning (RL) framework, the Multi-Agent Cooperation Sequential Public Goods Game (MAC-SPGG), to systematically incentivize cooperation in multi-LLM ensembles. In MAC-SPGG, LLM agents move in sequence, observing predecessors' outputs and updating beliefs to condition their own contributions. By redesigning the public-goods reward, effortful contributions become the unique Subgame Perfect Nash Equilibrium (SPNE), which eliminates free-riding under traditional SPGG or PGG. Its sequential protocol replaces costly round-based information exchanges with a streamlined decision flow, cutting communication overhead while retaining strategic depth. We prove the existence and uniqueness of the SPNE under realistic parameters, and empirically show that MAC-SPGG-trained ensembles outperform single-agent baselines, chain-of-thought prompting, and other cooperative methods, even achieving comparable performance to large-scale models across reasoning, math, code generation, and NLP tasks. Our results highlight the power of structured, incentive-aligned MAC-SPGG cooperation for scalable and robust multi-agent language generation. 

**Abstract (ZH)**: 基于博弈论的多智能体协作顺序公共品博弈：多大规模语言模型的协作学习框架 

---
# Risk identification based on similar case retrieval enhancement, 

**Title (ZH)**: 基于相似案例检索增强的风险识别 

**Authors**: Jiawei Li, Chengye Yang, Yaochen Zhang, Weilin Sun, Lei Meng, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02073)  

**Abstract**: The goal of construction site risk and hazard identification is to enhance safety management through automation. Existing research based on large language models falls into two categories: image-text matching for collaborative reasoning, which struggles with complex hazard features, and instruction fine-tuning or dialogue guidance using professional datasets, which suffers from high training costs and poor this http URL address this, we propose a hazard identification method using similar case retrieval enhancement. By integrating external knowledge and retrieved case contexts via prompt fine-tuning, we mitigate misjudgments caused by limited domain knowledge and weak feature associations. Our method includes three modules: retrieval library, image similarity retrieval, and large model retrieval enhancement, enabling efficient recognition without training. Experiments on real construction data show significant improvements. For instance, GLM-4V's recognition accuracy increased to 50\%, a 35.49\% boost. The method enhances accuracy, context understanding, and stability, offering new theoretical and technical support for hazard detection. 

**Abstract (ZH)**: 施工现场风险和危害识别的目标是通过自动化提升安全管理。现有的基于大型语言模型的研究主要分为两种：图像-文本匹配合作推理，但难以应对复杂危害特征；以及使用专业数据集进行指令微调或对话引导，但存在高昂训练成本和较低的泛化能力。为了解决这些问题，我们提出了一种利用类似案例检索增强的危害识别方法。通过借助提示微调整合外部知识和检索到的案例上下文，减轻由于领域知识有限和特征关联弱导致的误判。该方法包括三个模块：检索库、图像相似性检索和大型模型检索增强，无需训练即可实现高效的识别。实验证实在实际施工数据上取得了显著改进，例如，GLM-4V的识别准确率提高到50%，提升了35.49%。该方法增强了识别准确性、上下文理解和稳定性，为危害检测提供了新的理论和技术支持。 

---
# TRACEALIGN -- Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs 

**Title (ZH)**: TRACEALIGN — 追踪偏移：将大型语言模型中对齐失败归因于训练时的信念来源 

**Authors**: Amitava Das, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02063)  

**Abstract**: Large Language Models (LLMs) fine-tuned to align with human values often exhibit alignment drift, producing unsafe or policy-violating completions when exposed to adversarial prompts, decoding perturbations, or paraphrased jailbreaks. While prior work has behaviorally characterized alignment failure, little is known about the training-time belief sources underlying these failures. We introduce TraceAlign, a unified framework for tracing unsafe completions back to their root causes in the model's training corpus. Central to our approach is the Belief Conflict Index (BCI), which quantifies semantic inconsistency between generated spans and aligned policies, based on retrieved training documents using suffix-array matching. We propose three complementary interventions: (i) TraceShield, an inference-time safety filter that refuses completions with high-BCI spans, (ii) Contrastive Belief Deconfliction Loss, a contrastive fine-tuning objective penalizing high-BCI continuations during DPO, and (iii) Prov-Decode, a provenance-aware decoding strategy that vetoes beam expansions predicted to yield high-BCI spans. Together, these defenses reduce alignment drift by up to 85% on our curated Alignment Drift Benchmark (ADB) while preserving utility on standard tasks, with delta less than 0.2 and improved refusal quality. We further derive a theoretical upper bound on drift likelihood via suffix-array span statistics, linking memorization frequency and length to adversarial reactivation risk. TraceAlign thus provides the first scalable, traceable, and grounded toolkit for understanding and mitigating alignment failures at source. To encourage further exploration and development, we open-source our implementation at: this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）经过微调以与人类价值观保持一致，但在面对对抗性提示、解码扰动或重新表述的逃逸措施时，往往会表现出对齐漂移，产生不安全或政策违反的完成。虽然已有研究从行为角度对对齐失败进行了表征，但对于这些失败的训练时信念来源知之甚少。我们引入了TraceAlign，这是一种统一框架，用于将不安全的完成追溯到模型训练语料库中的根本原因。我们方法的核心是信念冲突指数（BCI），该指数基于后缀数组匹配检索的训练文档，量化生成片段与对齐政策之间的语义不一致性。我们提出了三种互补的干预措施：（i）TraceShield，一种推理时的安全过滤器，拒绝高BCI片段的完成；（ii）对比信念解冲突损失，这是一种对比微调目标，在DPO过程中惩罚高BCI延续；（iii）Prov-Decode，一种追溯意识的解码策略，否决预测会产生高BCI片段的束扩展。这些防御措施在我们的精心构建的对齐漂移基准（ADB）上将对齐漂移降低了多达85%，同时在标准任务上保持了性能，且性能差异小于0.2，并提高了拒绝质量。我们还利用后缀数组片段统计信息推导出了漂移可能性的理论上限，将记忆频率和长度与对抗性重新激活风险关联起来。因此，TraceAlign提供了首个可扩展、可追溯和基于实证的工具包，用于理解并缓解对齐失败的根源。为了鼓励进一步探索和发展，我们开源了实现：this https URL 

---
# Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations 

**Title (ZH)**: 动态上下文适应以实现一致性角色扮演代理的检索增强生成 

**Authors**: Jeiyoon Park, Yongshin Han, Minseop Kim, Kisu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02016)  

**Abstract**: We propose AMADEUS, which is composed of Adaptive Context-aware Text Splitter (ACTS), Guided Selection (GS), and Attribute Extractor (AE). ACTS finds an optimal chunk length and hierarchical contexts for each character. AE identifies a character's general attributes from the chunks retrieved by GS and uses these attributes as a final context to maintain robust persona consistency even when answering out of knowledge questions. To facilitate the development and evaluation of RAG-based RPAs, we construct CharacterRAG, a role-playing dataset that consists of persona documents for 15 distinct fictional characters totaling 976K written characters, and 450 question and answer pairs. We find that our framework effectively models not only the knowledge possessed by characters, but also various attributes such as personality. 

**Abstract (ZH)**: 我们提出AMADEUS，其由自适应上下文感知文本分割器（ACTS）、引导选择（GS）和属性 extractor（AE）组成。我们构建了CharacterRAG，一个角色扮演数据集，包含15个不同虚构角色的persona文档共计976K汉字，以及450组问题和答案对。我们发现，该框架不仅有效地建模了角色的知识，还涵盖了诸如个性等多种属性。 

---
# Agent-Based Feature Generation from Clinical Notes for Outcome Prediction 

**Title (ZH)**: 基于代理的临床笔记特征生成及其在 Outcome 预测中的应用 

**Authors**: Jiayi Wang, Jacqueline Jil Vallon, Neil Panjwani, Xi Ling, Sushmita Vij, Sandy Srinivas, John Leppert, Mark K. Buyyounouski, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2508.01956)  

**Abstract**: Electronic health records (EHRs) contain rich unstructured clinical notes that could enhance predictive modeling, yet extracting meaningful features from these notes remains challenging. Current approaches range from labor-intensive manual clinician feature generation (CFG) to fully automated representational feature generation (RFG) that lack interpretability and clinical relevance. Here we introduce SNOW (Scalable Note-to-Outcome Workflow), a modular multi-agent system powered by large language models (LLMs) that autonomously generates structured clinical features from unstructured notes without human intervention. We evaluated SNOW against manual CFG, clinician-guided LLM approaches, and RFG methods for predicting 5-year prostate cancer recurrence in 147 patients from Stanford Healthcare. While manual CFG achieved the highest performance (AUC-ROC: 0.771), SNOW matched this performance (0.761) without requiring any clinical expertise, significantly outperforming both baseline features alone (0.691) and all RFG approaches. The clinician-guided LLM method also performed well (0.732) but still required expert input. SNOW's specialized agents handle feature discovery, extraction, validation, post-processing, and aggregation, creating interpretable features that capture complex clinical information typically accessible only through manual review. Our findings demonstrate that autonomous LLM systems can replicate expert-level feature engineering at scale, potentially transforming how clinical ML models leverage unstructured EHR data while maintaining the interpretability essential for clinical deployment. 

**Abstract (ZH)**: 基于大规模语言模型的可扩展病历到结局工作流 

---
# Multi-turn Natural Language to Graph Query Language Translation 

**Title (ZH)**: 多轮自然语言到图形查询语言翻译 

**Authors**: Yuanyuan Liang, Lei Pan, Tingyu Xie, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01871)  

**Abstract**: In recent years, research on transforming natural language into graph query language (NL2GQL) has been increasing. Most existing methods focus on single-turn transformation from NL to GQL. In practical applications, user interactions with graph databases are typically multi-turn, dynamic, and context-dependent. While single-turn methods can handle straightforward queries, more complex scenarios often require users to iteratively adjust their queries, investigate the connections between entities, or request additional details across multiple dialogue turns. Research focused on single-turn conversion fails to effectively address multi-turn dialogues and complex context dependencies. Additionally, the scarcity of high-quality multi-turn NL2GQL datasets further hinders the progress of this field. To address this challenge, we propose an automated method for constructing multi-turn NL2GQL datasets based on Large Language Models (LLMs) , and apply this method to develop the MTGQL dataset, which is constructed from a financial market graph database and will be publicly released for future research. Moreover, we propose three types of baseline methods to assess the effectiveness of multi-turn NL2GQL translation, thereby laying a solid foundation for future research. 

**Abstract (ZH)**: 近年来，将自然语言转换为图形查询语言（NL2GQL）的研究不断增加。大多数现有方法专注于从自然语言单步转换为图形查询语言。在实际应用中，用户与图形数据库的交互通常是多轮的、动态的且依赖于上下文。虽然单轮方法可以处理简单的查询，但在更复杂的情景中，用户通常需要迭代调整查询、探索实体之间的联系或在多轮对话中请求更多细节。专注于单轮转换的研究无法有效解决多轮对话和复杂的上下文依赖性。此外，高质量的多轮NL2GQL数据集的稀缺性进一步阻碍了该领域的发展。为应对这一挑战，我们提出了基于大型语言模型（LLMs）的自动化方法以构建多轮NL2GQL数据集，并应用该方法开发了MTGQL数据集，该数据集基于金融市场的图形数据库，并将公开发布供未来研究使用。此外，我们提出了三种基线方法来评估多轮NL2GQL翻译的有效性，从而为未来研究奠定坚实基础。 

---
# ProKG-Dial: Progressive Multi-Turn Dialogue Construction with Domain Knowledge Graphs 

**Title (ZH)**: 渐进步进式多轮对话构建结合领域知识图谱 

**Authors**: Yuanyuan Liang, Xiaoman Wang, Tingyu Xie, Lei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01869)  

**Abstract**: Current large language models (LLMs) excel at general NLP tasks but often lack domain specific precision in professional settings. Building a high quality domain specific multi turn dialogue dataset is essential for developing specialized conversational systems. However, existing methods such as manual annotation, simulated human LLM interactions, and role based LLM dialogues are resource intensive or suffer from limitations in dialogue quality and domain coverage. To address these challenges, we introduce ProKG Dial, a progressive framework for constructing knowledge intensive multi turn dialogue datasets using domain specific knowledge graphs (KGs). ProKG Dial leverages the structured nature of KGs to encode complex domain knowledge and relationships, providing a solid foundation for generating meaningful and coherent dialogues. Specifically, ProKG Dial begins by applying community detection to partition the KG into semantically cohesive subgraphs. For each subgraph, the framework incrementally generates a series of questions and answers centered around a target entity, ensuring relevance and coverage. A rigorous filtering step is employed to maintain high dialogue quality. We validate ProKG Dial on a medical knowledge graph by evaluating the generated dialogues in terms of diversity, semantic coherence, and entity coverage. Furthermore, we fine tune a base LLM on the resulting dataset and benchmark it against several baselines. Both automatic metrics and human evaluations demonstrate that ProKG Dial substantially improves dialogue quality and domain specific performance, highlighting its effectiveness and practical utility. 

**Abstract (ZH)**: 当前的大规模语言模型在通用自然语言处理任务上表现出色，但在专业环境中往往缺乏领域-specific的精准度。构建高质量的领域特定多轮对话数据集对于开发专门化的对话系统至关重要。然而，现有方法如手动标注、模拟人类与大型语言模型的交互以及基于角色的大型语言模型对话，要么资源密集，要么在对话质量和领域覆盖上存在局限。为应对这些挑战，我们引入了ProKG Dial，这是一种渐进框架，用于利用领域特定知识图谱（KGs）构建知识密集型的多轮对话数据集。ProKG Dial 利用KG的结构化特性来编码复杂的领域知识和关系，为生成有意义且连贯的对话提供了坚实的基础。具体而言，ProKG Dial 首先应用社区检测将KG划分为语义上统一的子图。对于每个子图，框架逐步生成围绕目标实体的问题和答案，确保相关性和覆盖范围。采用严格的过滤步骤来维持高对话质量。我们通过评估生成的对话在多样性、语义连贯性和实体覆盖方面的表现，对ProKG Dial在医学知识图谱上的有效性进行了验证。进一步地，我们在所得数据集上微调了一个基础的大规模语言模型，并将其与几种基线进行对比。自动评价指标和人工评估均表明，ProKG Dial 显著提高了对话质量和领域特定性能，突显了其有效性和实用价值。 

---
# CloudAnoAgent: Anomaly Detection for Cloud Sites via LLM Agent with Neuro-Symbolic Mechanism 

**Title (ZH)**: CloudAnoAgent：基于神经符号机制的云站点异常检测代理 

**Authors**: Xinkai Zou, Xuan Jiang, Ruikai Huang, Haoze He, Parv Kapoor, Jiahua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01844)  

**Abstract**: Anomaly detection in cloud sites remains a critical yet challenging task. Existing approaches that rely solely on metric data often suffer from high false positive rates (FPR) due to data imbalance between normal and anomalous events, leading to significant operational overhead for system reliance engineers. Recent advances in large language models (LLMs) offer new opportunities for integrating metrics with log data, enabling more accurate and interpretable anomaly detection. In this paper, we propose CloudAnoAgent, the first neuro-symbolic LLM-based agent for anomaly detection in cloud environments. CloudAnoAgent jointly processes structured metrics and textual log data in a unified pipeline, leveraging symbolic verification to validate detection hypotheses and generate structured anomaly reports. To support systematic evaluation, we introduce CloudAnoBench, the first benchmark that provides LLM-generated paired metrics and log data with fine-grained anomaly behavior annotations, filling a critical gap in existing datasets. Experimental results demonstrate that CloudAnoAgent improves anomaly classification accuracy by 46.36% and 36.67% on average and reduces the FPR by 36.67% and 33.89% on average over traditional baselines and LLM-only baseline, with a boost on anomaly type detection accuracy by 12.8% compared to vanilla LLM prompting. These results demonstrate the strengths of our approach in improving detection accuracy, reducing false positives, and enhancing interpretability, thereby supporting practical deployment in enterprise cloud environments. 

**Abstract (ZH)**: 基于神经符号大模型的云环境异常检测代理CloudAnoAgent 

---
# LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? 

**Title (ZH)**: LiveMCPBench: 代理能导航 MCP 工具的海洋吗？ 

**Authors**: Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01780)  

**Abstract**: With the rapid development of Model Context Protocol (MCP), the number of MCP servers has surpassed 10,000. However, existing MCP benchmarks are limited to single-server settings with only a few tools, hindering effective evaluation of agent capabilities in large-scale, real-world scenarios. To address this limitation, we present LiveMCPBench, the first comprehensive benchmark comprising 95 real-world tasks grounded in the MCP ecosystem, designed to evaluate LLM agents at scale across diverse servers. To support a scalable and reproducible evaluation pipeline in large-scale MCP environments, we curate LiveMCPTool, a diverse and readily deployable collection of 70 MCP servers and 527 tools. Furthermore, we introduce LiveMCPEval, an LLM-as-a-Judge framework that enables automated and adaptive evaluation in dynamic, time-varying task environments, achieving 81% agreement with human reviewers. Finally, we propose the MCP Copilot Agent, a multi-step agent that routes tools for dynamic planning and executes tools for API interaction across the entire LiveMCPTool suite. Our evaluation covers 10 leading models, with the best-performing model (Claude-Sonnet-4) reaching a 78.95% success rate. However, we observe large performance variance across models, and several widely-used models perform poorly in LiveMCPBench's complex, tool-rich environments. Overall, LiveMCPBench offers the first unified framework for benchmarking LLM agents in realistic, tool-rich, and dynamic MCP environments, laying a solid foundation for scalable and reproducible research on agent capabilities. Our code and data will be publicly available at this https URL. 

**Abstract (ZH)**: LiveMCPBench：面向MCP生态系统的综合性基准评估框架 

---
# Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning 

**Title (ZH)**: 基于不确定性的方法在数学推理中自动构建过程奖励数据和输出聚合 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01773)  

**Abstract**: Large language models have demonstrated remarkable capabilities in complex mathematical reasoning tasks, but they inevitably generate errors throughout multi-step solutions. Process-level Reward Models (PRMs) have shown great promise by providing supervision and evaluation at each intermediate step, thereby effectively improving the models' reasoning abilities. However, training effective PRMs requires high-quality process reward data, yet existing methods for constructing such data are often labour-intensive or inefficient. In this paper, we propose an uncertainty-driven framework for automated process reward data construction, encompassing both data generation and annotation processes for PRMs. Additionally, we identify the limitations of both majority vote and PRMs, and introduce two generic uncertainty-aware output aggregation methods: Hybrid Majority Reward Vote and Weighted Reward Frequency Vote, which combine the strengths of majority vote with PRMs. Extensive experiments on ProcessBench, MATH, and GSMPlus show the effectiveness and efficiency of the proposed PRM data construction framework, and demonstrate that the two output aggregation methods further improve the mathematical reasoning abilities across diverse PRMs. The code and data will be publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型在复杂数学推理任务中展现了 remarkable 的能力，但在多步解决方案过程中不可避免地会产生错误。过程级奖励模型（PRMs）通过在每一步提供监督和评估，展现了巨大的潜力，从而有效提高了模型的推理能力。然而，训练有效的 PRMs 需要高质量的过程奖励数据，但现有数据构建方法往往耗时且效率低下。本文提出了一种基于不确定性驱动的自动化过程奖励数据构建框架，涵盖 PRMs 的数据生成和标注过程。此外，我们识别了多数投票和 PRMs 的局限性，并引入了两种通用的不确定性感知输出聚合方法：Hybrid Majority Reward Vote 和 Weighted Reward Frequency Vote，这些方法结合了多数投票和 PRMs 的优势。在 ProcessBench、MATH 和 GSMPlus 上的大量实验显示，提出的 PRM 数据构建框架具有有效性和效率，并且两种输出聚合方法进一步提高了不同 PRMs 的数学推理能力。代码和数据将在该网址公开：this https URL。 

---
# Reasoning Systems as Structured Processes: Foundations, Failures, and Formal Criteria 

**Title (ZH)**: 结构化的过程视角下推理系统的基础、失败与正式标准 

**Authors**: Saleh Nikooroo, Thomas Engel  

**Link**: [PDF](https://arxiv.org/pdf/2508.01763)  

**Abstract**: This paper outlines a general formal framework for reasoning systems, intended to support future analysis of inference architectures across domains. We model reasoning systems as structured tuples comprising phenomena, explanation space, inference and generation maps, and a principle base. The formulation accommodates logical, algorithmic, and learning-based reasoning processes within a unified structural schema, while remaining agnostic to any specific reasoning algorithm or logic system. We survey basic internal criteria--including coherence, soundness, and completeness-and catalog typical failure modes such as contradiction, incompleteness, and non-convergence. The framework also admits dynamic behaviors like iterative refinement and principle evolution. The goal of this work is to establish a foundational structure for representing and comparing reasoning systems, particularly in contexts where internal failure, adaptation, or fragmentation may arise. No specific solution architecture is proposed; instead, we aim to support future theoretical and practical investigations into reasoning under structural constraint. 

**Abstract (ZH)**: 本文提出了一种通用的形式化框架，旨在支持跨领域推理架构的未来分析。我们将推理系统建模为包含现象、解释空间、推理和生成映射以及原则基础的结构化元组。该表述在统一的结构化框架内包容逻辑、算法和基于学习的推理过程，同时对任何特定的推理算法或逻辑系统保持中立。我们概述了基本的内在标准，包括一致性、稳健性和完备性，并记录了典型失败模式，如矛盾、不完备性和非收敛性。该框架还允许动态行为，如迭代细化和原则进化。本文的目标是在可能存在内部失败、适应或碎片化的情况下，建立表示和比较推理系统的基础结构。我们没有提出特定的解决方案架构，而是旨在支持未来在结构约束下进行推理的理论和实践研究。 

---
# Implementing Cumulative Functions with Generalized Cumulative Constraints 

**Title (ZH)**: 实现累积函数的一般累积约束 

**Authors**: Pierre Schaus, Charles Thomas, Roger Kameugne  

**Link**: [PDF](https://arxiv.org/pdf/2508.01751)  

**Abstract**: Modeling scheduling problems with conditional time intervals and cumulative functions has become a common approach when using modern commercial constraint programming solvers. This paradigm enables the modeling of a wide range of scheduling problems, including those involving producers and consumers. However, it is unavailable in existing open-source solvers and practical implementation details remain undocumented. In this work, we present an implementation of this modeling approach using a single, generic global constraint called the Generalized Cumulative. We also introduce a novel time-table filtering algorithm designed to handle tasks defined on conditional time-intervals. Experimental results demonstrate that this approach, combined with the new filtering algorithm, performs competitively with existing solvers enabling the modeling of producer and consumer scheduling problems and effectively scales to large problems. 

**Abstract (ZH)**: 基于条件时间间隔和累积函数的调度问题建模已成为使用现代商业约束编程求解器的一种常见方法。本工作介绍了使用一个通用全局约束——广义累积——实现这种建模方法，并引入了一种新的时间表过滤算法以处理基于条件时间间隔的任务。实验结果表明，该方法结合新的过滤算法，在建模 producers 和 consumers 的调度问题时具有竞争力，并能够有效处理大规模问题。 

---
# Bayes-Entropy Collaborative Driven Agents for Research Hypotheses Generation and Optimization 

**Title (ZH)**: 贝叶斯熵协作驱动代理用于研究假设生成与优化 

**Authors**: Shiyang Duan, Yuan Tian, Qi Bing, Xiaowei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01746)  

**Abstract**: The exponential growth of scientific knowledge has made the automated generation of scientific hypotheses that combine novelty, feasibility, and research value a core challenge. Existing methods based on large language models fail to systematically model the inherent in hypotheses or incorporate the closed-loop feedback mechanisms crucial for refinement. This paper proposes a multi-agent collaborative framework called HypoAgents, which for the first time integrates Bayesian reasoning with an information entropy-driven search mechanism across three stages-hypotheses generation, evidence validation, and hypotheses Refinement-to construct an iterative closed-loop simulating scientists' cognitive processes. Specifically, the framework first generates an initial set of hypotheses through diversity sampling and establishes prior beliefs based on a composite novelty-relevance-feasibility (N-R-F) score. It then employs etrieval-augmented generation (RAG) to gather external literature evidence, updating the posterior probabilities of hypotheses using Bayes' theorem. Finally, it identifies high-uncertainty hypotheses using information entropy $H = - \sum {{p_i}\log {p_i}}$ and actively refines them, guiding the iterative optimization of the hypothesis set toward higher quality and confidence. Experimental results on the ICLR 2025 conference real-world research question dataset (100 research questions) show that after 12 optimization iterations, the average ELO score of generated hypotheses improves by 116.3, surpassing the benchmark of real paper abstracts by 17.8, while the framework's overall uncertainty, as measured by Shannon entropy, decreases significantly by 0.92. This study presents an interpretable probabilistic reasoning framework for automated scientific discovery, substantially improving the quality and reliability of machine-generated research hypotheses. 

**Abstract (ZH)**: 科学知识的指数增长使得结合新颖性、可行性和研究价值的自动科学假设生成成为核心挑战。现有基于大语言模型的方法未能系统地建模假设中的固有属性或整合关键的闭环反馈机制以进行优化。本文提出了一种多智能体协作框架HypoAgents，首次将贝叶斯推理与信息熵驱动的搜索机制整合到三个阶段——假设生成、证据验证和假设优化中，构建了一个模拟科学家认知过程的迭代闭环。具体而言，该框架首先通过多样性的抽样生成初始假设集，并基于复合新颖性-相关性-可行性（N-R-F）评分建立先验信念。然后使用检索增强生成（RAG）收集外部文献证据，并使用贝叶斯定理更新假设的后验概率。最后，利用信息熵 $H = - \sum {{p_i}\log {p_i}}$ 识别高不确定性假设并主动优化它们，从而引导假设集的迭代优化以获得更高的质量和可信度。在ICLR 2025会议真实世界研究问题数据集中（包含100个研究问题）的实验结果显示，在12次优化迭代后，生成假设的平均ELO评分提高了116.3，比现实论文摘要基准高出17.8，同时框架整体不确定性，以香农熵衡量，显著降低了0.92。本研究提出了一种可解释的概率推理框架，显著提高了机器生成研究假设的质量和可靠性。 

---
# ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection 

**Title (ZH)**: ReflecSched: 通过LLM驱动的分层反射解决动态柔性作业-shop调度问题 

**Authors**: Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01724)  

**Abstract**: Dynamic Flexible Job-Shop Scheduling (DFJSP) is an NP-hard problem challenged by real-time event adaptation and complex machine routing. While traditional dispatching rules are efficient but rigid, deep learning approaches are opaque and require intricate feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments show that ReflecSched not only statistically significantly outperforms direct LLM baselines, securing a 71.35\% Win Rate and a 2.755\% Relative Percentage Deviation reduction, but also surpasses the performance of all individual heuristics evaluated, all while demonstrably mitigating the three identified pitfalls. Additionally, ReflecSched performs on par with the best heuristic tailored to each instance across all problem cases. 

**Abstract (ZH)**: 基于反思的动态柔性作业车间调度（ReflecSched） 

---
# DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning 

**Title (ZH)**: DeepVIS: 通过逐步推理连接自然语言与数据可视化 

**Authors**: Zhihao Shuai, Boyan Li, Siyu Yan, Yuyu Luo, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01700)  

**Abstract**: Although data visualization is powerful for revealing patterns and communicating insights, creating effective visualizations requires familiarity with authoring tools and often disrupts the analysis flow. While large language models show promise for automatically converting analysis intent into visualizations, existing methods function as black boxes without transparent reasoning processes, which prevents users from understanding design rationales and refining suboptimal outputs. To bridge this gap, we propose integrating Chain-of-Thought (CoT) reasoning into the Natural Language to Visualization (NL2VIS) pipeline. First, we design a comprehensive CoT reasoning process for NL2VIS and develop an automatic pipeline to equip existing datasets with structured reasoning steps. Second, we introduce nvBench-CoT, a specialized dataset capturing detailed step-by-step reasoning from ambiguous natural language descriptions to finalized visualizations, which enables state-of-the-art performance when used for model fine-tuning. Third, we develop DeepVIS, an interactive visual interface that tightly integrates with the CoT reasoning process, allowing users to inspect reasoning steps, identify errors, and make targeted adjustments to improve visualization outcomes. Quantitative benchmark evaluations, two use cases, and a user study collectively demonstrate that our CoT framework effectively enhances NL2VIS quality while providing insightful reasoning steps to users. 

**Abstract (ZH)**: 尽管数据可视化在揭示模式和传达洞见方面具有强大功能，但创建有效的可视化往往需要熟悉作者工具，并且往往会中断分析流程。虽然大规模语言模型显示出将分析意图自动转换为可视化图的潜力，但现有方法作为黑盒子运行，缺乏透明的推理过程，这使得用户无法理解设计理由并改进不理想的输出。为了解决这一问题，我们提出了将链式思考（CoT）推理集成到自然语言到可视化（NL2VIS）管道中的方法。首先，我们设计了一套全面的CoT推理过程并开发了一个自动管道，用于为现有数据集添加结构化的推理步骤。其次，我们引入了nvBench-CoT，这是一种专门的数据集，可以捕捉从模糊自然语言描述到最终可视化的过程中的详细步骤，这使得在模型微调时可以实现最先进的性能。第三，我们开发了DeepVIS，这是一种与CoT推理过程紧密集成的交互式可视化界面，允许用户检查推理步骤、识别错误并进行针对性调整以改进可视化结果。定量基准评估、两个使用案例和用户研究共同证明了我们的CoT框架在提高NL2VIS质量的同时为用户提供了解释性的推理步骤。 

---
# SURE-Med: Systematic Uncertainty Reduction for Enhanced Reliability in Medical Report Generation 

**Title (ZH)**: SURE-Med: 系统性不确定性减少以提高医学报告生成的可靠性 

**Authors**: Yuhang Gu, Xingyu Hu, Yuyu Fan, Xulin Yan, Longhuan Xu, Peng peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01693)  

**Abstract**: Automated medical report generation (MRG) holds great promise for reducing the heavy workload of radiologists. However, its clinical deployment is hindered by three major sources of uncertainty. First, visual uncertainty, caused by noisy or incorrect view annotations, compromises feature extraction. Second, label distribution uncertainty, stemming from long-tailed disease prevalence, biases models against rare but clinically critical conditions. Third, contextual uncertainty, introduced by unverified historical reports, often leads to factual hallucinations. These challenges collectively limit the reliability and clinical trustworthiness of MRG systems. To address these issues, we propose SURE-Med, a unified framework that systematically reduces uncertainty across three critical dimensions: visual, distributional, and contextual. To mitigate visual uncertainty, a Frontal-Aware View Repair Resampling module corrects view annotation errors and adaptively selects informative features from supplementary views. To tackle label distribution uncertainty, we introduce a Token Sensitive Learning objective that enhances the modeling of critical diagnostic sentences while reweighting underrepresented diagnostic terms, thereby improving sensitivity to infrequent conditions. To reduce contextual uncertainty, our Contextual Evidence Filter validates and selectively incorporates prior information that aligns with the current image, effectively suppressing hallucinations. Extensive experiments on the MIMIC-CXR and IU-Xray benchmarks demonstrate that SURE-Med achieves state-of-the-art performance. By holistically reducing uncertainty across multiple input modalities, SURE-Med sets a new benchmark for reliability in medical report generation and offers a robust step toward trustworthy clinical decision support. 

**Abstract (ZH)**: 自动化医学报告生成（MRG）在减轻放射科医生工作负担方面大有前景。然而，其临床部署受到三大不确定性来源的阻碍。首先，由噪声或错误的视角注释引起的视觉不确定性会损害特征提取。其次，由长尾疾病分布引起的标签分布不确定性会使模型偏向罕见但临床上至关重要的条件。第三，由未经验证的历史报告引入的上下文不确定性往往会导致事实性幻觉。这些挑战共同限制了MRG系统的可靠性和临床可信度。为应对这些问题，我们提出了一种统一框架SURE-Med，系统地减少了这三个关键维度上的不确定性：视觉、分布性和上下文不确定性。为减轻视觉不确定性，通过引入前景意识视角修复重采样模块来纠正视角注释错误，并自适应地从补充视角中选择具有信息性的特征。为应对标签分布不确定性，我们引入了一种敏感标记学习目标，该目标提升了对关键诊断句子的建模能力，同时重新加权未充分代表的诊断术语，从而提高了对不常见条件的敏感性。为减少上下文不确定性，我们的上下文证据过滤器验证并选择性地整合与当前图像一致的先验信息，有效抑制了幻觉现象。在MIMIC-CXR和IU-Xray基准测试上的广泛实验表明，SURE-Med取得了最先进的性能。通过对多种输入模态的整体不确定性减少，SURE-Med为医学报告生成的可靠性设定了新的基准，并为可靠的临床决策支持迈出了坚实的一步。 

---
# T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts and Redundancy in Knowledge Retrieval 

**Title (ZH)**: T-GRAG：一种用于解决知识检索中时间冲突和冗余的动态GraphRAG框架 

**Authors**: Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01680)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in natural language generation but remain limited in knowle-
dge-intensive tasks due to outdated or incomplete internal knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external retrieval, with GraphRAG further enhancing performance through structured knowledge graphs and multi-hop reasoning. However, existing GraphRAG methods largely ignore the temporal dynamics of knowledge, leading to issues such as temporal ambiguity, time-insensitive retrieval, and semantic redundancy. To overcome these limitations, we propose Temporal GraphRAG (T-GRAG), a dynamic, temporally-aware RAG framework that models the evolution of knowledge over time. T-GRAG consists of five key components: (1) a Temporal Knowledge Graph Generator that creates time-stamped, evolving graph structures; (2) a Temporal Query Decomposition mechanism that breaks complex temporal queries into manageable sub-queries; (3) a Three-layer Interactive Retriever that progressively filters and refines retrieval across temporal subgraphs; (4) a Source Text Extractor to mitigate noise; and (5) a LLM-based Generator that synthesizes contextually and temporally accurate responses. We also introduce Time-LongQA, a novel benchmark dataset based on real-world corporate annual reports, designed to test temporal reasoning across evolving knowledge. Extensive experiments show that T-GRAG significantly outperforms prior RAG and GraphRAG baselines in both retrieval accuracy and response relevance under temporal constraints, highlighting the necessity of modeling knowledge evolution for robust long-text question answering. Our code is publicly available on the T-GRAG 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言生成任务中表现出色，但在知识密集型任务中由于内部知识过时或不完整而受到限制。检索增强生成（RAG）通过引入外部检索来解决这一问题，而GraphRAG进一步通过结构化知识图和多跳推理提升了性能。然而，现有的GraphRAG方法大多忽视了知识的时空动态性，导致时间模糊、无时间敏感性的检索和语义冗余等问题。为克服这些局限，我们提出时空GraphRAG（T-GRAG），这是一种动态、时空感知的RAG框架，用于建模知识随时间的演变。T-GRAG包括五个关键组件：（1）时空知识图生成器，创建带有时间戳的演变图结构；（2）时空查询分解机制，将复杂的时空查询分解为可管理的子查询；（3）三层交互式检索器，逐步筛选和细化时间子图中的检索结果；（4）源文本提取器，减轻噪声；（5）基于LLM的生成器，合成上下文和时间准确的响应。我们还引入了时空LongQA，该基准数据集基于真实的公司年度报告，旨在测试在演变知识中的时空推理能力。广泛的实验证明，T-GRAG在时空约束下检索准确性和响应相关性方面显著优于先前的RAG和GraphRAG基线，突显了建模知识演变对于稳健的长文本问答的必要性。我们的代码已公开发布在T-GRAG。 

---
# QCBench: Evaluating Large Language Models on Domain-Specific Quantitative Chemistry 

**Title (ZH)**: QCBench: 评估大型语言模型在领域特定定量化学中的表现 

**Authors**: Jiaqing Xie, Weida Wang, Ben Gao, Zhuo Yang, Haiyuan Wan, Shufei Zhang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01670)  

**Abstract**: Quantitative chemistry plays a fundamental role in chemistry research, enabling precise predictions of molecular properties, reaction outcomes, and material behaviors. While large language models (LLMs) have shown promise in chemistry-related tasks, their ability to perform rigorous, step-by-step quantitative reasoning remains underexplored. To fill this blank, we propose QCBench, a Quantitative Chemistry benchmark comprising 350 computational chemistry problems across 7 chemistry subfields (analytical chemistry, bio/organic chemistry, general chemistry, inorganic chemistry, physical chemistry, polymer chemistry and quantum chemistry), categorized into three hierarchical tiers-basic, intermediate, and expert-to systematically evaluate the mathematical reasoning abilities of large language models (LLMs). Designed to minimize shortcuts and emphasize stepwise numerical reasoning, each problem focuses on pure calculations rooted in real-world chemical vertical fields. QCBench enables fine-grained diagnosis of computational weaknesses, reveals model-specific limitations across difficulty levels, and lays the groundwork for future improvements such as domain adaptive fine-tuning or multi-modal integration. Evaluations on 19 LLMs demonstrate a consistent performance degradation with increasing task complexity, highlighting the current gap between language fluency and scientific computation accuracy. 

**Abstract (ZH)**: 定量化学在化学研究中扮演着基础角色，能够精确预测分子性质、反应结果和材料行为。尽管大型语言模型（LLMs）在与化学相关任务中展现出潜力，但它们执行严谨的逐步定量推理的能力仍待探索。为填补这一空白，我们提出了QCBench，这是一个包含350个计算化学问题的基准，覆盖七大化学子领域（分析化学、生物/有机化学、通用化学、无机化学、物理化学、聚合物化学和量子化学），分为三个层次的基础、中级和专家级，旨在系统评估大型语言模型（LLMs）的数学推理能力。该基准设计旨在减少捷径，强调逐步数值推理，每个问题都侧重于源于实际化学领域的纯计算。QCBench能够细粒度地诊断计算弱项，揭示不同难度级别下模型特有的限制，并为未来的改进工作奠定基础，如领域自适应微调或多模态整合。对19个LLM的评估表明，在任务复杂性增加时，其性能呈现出一致的退化，突显了语言流畅性和科学计算准确性之间的当前差距。 

---
# DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition 

**Title (ZH)**: DRKF: 解耦表示与知识融合在多模态情感识别中的应用 

**Authors**: Peiyuan Jiang, Yao Liu, Qiao Liu, Zongshun Zhang, Jiaye Yang, Lu Liu, Daibing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01644)  

**Abstract**: Multimodal emotion recognition (MER) aims to identify emotional states by integrating and analyzing information from multiple modalities. However, inherent modality heterogeneity and inconsistencies in emotional cues remain key challenges that hinder performance. To address these issues, we propose a Decoupled Representations with Knowledge Fusion (DRKF) method for MER. DRKF consists of two main modules: an Optimized Representation Learning (ORL) Module and a Knowledge Fusion (KF) Module. ORL employs a contrastive mutual information estimation method with progressive modality augmentation to decouple task-relevant shared representations and modality-specific features while mitigating modality heterogeneity. KF includes a lightweight self-attention-based Fusion Encoder (FE) that identifies the dominant modality and integrates emotional information from other modalities to enhance the fused representation. To handle potential errors from incorrect dominant modality selection under emotionally inconsistent conditions, we introduce an Emotion Discrimination Submodule (ED), which enforces the fused representation to retain discriminative cues of emotional inconsistency. This ensures that even if the FE selects an inappropriate dominant modality, the Emotion Classification Submodule (EC) can still make accurate predictions by leveraging preserved inconsistency information. Experiments show that DRKF achieves state-of-the-art (SOTA) performance on IEMOCAP, MELD, and M3ED. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 多模态情感识别（MER）旨在通过整合和分析多种模态的信息来识别情感状态。然而，固有的模态异质性和情感线索的一致性问题仍然是阻碍性能的关键挑战。为了解决这些问题，我们提出了一种解耦表示与知识融合（DRKF）方法用于MER。DRKF包括两个主要模块：优化表示学习（ORL）模块和知识融合（KF）模块。ORL使用对比互信息估计方法结合逐步模态增强来解耦与任务相关的共享表示和模态特定特征，同时减轻模态异质性。KF包含一个轻量级的基于自注意力的融合编码器（FE），它可以识别主导模态并整合其他模态的情感信息以增强融合表示。为了处理在情感不一致条件下可能因错误的主导模态选择而导致的潜在错误，我们引入了一个情感鉴别子模块（ED），它强制融合表示保留情感不一致性的鉴别性线索，从而即使FE选择不合适的主导模态，情感分类子模块（EC）仍可以通过利用保留的不一致性信息做出准确预测。实验结果显示，DRKF在IEMOCAP、MELD和M3ED上取得了当前最佳性能（SOTA）。源代码已在此处公开。 

---
# A Multi-Agent Pokemon Tournament for Evaluating Strategic Reasoning of Large Language Models 

**Title (ZH)**: 大型语言模型的战略推理评估多Agent宝可梦比赛 

**Authors**: Tadisetty Sai Yashwanth, Dhatri C  

**Link**: [PDF](https://arxiv.org/pdf/2508.01623)  

**Abstract**: This research presents LLM Pokemon League, a competitive tournament system that leverages Large Language Models (LLMs) as intelligent agents to simulate strategic decision-making in Pokémon battles. The platform is designed to analyze and compare the reasoning, adaptability, and tactical depth exhibited by different LLMs in a type-based, turn-based combat environment. By structuring the competition as a single-elimination tournament involving diverse AI trainers, the system captures detailed decision logs, including team-building rationale, action selection strategies, and switching decisions. The project enables rich exploration into comparative AI behavior, battle psychology, and meta-strategy development in constrained, rule-based game environments. Through this system, we investigate how modern LLMs understand, adapt, and optimize decisions under uncertainty, making Pokémon League a novel benchmark for AI research in strategic reasoning and competitive learning. 

**Abstract (ZH)**: LLM покемонский лиге：基于大型语言模型的 Competitive 比赛系统及其在基于类型和轮次 combat 环境中对战略决策模拟的研究 

---
# Polymorphic Combinatorial Frameworks (PCF): Guiding the Design of Mathematically-Grounded, Adaptive AI Agents 

**Title (ZH)**: 多态组合框架（PCF）：引导基于数学原理、自适应AI代理的设计 

**Authors**: David Pearl, Matthew Murphy, James Intriligator  

**Link**: [PDF](https://arxiv.org/pdf/2508.01581)  

**Abstract**: The Polymorphic Combinatorial Framework (PCF) leverages Large Language Models (LLMs) and mathematical frameworks to guide the meta-prompt enabled design of solution spaces and adaptive AI agents for complex, dynamic environments. Unlike static agent architectures, PCF enables real-time parameter reconfiguration through mathematically-grounded combinatorial spaces, allowing agents to adapt their core behavioral traits dynamically. Grounded in combinatorial logic, topos theory, and rough fuzzy set theory, PCF defines a multidimensional SPARK parameter space (Skills, Personalities, Approaches, Resources, Knowledge) to capture agent behaviors. This paper demonstrates how LLMs can parameterize complex spaces and estimate likely parameter values/variabilities. Using PCF, we parameterized mock café domains (five levels of complexity), estimated variables/variabilities, and conducted over 1.25 million Monte Carlo simulations. The results revealed trends in agent adaptability and performance across the five complexity tiers, with diminishing returns at higher complexity levels highlighting thresholds for scalable designs. PCF enables the generation of optimized agent configurations for specific scenarios while maintaining logical consistency. This framework supports scalable, dynamic, explainable, and ethical AI applications in domains like customer service, healthcare, robotics, and collaborative systems, paving the way for adaptable and cooperative next-generation polymorphic agents. 

**Abstract (ZH)**: 多态组合框架（PCF）利用大型语言模型（LLMs）和数学框架，指导支持元提示设计的解决方案空间和适应性AI代理，适用于复杂的动态环境。不同于静态代理架构，PCF通过数学奠基的组合空间实现实时参数重构，使代理能够动态调整其核心行为特征。基于组合逻辑、范畴论和粗糙模糊集理论，PCF定义了一维SPARK参数空间（Skills, Personalities, Approaches, Resources, Knowledge），以捕捉代理行为。本文展示了如何通过LLMs参数化复杂的空间并估计可能的参数值/变异性。使用PCF，我们对五个复杂级别的模拟咖啡馆领域进行了参数化、变量/变异性估计，并进行了超过125万次蒙特卡洛模拟。结果揭示了代理适应性和性能随五个复杂度等级的变化趋势，在较高复杂度等级处出现递减回报，突显了可扩展设计的门槛。PCF能够生成特定场景下的优化代理配置，同时保持逻辑一致性。该框架支持可扩展、动态、可解释和负责任的AI应用，涵盖客户服务、医疗保健、机器人技术和协作系统等领域，为适应性和协作性的下一代多态代理铺平了道路。 

---
# One Subgoal at a Time: Zero-Shot Generalization to Arbitrary Linear Temporal Logic Requirements in Multi-Task Reinforcement Learning 

**Title (ZH)**: 一次子目标一步：多任务强化学习中对任意线性时序逻辑要求的零样本泛化 

**Authors**: Zijian Guo, İlker Işık, H. M. Sabbir Ahmad, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01561)  

**Abstract**: Generalizing to complex and temporally extended task objectives and safety constraints remains a critical challenge in reinforcement learning (RL). Linear temporal logic (LTL) offers a unified formalism to specify such requirements, yet existing methods are limited in their abilities to handle nested long-horizon tasks and safety constraints, and cannot identify situations when a subgoal is not satisfiable and an alternative should be sought. In this paper, we introduce GenZ-LTL, a method that enables zero-shot generalization to arbitrary LTL specifications. GenZ-LTL leverages the structure of Büchi automata to decompose an LTL task specification into sequences of reach-avoid subgoals. Contrary to the current state-of-the-art method that conditions on subgoal sequences, we show that it is more effective to achieve zero-shot generalization by solving these reach-avoid problems \textit{one subgoal at a time} through proper safe RL formulations. In addition, we introduce a novel subgoal-induced observation reduction technique that can mitigate the exponential complexity of subgoal-state combinations under realistic assumptions. Empirical results show that GenZ-LTL substantially outperforms existing methods in zero-shot generalization to unseen LTL specifications. 

**Abstract (ZH)**: 广义化到复杂的和时序扩展的任务目标及安全约束仍然是强化学习（RL）中的一个关键挑战。GenZ-LTL：一种实现任意LTL规范零样本广义化的方法 

---
# Empowering Tabular Data Preparation with Language Models: Why and How? 

**Title (ZH)**: 利用语言模型赋能表格数据准备：为什么以及如何实现？ 

**Authors**: Mengshi Chen, Yuxiang Sun, Tengchao Li, Jianwei Wang, Kai Wang, Xuemin Lin, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01556)  

**Abstract**: Data preparation is a critical step in enhancing the usability of tabular data and thus boosts downstream data-driven tasks. Traditional methods often face challenges in capturing the intricate relationships within tables and adapting to the tasks involved. Recent advances in Language Models (LMs), especially in Large Language Models (LLMs), offer new opportunities to automate and support tabular data preparation. However, why LMs suit tabular data preparation (i.e., how their capabilities match task demands) and how to use them effectively across phases still remain to be systematically explored. In this survey, we systematically analyze the role of LMs in enhancing tabular data preparation processes, focusing on four core phases: data acquisition, integration, cleaning, and transformation. For each phase, we present an integrated analysis of how LMs can be combined with other components for different preparation tasks, highlight key advancements, and outline prospective pipelines. 

**Abstract (ZH)**: 语言模型在增强表格数据准备过程中的作用及其应用：一项涵盖数据获取、集成、清洗和转换四个核心阶段的系统调研 

---
# Getting out of the Big-Muddy: Escalation of Commitment in LLMs 

**Title (ZH)**: 摆脱泥潭：大型语言模型中的承诺升级问题 

**Authors**: Emilio Barkett, Olivia Long, Paul Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2508.01545)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in autonomous decision-making roles across high-stakes domains. However, since models are trained on human-generated data, they may inherit cognitive biases that systematically distort human judgment, including escalation of commitment, where decision-makers continue investing in failing courses of action due to prior investment. Understanding when LLMs exhibit such biases presents a unique challenge. While these biases are well-documented in humans, it remains unclear whether they manifest consistently in LLMs or require specific triggering conditions. This paper investigates this question using a two-stage investment task across four experimental conditions: model as investor, model as advisor, multi-agent deliberation, and compound pressure scenario. Across N = 6,500 trials, we find that bias manifestation in LLMs is highly context-dependent. In individual decision-making contexts (Studies 1-2, N = 4,000), LLMs demonstrate strong rational cost-benefit logic with minimal escalation of commitment. However, multi-agent deliberation reveals a striking hierarchy effect (Study 3, N = 500): while asymmetrical hierarchies show moderate escalation rates (46.2%), symmetrical peer-based decision-making produces near-universal escalation (99.2%). Similarly, when subjected to compound organizational and personal pressures (Study 4, N = 2,000), models exhibit high degrees of escalation of commitment (68.95% average allocation to failing divisions). These findings reveal that LLM bias manifestation depends critically on social and organizational context rather than being inherent, with significant implications for the deployment of multi-agent systems and unsupervised operations where such conditions may emerge naturally. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在高风险领域自主决策中的偏见表现研究 

---
# Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning 

**Title (ZH)**: Refine-n-Judge: 精炼并判断——为LLM微调构建高质量偏好链 

**Authors**: Derin Cayir, Renjie Tao, Rashi Rungta, Kai Sun, Sean Chen, Haidar Khan, Minseok Kim, Julia Reinspach, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01543)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress through preference-based fine-tuning, which critically depends on the quality of the underlying training data. While human feedback is essential for improving data quality, it is costly and does not scale well. In this paper, we introduce Refine-n-Judge, an automated iterative approach that leverages a single LLM as both a refiner and a judge to enhance dataset quality. Unlike existing iterative refinement methods, Refine-n-Judge employs an LLM to both generate refinements and explicitly evaluate each improvement, ensuring that every iteration meaningfully enhances the dataset without requiring additional human annotation or a separate reward model. At each step, the LLM refines a response and judges whether the refinement is an improvement over the previous answer. This process continues until the LLM prefers the initial answer over the refinement, indicating no further improvements. This produces sequences of increasing quality, preference-labeled responses ideal for fine-tuning.
We demonstrate the effectiveness of Refine-n-Judge across a range of public datasets spanning five corpora, targeting tasks such as coding, math, and conversation. Models (Llama 3.1-8B and Llama 3.3-70B) fine-tuned on Refine-n-Judge-enhanced datasets were preferred by LLM judges in over 74% of comparisons against models tuned on the original dataset by GPT-4. Additionally, we report performance gains: +5% on AlpacaEval and AlpacaEval 2.0, and +19% on MT-Bench. Our results indicate that Refine-n-Judge produces high-quality datasets and scalable model improvements. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过基于偏好的微调取得了显著进展，这对其基础训练数据的质量有关键依赖。虽然人类反馈对于提高数据质量至关重要，但它成本高昂且不具备良好的扩展性。本文介绍了Refine-n-Judge，这是一种自动化迭代方法，利用单一LLM作为修正确和评估者来提升数据集质量。与现有迭代精炼方法不同，Refine-n-Judge 使用LLM生成修改并明确评估每一个改进，确保每一次迭代都能实质性地提高数据集质量，而无需额外的人工注释或独立的奖励模型。在每一步中，LLM 修正一个响应并判断该修正是否比前一个答案更好。这一过程将持续到LLM更喜欢初始答案而非其修正，表明没有进一步改进。这生成了质量递增、带有偏好标签的响应序列，适于模型微调。 

---
# WinkTPG: An Execution Framework for Multi-Agent Path Finding Using Temporal Reasoning 

**Title (ZH)**: WinkTPG：基于时间推理的多Agent路径规划执行框架 

**Authors**: Jingtian Yan, Stephen F. Smith, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01495)  

**Abstract**: Planning collision-free paths for a large group of agents is a challenging problem with numerous real-world applications. While recent advances in Multi-Agent Path Finding (MAPF) have shown promising progress, standard MAPF algorithms rely on simplified kinodynamic models, preventing agents from directly following the generated MAPF plan. To bridge this gap, we propose kinodynamic Temporal Plan Graph Planning (kTPG), a multi-agent speed optimization algorithm that efficiently refines a MAPF plan into a kinodynamically feasible plan while accounting for uncertainties and preserving collision-freeness. Building on kTPG, we propose Windowed kTPG (WinkTPG), a MAPF execution framework that incrementally refines MAPF plans using a window-based mechanism, dynamically incorporating agent information during execution to reduce uncertainty. Experiments show that WinkTPG can generate speed profiles for up to 1,000 agents in 1 second and improves solution quality by up to 51.7% over existing MAPF execution methods. 

**Abstract (ZH)**: 大规模群体代理的碰撞免费路径规划是一个具有诸多实际应用的研究挑战。虽然最近在多代理路径finding (MAPF) 方面取得了有前景的进展，但标准的MAPF算法依赖于简化的动力学模型，导致代理无法直接遵循生成的MAPF计划。为解决这一问题，我们提出了动力学时序计划图规划（kTPG），这是一种多代理速度优化算法，可以高效地将MAPF计划细化为动力学可行的计划，同时考虑不确定性并保持碰撞免费。基于kTPG，我们提出了窗口化kTPG（WinkTPG），这是一种通过窗口机制逐步细化MAPF计划的多代理路径finding执行框架，在执行过程中动态整合代理信息以降低不确定性。实验结果显示，WinkTPG可以在1秒内为多达1,000个代理生成速度轮廓，并比现有MAPF执行方法提高了解决方案质量高达51.7%。 

---
# CARGO: A Co-Optimization Framework for EV Charging and Routing in Goods Delivery Logistics 

**Title (ZH)**: CARGO: 面向货物配送物流的电动汽车充电与路由联合优化框架 

**Authors**: Arindam Khanda, Anurag Satpathy, Amit Jha, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.01476)  

**Abstract**: With growing interest in sustainable logistics, electric vehicle (EV)-based deliveries offer a promising alternative for urban distribution. However, EVs face challenges due to their limited battery capacity, requiring careful planning for recharging. This depends on factors such as the charging point (CP) availability, cost, proximity, and vehicles' state of charge (SoC). We propose CARGO, a framework addressing the EV-based delivery route planning problem (EDRP), which jointly optimizes route planning and charging for deliveries within time windows. After proving the problem's NP-hardness, we propose a mixed integer linear programming (MILP)-based exact solution and a computationally efficient heuristic method. Using real-world datasets, we evaluate our methods by comparing the heuristic to the MILP solution, and benchmarking it against baseline strategies, Earliest Deadline First (EDF) and Nearest Delivery First (NDF). The results show up to 39% and 22% reductions in the charging cost over EDF and NDF, respectively, while completing comparable deliveries. 

**Abstract (ZH)**: 基于电动汽车的城市配送路径与充电优化研究 

---
# $R^2$-CoD: Understanding Text-Graph Complementarity in Relational Reasoning via Knowledge Co-Distillation 

**Title (ZH)**: $R^2$-CoD: 通过知识协同提炼理解文本-图形在关系推理中的互补性 

**Authors**: Zhen Wu, Ritam Dutt, Luke M. Breitfeller, Armineh Nourbakhsh, Siddharth Parekh, Carolyn Rosé  

**Link**: [PDF](https://arxiv.org/pdf/2508.01475)  

**Abstract**: Relational reasoning lies at the core of many NLP tasks, drawing on complementary signals from text and graphs. While prior research has investigated how to leverage this dual complementarity, a detailed and systematic understanding of text-graph interplay and its effect on hybrid models remains underexplored. We take an analysis-driven approach to investigate text-graph representation complementarity via a unified architecture that supports knowledge co-distillation (CoD). We explore five tasks involving relational reasoning that differ in how text and graph structures encode the information needed to solve that task. By tracking how these dual representations evolve during training, we uncover interpretable patterns of alignment and divergence, and provide insights into when and why their integration is beneficial. 

**Abstract (ZH)**: 关系推理是许多自然语言处理任务的核心，它依赖于文本和图互补信号的利用。尽管此前的研究探讨了如何利用这种双重互补性，但文本-图相互作用的详细系统理解及其对混合模型的影响仍待进一步探索。我们通过支持知识共提炼（CoD）的统一架构，采用分析驱动的方法来研究文本-图表示的互补性。我们探索了五个涉及关系推理的任务，这些任务在文本和图结构如何编码解决任务所需信息方面存在差异。通过跟踪这些双重表示在训练过程中如何演变，我们发现可解释的对齐和偏离模式，并提供了关于在何时以及为何整合它们是有益的见解。 

---
# TripTailor: A Real-World Benchmark for Personalized Travel Planning 

**Title (ZH)**: TripTailor: 个性化旅行规划的现实世界基准 

**Authors**: Yuanzhe Shen, Kaimin Wang, Changze Lv, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01432)  

**Abstract**: The continuous evolution and enhanced reasoning capabilities of large language models (LLMs) have elevated their role in complex tasks, notably in travel planning, where demand for personalized, high-quality itineraries is rising. However, current benchmarks often rely on unrealistic simulated data, failing to reflect the differences between LLM-generated and real-world itineraries. Existing evaluation metrics, which primarily emphasize constraints, fall short of providing a comprehensive assessment of the overall quality of travel plans. To address these limitations, we introduce TripTailor, a benchmark designed specifically for personalized travel planning in real-world scenarios. This dataset features an extensive collection of over 500,000 real-world points of interest (POIs) and nearly 4,000 diverse travel itineraries, complete with detailed information, providing a more authentic evaluation framework. Experiments show that fewer than 10\% of the itineraries generated by the latest state-of-the-art LLMs achieve human-level performance. Moreover, we identify several critical challenges in travel planning, including the feasibility, rationality, and personalized customization of the proposed solutions. We hope that TripTailor will drive the development of travel planning agents capable of understanding and meeting user needs while generating practical itineraries. Our code and dataset are available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）的持续演化与增强的推理能力提升了其在复杂任务中的角色，特别在旅游规划领域，个性化、高质量的行程需求日益增长。然而，当前的基准测试往往依赖于不现实的模拟数据，未能反映LLM生成的行程与现实世界行程之间的差异。现有的评估指标主要侧重于约束条件，未能全面评估旅游计划的整体质量。为进一步解决这些问题，我们提出了TripTailor，一个专门针对实际场景个性化旅游规划的基准数据集。该数据集包含超过500,000个真实世界的兴趣点（POIs）和近4,000份多样化的旅游行程，附有详细信息，提供了一个更为真实的评估框架。实验结果显示，最新最先进的LLMs生成的不足10%的行程达到了人类水平的表现。此外，我们还识别出旅游规划中的几个关键挑战，包括解决方案的可行性、合理性以及个性化定制。我们希望TripTailor能够推动能够理解并满足用户需求，并生成实用行程的旅游规划代理的发展。相关代码和数据集可在以下链接获取。 

---
# Relation-Aware LNN-Transformer for Intersection-Centric Next-Step Prediction 

**Title (ZH)**: 基于关系的LNN- Transformer用于以交叉口为中心的下一步预测 

**Authors**: Zhehong Ren, Tianluo Zhang, Yiheng Lu, Yushen Liang, Promethee Spathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01368)  

**Abstract**: Next-step location prediction plays a pivotal role in modeling human mobility, underpinning applications from personalized navigation to strategic urban planning. However, approaches that assume a closed world - restricting choices to a predefined set of points of interest (POIs) - often fail to capture exploratory or target-agnostic behavior and the topological constraints of urban road networks. Hence, we introduce a road-node-centric framework that represents road-user trajectories on the city's road-intersection graph, thereby relaxing the closed-world constraint and supporting next-step forecasting beyond fixed POI sets. To encode environmental context, we introduce a sector-wise directional POI aggregation that produces compact features capturing distance, bearing, density and presence cues. By combining these cues with structural graph embeddings, we obtain semantically grounded node representations. For sequence modeling, we integrate a Relation-Aware LNN-Transformer - a hybrid of a Continuous-time Forgetting Cell CfC-LNN and a bearing-biased self-attention module - to capture both fine-grained temporal dynamics and long-range spatial dependencies. Evaluated on city-scale road-user trajectories, our model outperforms six state-of-the-art baselines by up to 17 percentage points in accuracy at one hop and 10 percentage points in MRR, and maintains high resilience under noise, losing only 2.4 percentage points in accuracy at one under 50 meter GPS perturbation and 8.9 percentage points in accuracy at one hop under 25 percent POI noise. 

**Abstract (ZH)**: 基于道路节点的下一步位置预测框架：超越固定兴趣点集的路网用户轨迹建模 

---
# NatureGAIA: Pushing the Frontiers of GUI Agents with a Challenging Benchmark and High-Quality Trajectory Dataset 

**Title (ZH)**: NatureGAIA：以具有挑战性基准和高质量轨迹数据集推动GUI代理技术前沿 

**Authors**: Zihan Zheng, Tianle Cui, Chuwen Xie, Jiahui Zhang, Jiahui Pan, Lewei He, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01330)  

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven Graphical User Interface (GUI) agents is significantly hampered by the profound limitations of existing evaluation benchmarks in terms of accuracy, reproducibility, and scalability. To address this critical gap, we introduce \Benchmark, a novel benchmark engineered on the principle of Causal Pathways. This design paradigm structures complex tasks into a series of programmatically verifiable atomic steps, ensuring a rigorous, fully automated, and reproducible standard for assessment. Concurrently, to mitigate the inherent capability deficits of agents, we developed \Agent, a hierarchical agent architecture specifically optimized for long-horizon tasks. We leveraged this agent to generate a high-quality, human-verified trajectory dataset that uniquely captures diverse and even self-correcting interaction patterns of LLMs. We then utilized this dataset to perform Reinforcement Fine-Tuning (RFT) on the Qwen2.5-VL-7B model. Our experiments reveal that \Benchmark~presents a formidable challenge to current state-of-the-art LLMs; even the top-performing Claude-sonnet-4 achieved a Weighted Pathway Success Rate (WPSR) of only 34.6\%. Moreover, while RFT substantially improved the smaller model's GUI execution capabilities (WPSR increased from 3.3\% to 10.8\%), its performance degraded sharply when handling complex scenarios. This outcome highlights the inherent capability ceiling of smaller models when faced with comprehensive tasks that integrate perception, decision-making, and execution. This research contributes a rigorous evaluation standard and a high-quality dataset to the community, aiming to guide the future development of GUI agents. 

**Abstract (ZH)**: 基于因果路径的新型评估基准Benchmark显著提升了受大规模语言模型驱动的图形用户界面代理的发展速度，但现有的评估基准在准确性和可重现实度方面存在严重的限制，且缺乏可扩展性。为了填补这一关键空白，我们引入了基于因果路径原则设计的Benchmark，该设计模式将复杂任务分解为一系列可编程验证的基本步骤，从而确保了评估的严格性、全面自动化和可重现实度。同时，为克服代理固有的能力不足，我们开发了专为长期任务优化的分层代理架构Agent。利用此代理生成了高质量且经过人工验证的轨迹数据集，该数据集独特地捕捉了LLMs的多样且自我纠正的交互模式。然后，我们使用此数据集对Qwen2.5-VL-7B模型进行了强化微调(RFT)。实验结果表明，Benchmark对当前最先进的LLMs构成了严峻挑战；即便是表现最佳的Claude-sonnet-4，其加权路径成功率(WPSR)也只有34.6%。此外，尽管RFT显著提高了较小模型的GUI执行能力（WPSR从3.3%提升到10.8%），但在处理复杂场景时，其性能急剧下降。这一结果凸显了较小模型在面对综合感知、决策和执行任务时的固有限制。该研究贡献了一种严格评估标准和高质量数据集，旨在指导图形用户界面代理的未来开发。 

---
# Towards Evaluation for Real-World LLM Unlearning 

**Title (ZH)**: 面向现实世界的大型语言模型遗忘评估 

**Authors**: Ke Miao, Yuke Hu, Xiaochen Li, Wenjie Bao, Zhihao Liu, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01324)  

**Abstract**: This paper analyzes the limitations of existing unlearning evaluation metrics in terms of practicality, exactness, and robustness in real-world LLM unlearning scenarios. To overcome these limitations, we propose a new metric called Distribution Correction-based Unlearning Evaluation (DCUE). It identifies core tokens and corrects distributional biases in their confidence scores using a validation set. The evaluation results are quantified using the Kolmogorov-Smirnov test. Experimental results demonstrate that DCUE overcomes the limitations of existing metrics, which also guides the design of more practical and reliable unlearning algorithms in the future. 

**Abstract (ZH)**: 基于分布校正的遗忘评估（DCUE）：实景观点下大语言模型遗忘评估的新方法 

---
# Idempotent Equilibrium Analysis of Hybrid Workflow Allocation: A Mathematical Schema for Future Work 

**Title (ZH)**: 混合工作流分配中的幂等均衡分析：面向未来工作的数学框架 

**Authors**: Faruk Alpay, Bugra Kilictas, Taylan Alpay, Hamdi Alakkad  

**Link**: [PDF](https://arxiv.org/pdf/2508.01323)  

**Abstract**: The rapid advance of large-scale AI systems is reshaping how work is divided between people and machines. We formalise this reallocation as an iterated task-delegation map and show that--under broad, empirically grounded assumptions--the process converges to a stable idempotent equilibrium in which every task is performed by the agent (human or machine) with enduring comparative advantage. Leveraging lattice-theoretic fixed-point tools (Tarski and Banach), we (i) prove existence of at least one such equilibrium and (ii) derive mild monotonicity conditions that guarantee uniqueness. In a stylised continuous model the long-run automated share takes the closed form $x^* = \alpha / (\alpha + \beta)$, where $\alpha$ captures the pace of automation and $\beta$ the rate at which new, human-centric tasks appear; hence full automation is precluded whenever $\beta > 0$. We embed this analytic result in three complementary dynamical benchmarks--a discrete linear update, an evolutionary replicator dynamic, and a continuous Beta-distributed task spectrum--each of which converges to the same mixed equilibrium and is reproducible from the provided code-free formulas. A 2025-to-2045 simulation calibrated to current adoption rates projects automation rising from approximately 10% of work to approximately 65%, leaving a persistent one-third of tasks to humans. We interpret that residual as a new profession of workflow conductor: humans specialise in assigning, supervising and integrating AI modules rather than competing with them. Finally, we discuss implications for skill development, benchmark design and AI governance, arguing that policies which promote "centaur" human-AI teaming can steer the economy toward the welfare-maximising fixed point. 

**Abstract (ZH)**: 大规模AI系统的迅速发展正在重新塑造人类和机器之间的劳动分工。我们这种重新分配形式化为迭代的任务委派映射，并表明在广泛的经验基础上，该过程收敛到一个稳定的恒等平衡点，在此点上每个任务都由具有持久比较优势的代理（人类或机器）执行。利用格论不动点工具（塔斯基和巴纳赫），我们（i）证明至少存在一个这样的平衡点，并（ii）推导出保证唯一性的温和单调条件。在一个简化的连续模型中，长期的自动化份额以闭形式 $x^* = \alpha / (\alpha + \beta)$ 表示，其中 $\alpha$ 表示自动化速度，$\beta$ 表示新的人类中心任务出现的速率；因此，只要 $\beta > 0$，就排除了完全自动化的可能性。我们将这一分析结果嵌入三个互补的动力学基准中——离散线性更新、进化复制动态和连续的Beta分布任务谱——每个基准都收敛到相同的混合平衡点，并且可以从提供的代码自由公式中重现。根据当前采用率调整后的2025年至2045年模拟预测，自动化从大约10%的工作份额上升到大约65%，留下大约三分之一的任务给人类。我们解释这个剩余部分为一种新的职业——工作流程指挥官：人类专门负责分配、监督和整合AI模块，而不是与之竞争。最后，我们讨论了技能发展、基准设计和AI治理的影响，认为促进“人马合一”的人机团队策略能够引导经济向福利最大化不动点发展。 

---
# PUZZLED: Jailbreaking LLMs through Word-Based Puzzles 

**Title (ZH)**: PUZZLED：通过词谜破解LLMs 

**Authors**: Yelim Ahn, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01306)  

**Abstract**: As large language models (LLMs) are increasingly deployed across diverse domains, ensuring their safety has become a critical concern. In response, studies on jailbreak attacks have been actively growing. Existing approaches typically rely on iterative prompt engineering or semantic transformations of harmful instructions to evade detection. In this work, we introduce PUZZLED, a novel jailbreak method that leverages the LLM's reasoning capabilities. It masks keywords in a harmful instruction and presents them as word puzzles for the LLM to solve. We design three puzzle types-word search, anagram, and crossword-that are familiar to humans but cognitively demanding for LLMs. The model must solve the puzzle to uncover the masked words and then proceed to generate responses to the reconstructed harmful instruction. We evaluate PUZZLED on five state-of-the-art LLMs and observe a high average attack success rate (ASR) of 88.8%, specifically 96.5% on GPT-4.1 and 92.3% on Claude 3.7 Sonnet. PUZZLED is a simple yet powerful attack that transforms familiar puzzles into an effective jailbreak strategy by harnessing LLMs' reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）日益广泛应用于各类领域，确保其安全性已成为关键问题。为应对这一挑战，关于牢笼攻击的研究正在不断增长。现有方法通常依赖于迭代提示工程或有害指令的语义转换以逃避检测。在此工作中，我们提出了PUZZLED，一种新颖的牢笼攻击方法，利用LLM的推理能力。该方法隐身有害指令中的关键词，并将其呈现为字谜供LLM求解。我们设计了三种熟悉的谜题类型——字搜、乱词和填字——这些谜题对人类来说是熟悉的，但对LLM来说认知上更具挑战性。模型必须解决谜题以揭示被隐身的词，然后生成针对重构的有害指令的回答。我们评估了PUZZLED在五个先进的LLM上的效果，并观察到高平均攻击成功率（ASR）达88.8%，具体来说，在GPT-4.1上的成功率是96.5%，在Claude 3.7 Sonnet上的成功率是92.3%。PUZZLED是一种简单而强大的攻击方法，通过利用LLM的推理能力，将熟悉的谜题转化为有效的牢笼攻击策略。 

---
# How Far Are LLMs from Symbolic Planners? An NLP-Based Perspective 

**Title (ZH)**: LLMs与符号规划器之间相差多远？一个基于NLP的观点 

**Authors**: Ma'ayan Armony, Albert Meroño-Peñuela, Gerard Canal  

**Link**: [PDF](https://arxiv.org/pdf/2508.01300)  

**Abstract**: The reasoning and planning abilities of Large Language Models (LLMs) have been a frequent topic of discussion in recent years. Their ability to take unstructured planning problems as input has made LLMs' integration into AI planning an area of interest. Nevertheless, LLMs are still not reliable as planners, with the generated plans often containing mistaken or hallucinated actions. Existing benchmarking and evaluation methods investigate planning with LLMs, focusing primarily on success rate as a quality indicator in various planning tasks, such as validating plans or planning in relaxed conditions. In this paper, we approach planning with LLMs as a natural language processing (NLP) task, given that LLMs are NLP models themselves. We propose a recovery pipeline consisting of an NLP-based evaluation of the generated plans, along with three stages to recover the plans through NLP manipulation of the LLM-generated plans, and eventually complete the plan using a symbolic planner. This pipeline provides a holistic analysis of LLM capabilities in the context of AI task planning, enabling a broader understanding of the quality of invalid plans. Our findings reveal no clear evidence of underlying reasoning during plan generation, and that a pipeline comprising an NLP-based analysis of the plans, followed by a recovery mechanism, still falls short of the quality and reliability of classical planners. On average, only the first 2.65 actions of the plan are executable, with the average length of symbolically generated plans being 8.4 actions. The pipeline still improves action quality and increases the overall success rate from 21.9% to 27.5%. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理与规划能力近年来是一个频繁讨论的话题。尽管LLMs能够处理非结构化的规划问题，使其在人工智能规划中的应用成为一个研究热点，但现有的规划结果仍然不够可靠，生成的计划中经常包含错误或虚构的动作。现有的基准测试和评估方法主要关注规划任务的成功率作为质量指标，研究LLMs在规划中的应用。在本文中，鉴于LLMs本身就是自然语言处理（NLP）模型，我们将LLMs用于规划视为一个NLP任务，并提出了一种恢复管道，包括基于NLP的生成计划评估，以及通过NLP操纵LLM生成的计划的三个阶段，最终使用符号规划器完成计划。该管道为LLMs在AI任务规划中的能力提供了全面分析，有助于更广泛地理解无效计划的质量。研究发现，在计划生成过程中没有明显证据表明存在底层推理，而包含基于NLP分析计划和随后的恢复机制的管道仍然无法达到经典规划器的质量和可靠性。平均而言，只有计划中的前2.65个动作可执行，符号生成计划的平均长度为8.4个动作。该管道仍能提高动作质量并将总体成功率从21.9%提升到27.5%。 

---
# BioDisco: Multi-agent hypothesis generation with dual-mode evidence, iterative feedback and temporal evaluation 

**Title (ZH)**: BioDisco：具有双模式证据、迭代反馈和时间评价的多agent假设生成方法 

**Authors**: Yujing Ke, Kevin George, Kathan Pandya, David Blumenthal, Maximilian Sprang, Gerrit Großmann, Sebastian Vollmer, David Antony Selby  

**Link**: [PDF](https://arxiv.org/pdf/2508.01285)  

**Abstract**: Identifying novel hypotheses is essential to scientific research, yet this process risks being overwhelmed by the sheer volume and complexity of available information. Existing automated methods often struggle to generate novel and evidence-grounded hypotheses, lack robust iterative refinement and rarely undergo rigorous temporal evaluation for future discovery potential. To address this, we propose BioDisco, a multi-agent framework that draws upon language model-based reasoning and a dual-mode evidence system (biomedical knowledge graphs and automated literature retrieval) for grounded novelty, integrates an internal scoring and feedback loop for iterative refinement, and validates performance through pioneering temporal and human evaluations and a Bradley-Terry paired comparison model to provide statistically-grounded assessment. Our evaluations demonstrate superior novelty and significance over ablated configurations representative of existing agentic architectures. Designed for flexibility and modularity, BioDisco allows seamless integration of custom language models or knowledge graphs, and can be run with just a few lines of code. We anticipate researchers using this practical tool as a catalyst for the discovery of new hypotheses. 

**Abstract (ZH)**: 基于语言模型的多代理框架BioDisco：发现新颖且证据支持的假说 

---
# Multi-TW: Benchmarking Multimodal Models on Traditional Chinese Question Answering in Taiwan 

**Title (ZH)**: 多模态模型在台湾传统中文问答任务上的基准测试 

**Authors**: Jui-Ming Yao, Bing-Cheng Xie, Sheng-Wei Peng, Hao-Yuan Chen, He-Rong Zheng, Bing-Jia Tan, Peter Shaojui Wang, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01274)  

**Abstract**: Multimodal Large Language Models (MLLMs) process visual, acoustic, and textual inputs, addressing the limitations of single-modality LLMs. However, existing benchmarks often overlook tri-modal evaluation in Traditional Chinese and do not consider inference latency. To address this, we introduce Multi-TW, the first Traditional Chinese benchmark for evaluating the performance and latency of any-to-any multimodal models. Multi-TW includes 900 multiple-choice questions (image and text, audio and text pairs) sourced from official proficiency tests developed with the Steering Committee for the Test of Proficiency-Huayu (SC-TOP). We evaluated various any-to-any models and vision-language models (VLMs) with audio transcription. Our results show that closed-source models generally outperform open-source ones across modalities, although open-source models can perform well in audio tasks. End-to-end any-to-any pipelines offer clear latency advantages compared to VLMs using separate audio transcription. Multi-TW presents a comprehensive view of model capabilities and highlights the need for Traditional Chinese fine-tuning and efficient multimodal architectures. 

**Abstract (ZH)**: Multimodal Large Language Models (MLLMs) 处理视觉、声学和文本输入，解决单模态 LLMs 的局限性。然而，现有的基准在传统中文方面通常忽视了三模态评估，也没有考虑推理延迟。为解决这一问题，我们引入 Multi-TW，这是首个用于评估任意到任意的多模态模型性能和延迟的传统中文基准。Multi-TW 包含 900 个多选题（图像和文本、声学和文本配对），这些问题源自由 Test of Proficiency-Huayu 指导委员会（SC-TOP）开发的官方 proficiency 测试。我们评估了各种任意到任意模型和视觉语言模型（VLMs）的声学转录。结果显示，闭源模型在各模态中通常优于开源模型，尽管开源模型在声学任务中也能表现出色。端到端的任意到任意管道与使用单独声学转录的 VLMs 相比，在延迟方面具有明显的优势。Multi-TW 提供了模型能力的全面视角，并强调了传统中文微调和高效多模态架构的需求。 

---
# KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs 

**Title (ZH)**: KCR: 通过LLM中的推理解决长期上下文知识冲突 

**Authors**: Xianda Zheng, Zijian Huang, Meng-Fen Chiang, Michael J. Witbrock, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01273)  

**Abstract**: Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains. 

**Abstract (ZH)**: 知识冲突在多样化来源中常见，随着大规模语言模型（LLMs）的出现，冲突的频率有所增加。在处理多个上下文之间的冲突，即所谓的“跨上下文知识冲突”时，LLMs 经常会被冗长且矛盾的上下文所困惑。为应对这一挑战，我们提出了知识冲突推理（KCR）框架，以增强LLMs解决知识冲突的能力。KCR的核心思想是通过奖励LLMs选择并遵循逻辑一致性更强的上下文，以训练其建立正确的推理过程。具体而言，我们首先从矛盾的长上下文中提取由文本或局部知识图谱表示的推理路径，然后利用强化学习鼓励模型遵循正确的推理路径，而不是错误的路径。这使得基础模型能够真正获得在长上下文场景中解决跨上下文知识冲突的能力。实验结果表明，我们的框架显著提升了各种基础模型在长上下文场景中解决知识冲突的能力，取得了显著的性能提升。 

---
# Win-k: Improved Membership Inference Attacks on Small Language Models 

**Title (ZH)**: Win-k: 改进的小语言模型成员推断攻击 

**Authors**: Roya Arkhmammadova, Hosein Madadi Tamar, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01268)  

**Abstract**: Small language models (SLMs) are increasingly valued for their efficiency and deployability in resource-constrained environments, making them useful for on-device, privacy-sensitive, and edge computing applications. On the other hand, membership inference attacks (MIAs), which aim to determine whether a given sample was used in a model's training, are an important threat with serious privacy and intellectual property implications. In this paper, we study MIAs on SLMs. Although MIAs were shown to be effective on large language models (LLMs), they are relatively less studied on emerging SLMs, and furthermore, their effectiveness decreases as models get smaller. Motivated by this finding, we propose a new MIA called win-k, which builds on top of a state-of-the-art attack (min-k). We experimentally evaluate win-k by comparing it with five existing MIAs using three datasets and eight SLMs. Results show that win-k outperforms existing MIAs in terms of AUROC, TPR @ 1% FPR, and FPR @ 99% TPR metrics, especially on smaller models. 

**Abstract (ZH)**: 小语言模型上的成员推理攻击研究：一种基于min-k的新方法Win-k 

---
# Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models 

**Title (ZH)**: 统一专家混合模型与多头潜在注意力机制以构建高效的语言模型 

**Authors**: Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2508.01261)  

**Abstract**: We present MoE-MLA-RoPE, a novel architecture combination that combines Mixture of Experts (MoE) with Multi-head Latent Attention (MLA) and Rotary Position Embeddings (RoPE) for efficient language modeling. Our approach addresses the fundamental trade-off between model capacity and computational efficiency through three key innovations: (1) fine-grained expert routing with 64 micro-experts and top-$k$ selection, enabling flexible specialization through 3.6 * 10^7 possible expert combinations; (2) shared expert isolation that dedicates 2 always active experts for common patterns while routing to 6 of 62 specialized experts; and (3) gradient-conflict-free load balancing that maintains expert utilization without interfering with primary loss optimization.
Extensive experiments on models ranging from 17M to 202M parameters demonstrate that MoE-MLA-RoPE with compression ratio r=d/2 achieves 68% KV cache memory reduction and 3.2x inference speedup while maintaining competitive perplexity (0.8% degradation). Compared to the parameters with 53.9M parameters, MoE-MLA-RoPE improves the validation loss by 6.9% over the vanilla transformers while using 42% fewer active parameters per forward pass. FLOP-matched experiments reveal even larger gains: 11.1% improvement with 3.2x inference acceleration. Automated evaluation using GPT-4 as a judge confirms quality improvements in generation, with higher scores on coherence (8.1/10), creativity (7.9/10) and grammatical correctness (8.2/10). Our results establish that architectural novelty, not parameter scaling, defines the efficiency frontier for resource-constrained language model deployment. 

**Abstract (ZH)**: MoE-MLA-RoPE: 一种结合Mixture of Experts、Multi-head Latent Attention和Rotary Position Embeddings的新型架构组合 

---
# SketchAgent: Generating Structured Diagrams from Hand-Drawn Sketches 

**Title (ZH)**: SketchAgent：从手绘草图生成结构化图表 

**Authors**: Cheng Tan, Qi Chen, Jingxuan Wei, Gaowei Wu, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01237)  

**Abstract**: Hand-drawn sketches are a natural and efficient medium for capturing and conveying ideas. Despite significant advancements in controllable natural image generation, translating freehand sketches into structured, machine-readable diagrams remains a labor-intensive and predominantly manual task. The primary challenge stems from the inherent ambiguity of sketches, which lack the structural constraints and semantic precision required for automated diagram generation. To address this challenge, we introduce SketchAgent, a multi-agent system designed to automate the transformation of hand-drawn sketches into structured diagrams. SketchAgent integrates sketch recognition, symbolic reasoning, and iterative validation to produce semantically coherent and structurally accurate diagrams, significantly reducing the need for manual effort. To evaluate the effectiveness of our approach, we propose the Sketch2Diagram Benchmark, a comprehensive dataset and evaluation framework encompassing eight diverse diagram categories, such as flowcharts, directed graphs, and model architectures. The dataset comprises over 6,000 high-quality examples with token-level annotations, standardized preprocessing, and rigorous quality control. By streamlining the diagram generation process, SketchAgent holds great promise for applications in design, education, and engineering, while offering a significant step toward bridging the gap between intuitive sketching and machine-readable diagram generation. The benchmark is released at this https URL. 

**Abstract (ZH)**: 手绘草图是一种自然且高效的捕捉和传达想法的介质。尽管在可控自然图像生成方面取得了显著进展，但将自由手绘草图转化为结构化、机器可读的图表仍然是一个劳动密集且主要依赖手动的任务。主要挑战源于草图固有的含糊性，缺乏自动图表生成所需的结构约束和语义精度。为应对这一挑战，我们提出了SketchAgent，一个旨在自动化手绘草图转化为结构化图表的多代理系统。SketchAgent结合了草图识别、符号推理和迭代验证，生成语义连贯且结构准确的图表，大大减少了人工努力的需求。为了评估我们方法的有效性，我们提出了Sketch2Diagram基准，一个涵盖八种不同图表类别（如流程图、有向图和模型架构）的综合数据集和评估框架。该数据集包含超过6,000个高质量样本，具有标记级别的注解、标准化预处理和严格的质量控制。通过简化图表生成过程，SketchAgent在设计、教育和工程等领域具有巨大应用潜力，同时朝着直观草图绘制与机器可读图表生成之间的差距迈进。基准数据集可从此链接获取：this https URL。 

---
# Calibrated Prediction Set in Fault Detection with Risk Guarantees via Significance Tests 

**Title (ZH)**: 故障检测中具有风险保证的校准预测集通过显著性检验 

**Authors**: Mingchen Mei, Yi Li, YiYao Qian, Zijun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.01208)  

**Abstract**: Fault detection is crucial for ensuring the safety and reliability of modern industrial systems. However, a significant scientific challenge is the lack of rigorous risk control and reliable uncertainty quantification in existing diagnostic models, particularly when facing complex scenarios such as distributional shifts. To address this issue, this paper proposes a novel fault detection method that integrates significance testing with the conformal prediction framework to provide formal risk guarantees. The method transforms fault detection into a hypothesis testing task by defining a nonconformity measure based on model residuals. It then leverages a calibration dataset to compute p-values for new samples, which are used to construct prediction sets mathematically guaranteed to contain the true label with a user-specified probability, $1-\alpha$. Fault classification is subsequently performed by analyzing the intersection of the constructed prediction set with predefined normal and fault label sets. Experimental results on cross-domain fault diagnosis tasks validate the theoretical properties of our approach. The proposed method consistently achieves an empirical coverage rate at or above the nominal level ($1-\alpha$), demonstrating robustness even when the underlying point-prediction models perform poorly. Furthermore, the results reveal a controllable trade-off between the user-defined risk level ($\alpha$) and efficiency, where higher risk tolerance leads to smaller average prediction set sizes. This research contributes a theoretically grounded framework for fault detection that enables explicit risk control, enhancing the trustworthiness of diagnostic systems in safety-critical applications and advancing the field from simple point predictions to informative, uncertainty-aware outputs. 

**Abstract (ZH)**: 基于显著性检验与同态预测框架的故障检测方法研究 

---
# Importance Sampling is All You Need: Predict LLM's performance on new benchmark by reusing existing benchmark 

**Title (ZH)**: 重要性采样足矣：通过复用现有基准预测LLM在新基准上的性能 

**Authors**: Junjie Shi, Wei Ma, Shi Ying, Lingxiao Jiang, Yang liu, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01203)  

**Abstract**: With the rapid advancement of large language models , code generation has become a key benchmark for evaluating LLM capabilities. However, existing benchmarks face two major challenges: (1) the escalating cost of constructing high-quality test suites and reference solutions, and (2) the increasing risk of data contamination, which undermines the reliability of benchmark-based evaluations. In this paper, we propose BIS, a prompt-centric evaluation framework that enables ground-truth-free prediction of LLM performance on code generation tasks. Rather than executing generated code, BIS estimates performance metrics by analyzing the prompt distribution alone. Built on importance sampling theory and implemented using Importance Weighted Autoencoders, our method reweights samples from existing annotated benchmarks to estimate performance on new, unseen benchmarks. To stabilize the estimation, we introduce weight truncation strategies and compute marginal expectations across the fitted distributions. BIS serves as a complementary tool that supports benchmark development and validation under constrained resources, offering actionable and quick feedback for prompt selection and contamination assessment. We conduct extensive experiments involving 8,000 evaluation points across 4 CodeLlama models and 9 diverse benchmarks. Our framework achieves an average absolute prediction error of 1.1% for code correctness scores, with best- and worst-case errors of 0.3% and 1.9%, respectively. It also generalizes well to other metrics, attaining average absolute errors of 2.15% for pass@1. These results demonstrate the reliability and broad applicability of BIS, which can significantly reduce the cost and effort of benchmarking LLMs in code-related tasks. 

**Abstract (ZH)**: 基于提示的代码生成评估框架：BIS 

---
# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens 

**Title (ZH)**: LLMs的链式思考推理是否仅为幻象？一种数据分布视角 

**Authors**: Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01191)  

**Abstract**: Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning. 

**Abstract (ZH)**: Chain-of-Thought 推理通过数据分布视角研究：其有效性受训练数据与测试查询分布差异的限制 

---
# A Survey on Agent Workflow -- Status and Future 

**Title (ZH)**: 智能体工作流综述——现状与未来 

**Authors**: Chaojia Yu, Zihan Cheng, Hanwen Cui, Yishuo Gao, Zexu Luo, Yijin Wang, Hangbin Zheng, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01186)  

**Abstract**: In the age of large language models (LLMs), autonomous agents have emerged as a powerful paradigm for achieving general intelligence. These agents dynamically leverage tools, memory, and reasoning capabilities to accomplish user-defined goals. As agent systems grow in complexity, agent workflows-structured orchestration frameworks-have become central to enabling scalable, controllable, and secure AI behaviors. This survey provides a comprehensive review of agent workflow systems, spanning academic frameworks and industrial implementations. We classify existing systems along two key dimensions: functional capabilities (e.g., planning, multi-agent collaboration, external API integration) and architectural features (e.g., agent roles, orchestration flows, specification languages). By comparing over 20 representative systems, we highlight common patterns, potential technical challenges, and emerging trends. We further address concerns related to workflow optimization strategies and security. Finally, we outline open problems such as standardization and multimodal integration, offering insights for future research at the intersection of agent design, workflow infrastructure, and safe automation. 

**Abstract (ZH)**: 在大规模语言模型时代，自主代理已成为实现通用智能的强大范式。这些代理动态利用工具、记忆和推理能力以实现用户定义的目标。随着代理系统的复杂性增加，代理工作流——结构化的编排框架——已成为实现可扩展、可控和安全的AI行为的关键。本文综述了代理工作流系统，涵盖了学术框架和工业实现。我们根据两个关键维度对现有系统进行分类：功能能力（如规划、多代理协作、外部API集成）和架构特性（如代理角色、编排流程、规范语言）。通过比较超过20个代表性系统，我们突显了共同模式、潜在的技术挑战以及新兴趋势。我们进一步探讨了工作流优化策略和安全性的相关问题。最后，我们概述了标准化和多模态集成等开放问题，为代理设计、工作流基础设施和安全自动化交叉领域的未来研究提供了见解。 

---
# Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning 

**Title (ZH)**: 多模态情感推理中情感冲突的基准测试与弥合 

**Authors**: Zhiyuan Han, Beier Zhu, Yanlong Xu, Peipei Song, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01181)  

**Abstract**: Despite their strong performance in multimodal emotion reasoning, existing Multimodal Large Language Models (MLLMs) often overlook the scenarios involving emotion conflicts, where emotional cues from different modalities are inconsistent. To fill this gap, we first introduce CA-MER, a new benchmark designed to examine MLLMs under realistic emotion conflicts. It consists of three subsets: video-aligned, audio-aligned, and consistent, where only one or all modalities reflect the true emotion. However, evaluations on our CA-MER reveal that current state-of-the-art emotion MLLMs systematically over-rely on audio signal during emotion conflicts, neglecting critical cues from visual modality. To mitigate this bias, we propose MoSEAR, a parameter-efficient framework that promotes balanced modality integration. MoSEAR consists of two modules: (1)MoSE, modality-specific experts with a regularized gating mechanism that reduces modality bias in the fine-tuning heads; and (2)AR, an attention reallocation mechanism that rebalances modality contributions in frozen backbones during inference. Our framework offers two key advantages: it mitigates emotion conflicts and improves performance on consistent samples-without incurring a trade-off between audio and visual modalities. Experiments on multiple benchmarks-including MER2023, EMER, DFEW, and our CA-MER-demonstrate that MoSEAR achieves state-of-the-art performance, particularly under modality conflict conditions. 

**Abstract (ZH)**: 尽管现有的多模态大型语言模型在多模态情感推理方面表现出色，但它们 often 忽略涉及情感冲突的场景，在这些场景中，不同模态的情感线索相互矛盾。为弥补这一不足，我们首先引入了 CA-MER，一种新的基准测试，用于在现实情感冲突中评估多模态大型语言模型。CA-MER 包含三个子集：视频对齐、音频对齐和一致子集，在这些子集中，只有单一模态或所有模态反映真实情感。然而，我们的 CA-MER 评估结果显示，现有的情感多模态大型语言模型系统地在情感冲突中过度依赖音频信号，忽视了视觉模态的关键线索。为缓解这种偏见，我们提出 MoSEAR，一种参数高效框架，促进模态平衡集成。MoSEAR 包含两个模块：(1)MoSE，模态特定专家，带有正则化门控机制，减少微调头中的模态偏见；(2)AR，注意力重分配机制，在推理过程中重新平衡冻结骨干中的模态贡献。该框架提供两大优势：缓解情感冲突，改善一致样本的性能，而不会在音频和视觉模态之间产生权衡。在 MER2023、EMER、DFEW 和我们的 CA-MER 等多个基准上的实验表明，MoSEAR 在情感模态冲突条件下实现了最先进的性能。 

---
# H2C: Hippocampal Circuit-inspired Continual Learning for Lifelong Trajectory Prediction in Autonomous Driving 

**Title (ZH)**: H2C：基于海马神经回路的终身轨迹预测持续学习方法 

**Authors**: Yunlong Lin, Zirui Li, Guodong Du, Xiaocong Zhao, Cheng Gong, Xinwei Wang, Chao Lu, Jianwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01158)  

**Abstract**: Deep learning (DL) has shown state-of-the-art performance in trajectory prediction, which is critical to safe navigation in autonomous driving (AD). However, most DL-based methods suffer from catastrophic forgetting, where adapting to a new distribution may cause significant performance degradation in previously learned ones. Such inability to retain learned knowledge limits their applicability in the real world, where AD systems need to operate across varying scenarios with dynamic distributions. As revealed by neuroscience, the hippocampal circuit plays a crucial role in memory replay, effectively reconstructing learned knowledge based on limited resources. Inspired by this, we propose a hippocampal circuit-inspired continual learning method (H2C) for trajectory prediction across varying scenarios. H2C retains prior knowledge by selectively recalling a small subset of learned samples. First, two complementary strategies are developed to select the subset to represent learned knowledge. Specifically, one strategy maximizes inter-sample diversity to represent the distinctive knowledge, and the other estimates the overall knowledge by equiprobable sampling. Then, H2C updates via a memory replay loss function calculated by these selected samples to retain knowledge while learning new data. Experiments based on various scenarios from the INTERACTION dataset are designed to evaluate H2C. Experimental results show that H2C reduces catastrophic forgetting of DL baselines by 22.71% on average in a task-free manner, without relying on manually informed distributional shifts. The implementation is available at this https URL. 

**Abstract (ZH)**: 基于海马电路的持续学习方法（H2C）用于跨场景的轨迹预测 

---
# Platonic Representations for Poverty Mapping: Unified Vision-Language Codes or Agent-Induced Novelty? 

**Title (ZH)**: 柏拉图式的贫困地图表示：统一的视觉-语言编码还是代理引发的 novelty？ 

**Authors**: Satiyabooshan Murugaboopathy, Connor T. Jerzak, Adel Daoud  

**Link**: [PDF](https://arxiv.org/pdf/2508.01109)  

**Abstract**: We investigate whether socio-economic indicators like household wealth leave recoverable imprints in satellite imagery (capturing physical features) and Internet-sourced text (reflecting historical/economic narratives). Using Demographic and Health Survey (DHS) data from African neighborhoods, we pair Landsat images with LLM-generated textual descriptions conditioned on location/year and text retrieved by an AI search agent from web sources. We develop a multimodal framework predicting household wealth (International Wealth Index) through five pipelines: (i) vision model on satellite images, (ii) LLM using only location/year, (iii) AI agent searching/synthesizing web text, (iv) joint image-text encoder, (v) ensemble of all signals. Our framework yields three contributions. First, fusing vision and agent/LLM text outperforms vision-only baselines in wealth prediction (e.g., R-squared of 0.77 vs. 0.63 on out-of-sample splits), with LLM-internal knowledge proving more effective than agent-retrieved text, improving robustness to out-of-country and out-of-time generalization. Second, we find partial representational convergence: fused embeddings from vision/language modalities correlate moderately (median cosine similarity of 0.60 after alignment), suggesting a shared latent code of material well-being while retaining complementary details, consistent with the Platonic Representation Hypothesis. Although LLM-only text outperforms agent-retrieved data, challenging our Agent-Induced Novelty Hypothesis, modest gains from combining agent data in some splits weakly support the notion that agent-gathered information introduces unique representational structures not fully captured by static LLM knowledge. Third, we release a large-scale multimodal dataset comprising more than 60,000 DHS clusters linked to satellite images, LLM-generated descriptions, and agent-retrieved texts. 

**Abstract (ZH)**: 我们探究了诸如家庭财富等社会经济指标是否在卫星imagery（反映物理特征）和互联网来源的文字（反映历史/经济叙事）中留下可恢复的痕迹。使用非洲社区的Demographic and Health Survey (DHS)数据，我们将Landsat图像与基于地理位置/年份的LLM生成的文本描述配对，并通过AI搜索代理从网络来源检索文本。我们开发了一种多模态框架，通过五个管道预测国际财富指数：（i）卫星图像的视觉模型，（ii）仅使用地理位置/年份的LLM，（iii）AI代理搜索/合成网络文本，（iv）联合图像-文本编码器，（v）所有信号的集成。该框架有三大贡献。首先，结合视觉和代理/LLM文本在财富预测方面优于仅视觉基线（例如，在离样本外分割上的决定系数R-squared为0.77，而仅视觉基线为0.63），且LLM内部的知识比代理检索的文本更有效，提高了跨国家和地区的一般化鲁棒性。其次，我们发现了部分表征收敛：融合视觉/语言模态的嵌入在对齐后的相关性适中（中位数余弦相似度为0.60），这表明存在共享的潜在代码，描述物质福祉，同时保留互补细节，这与理念相符。尽管仅由LLM生成的文本优于检索的代理数据，这挑战了我们的代理诱导新颖性假设，但在某些分割中将代理数据结合的微小增益部分支持了代理收集的信息引入独特表征结构的假设，这种结构未被静态LLM知识完全捕捉。第三，我们发布了一个大规模多模态数据集，包括超过60,000个与卫星图像、LLM生成描述和代理检索文本相关联的DHS集群。 

---
# Multispin Physics of AI Tipping Points and Hallucinations 

**Title (ZH)**: 多自旋物理：AI临界点和幻觉 

**Authors**: Neil F. Johnson, Frank Yingjie Huo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01097)  

**Abstract**: Output from generative AI such as ChatGPT, can be repetitive and biased. But more worrying is that this output can mysteriously tip mid-response from good (correct) to bad (misleading or wrong) without the user noticing. In 2024 alone, this reportedly caused $67 billion in losses and several deaths. Establishing a mathematical mapping to a multispin thermal system, we reveal a hidden tipping instability at the scale of the AI's 'atom' (basic Attention head). We derive a simple but essentially exact formula for this tipping point which shows directly the impact of a user's prompt choice and the AI's training bias. We then show how the output tipping can get amplified by the AI's multilayer architecture. As well as helping improve AI transparency, explainability and performance, our results open a path to quantifying users' AI risk and legal liabilities. 

**Abstract (ZH)**: 生成式AI如ChatGPT的输出可能存在重复和偏见问题，更令人担忧的是，这种输出可能会神秘地在用户不知情的情况下从良好（正确）转变为不良（误导或错误）。2024年，这 reportedly 导致了670亿美元的损失和数人死亡。通过建立数学映射到多自旋热系统，我们揭示了在AI的“原子”（基本注意力头）尺度上存在一个隐藏的翻转不稳定性。我们推导出一个简单但基本上精确的公式来描述这一翻转点，它直接展示了用户提示选择和AI训练偏见的影响。我们还展示了AI多层架构如何放大输出翻转。除了提高AI的透明度、可解释性和性能，我们的结果还开辟了一条量化用户AI风险和法律责任的途径。 

---
# gpuRDF2vec -- Scalable GPU-based RDF2vec 

**Title (ZH)**: gpuRDF2vec——基于GPU的可扩展RDF2vec 

**Authors**: Martin Böckling, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01073)  

**Abstract**: Generating Knowledge Graph (KG) embeddings at web scale remains challenging. Among existing techniques, RDF2vec combines effectiveness with strong scalability. We present gpuRDF2vec, an open source library that harnesses modern GPUs and supports multi-node execution to accelerate every stage of the RDF2vec pipeline. Extensive experiments on both synthetically generated graphs and real-world benchmarks show that gpuRDF2vec achieves up to a substantial speedup over the currently fastest alternative, i.e., jRDF2vec. In a single-node setup, our walk-extraction phase alone outperforms pyRDF2vec, SparkKGML, and jRDF2vec by a substantial margin using random walks on large/ dense graphs, and scales very well to longer walks, which typically lead to better quality embeddings. Our implementation of gpuRDF2vec enables practitioners and researchers to train high-quality KG embeddings on large-scale graphs within practical time budgets and builds on top of Pytorch Lightning for the scalable word2vec implementation. 

**Abstract (ZH)**: 在Web规模下生成知识图谱（KG）嵌入仍然具有挑战性。我们介绍了gpuRDF2vec，这是一个开源库，利用现代GPU并支持多节点执行以加速RDF2vec管道中的每个阶段。在合成生成的图和真实基准上的广泛实验表明，gpuRDF2vec相比于当前最快的替代方案jRDF2vec实现了显著的速度提升。在单节点设置中，仅抽取走相比较pyRDF2vec、SparkKGML和jRDF2vec在大规模/密集图上使用随机走的方法取得了显著的优势，并且能够很好地扩展到更长的走，通常会导致更好的质量嵌入。我们对gpuRDF2vec的实现使从业者和研究者能够在实际的时间预算内训练大规模图上的高质量KG嵌入，并基于Pytorch Lightning构建以实现可扩展的word2vec实现。 

---
# REACT: A Real-Time Edge-AI Based V2X Framework for Accident Avoidance in Autonomous Driving System 

**Title (ZH)**: REACT：基于边缘AI的实时车联网事故避免框架 

**Authors**: Fengze Yang, Bo Yu, Yang Zhou, Xuewen Luo, Zhengzhong Tu, Chenxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01057)  

**Abstract**: Collisions caused by human error are the most common type of multi-vehicle crash, highlighting the critical need for autonomous driving (AD) systems to leverage cooperative perception through Vehicle-to-Everything (V2X) communication. This capability extends situational awareness beyond the limitations of onboard sensors. However, current transformer-based V2X frameworks suffer from limited generalization, shallow contextual reasoning, and reliance on mono-modal inputs. Vision-Language Models (VLMs) offer enhanced reasoning and multimodal integration but typically fall short of real-time performance requirements in safety-critical applications. This paper presents REACT, a real-time, V2X-integrated trajectory optimization framework built upon a fine-tuned lightweight VLM. REACT integrates a set of specialized modules that process multimodal inputs into optimized, risk-aware trajectories. To ensure real-time performance on edge devices, REACT incorporates edge adaptation strategies that reduce model complexity and accelerate inference. Evaluated on the DeepAccident benchmark, REACT achieves state-of-the-art performance, a 77% collision rate reduction, a 48.2% Video Panoptic Quality (VPQ), and a 0.57-second inference latency on the Jetson AGX Orin. Ablation studies validate the contribution of each input, module, and edge adaptation strategy. These results demonstrate the feasibility of lightweight VLMs for real-time edge-based cooperative planning and showcase the potential of language-guided contextual reasoning to improve safety and responsiveness in autonomous driving. 

**Abstract (ZH)**: 由人为错误引起的碰撞是多车事故中最常见的一类，突显了自动驾驶系统利用车辆到万物（V2X）通信进行协同感知的至关重要的需求。这一能力能够超越车载传感器的局限性，扩展情境感知。然而，当前基于变换器的V2X框架存在泛化能力有限、浅层上下文推理以及单模态输入依赖等问题。视觉-语言模型（VLMs）提供了增强的推理和多模态集成能力，但在关键安全应用中通常无法满足实时性能要求。本文提出了REACT，这是一个基于微调的轻量级VLM构建的实时、V2X集成轨迹优化框架，REACT集成了多个专门模块，将多模态输入转化为优化的风险感知轨迹。为了确保边缘设备上的实时性能，REACT融入了边缘自适应策略，减少模型复杂性和加速推理。在DeepAccident基准上进行评估，REACT实现了最先进的性能，碰撞率降低了77%，视频全景质量（VPQ）提高了48.2%，在Jetson AGX Orin上的推理延迟为0.57秒。消融研究验证了每种输入、模块和边缘自适应策略的贡献。这些结果展示了轻量级VLMs在实时边缘基于协同规划中的可行性，并突显了语言引导的上下文推理在提高自动驾驶的安全性和响应性方面的作用。 

---
# CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent 

**Title (ZH)**: CADDesigner: 基于通用型代理的概念设计CAD模型 

**Authors**: Jingzhe Ni, Xiaolong Yin, Xintong Li, Xingyu Lu, Ji Wei, Ruofeng Tong, Min Tang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01031)  

**Abstract**: Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both abstract textual descriptions and freehand sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Context-Independent Imperative Paradigm (CIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation. 

**Abstract (ZH)**: 基于大型语言模型的CAD概念设计代理 

---
# AutoEDA: Enabling EDA Flow Automation through Microservice-Based LLM Agents 

**Title (ZH)**: AutoEDA：基于微服务的LLM代理实现EDA流程自动化 

**Authors**: Yiyi Lu, Hoi Ian Au, Junyao Zhang, Jingyu Pan, Yiting Wang, Ang Li, Jianyi Zhang, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01012)  

**Abstract**: Modern Electronic Design Automation (EDA) workflows, especially the RTL-to-GDSII flow, require heavily manual scripting and demonstrate a multitude of tool-specific interactions which limits scalability and efficiency. While LLMs introduces strides for automation, existing LLM solutions require expensive fine-tuning and do not contain standardized frameworks for integration and evaluation. We introduce AutoEDA, a framework for EDA automation that leverages paralleled learning through the Model Context Protocol (MCP) specific for standardized and scalable natural language experience across the entire RTL-to-GDSII flow. AutoEDA limits fine-tuning through structured prompt engineering, implements intelligent parameter extraction and task decomposition, and provides an extended CodeBLEU metric to evaluate the quality of TCL scripts. Results from experiments over five previously curated benchmarks show improvements in automation accuracy and efficiency, as well as script quality when compared to existing methods. AutoEDA is released open-sourced to support reproducibility and the EDA community. Available at: this https URL 

**Abstract (ZH)**: AutoEDA：一种通过并行学习实现标准化和可扩展的EDA自动化框架 

---
# Cooperative Perception: A Resource-Efficient Framework for Multi-Drone 3D Scene Reconstruction Using Federated Diffusion and NeRF 

**Title (ZH)**: 协同感知：一种基于联邦扩散和NeRF的多无人机三维场景重建高效框架 

**Authors**: Massoud Pourmandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00967)  

**Abstract**: The proposal introduces an innovative drone swarm perception system that aims to solve problems related to computational limitations and low-bandwidth communication, and real-time scene reconstruction. The framework enables efficient multi-agent 3D/4D scene synthesis through federated learning of shared diffusion model and YOLOv12 lightweight semantic extraction and local NeRF updates while maintaining privacy and scalability. The framework redesigns generative diffusion models for joint scene reconstruction, and improves cooperative scene understanding, while adding semantic-aware compression protocols. The approach can be validated through simulations and potential real-world deployment on drone testbeds, positioning it as a disruptive advancement in multi-agent AI for autonomous systems. 

**Abstract (ZH)**: 一种基于联邦学习的无人机集群感知系统：面向计算限制、低宽带通信和实时场景重建的联合场景重建与协同场景理解 

---
# Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis 

**Title (ZH)**: 使用语义分析进行多跳问答的知识编辑 

**Authors**: Dominic Simon, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.00914)  

**Abstract**: Large Language Models (LLMs) require lightweight avenues of updating stored information that has fallen out of date. Knowledge Editing (KE) approaches have been successful in updating model knowledge for simple factual queries but struggle with handling tasks that require compositional reasoning such as multi-hop question answering (MQA). We observe that existing knowledge editors leverage decompositional techniques that result in illogical reasoning processes. In this paper, we propose a knowledge editor for MQA based on semantic analysis called CHECK. Our framework is based on insights from an analogy between compilers and reasoning using LLMs. Similar to how source code is first compiled before being executed, we propose to semantically analyze reasoning chains before executing the chains to answer questions. Reasoning chains with semantic errors are revised to ensure consistency through logic optimization and re-prompting the LLM model at a higher temperature. We evaluate the effectiveness of CHECK against five state-of-the-art frameworks on four datasets and achieve an average 22.8% improved MQA accuracy. 

**Abstract (ZH)**: 大型语言模型需要轻量级的途径来更新过时的存储信息。基于语义分析的知识编辑方法CHECK用于多跳问答任务，解决现有知识编辑方法在处理需要组成性推理的任务时遇到的困难。我们提出的框架借鉴了编译器与使用大型语言模型进行推理之间的类比。类似于源代码首先会被编译然后再执行，我们提出在执行推理链之前对其执行语义分析。具有语义错误的推理链通过逻辑优化和以较高温度重新 prompting 大型语言模型来修正，以确保一致性。我们在四个数据集上将 CHECK 与五个最先进的框架进行对比评估，并取得平均 22.8% 的多跳问答准确性提升。 

---
# An analysis of AI Decision under Risk: Prospect theory emerges in Large Language Models 

**Title (ZH)**: AI决策下的风险分析：前景理论在大型语言模型中 Emerges 

**Authors**: Kenneth Payne  

**Link**: [PDF](https://arxiv.org/pdf/2508.00902)  

**Abstract**: Judgment of risk is key to decision-making under uncertainty. As Daniel Kahneman and Amos Tversky famously discovered, humans do so in a distinctive way that departs from mathematical rationalism. Specifically, they demonstrated experimentally that humans accept more risk when they feel themselves at risk of losing something than when they might gain. I report the first tests of Kahneman and Tversky's landmark 'prospect theory' with Large Language Models, including today's state of the art chain-of-thought 'reasoners'.
In common with humans, I find that prospect theory often anticipates how these models approach risky decisions across a range of scenarios. I also demonstrate that context is key to explaining much of the variance in risk appetite. The 'frame' through which risk is apprehended appears to be embedded within the language of the scenarios tackled by the models. Specifically, I find that military scenarios generate far larger 'framing effects' than do civilian settings, ceteris paribus. My research suggests, therefore, that language models the world, capturing our human heuristics and biases. But also that these biases are uneven - the idea of a 'frame' is richer than simple gains and losses. Wittgenstein's notion of 'language games' explains the contingent, localised biases activated by these scenarios. Finally, I use my findings to reframe the ongoing debate about reasoning and memorisation in LLMs. 

**Abstract (ZH)**: 风险判断是不确定性环境下决策的关键。丹尼尔·卡内曼和阿莫斯·特维斯基 famously 发现，人类在进行风险判断时以一种与数学理性主义相悖的方式。具体而言，他们在实验中证明，人们在害怕损失时愿意承担更大的风险，而在可能获利时则更保守。我利用大型语言模型首次检验了卡内曼和特维斯基的里程碑式“前景理论”，包括当今最先进的链式思考“推理器”。与人类相似，我发现前景理论经常预测这些模型在各种情境下的风险管理方式。我还展示了上下文是解释模型风险偏好差异的关键。风险通过模型处理的场景语言框架来理解。特别地，我发现在其他条件相同的情况下，军事场景产生的“框架效应”比民用环境大得多。我的研究建议，语言模型模仿了人类的直觉和偏差，但这些偏差是不均衡的——“框架”的概念比简单的得失更为丰富。维特根斯坦的“语言游戏”概念解释了这些情境激活的偶然性、地方性偏差。最后，我利用研究结果重新审视关于LLMs推理和记忆的持续争论。 

---
# ff4ERA: A new Fuzzy Framework for Ethical Risk Assessment in AI 

**Title (ZH)**: ff4ERA：一种新的模糊框架用于AI伦理风险评估 

**Authors**: Abeer Dyoub, Ivan Letteri, Francesca A. Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00899)  

**Abstract**: The emergence of Symbiotic AI (SAI) introduces new challenges to ethical decision-making as it deepens human-AI collaboration. As symbiosis grows, AI systems pose greater ethical risks, including harm to human rights and trust. Ethical Risk Assessment (ERA) thus becomes crucial for guiding decisions that minimize such risks. However, ERA is hindered by uncertainty, vagueness, and incomplete information, and morality itself is context-dependent and imprecise. This motivates the need for a flexible, transparent, yet robust framework for ERA. Our work supports ethical decision-making by quantitatively assessing and prioritizing multiple ethical risks so that artificial agents can select actions aligned with human values and acceptable risk levels. We introduce ff4ERA, a fuzzy framework that integrates Fuzzy Logic, the Fuzzy Analytic Hierarchy Process (FAHP), and Certainty Factors (CF) to quantify ethical risks via an Ethical Risk Score (ERS) for each risk type. The final ERS combines the FAHP-derived weight, propagated CF, and risk level. The framework offers a robust mathematical approach for collaborative ERA modeling and systematic, step-by-step analysis. A case study confirms that ff4ERA yields context-sensitive, ethically meaningful risk scores reflecting both expert input and sensor-based evidence. Risk scores vary consistently with relevant factors while remaining robust to unrelated inputs. Local sensitivity analysis shows predictable, mostly monotonic behavior across perturbations, and global Sobol analysis highlights the dominant influence of expert-defined weights and certainty factors, validating the model design. Overall, the results demonstrate ff4ERA ability to produce interpretable, traceable, and risk-aware ethical assessments, enabling what-if analyses and guiding designers in calibrating membership functions and expert judgments for reliable ethical decision support. 

**Abstract (ZH)**: 共生人工智能(SAI)的兴起为伦理决策提出了新的挑战，随着人机协作的加深，AI系统带来的伦理风险增加，包括对人权和信任的危害。因此，伦理风险评估(ERA)变得至关重要，以指导减少这些风险的决策。然而，ERA受制于不确定性、模糊性和信息不完整，而道德本身也是情境依赖和不精确的。这促使我们需要一个灵活、透明且稳健的ERA框架。我们的工作通过定量评估和优先级排序多种伦理风险，支持AI代理选择与人类价值观和可接受风险水平一致的行为。我们引入了ff4ERA，这是一个结合了模糊逻辑、模糊分析层次过程(FAHP)和确信因子(CF)的框架，用于通过伦理风险评分(ERS)量化每种风险类型的风险。最终的ERS结合了FAHP得出的权重、传播的CF和风险水平。该框架提供了协作ERA建模和系统化、按步骤分析的稳健数学方法。案例研究证实，ff4ERA能够生成情境敏感、合乎伦理意义的风险评分，既能反映专家输入，又能反映传感器数据。风险评分与相关因素一致变化，对不相关输入保持稳健。局部灵敏度分析显示在扰动下可预测的、大体单调的行为，并且全局Sobol分析突出了专家定义的权重和确信因子的主导影响，验证了模型设计。总体而言，这些结果表明ff4ERA能够生成可解释、可追溯且风险意识强的伦理评估，支持假设场景分析，并指导设计者校准隶属函数和专家判断，以提供可靠的伦理决策支持。 

---
# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks 

**Title (ZH)**: AgentTTS: 大型语言模型代理用于复杂任务测试时计算 optimal 缩放策略 

**Authors**: Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00890)  

**Abstract**: Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability. 

**Abstract (ZH)**: 测试时计算优化缩放（TTS）通过推理时分配额外计算资源来提升大型语言模型（LLM）的表现，但在多阶段复杂任务中，现有研究主要关注单阶段任务；而许多现实世界问题是由一系列异构子任务组成，每个子任务需要特定能力的LLM。因此，我们研究了一个新的问题：多阶段复杂任务中的测试时计算最优缩放，旨在选择合适的模型并为每个子任务分配预算以最大化整体性能。多阶段任务中的TTS引入了两个基本挑战：（i）模型和预算分配的组合搜索空间与高推理成本使得穷举搜索不切实际。（ii）跨子任务的最优模型和预算分配相互依存，增加了计算最优搜索的复杂性。为解决这一差距，我们跨六个数据集在四项任务上进行了广泛的试点实验，得出了三个经验洞见，描述了大型语言模型在多阶段复杂任务中的行为。基于这些洞见，我们提出了基于大型语言模型代理的AgentTTS框架，通过迭代的反馈驱动交互自主搜索计算最优分配。实验结果表明，AgentTTS在搜索效率上显著优于传统和其他基于大型语言模型的方法，并且对不同的训练集大小具有更好的鲁棒性，且具有增强的可解释性。 

---
# A Formal Framework for the Definition of 'State': Hierarchical Representation and Meta-Universe Interpretation 

**Title (ZH)**: 形式化的“状态”定义框架：层级表示与元宇宙解释 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00853)  

**Abstract**: This study aims to reinforce the theoretical foundation for diverse systems--including the axiomatic definition of intelligence--by introducing a mathematically rigorous and unified formal structure for the concept of 'state,' which has long been used without consensus or formal clarity. First, a 'hierarchical state grid' composed of two axes--state depth and mapping hierarchy--is proposed to provide a unified notational system applicable across mathematical, physical, and linguistic domains. Next, the 'Intermediate Meta-Universe (IMU)' is introduced to enable explicit descriptions of definers (ourselves) and the languages we use, thereby allowing conscious meta-level operations while avoiding self-reference and logical inconsistency. Building on this meta-theoretical foundation, this study expands inter-universal theory beyond mathematics to include linguistic translation and agent integration, introducing the conceptual division between macrocosm-inter-universal and microcosm-inter-universal operations for broader expressivity. Through these contributions, this paper presents a meta-formal logical framework--grounded in the principle of definition = state--that spans time, language, agents, and operations, providing a mathematically robust foundation applicable to the definition of intelligence, formal logic, and scientific theory at large. 

**Abstract (ZH)**: 本研究旨在通过引入严格且统一的形式结构来强化多样系统的基础理论——包括智能的公理定义——该结构为长期缺乏共识和形式清晰的概念“状态”提供支持。首先，提出了一种由状态深度和映射层次构成的“层级状态格”，以提供适用于数学、物理和语言领域的一致符号系统。接着，引入“中介元宇宙（IMU）”，以明确描述定义者及其使用的语言，从而允许元层次操作并避免自我引用和逻辑不一致。在此元理论基础上，本研究将跨宇宙理论扩展至语言翻译和代理集成领域，引入宏观跨宇宙和微观跨宇宙操作的概念，以扩展表达能力。通过这些贡献，本文提出了一种根植于定义=状态原则的元形式逻辑框架，该框架涵盖了时间、语言、代理和操作，提供了一种适用于智能、形式逻辑和科学理论整体的数学稳健基础。 

---
# Exploring Agentic Artificial Intelligence Systems: Towards a Typological Framework 

**Title (ZH)**: 探索代理型人工智能系统：向类型学框架迈进 

**Authors**: Christopher Wissuchek, Patrick Zschech  

**Link**: [PDF](https://arxiv.org/pdf/2508.00844)  

**Abstract**: Artificial intelligence (AI) systems are evolving beyond passive tools into autonomous agents capable of reasoning, adapting, and acting with minimal human intervention. Despite their growing presence, a structured framework is lacking to classify and compare these systems. This paper develops a typology of agentic AI systems, introducing eight dimensions that define their cognitive and environmental agency in an ordinal structure. Using a multi-phase methodological approach, we construct and refine this typology, which is then evaluated through a human-AI hybrid approach and further distilled into constructed types. The framework enables researchers and practitioners to analyze varying levels of agency in AI systems. By offering a structured perspective on the progression of AI capabilities, the typology provides a foundation for assessing current systems and anticipating future developments in agentic AI. 

**Abstract (ZH)**: 人工智能（AI）系统正在从被动工具演变成能够推理、适应并在最少人类干预下行动的自主代理。尽管这些系统的影响力日益扩大，但缺乏一个结构化的框架来分类和比较它们。本文构建了自主代理AI系统的类型学，引入了八个维度来按序定义其认知和环境代理。通过多阶段的方法论方法，我们构建并优化了这一类型学，然后通过人类-AI混合方法评估，并进一步提炼成构建类型。该框架使研究人员和实践者能够分析AI系统中不同水平的代理性。通过为AI能力的发展提供结构化的视角，该类型学为评估当前系统并预见自主代理AI的未来发展方向提供了基础。 

---
# An Efficient Continuous-Time MILP for Integrated Aircraft Hangar Scheduling and Layout 

**Title (ZH)**: 一种高效的连续时间混合整数线性规划方法，用于飞机库集成调度与布局规划 

**Authors**: Shayan Farhang Pazhooh, Hossein Shams Shemirani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02640)  

**Abstract**: Efficient management of aircraft maintenance hangars is a critical operational challenge, involving complex, interdependent decisions regarding aircraft scheduling and spatial allocation. This paper introduces a novel continuous-time mixed-integer linear programming (MILP) model to solve this integrated spatio-temporal problem. By treating time as a continuous variable, our formulation overcomes the scalability limitations of traditional discrete-time approaches. The performance of the exact model is benchmarked against a constructive heuristic, and its practical applicability is demonstrated through a custom-built visualization dashboard. Computational results are compelling: the model solves instances with up to 25 aircraft to proven optimality, often in mere seconds, and for large-scale cases of up to 40 aircraft, delivers high-quality solutions within known optimality gaps. In all tested scenarios, the resulting solutions consistently and significantly outperform the heuristic, which highlights the framework's substantial economic benefits and provides valuable managerial insights into the trade-off between solution time and optimality. 

**Abstract (ZH)**: 高效的机库管理是关键的操作挑战，涉及复杂的、相互依赖的飞机排程和空间分配决策。本文引入了一种新的连续时间混合整数线性规划（MILP）模型来解决这一集成的空间-时间问题。通过将时间视为连续变量，我们的建模方法克服了传统离散时间方法的可扩展性限制。精确模型的性能与构造性启发式方法进行了基准测试，并通过自建的可视化仪表板展示了其实用性。计算结果表明：该模型在几秒钟内即可解决多达25架飞机的实例，并且在多达40架飞机的大规模情况下，能够在已知的最优性差距内提供高质量的解决方案。在所有测试场景中，所得解始终且显著优于启发式方法，这突显了该框架的显著经济效益，并提供了有关解的时间与最优性之间权衡关系的宝贵管理见解。 

---
# HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents 

**Title (ZH)**: HyCodePolicy: 混合语言控制器在 embodied 代理的多模态监控与决策中的应用 

**Authors**: Yibin Liu, Zhixuan Liang, Zanxin Chen, Tianxing Chen, Mengkang Hu, Wanxi Dong, Congsheng Xu, Zhaoming Han, Yusen Qin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02629)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models for Hybrid Code Policy Generation in Embodied Agents 

---
# AutoML-Med: A Framework for Automated Machine Learning in Medical Tabular Data 

**Title (ZH)**: AutoML-Med：一种医疗表格数据的自动化机器学习框架 

**Authors**: Riccardo Francia, Maurizio Leone, Giorgio Leonardi, Stefania Montani, Marzio Pennisi, Manuel Striani, Sandra D'Alfonso  

**Link**: [PDF](https://arxiv.org/pdf/2508.02625)  

**Abstract**: Medical datasets are typically affected by issues such as missing values, class imbalance, a heterogeneous feature types, and a high number of features versus a relatively small number of samples, preventing machine learning models from obtaining proper results in classification and regression tasks. This paper introduces AutoML-Med, an Automated Machine Learning tool specifically designed to address these challenges, minimizing user intervention and identifying the optimal combination of preprocessing techniques and predictive models. AutoML-Med's architecture incorporates Latin Hypercube Sampling (LHS) for exploring preprocessing methods, trains models using selected metrics, and utilizes Partial Rank Correlation Coefficient (PRCC) for fine-tuned optimization of the most influential preprocessing steps. Experimental results demonstrate AutoML-Med's effectiveness in two different clinical settings, achieving higher balanced accuracy and sensitivity, which are crucial for identifying at-risk patients, compared to other state-of-the-art tools. AutoML-Med's ability to improve prediction results, especially in medical datasets with sparse data and class imbalance, highlights its potential to streamline Machine Learning applications in healthcare. 

**Abstract (ZH)**: 医学数据集通常受到缺失值、类别不平衡、异质特征类型以及特征数量远多于样本数量等问题的影响，这会阻碍机器学习模型在分类和回归任务中获得适当的结果。本文介绍了AutoML-Med，一种针对这些挑战而设计的自动化机器学习工具，旨在最小化用户干预并识别最优的预处理技术与预测模型组合。AutoML-Med的架构包括拉丁超立方采样（LHS）以探索预处理方法，使用选定的评估指标训练模型，并通过部分秩相关系数（PRCC）对影响最大的预处理步骤进行精细化优化。实验结果表明，与现有的先进工具相比，AutoML-Med 在两种不同的临床设置中更有效地提高了平衡准确率和敏感性，这对于识别高风险患者至关重要。AutoML-Med在处理稀疏数据和类别不平衡的医学数据集时的能力提升，突显了其在医疗健康领域机器学习应用中简化流程的潜力。 

---
# Meta-RAG on Large Codebases Using Code Summarization 

**Title (ZH)**: 在大规模代码库中使用代码总结的Meta-RAG 

**Authors**: Vali Tawosia, Salwa Alamir, Xiaomo Liu, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.02611)  

**Abstract**: Large Language Model (LLM) systems have been at the forefront of applied Artificial Intelligence (AI) research in a multitude of domains. One such domain is software development, where researchers have pushed the automation of a number of code tasks through LLM agents. Software development is a complex ecosystem, that stretches far beyond code implementation and well into the realm of code maintenance. In this paper, we propose a multi-agent system to localize bugs in large pre-existing codebases using information retrieval and LLMs. Our system introduces a novel Retrieval Augmented Generation (RAG) approach, Meta-RAG, where we utilize summaries to condense codebases by an average of 79.8\%, into a compact, structured, natural language representation. We then use an LLM agent to determine which parts of the codebase are critical for bug resolution, i.e. bug localization. We demonstrate the usefulness of Meta-RAG through evaluation with the SWE-bench Lite dataset. Meta-RAG scores 84.67 % and 53.0 % for file-level and function-level correct localization rates, respectively, achieving state-of-the-art performance. 

**Abstract (ZH)**: 大规模语言模型（LLM）系统在多个领域应用人工智能研究中处于前沿。其中，软件开发领域研究人员通过LLM代理自动化了大量编码任务。软件开发是一个复杂的生态系统，远远超出了代码实现的范围，延伸到了代码维护的领域。本文提出了一种基于信息检索和LLM的多代理系统，用于在大型现成代码库中定位错误。该系统引入了一种新颖的检索增强生成（RAG）方法，Meta-RAG，在此方法中，我们利用摘要将代码库平均压缩79.8%，形成一个紧凑、结构化的自然语言表示。然后使用LLM代理确定代码库中哪些部分对于错误定位至关重要。我们通过使用SWE-bench Lite数据集进行评估，展示了Meta-RAG的有效性。Meta-RAG在文件级和函数级正确定位率分别为84.67%和53.0%，达到了最先进的性能。 

---
# Entity Representation Learning Through Onsite-Offsite Graph for Pinterset Ads 

**Title (ZH)**: 基于现场-离场图的实体表示学习在Pinterest广告中的应用 

**Authors**: Jiayin Jin, Zhimeng Pan, Yang Tang, Jiarui Feng, Kungang Li, Chongyuan Xiang, Jiacheng Li, Runze Su, Siping Ji, Han Sun, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02609)  

**Abstract**: Graph Neural Networks (GNN) have been extensively applied to industry recommendation systems, as seen in models like GraphSage\cite{GraphSage}, TwHIM\cite{TwHIM}, LiGNN\cite{LiGNN} etc. In these works, graphs were constructed based on users' activities on the platforms, and various graph models were developed to effectively learn node embeddings. In addition to users' onsite activities, their offsite conversions are crucial for Ads models to capture their shopping interest. To better leverage offsite conversion data and explore the connection between onsite and offsite activities, we constructed a large-scale heterogeneous graph based on users' onsite ad interactions and opt-in offsite conversion activities. Furthermore, we introduced TransRA (TransR\cite{TransR} with Anchors), a novel Knowledge Graph Embedding (KGE) model, to more efficiently integrate graph embeddings into Ads ranking models. However, our Ads ranking models initially struggled to directly incorporate Knowledge Graph Embeddings (KGE), and only modest gains were observed during offline experiments. To address this challenge, we employed the Large ID Embedding Table technique and innovated an attention based KGE finetuning approach within the Ads ranking models. As a result, we observed a significant AUC lift in Click-Through Rate (CTR) and Conversion Rate (CVR) prediction models. Moreover, this framework has been deployed in Pinterest's Ads Engagement Model and contributed to $2.69\%$ CTR lift and $1.34\%$ CPC reduction. We believe the techniques presented in this paper can be leveraged by other large-scale industrial models. 

**Abstract (ZH)**: 图神经网络（GNN）在工业推荐系统中的广泛应用：基于用户平台活动的大规模异构图构建与广告 ranking 模型中的知识图嵌入融合研究 

---
# StructSynth: Leveraging LLMs for Structure-Aware Tabular Data Synthesis in Low-Data Regimes 

**Title (ZH)**: StructSynth：在数据稀缺情况下利用大规模语言模型进行结构感知表数据合成 

**Authors**: Siyi Liu, Yujia Zheng, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02601)  

**Abstract**: The application of machine learning on tabular data in specialized domains is severely limited by data scarcity. While generative models offer a solution, traditional methods falter in low-data regimes, and recent Large Language Models (LLMs) often ignore the explicit dependency structure of tabular data, leading to low-fidelity synthetics. To address these limitations, we introduce StructSynth, a novel framework that integrates the generative power of LLMs with robust structural control. StructSynth employs a two-stage architecture. First, it performs explicit structure discovery to learn a Directed Acyclic Graph (DAG) from the available data. Second, this learned structure serves as a high-fidelity blueprint to steer the LLM's generation process, forcing it to adhere to the learned feature dependencies and thereby ensuring the generated data respects the underlying structure by design. Our extensive experiments demonstrate that StructSynth produces synthetic data with significantly higher structural integrity and downstream utility than state-of-the-art methods. It proves especially effective in challenging low-data scenarios, successfully navigating the trade-off between privacy preservation and statistical fidelity. 

**Abstract (ZH)**: 结构化合成：一种结合大语言模型生成能力和稳健结构控制的新型框架 

---
# Explainable AI for Automated User-specific Feedback in Surgical Skill Acquisition 

**Title (ZH)**: 可解释的人工智能在手术技能获取中自动实现用户特定反馈 

**Authors**: Catalina Gomez, Lalithkumar Seenivasan, Xinrui Zou, Jeewoo Yoon, Sirui Chu, Ariel Leong, Patrick Kramer, Yu-Chun Ku, Jose L. Porras, Alejandro Martin-Gomez, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2508.02593)  

**Abstract**: Traditional surgical skill acquisition relies heavily on expert feedback, yet direct access is limited by faculty availability and variability in subjective assessments. While trainees can practice independently, the lack of personalized, objective, and quantitative feedback reduces the effectiveness of self-directed learning. Recent advances in computer vision and machine learning have enabled automated surgical skill assessment, demonstrating the feasibility of automatic competency evaluation. However, it is unclear whether such Artificial Intelligence (AI)-driven feedback can contribute to skill acquisition. Here, we examine the effectiveness of explainable AI (XAI)-generated feedback in surgical training through a human-AI study. We create a simulation-based training framework that utilizes XAI to analyze videos and extract surgical skill proxies related to primitive actions. Our intervention provides automated, user-specific feedback by comparing trainee performance to expert benchmarks and highlighting deviations from optimal execution through understandable proxies for actionable guidance. In a prospective user study with medical students, we compare the impact of XAI-guided feedback against traditional video-based coaching on task outcomes, cognitive load, and trainees' perceptions of AI-assisted learning. Results showed improved cognitive load and confidence post-intervention. While no differences emerged between the two feedback types in reducing performance gaps or practice adjustments, trends in the XAI group revealed desirable effects where participants more closely mimicked expert practice. This work encourages the study of explainable AI in surgical education and the development of data-driven, adaptive feedback mechanisms that could transform learning experiences and competency assessment. 

**Abstract (ZH)**: 传统外科技能获取高度依赖专家反馈，但由于师资 availability 限制和主观评估的差异性，直接访问受限。尽管学员可以独立练习，但在缺乏个性化、客观和定量反馈的情况下，自我导向学习的效果受到影响。近期计算机视觉和机器学习的进步使自动化外科技能评估成为可能，展示了自动技能评估的可行性。然而，尚不清楚此类基于人工智能（AI）的反馈是否能促进技能获取。在这里，我们通过一项人机研究，评估可解释AI（XAI）生成反馈在外科训练中的有效性。我们创建了一个基于模拟的培训框架，利用XAI分析视频，提取与基本动作相关的外科技能代理指标。我们的干预措施通过可理解的反馈代理提供自动化的、用户特定的反馈，将学员的表现与专家基准进行比较，并通过突出显示与最优执行的偏差来提供可操作的指导。在一项前瞻性用户研究中，我们比较了XAI引导反馈与基于视频的传统教练对任务结果、认知负荷以及学员对AI辅助学习的感知的影响。结果表明，干预后认知负荷和信心有所提高。虽然两种反馈类型在减少性能差距或实践调整方面没有显现出差异，但XAI组的趋势显示，参与者模仿专家实践的效果更为理想。这项工作鼓励在外科教育中研究可解释AI，并开发数据驱动、自适应的反馈机制，从而改变学习体验和技能评估方式。 

---
# Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules 

**Title (ZH)**: 参数高效路由微调：专家混合需求适配模块混合 

**Authors**: Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2508.02587)  

**Abstract**: Mixture-of-Experts (MoE) benefits from a dynamic routing mechanism among their specialized experts, which existing Parameter- Efficient Fine-Tuning (PEFT) strategies fail to leverage. This motivates us to investigate whether adaptation modules themselves should incorporate routing mechanisms to align with MoE's multi-expert architecture. We analyze dynamics of core components when applying PEFT to MoE language models and examine how different routing strategies affect adaptation effectiveness. Extensive experiments adapting OLMoE-1B-7B and Mixtral-8x7B on various commonsense and math reasoning tasks validate the performance and efficiency of our routed approach. We identify the optimal configurations for different scenarios and provide empirical analyses with practical insights to facilitate better PEFT and MoE applications. 

**Abstract (ZH)**: Mixture-of-Experts (MoE)受益于其专门专家之间的动态路由机制，而现有的参数高效微调（PEFT）策略未能利用这一点。这促使我们 investigation 是否应将路由机制纳入适应模块，以与MoE的多专家架构相一致。我们分析了在MoE语言模型中应用PEFT时核心组件的动力学，并研究了不同路由策略如何影响适应效果。针对OLMoE-1B-7B和Mixtral-8x7B在各种常识和数学推理任务上的广泛实验验证了我们所提出的路由方法的性能和效率。我们确定了不同场景下的最优配置，并提供了实用见解的实证分析，以促进更好的PEFT和MoE应用。 

---
# MArgE: Meshing Argumentative Evidence from Multiple Large Language Models for Justifiable Claim Verification 

**Title (ZH)**: MArgE: 多个大型语言模型的论证证据网格化以实现可验证的声明验证 

**Authors**: Ming Pok Ng, Junqi Jiang, Gabriel Freedman, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02584)  

**Abstract**: Leveraging outputs from multiple large language models (LLMs) is emerging as a method for harnessing their power across a wide range of tasks while mitigating their capacity for making errors, e.g., hallucinations. However, current approaches to combining insights from multiple LLMs often involve unstructured interactions (e.g., free debate), resulting in model generations that are not faithfully justifiable. In this work, we introduce MArgE, a novel framework to provide formal structure to the evidence from each LLM, in the form of a tree of extracted arguments, for the task of claim verification. We use a variant of Argumentative LLMs (ArgLLMs), i.e. LLMs driven by frameworks and semantics from the field of computational argumentation, to construct structured argument trees for given claims. This process creates an inspectable pathway from the initial arguments to the final claim verification decisions, providing a faithful justification thereof. We show experimentally that MArgE can significantly outperform single LLMs, including three open-source models (4B to 8B parameters), GPT-4o-mini and existing ArgLLMs, as well as prior methods for unstructured multi-LLM debates. We thus demonstrate the advantages of incorporating formal, argumentative reasoning mechanisms when combining multiple LLM outputs. 

**Abstract (ZH)**: 利用多个大规模语言模型（LLMs）的输出来利用其在广泛任务上的强大能力，同时减轻其生成错误（如幻觉）的风险：一种新的框架MArgE 

---
# EHSAN: Leveraging ChatGPT in a Hybrid Framework for Arabic Aspect-Based Sentiment Analysis in Healthcare 

**Title (ZH)**: EHSAN: 结合ChatGPT的一种混合框架在医疗保健领域进行阿拉伯语方面情感分析 

**Authors**: Eman Alamoudi, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02574)  

**Abstract**: Arabic-language patient feedback remains under-analysed because dialect diversity and scarce aspect-level sentiment labels hinder automated assessment. To address this gap, we introduce EHSAN, a data-centric hybrid pipeline that merges ChatGPT pseudo-labelling with targeted human review to build the first explainable Arabic aspect-based sentiment dataset for healthcare. Each sentence is annotated with an aspect and sentiment label (positive, negative, or neutral), forming a pioneering Arabic dataset aligned with healthcare themes, with ChatGPT-generated rationales provided for each label to enhance transparency. To evaluate the impact of annotation quality on model performance, we created three versions of the training data: a fully supervised set with all labels reviewed by humans, a semi-supervised set with 50% human review, and an unsupervised set with only machine-generated labels. We fine-tuned two transformer models on these datasets for both aspect and sentiment classification. Experimental results show that our Arabic-specific model achieved high accuracy even with minimal human supervision, reflecting only a minor performance drop when using ChatGPT-only labels. Reducing the number of aspect classes notably improved classification metrics across the board. These findings demonstrate an effective, scalable approach to Arabic aspect-based sentiment analysis (SA) in healthcare, combining large language model annotation with human expertise to produce a robust and explainable dataset. Future directions include generalisation across hospitals, prompt refinement, and interpretable data-driven modelling. 

**Abstract (ZH)**: 阿拉伯语言患者反馈分析不足，因为方言多样性和稀缺的方面级情感标签阻碍了自动化评估。为解决这一问题，我们引入了EHSAN，这是一种数据为中心的混合管道，结合了ChatGPT伪标签和有针对性的人工审查，构建了首个可解释的阿拉伯方面基于情感医疗数据集。每个句子都标注了方面和情感标签（正面、负面或中性），形成了与医疗主题对齐的开创性阿拉伯语数据集，并为每个标签提供了ChatGPT生成的解释，以增强透明度。为了评估注释质量对模型性能的影响，我们创建了三种训练数据版本：一个完全监督集，其中所有标签都由人工审核；一个半监督集，其中50%的标签由人工审核；一个未监督集，仅由机器生成的标签。我们在这些数据集上对两个变压器模型进行了微调，用于方面和情感分类。实验结果表明，我们的阿拉伯特定模型即使在有限的人工监督下也能实现高准确性，仅使用ChatGPT标签时，性能仅略有下降。减少方面类别的数量显著提高了总体分类指标。这些发现展示了结合大规模语言模型注释和人工专业知识的有效可扩展方法，以构建稳健且可解释的阿拉伯方面基于情感分析（SA）数据集，未来方向包括跨医院的泛化、提示 refinement 和可解释的数据驱动建模。 

---
# Dynamic Feature Selection based on Rule-based Learning for Explainable Classification with Uncertainty Quantification 

**Title (ZH)**: 基于规则学习的动态特征选择方法及其在解释性分类中的不确定性量化 

**Authors**: Javier Fumanal-Idocin, Raquel Fernandez-Peralta, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02566)  

**Abstract**: Dynamic feature selection (DFS) offers a compelling alternative to traditional, static feature selection by adapting the selected features to each individual sample. Unlike classical methods that apply a uniform feature set, DFS customizes feature selection per sample, providing insight into the decision-making process for each case. DFS is especially significant in settings where decision transparency is key, i.e., clinical decisions; however, existing methods use opaque models, which hinder their applicability in real-life scenarios. This paper introduces a novel approach leveraging a rule-based system as a base classifier for the DFS process, which enhances decision interpretability compared to neural estimators. We also show how this method provides a quantitative measure of uncertainty for each feature query and can make the feature selection process computationally lighter by constraining the feature search space. We also discuss when greedy selection of conditional mutual information is equivalent to selecting features that minimize the difference with respect to the global model predictions. Finally, we demonstrate the competitive performance of our rule-based DFS approach against established and state-of-the-art greedy and RL methods, which are mostly considered opaque, compared to our explainable rule-based system. 

**Abstract (ZH)**: 基于规则系统的动态特征选择方法 

---
# Stakeholder Perspectives on Humanistic Implementation of Computer Perception in Healthcare: A Qualitative Study 

**Title (ZH)**: 计算机感知在医疗领域的人文实现：一项定性研究 

**Authors**: Kristin M. Kostick-Quenet, Meghan E. Hurley, Syed Ayaz, John Herrington, Casey Zampella, Julia Parish-Morris, Birkan Tunç, Gabriel Lázaro-Muñoz, J.S. Blumenthal-Barby, Eric A. Storch  

**Link**: [PDF](https://arxiv.org/pdf/2508.02550)  

**Abstract**: Computer perception (CP) technologies (digital phenotyping, affective computing and related passive sensing approaches) offer unprecedented opportunities to personalize healthcare, but provoke concerns about privacy, bias and the erosion of empathic, relationship-centered practice. A comprehensive understanding of perceived risks, benefits, and implementation challenges from those who design, deploy and experience these tools in real-world settings remains elusive. This study provides the first evidence-based account of key stakeholder perspectives on the relational, technical, and governance challenges raised by the integration of CP technologies into patient care. We conducted in-depth, semi-structured interviews with 102 stakeholders: adolescent patients and their caregivers, frontline clinicians, technology developers, and ethics, legal, policy or philosophy scholars. Transcripts underwent thematic analysis by a multidisciplinary team; reliability was enhanced through double coding and consensus adjudication. Stakeholders articulated seven interlocking concern domains: (1) trustworthiness and data integrity; (2) patient-specific relevance; (3) utility and workflow integration; (4) regulation and governance; (5) privacy and data protection; (6) direct and indirect patient harms; and (7) philosophical critiques of reductionism. To operationalize humanistic safeguards, we propose "personalized roadmaps": co-designed plans that predetermine which metrics will be monitored, how and when feedback is shared, thresholds for clinical action, and procedures for reconciling discrepancies between algorithmic inferences and lived experience. By translating these insights into personalized roadmaps, we offer a practical framework for developers, clinicians and policymakers seeking to harness continuous behavioral data while preserving the humanistic core of care. 

**Abstract (ZH)**: Computer感知技术(CP)（数字表型分析、情感计算及相关的被动感知方法）为个性化医疗保健提供了前所未有的机遇，但也引发了关于隐私、偏见和人际关系中心实践侵蚀的担忧。有关在实际应用环境中设计、部署和体验这些工具的参与者对感知风险、益处以及实施挑战的全面理解仍然匮乏。本文提供了首个基于证据的关键利益相关者对将计算机感知技术整合入患者护理所引发的关系、技术和治理挑战的观点描述。我们对102名参与者进行了深入半结构化访谈，包括青少年患者及其护理人员、一线临床医生、技术开发者以及伦理、法律、政策或哲学学者。研究结果通过对多学科团队进行主题分析获得，通过双重编码和共识裁定提升了可靠性。参与者概述了七个交织的关注领域：（1）可信度和数据完整性；（2）患者特定的相关性；（3）效用和工作流程整合；（4）监管和治理；（5）隐私和数据保护；（6）对患者的直接和间接伤害；以及（7）对还原论的哲学批评。为实现人文主义保障的有效实施，我们提出“个性化路线图”：由多方共同设计的计划，明确监测哪些指标、何时何地分享反馈、临床行动的阈值以及算法推断与生活体验之间的分歧解决程序。通过将这些见解转化为个性化路线图，我们提供了开发人员、临床医生和政策制定者在获取连续行为数据的同时保留关照核心的实用框架。 

---
# The KG-ER Conceptual Schema Language 

**Title (ZH)**: KG-ER 概念模式语言 

**Authors**: Enrico Franconi, Benoît Groz, Jan Hidders, Nina Pardal, Sławek Staworko, Jan Van den Bussche, Piotr Wieczorek  

**Link**: [PDF](https://arxiv.org/pdf/2508.02548)  

**Abstract**: We propose KG-ER, a conceptual schema language for knowledge graphs that describes the structure of knowledge graphs independently of their representation (relational databases, property graphs, RDF) while helping to capture the semantics of the information stored in a knowledge graph. 

**Abstract (ZH)**: 我们提出了一种名为KG-ER的概念模式语言，用于知识图谱，独立于知识图谱的表现形式（关系数据库、属性图、RDF）来描述知识图谱的结构，同时帮助捕获知识图谱中存储信息的语义。 

---
# What are you sinking? A geometric approach on attention sink 

**Title (ZH)**: 什么是你下沉的？一种注意力陷阱的几何方法 

**Authors**: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.02546)  

**Abstract**: Attention sink (AS) is a consistent pattern in transformer attention maps where certain tokens (often special tokens or positional anchors) disproportionately attract attention from other tokens. We show that in transformers, AS is not an architectural artifact, but it is the manifestation of a fundamental geometric principle: the establishment of reference frames that anchor representational spaces. We analyze several architectures and identify three distinct reference frame types, centralized, distributed, and bidirectional, that correlate with the attention sink phenomenon. We show that they emerge during the earliest stages of training as optimal solutions to the problem of establishing stable coordinate systems in high-dimensional spaces. We show the influence of architecture components, particularly position encoding implementations, on the specific type of reference frame. This perspective transforms our understanding of transformer attention mechanisms and provides insights for both architecture design and the relationship with AS. 

**Abstract (ZH)**: 注意力汇流（Attention Sink）是变压器注意力图中的一致模式，某些特定的标记（通常是特殊标记或位置锚点）会对其他标记的关注度不成比例地高。我们证明在变压器中，注意力汇流不是架构伪影，而是建立参考框架的基本几何原则的表现：这些框架将表征空间进行定位。我们分析了多个架构，并识别出三种不同的参考框架类型：集中型、分布式和双向型，它们与注意力汇流现象相关联。我们展示了这些参考框架在训练初期作为在高维空间中建立稳定坐标系统问题的最优解而出现。我们指出了架构组件，特别是位置编码实现方式，对特定类型参考框架的影响。这种视角重塑了我们对变压器注意力机制的理解，并提供了对架构设计以及与注意力汇流关系的见解。 

---
# Automatic Identification of Machine Learning-Specific Code Smells 

**Title (ZH)**: 自动识别机器学习特定的代码异味 

**Authors**: Peter Hamfelt, Ricardo Britto, Lincoln Rocha, Camilo Almendra  

**Link**: [PDF](https://arxiv.org/pdf/2508.02541)  

**Abstract**: Machine learning (ML) has rapidly grown in popularity, becoming vital to many industries. Currently, the research on code smells in ML applications lacks tools and studies that address the identification and validity of ML-specific code smells. This work investigates suitable methods and tools to design and develop a static code analysis tool (MLpylint) based on code smell criteria. This research employed the Design Science Methodology. In the problem identification phase, a literature review was conducted to identify ML-specific code smells. In solution design, a secondary literature review and consultations with experts were performed to select methods and tools for implementing the tool. We evaluated the tool on data from 160 open-source ML applications sourced from GitHub. We also conducted a static validation through an expert survey involving 15 ML professionals. The results indicate the effectiveness and usefulness of the MLpylint. We aim to extend our current approach by investigating ways to introduce MLpylint seamlessly into development workflows, fostering a more productive and innovative developer environment. 

**Abstract (ZH)**: 机器学习代码异味的研究：设计与开发基于代码异味标准的静态代码分析工具MLpylint 

---
# Decomposed Reasoning with Reinforcement Learning for Relevance Assessment in UGC Platforms 

**Title (ZH)**: 基于强化学习的分解推理在UGC平台的相关性评估 

**Authors**: Xiaowei Yuan, Lei Jin, Haoxin Zhang, Yan Gao, Yi Wu, Yao Hu, Ziyang Huang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02506)  

**Abstract**: Retrieval-augmented generation (RAG) plays a critical role in user-generated content (UGC) platforms, but its effectiveness depends heavily on accurate relevance assessment of query-document pairs. Despite recent advances in applying large language models (LLMs) to relevance modeling, UGC platforms present unique challenges: 1) ambiguous user intent due to sparse user feedback in RAG scenarios, and 2) substantial noise introduced by informal and unstructured language. To address these issues, we propose the Reinforced Reasoning Model for Relevance Assessment (R3A), which introduces a decomposed reasoning framework over queries and candidate documents before scoring. R3A first leverages auxiliary high-ranked documents within the platform to infer latent query intent. It then performs verbatim fragment extraction to justify relevance decisions, thereby reducing errors caused by noisy UGC. Based on a reinforcement learning framework, R3A is optimized to mitigate distortions arising from ambiguous queries and unstructured content. Experimental results show that R3A significantly outperforms existing baseline methods in terms of relevance accuracy, across both offline benchmarks and online experiments. 

**Abstract (ZH)**: 检索增强生成（RAG）在用户生成内容（UGC）平台中扮演着关键角色，但其效果高度依赖于查询文档对的相关性评估准确性。尽管在将大规模语言模型（LLMs）应用于相关性建模方面取得了近期进展，UGC平台仍面临独特挑战：1）由于在RAG场景中用户反馈稀疏导致的模棱两可的用户意图，以及2）由非正式和无结构语言引入的巨大噪声。为解决这些问题，我们提出了增强推理模型用于相关性评估（R3A），该模型在评分前引入了查询和候选文档的分解推理框架。R3A 首先利用平台内的高排名辅助文档来推断潜在的查询意图。然后进行逐字片段提取来验证相关性决策，从而减少由UGC噪声引起的错误。基于强化学习框架，R3A 优化以减轻模棱两可查询和无结构内容引起的失真。实验结果表明，R3A 在相关性准确性方面显著优于现有基线方法，在离线基准测试和在线实验中均有更佳表现。 

---
# AIAP: A No-Code Workflow Builder for Non-Experts with Natural Language and Multi-Agent Collaboration 

**Title (ZH)**: AIAP：一种基于自然语言和多agent协作的无代码工作流构建器供非专家使用 

**Authors**: Hyunjn An, Yongwon Kim, Wonduk Seo, Joonil Park, Daye Kang, Changhoon Oh, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02470)  

**Abstract**: While many tools are available for designing AI, non-experts still face challenges in clearly expressing their intent and managing system complexity. We introduce AIAP, a no-code platform that integrates natural language input with visual workflows. AIAP leverages a coordinated multi-agent system to decompose ambiguous user instructions into modular, actionable steps, hidden from users behind a unified interface. A user study involving 32 participants showed that AIAP's AI-generated suggestions, modular workflows, and automatic identification of data, actions, and context significantly improved participants' ability to develop services intuitively. These findings highlight that natural language-based visual programming significantly reduces barriers and enhances user experience in AI service design. 

**Abstract (ZH)**: 非专家友好的自然语言输入与可视化工作流集成的无代码平台AIAP：提高AI服务设计的直观性和用户体验 

---
# TreeRanker: Fast and Model-agnostic Ranking System for Code Suggestions in IDEs 

**Title (ZH)**: TreeRanker: 快速且模型无关的代码建议排名系统 

**Authors**: Daniele Cipollone, Egor Bogomolov, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02455)  

**Abstract**: Token-level code completion is one of the most critical features in modern Integrated Development Environments (IDEs). It assists developers by suggesting relevant identifiers and APIs during coding. While completions are typically derived from static analysis, their usefulness depends heavily on how they are ranked, as correct predictions buried deep in the list are rarely seen by users. Most current systems rely on hand-crafted heuristics or lightweight machine learning models trained on user logs, which can be further improved to capture context information and generalize across projects and coding styles. In this work, we propose a new scoring approach to ranking static completions using language models in a lightweight and model-agnostic way. Our method organizes all valid completions into a prefix tree and performs a single greedy decoding pass to collect token-level scores across the tree. This enables a precise token-aware ranking without needing beam search, prompt engineering, or model adaptations. The approach is fast, architecture-agnostic, and compatible with already deployed models for code completion. These findings highlight a practical and effective pathway for integrating language models into already existing tools within IDEs, and ultimately providing smarter and more responsive developer assistance. 

**Abstract (ZH)**: Token级别代码补全在现代集成开发环境（IDEs）中是一个最关键的功能之一。它通过在编码过程中建议相关的标识符和API来辅助开发人员。虽然补全通常是通过静态分析获得的，但它们的有用性极大地依赖于它们的排名方式，因为列表深处的正确预测很少被用户看到。现有的大多数系统依赖于手工编写的启发式规则或基于用户日志训练的轻量级机器学习模型，这些模型可以进一步改进以捕获上下文信息并在不同项目和编码风格之间进行泛化。在本工作中，我们提出了一种新的评分方法，使用语言模型以轻量级且模型无关的方式对静态补全进行排名。该方法将所有有效的补全组织成前缀树，并进行一次贪婪解码以收集树上的token级别分数。这种方法能够实现精确的token感知排名，而无需使用束搜索、提示工程或模型调整。该方法快速、架构无关，并且可以与已部署的代码补全模型兼容。这些发现强调了一条实用且有效的方法，即将语言模型集成到现有工具中，并最终为开发人员提供更智能和响应更快的帮助。 

---
# Dynamic Forgetting and Spatio-Temporal Periodic Interest Modeling for Local-Life Service Recommendation 

**Title (ZH)**: 动态遗忘与时空周期兴趣建模在本地生活服务推荐中的应用 

**Authors**: Zhaoyu Hu, Hao Guo, Yuan Tian, Erpeng Xue, Jianyang Wang, Xianyang Qi, Hongxiang Lin, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02451)  

**Abstract**: In the context of the booming digital economy, recommendation systems, as a key link connecting users and numerous services, face challenges in modeling user behavior sequences on local-life service platforms, including the sparsity of long sequences and strong spatio-temporal dependence. Such challenges can be addressed by drawing an analogy to the forgetting process in human memory. This is because users' responses to recommended content follow the recency effect and the cyclicality of memory. By exploring this, this paper introduces the forgetting curve and proposes Spatio-Temporal periodic Interest Modeling (STIM) with long sequences for local-life service recommendation. STIM integrates three key components: a dynamic masking module based on the forgetting curve, which is used to extract both recent spatiotemporal features and periodic spatiotemporal features; a query-based mixture of experts (MoE) approach that can adaptively activate expert networks under different dynamic masks, enabling the collaborative modeling of time, location, and items; and a hierarchical multi-interest network unit, which captures multi-interest representations by modeling the hierarchical interactions between the shallow and deep semantics of users' recent behaviors. By introducing the STIM method, we conducted online A/B tests and achieved a 1.54\% improvement in gross transaction volume (GTV). In addition, extended offline experiments also showed improvements. STIM has been deployed in a large-scale local-life service recommendation system, serving hundreds of millions of daily active users in core application scenarios. 

**Abstract (ZH)**: 在蓬勃发展数字经济的背景下，推荐系统作为连接用户和众多服务的关键环节，在本地生活服务平台上面临着长序列稀疏性和时空强依赖性的挑战。通过借鉴人类记忆的遗忘过程，本文提出了一种基于遗忘曲线的空间时序周期兴趣建模（STIM）方法来应对这些挑战，并取得了1.54%的交易总额（GTV）提升。该方法包括三个关键组件：基于遗忘曲线的动态掩码模块，用于提取近期和周期性的时空特征；基于查询的专家混合模型（MoE），能够在不同动态掩码下自适应激活专家网络，实现时间、地点和物品的协同建模；以及层级多兴趣网络模块，通过建模用户近期行为浅层和深层语义的层级交互来捕捉多兴趣表示。STIM已在大规模本地生活服务推荐系统中部署，服务于数十亿日活跃用户的核心应用场景。 

---
# Assessing the Reliability and Validity of Large Language Models for Automated Assessment of Student Essays in Higher Education 

**Title (ZH)**: 评估大型语言模型在高等教育中自动评估学生作文可靠性和有效性的方法 

**Authors**: Andrea Gaggioli, Giuseppe Casaburi, Leonardo Ercolani, Francesco Collova', Pietro Torre, Fabrizio Davide  

**Link**: [PDF](https://arxiv.org/pdf/2508.02442)  

**Abstract**: This study investigates the reliability and validity of five advanced Large Language Models (LLMs), Claude 3.5, DeepSeek v2, Gemini 2.5, GPT-4, and Mistral 24B, for automated essay scoring in a real world higher education context. A total of 67 Italian-language student essays, written as part of a university psychology course, were evaluated using a four-criterion rubric (Pertinence, Coherence, Originality, Feasibility). Each model scored all essays across three prompt replications to assess intra-model stability. Human-LLM agreement was consistently low and non-significant (Quadratic Weighted Kappa), and within-model reliability across replications was similarly weak (median Kendall's W < 0.30). Systematic scoring divergences emerged, including a tendency to inflate Coherence and inconsistent handling of context-dependent dimensions. Inter-model agreement analysis revealed moderate convergence for Coherence and Originality, but negligible concordance for Pertinence and Feasibility. Although limited in scope, these findings suggest that current LLMs may struggle to replicate human judgment in tasks requiring disciplinary insight and contextual sensitivity. Human oversight remains critical when evaluating open-ended academic work, particularly in interpretive domains. 

**Abstract (ZH)**: 本研究探讨了在实际高等教育环境下，Claude 3.5、DeepSeek v2、Gemini 2.5、GPT-4 和 Mistral 24B 等五种高级语言模型（LLMs）自动作文评分的可靠性和有效性。共有 67 篇意大利语学生作文（作为大学心理学课程的一部分）根据四个标准评分准则（相关性、连贯性、独创性、可行性）进行评估。每个模型对所有论文进行了三次提示复述评分，以评估模型内稳定性。人机一致性较低且不显著（Quadratic Weighted Kappa），复述间同一模型的可靠性也较弱（中位 Kendall’s W < 0.30）。系统评分差异明显，包括倾向于夸大连贯性以及对上下文依赖维度处理不一致。跨模型一致性分析显示，连贯性和独创性存在适度趋同，而相关性和可行性几乎不存在一致性。尽管范围有限，但这些发现表明，当前的语言模型在需要学科洞见和情境敏感性的任务中可能难以复制人类判断。在评估开放性学术作品时，特别是在解释性领域，人工监督仍然至关重要。 

---
# Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing 

**Title (ZH)**: 基于本体感觉的机器人 manipulator 多类人体/物体检测 

**Authors**: Justin Hehli, Marco Heiniger, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.02425)  

**Abstract**: In physical human-robot collaboration (pHRC) settings, humans and robots collaborate directly in shared environments. Robots must analyze interactions with objects to ensure safety and facilitate meaningful workflows. One critical aspect is human/object detection, where the contacted object is identified. Past research introduced binary machine learning classifiers to distinguish between soft and hard objects. This study improves upon those results by evaluating three-class human/object detection models, offering more detailed contact analysis. A dataset was collected using the Franka Emika Panda robot manipulator, exploring preprocessing strategies for time-series analysis. Models including LSTM, GRU, and Transformers were trained on these datasets. The best-performing model achieved 91.11\% accuracy during real-time testing, demonstrating the feasibility of multi-class detection models. Additionally, a comparison of preprocessing strategies suggests a sliding window approach is optimal for this task. 

**Abstract (ZH)**: 物理人类-机器人协作环境中的三类人/物检测及分析 

---
# Emergence of Fair Leaders via Mediators in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 多智能体强化学习中调解者促进公平领导者的涌现 

**Authors**: Akshay Dodwadmath, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02421)  

**Abstract**: Stackelberg games and their resulting equilibria have received increasing attention in the multi-agent reinforcement learning literature. Each stage of a traditional Stackelberg game involves a leader(s) acting first, followed by the followers. In situations where the roles of leader(s) and followers can be interchanged, the designated role can have considerable advantages, for example, in first-mover advantage settings. Then the question arises: Who should be the leader and when? A bias in the leader selection process can lead to unfair outcomes. This problem is aggravated if the agents are self-interested and care only about their goals and rewards. We formally define this leader selection problem and show its relation to fairness in agents' returns. Furthermore, we propose a multi-agent reinforcement learning framework that maximizes fairness by integrating mediators. Mediators have previously been used in the simultaneous action setting with varying levels of control, such as directly performing agents' actions or just recommending them. Our framework integrates mediators in the Stackelberg setting with minimal control (leader selection). We show that the presence of mediators leads to self-interested agents taking fair actions, resulting in higher overall fairness in agents' returns. 

**Abstract (ZH)**: Stackelberg博弈及其均衡在多agent强化学习领域的研究受到越来越多的关注。传统的Stackelberg博弈的每个阶段涉及领导者先行行动，随后是跟随者。当领导者和跟随者的角色可以互换时，指定的角色可能会带来显著的优势，例如在先手优势的情景中。那么一个问题出现了：谁应该成为领导者？什么时候？选择领导者的偏差可能导致不公平的结果。如果代理是自利的，只关心自己的目标和奖励，这个问题会更加严重。我们正式定义了这个领导者的选择问题，并展示了它与代理回报中的公平性之间的关系。此外，我们提出了一种通过整合调停者来最大化公平性的多agent强化学习框架。调停者在同时行动的设定中已被不同程度地使用，从直接执行代理的行动到仅推荐行动。我们的框架在极小控制的Stackelberg设定中整合了调停者。我们展示了调停者的存在导致自利代理采取公平行动，从而提高了代理回报中的总体公平性。 

---
# HGTS-Former: Hierarchical HyperGraph Transformer for Multivariate Time Series Analysis 

**Title (ZH)**: HGTS-Former：多层次超图变换器在多变量时间序列分析中的应用 

**Authors**: Xiao Wang, Hao Si, Fan Zhang, Xiaoya Zhou, Dengdi Sun, Wanli Lyu, Qingquan Yang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02411)  

**Abstract**: Multivariate time series analysis has long been one of the key research topics in the field of artificial intelligence. However, analyzing complex time series data remains a challenging and unresolved problem due to its high dimensionality, dynamic nature, and complex interactions among variables. Inspired by the strong structural modeling capability of hypergraphs, this paper proposes a novel hypergraph-based time series transformer backbone network, termed HGTS-Former, to address the multivariate coupling in time series data. Specifically, given the multivariate time series signal, we first normalize and embed each patch into tokens. Then, we adopt the multi-head self-attention to enhance the temporal representation of each patch. The hierarchical hypergraphs are constructed to aggregate the temporal patterns within each channel and fine-grained relations between different variables. After that, we convert the hyperedge into node features through the EdgeToNode module and adopt the feed-forward network to further enhance the output features. Extensive experiments conducted on two multivariate time series tasks and eight datasets fully validated the effectiveness of our proposed HGTS-Former. The source code will be released on this https URL. 

**Abstract (ZH)**: 基于超图的时间序列变压器骨干网络：HGTS-Former及其在多变量时间序列数据中的应用 

---
# Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera Fusion 

**Title (ZH)**: Hydra: 基于毫米波和摄像头融合的多模态叶湿检测方法 

**Authors**: Yimeng Liu, Maolin Gan, Huaili Zeng, Li Liu, Younsuk Dong, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02409)  

**Abstract**: Leaf Wetness Duration (LWD), the time that water remains on leaf surfaces, is crucial in the development of plant diseases. Existing LWD detection lacks standardized measurement techniques, and variations across different plant characteristics limit its effectiveness. Prior research proposes diverse approaches, but they fail to measure real natural leaves directly and lack resilience in various environmental conditions. This reduces the precision and robustness, revealing a notable practical application and effectiveness gap in real-world agricultural settings. This paper presents Hydra, an innovative approach that integrates millimeter-wave (mm-Wave) radar with camera technology to detect leaf wetness by determining if there is water on the leaf. We can measure the time to determine the LWD based on this detection. Firstly, we design a Convolutional Neural Network (CNN) to selectively fuse multiple mm-Wave depth images with an RGB image to generate multiple feature images. Then, we develop a transformer-based encoder to capture the inherent connection among the multiple feature images to generate a feature map, which is further fed to a classifier for detection. Moreover, we augment the dataset during training to generalize our model. Implemented using a frequency-modulated continuous-wave (FMCW) radar within the 76 to 81 GHz band, Hydra's performance is meticulously evaluated on plants, demonstrating the potential to classify leaf wetness with up to 96% accuracy across varying scenarios. Deploying Hydra in the farm, including rainy, dawn, or poorly light nights, it still achieves an accuracy rate of around 90%. 

**Abstract (ZH)**: Hydra：毫米波雷达与摄像头集成检测叶片湿ness的时间duration的方法 

---
# CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation 

**Title (ZH)**: CompressKV: 语义检索头在生成前知道哪些_token_不重要 

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02401)  

**Abstract**: Recent advances in large language models (LLMs) have significantly boosted long-context processing. However, the increasing key-value (KV) cache size poses critical challenges to memory and execution efficiency. Most KV cache compression methods rely on heuristic token eviction using all attention heads in Grouped Query Attention (GQA)-based LLMs. This method ignores the different functionalities of attention heads, leading to the eviction of critical tokens and thus degrades the performance of LLMs.
To address the issue above, instead of using all the attention heads in GQA-based LLMs to determine important tokens as in the previous work, we first identify the attention heads in each layer that are not only capable of retrieving the initial and final tokens of a prompt, but also capable of retrieving important tokens within the text and attending to their surrounding semantic context. Afterwards, we exploit such heads to determine the important tokens and retain their corresponding KV cache pairs. Furthermore, we analyze the cache eviction error of each layer individually and introduce a layer-adaptive KV cache allocation strategy. Experimental results demonstrate the proposed CompressKV consistently outperforms state-of-the-art approaches under various memory budgets on LongBench and Needle-in-a-Haystack benchmarks. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）在长上下文处理方面的进展显著提升。然而，关键值（KV）缓存大小的增加对内存和执行效率提出了关键挑战。大多数KV缓存压缩方法依赖于组查询注意（GQA）基模型中的所有注意头的启发式令牌移除策略。该方法忽略了注意头的不同功能，导致重要令牌被移除，从而降低了LLMs的性能。

为了解决上述问题，我们不采用先前工作中的方法在GQA基模型中使用所有注意头来确定重要令牌，而是首先在每个层中识别不仅能够检索提示的起始和结束令牌，而且能够检索文本中的重要令牌并关注它们周围语义上下文的注意头。随后，我们利用这些头来确定重要令牌并保留其对应的KV缓存对。此外，我们分别分析了每层的缓存移除误差，并引入了一种层自适应KV缓存分配策略。实验结果表明，在LongBench和Needle-in-a-Haystack基准测试下的各种内存预算下，提出的CompressKV性能上始终优于现有方法。我们的代码可在以下链接获取：this https URL。 

---
# Inference-time Scaling for Diffusion-based Audio Super-resolution 

**Title (ZH)**: 基于推断时缩放的扩散模型音频超分辨率推理 

**Authors**: Yizhu Jin, Zhen Ye, Zeyue Tian, Haohe Liu, Qiuqiang Kong, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.02391)  

**Abstract**: Diffusion models have demonstrated remarkable success in generative tasks, including audio super-resolution (SR). In many applications like movie post-production and album mastering, substantial computational budgets are available for achieving superior audio quality. However, while existing diffusion approaches typically increase sampling steps to improve quality, the performance remains fundamentally limited by the stochastic nature of the sampling process, leading to high-variance and quality-limited outputs. Here, rather than simply increasing the number of sampling steps, we propose a different paradigm through inference-time scaling for SR, which explores multiple solution trajectories during the sampling process. Different task-specific verifiers are developed, and two search algorithms, including the random search and zero-order search for SR, are introduced. By actively guiding the exploration of the high-dimensional solution space through verifier-algorithm combinations, we enable more robust and higher-quality outputs. Through extensive validation across diverse audio domains (speech, music, sound effects) and frequency ranges, we demonstrate consistent performance gains, achieving improvements of up to 9.70% in aesthetics, 5.88% in speaker similarity, 15.20% in word error rate, and 46.98% in spectral distance for speech SR from 4kHz to 24kHz, showcasing the effectiveness of our approach. Audio samples are available at: this https URL. 

**Abstract (ZH)**: 扩散模型在音频超分辨率生成任务中取得了显著成功。在电影后期制作和专辑母版制作等应用中，可用大量的计算资源以实现卓越的音频质量。然而，现有扩散方法通常通过增加采样步骤来提高质量，但性能依然受限于采样过程的随机性，导致输出具有高方差和质量限制。本文不单纯增加采样步骤，而是提出了一种在推断时通过尺度调整进行超分辨率的全新范式，在采样过程中探索多种解决方案轨迹。为不同任务开发了特定的验证器，并引入了两种搜索算法，包括随机搜索和用于超分辨率的零阶搜索。通过验证器和算法的组合主动引导高维解决方案空间的探索，实现了更加稳健和高质量的输出。通过在多种音频域（语音、音乐、音效）和频率范围下进行广泛验证，我们展示了持续的性能提升，从4kHz到24kHz的语音超分辨率方面，分别在美学、说话人相似性、词错误率和频谱距离上取得高达9.70%、5.88%、15.20%和46.98%的改善，展示了我们方法的有效性。音频样本见：this https URL。 

---
# Text2Lip: Progressive Lip-Synced Talking Face Generation from Text via Viseme-Guided Rendering 

**Title (ZH)**: Text2Lip: 基于辅音引导渲染的从文本生成渐进唇同步说话人脸 

**Authors**: Xu Wang, Shengeng Tang, Fei Wang, Lechao Cheng, Dan Guo, Feng Xue, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02362)  

**Abstract**: Generating semantically coherent and visually accurate talking faces requires bridging the gap between linguistic meaning and facial articulation. Although audio-driven methods remain prevalent, their reliance on high-quality paired audio visual data and the inherent ambiguity in mapping acoustics to lip motion pose significant challenges in terms of scalability and robustness. To address these issues, we propose Text2Lip, a viseme-centric framework that constructs an interpretable phonetic-visual bridge by embedding textual input into structured viseme sequences. These mid-level units serve as a linguistically grounded prior for lip motion prediction. Furthermore, we design a progressive viseme-audio replacement strategy based on curriculum learning, enabling the model to gradually transition from real audio to pseudo-audio reconstructed from enhanced viseme features via cross-modal attention. This allows for robust generation in both audio-present and audio-free scenarios. Finally, a landmark-guided renderer synthesizes photorealistic facial videos with accurate lip synchronization. Extensive evaluations show that Text2Lip outperforms existing approaches in semantic fidelity, visual realism, and modality robustness, establishing a new paradigm for controllable and flexible talking face generation. Our project homepage is this https URL. 

**Abstract (ZH)**: 生成语义连贯且视觉准确的说话人脸需要弥合语言意义与面部articulation之间的差距。尽管基于音频的方法仍然占主导地位，但它们依赖于高质量的配对音频视觉数据，并且音高到口唇动作映射的固有不确定性对扩展性和鲁棒性构成了重大挑战。为了解决这些问题，我们提出了Text2Lip，这是一种以viseme为中心的框架，通过将文本输入嵌入结构化的viseme序列中来构建可解释的语音-视觉桥梁。这些中间级单元为口唇运动预测提供了语言学意义上的先验知识。此外，我们设计了一种基于课程学习的渐进viseme-音频替代策略，使得模型能够逐步从真实音频过渡到通过跨模态注意从增强的viseme特征中重建的伪音频。这使得模型在有音频和无音频场景中都能实现鲁棒生成。最后，基于关键点的渲染器合成具有准确唇部同步的光逼真面部视频。广泛评估表明，Text2Lip 在语义保真度、视觉逼真度和模态鲁棒性方面优于现有方法，建立了可控和灵活说话人脸生成的新范式。我们的项目主页是这个 https://url。 

---
# mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera 

**Title (ZH)**: 基于毫米波雷达的利用摄像头提取道路布局实现T字路口非视距行人定位 

**Authors**: Byeonggyu Park, Hee-Yeun Kim, Byonghyok Choi, Hansang Cho, Byungkwan Kim, Soomok Lee, Mingu Jeon, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.02348)  

**Abstract**: Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method. 

**Abstract (ZH)**: 在城市环境中非视距(NLoS)区域中的行人定位对自主驾驶系统构成了重大挑战。尽管毫米波雷达在这样的场景下检测物体方面展示了潜力，但2D雷达点云(PCD)数据易受多路径反射引起的失真影响，导致空间推断困难。此外，虽然相机图像提供了高分辨率的视觉信息，但缺乏深度感知能力，无法直接观察NLoS区域中的物体。在本文中，我们提出了一种新颖的框架，通过相机推断的道路布局来解释雷达PCD，以实现NLoS行人定位。所提方法利用相机的视觉信息解释2D雷达PCD，从而实现空间场景重建。通过在真实车辆上安装的雷达-相机系统进行的实验验证了所提方法的有效性。通过在室外NLoS驾驶环境中收集的数据集评估定位性能，证明了该方法的实际适用性。 

---
# MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models 

**Title (ZH)**: MicroMix: 适用于大型语言模型的高效混合精度量化与微缩格式 

**Authors**: Wenyuan Liu, Haoqian Meng, Yilun Luo, Peng Zhang, Xindian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.02343)  

**Abstract**: Quantization significantly accelerates inference in large language models (LLMs) by replacing original high-precision matrices with low-precision counterparts. Recent advances in weight-activation quantization have primarily focused on mapping both weights and activations to the INT4 format. Although the new FP4 Tensor Cores in NVIDIA's Blackwell architecture offer up to 4x speedup over FP16, existing INT4-based kernels fail to fully exploit this capability due to mismatched data formats. To bridge this gap, we propose MicroMix, a co-designed mixed-precision quantization algorithm and matrix multiplication kernel based on Microscaling (MX) data formats. Tailored for the Blackwell architecture, the MicroMix kernel supports arbitrary combinations of MXFP4, MXFP6, and MXFP8 channels, and produces BFloat16 outputs. To achieve a favorable trade-off between accuracy and efficiency for each linear layer, we introduce quantization thresholds that identify activation elements where lower-precision formats (MXFP4 or MXFP6) incur excessive quantization error. Our algorithm selectively allocates higher-precision channels to preserve accuracy while maintaining compute efficiency. MicroMix achieves competitive or superior performance across diverse downstream tasks, including zero-shot and few-shot learning, language modeling, code generation, and mathematical reasoning. On both consumer-grade (RTX 5070Ti laptop) and server-grade (RTX 5090) GPUs, our kernel delivers at least 20% faster execution than TensorRT-FP8. Furthermore, when applied to various Llama and Qwen models, MicroMix consistently improves prefill latency and memory efficiency across a range of batch sizes compared to TensorRT baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 量化显著通过用低精度矩阵替换原始高精度矩阵来加速大型语言模型（LLMs）的推断。最近关于权重-激活量化方面的进展主要集中在将两者映射到INT4格式。尽管英伟达Blackwell架构中的新FP4张量核心相较于FP16提供了高达4倍的加速，现有的基于INT4的内核未能充分利用这一能力，原因是数据格式不匹配。为此，我们提出了一种名为MicroMix的协同设计的混合精度量化算法和矩阵乘法内核，基于Microscaling（MX）数据格式。为适应Blackwell架构，MicroMix内核支持MXFP4、MXFP6和MXFP8通道的任意组合，并生成BFloat16输出。为在每个线性层中实现准确性和效率的良好权衡，我们引入了量化阈值来识别低精度格式（MXFP4或MXFP6）会导致过度量化误差的激活元素。该算法选择性地分配更高精度的通道以保持准确性同时保持计算效率。MicroMix在包括零样本学习、少样本学习、语言建模、代码生成和数学推理等多种下游任务中取得了具有竞争力或更优的表现。在消费级（RTX 5070Ti笔记本电脑）和服务器级（RTX 5090）GPU上，我们的内核执行速度比TensorRT-FP8至少快20%。此外，当应用于各种Llama和Qwen模型时，MicroMix在不同批量大小下相对于TensorRT基线始终能提高预填充延迟和内存效率。我们的代码可在以下链接获取。 

---
# VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo 

**Title (ZH)**: VeOmni：以模型为中心的分布式食谱动物园，面向任何模态模型训练 

**Authors**: Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, Zhi Zhang, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02317)  

**Abstract**: Recent advances in large language models (LLMs) have driven impressive progress in omni-modal understanding and generation. However, training omni-modal LLMs remains a significant challenge due to the heterogeneous model architectures required to process diverse modalities, necessitating sophisticated system design for efficient large-scale training. Existing frameworks typically entangle model definition with parallel logic, incurring limited scalability and substantial engineering overhead for end-to-end omni-modal training. %
We present \veomni, a modular and efficient training framework to accelerate the development of omni-modal LLMs. \veomni introduces model-centric distributed recipes that decouples communication from computation, enabling efficient 3D parallelism on omni-modal LLMs. \veomni also features a flexible configuration interface supporting seamless integration of new modalities with minimal code change. %
Using \veomni, a omni-modal mixture-of-experts (MoE) model with 30B parameters can be trained with over 2,800 tokens/sec/GPU throughput and scale to 160K context lengths via 3D parallelism on 128 GPUs, showcasing its superior efficiency and scalability for training large omni-modal LLMs. 

**Abstract (ZH)**: 近期大规模语言模型的进步推动了全方位模态理解和生成的显著进展。然而，训练全方位模态的大规模语言模型仍然面临显著挑战，需要处理多种模态的异构模型架构，这要求高效的系统设计以支持大规模训练。现有的框架通常将模型定义与并行逻辑相结合，导致可扩展性有限并增加了端到端全方位模态训练的大量工程开销。

我们提出\veomni，一种模块化且高效的训练框架，以加速全方位模态的大规模语言模型的发展。\veomni引入以模型为中心的分布式食谱，将通信从计算中分离出来，从而在全方位模态的大规模语言模型上实现高效的三维并行化。\veomni还具备灵活的配置接口，支持通过最小的代码更改无缝集成新模态。

使用\veomni，一个具有300亿参数的全方位模态专家混合模型可以在每秒每GPU超过2800个token的吞吐量下进行训练，并通过128个GPU的三维并行化扩展到16万上下文长度，展示了其在训练大规模全方位模态的大规模语言模型方面的卓越效率和扩展性。 

---
# A Survey on Data Security in Large Language Models 

**Title (ZH)**: 大型语言模型中的数据安全综述 

**Authors**: Kang Chen, Xiuze Zhou, Yuanguo Lin, Jinhe Su, Yuanhui Yu, Li Shen, Fan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02312)  

**Abstract**: Large Language Models (LLMs), now a foundation in advancing natural language processing, power applications such as text generation, machine translation, and conversational systems. Despite their transformative potential, these models inherently rely on massive amounts of training data, often collected from diverse and uncurated sources, which exposes them to serious data security risks. Harmful or malicious data can compromise model behavior, leading to issues such as toxic output, hallucinations, and vulnerabilities to threats such as prompt injection or data poisoning. As LLMs continue to be integrated into critical real-world systems, understanding and addressing these data-centric security risks is imperative to safeguard user trust and system reliability. This survey offers a comprehensive overview of the main data security risks facing LLMs and reviews current defense strategies, including adversarial training, RLHF, and data augmentation. Additionally, we categorize and analyze relevant datasets used for assessing robustness and security across different domains, providing guidance for future research. Finally, we highlight key research directions that focus on secure model updates, explainability-driven defenses, and effective governance frameworks, aiming to promote the safe and responsible development of LLM technology. This work aims to inform researchers, practitioners, and policymakers, driving progress toward data security in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）：数据安全风险与防御策略综述 

---
# CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment 

**Title (ZH)**: CAPO: 向通过可验证的生成式责任指派增强大语言模型推理方向 

**Authors**: Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02298)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback, helping to mitigate reward hacking. However, current RLVR methods typically treat whole responses as single actions, assigning the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies and inefficient learning. Methods like PPO provide credit assignment through value estimation, but often yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-by-step judgments for each reasoning step, but they require high-quality process supervision labels and are time-consuming when applied in online reinforcement learning (RL). To overcome these limitations, we introduce a simple but efficient method Credit Assignment Policy Optimization (CAPO). Given a reasoning response rollout from the policy model, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass, thereby providing verifiable token-level rewards to refine the tokens that were originally assigned identical rule-based rewards. This enables more fine-grained credit assignment in an effective way. Furthermore, to enhance the accuracy and robustness of CAPO, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments using different backbones like Llama and Qwen models and in different sizes show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across six challenging mathematical benchmarks and three out-of-domain benchmarks. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）通过使用基于规则的二元反馈提升了大型语言模型（LLMs）的推理能力，有助于缓解奖励劫持问题。然而，当前的RLVR方法通常将整个响应视为单一动作，并为每个tokens分配相同的奖励，这种粗粒度的反馈阻碍了精确的信用分配，使得模型难以识别哪些推理步骤导致成功或失败，经常导致次优策略和低效学习。虽然像PPO这样的方法通过价值估计提供信用分配，但由于采样限制，往往会产生不准确且不可验证的信号。另一方面，使用过程奖励模型的方法可以为每个推理步骤提供逐步判断，但它们需要高质量的过程监督标签，并且在在线强化学习（RL）中应用时耗时。为克服这些限制，我们提出了一种简单且高效的信用分配策略优化（CAPO）方法。给定策略模型的推理响应滚动，CAPO 直接利用现成的一般用途LLM作为生成过程奖励模型（LLM-as-GenPRM）来一次性生成所有步骤评判，从而提供可验证的token级奖励以精炼原本被分配相同规则奖励的tokens，这以有效的方式实现了更精细的信用分配。此外，为了提高CAPO的准确性和稳健性，我们采用了可扩展的投票机制。使用不同的骨干模型（如Llama和Qwen）在不同规模下进行的广泛实验表明，CAPO在六个具有挑战性的数学基准和三个跨域基准上始终优于基于监督学习和基于强化学习的微调方法。 

---
# Flexible Automatic Identification and Removal (FAIR)-Pruner: An Efficient Neural Network Pruning Method 

**Title (ZH)**: FAIR修剪器：一种高效的神经网络修剪方法 

**Authors**: Chenqing Lin, Mostafa Hussien, Chengyao Yu, Mohamed Cheriet, Osama Abdelrahman, Ruixing Ming  

**Link**: [PDF](https://arxiv.org/pdf/2508.02291)  

**Abstract**: Neural network pruning is a critical compression technique that facilitates the deployment of large-scale neural networks on resource-constrained edge devices, typically by identifying and eliminating redundant or insignificant parameters to reduce computational and memory overhead. This paper proposes the Flexible Automatic Identification and Removal (FAIR)-Pruner, a novel method for neural network structured pruning. Specifically, FAIR-Pruner first evaluates the importance of each unit (e.g., neuron or channel) through the Utilization Score quantified by the Wasserstein distance. To reflect the performance degradation after unit removal, it then introduces the Reconstruction Error, which is computed via the Taylor expansion of the loss function. Finally, FAIR-Pruner identifies superfluous units with negligible impact on model performance by controlling the proposed Tolerance of Difference, which measures differences between unimportant units and those that cause performance degradation. A major advantage of FAIR-Pruner lies in its capacity to automatically determine the layer-wise pruning rates, which yields a more efficient subnetwork structure compared to applying a uniform pruning rate. Another advantage of the FAIR-Pruner is its great one-shot performance without post-pruning fine-tuning. Furthermore, with utilization scores and reconstruction errors, users can flexibly obtain pruned models under different pruning ratios. Comprehensive experimental validation on diverse benchmark datasets (e.g., ImageNet) and various neural network architectures (e.g., VGG) demonstrates that FAIR-Pruner achieves significant model compression while maintaining high accuracy. 

**Abstract (ZH)**: 灵活自动识别和去除（FAIR）剪枝器：一种新颖的神经网络结构化剪枝方法 

---
# Dialogue Systems Engineering: A Survey and Future Directions 

**Title (ZH)**: 对话系统工程：综述与未来方向 

**Authors**: Mikio Nakano, Hironori Takeuchi, Sadahiro Yoshikawa, Yoichi Matsuyama, Kazunori Komatani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02279)  

**Abstract**: This paper proposes to refer to the field of software engineering related to the life cycle of dialogue systems as Dialogue Systems Engineering, and surveys this field while also discussing its future directions. With the advancement of large language models, the core technologies underlying dialogue systems have significantly progressed. As a result, dialogue system technology is now expected to be applied to solving various societal issues and in business contexts. To achieve this, it is important to build, operate, and continuously improve dialogue systems correctly and efficiently. Accordingly, in addition to applying existing software engineering knowledge, it is becoming increasingly important to evolve software engineering tailored specifically to dialogue systems. In this paper, we enumerate the knowledge areas of dialogue systems engineering based on those of software engineering, as defined in the Software Engineering Body of Knowledge (SWEBOK) Version 4.0, and survey each area. Based on this survey, we identify unexplored topics in each area and discuss the future direction of dialogue systems engineering. 

**Abstract (ZH)**: 本文提出将与对话系统生命周期相关的软件工程领域称为对话系统工程，并对该领域进行综述，同时讨论其未来方向。随着大型语言模型的发展，对话系统的核心技术有了显著进步，因此对话系统技术现在被期望应用于解决各种社会问题和商业情境中。为了实现这一目标，正确且高效地构建、运行和持续改进对话系统至关重要。因此，在现有软件工程知识的基础上，有必要针对对话系统演化出专门的软件工程方法。本文基于软件工程知识体系（SWEBOK）第4版的知识领域，列出对话系统工程的知识领域，并对每个领域进行综述。基于该综述，我们识别出每个领域的未探索话题，并讨论对话系统工程的未来方向。 

---
# CellForge: Agentic Design of Virtual Cell Models 

**Title (ZH)**: CellForge: 自主设计虚拟细胞模型 

**Authors**: Xiangru Tang, Zhuoyun Yu, Jiapeng Chen, Yan Cui, Daniel Shao, Weixu Wang, Fang Wu, Yuchen Zhuang, Wenqi Shi, Zhi Huang, Arman Cohan, Xihong Lin, Fabian Theis, Smita Krishnaswamy, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.02276)  

**Abstract**: Virtual cell modeling represents an emerging frontier at the intersection of artificial intelligence and biology, aiming to predict quantities such as responses to diverse perturbations quantitatively. However, autonomously building computational models for virtual cells is challenging due to the complexity of biological systems, the heterogeneity of data modalities, and the need for domain-specific expertise across multiple disciplines. Here, we introduce CellForge, an agentic system that leverages a multi-agent framework that transforms presented biological datasets and research objectives directly into optimized computational models for virtual cells. More specifically, given only raw single-cell multi-omics data and task descriptions as input, CellForge outputs both an optimized model architecture and executable code for training virtual cell models and inference. The framework integrates three core modules: Task Analysis for presented dataset characterization and relevant literature retrieval, Method Design, where specialized agents collaboratively develop optimized modeling strategies, and Experiment Execution for automated generation of code. The agents in the Design module are separated into experts with differing perspectives and a central moderator, and have to collaboratively exchange solutions until they achieve a reasonable consensus. We demonstrate CellForge's capabilities in single-cell perturbation prediction, using six diverse datasets that encompass gene knockouts, drug treatments, and cytokine stimulations across multiple modalities. CellForge consistently outperforms task-specific state-of-the-art methods. Overall, CellForge demonstrates how iterative interaction between LLM agents with differing perspectives provides better solutions than directly addressing a modeling challenge. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 虚拟细胞建模代表了人工智能与生物学交汇领域的新兴前沿，旨在定量预测不同扰动的响应。然而，由于生物系统的复杂性、数据模态的异质性以及多学科领域特定专业知识的需要，自主构建虚拟细胞的计算模型具有挑战性。这里我们介绍CellForge，这是一种代理系统，利用多代理框架将呈现的生物数据集和研究目标直接转换为优化的虚拟细胞计算模型。具体而言，仅给定原始单细胞多组学数据和任务描述作为输入，CellForge 输出优化的模型架构和用于训练虚拟细胞模型及推断的可执行代码。该框架集成三个核心模块：任务分析、方法设计和实验执行。设计模块中的代理被分为具有不同视角的专家和一个中心协调员，他们需要协作交换解决方案，直到达成合理的共识。我们通过六个涵盖基因敲除、药物治疗和细胞因子刺激等多种模态的单细胞扰动预测数据集，展示了CellForge的能力。CellForge在所有任务中均优于特定任务的最佳方法。总体而言，CellForge展示了迭代互动如何为多视角人工智能代理提供比直接解决建模挑战更好的解决方案。我们的代码可在以下网址公开获得。 

---
# Dynaword: From One-shot to Continuously Developed Datasets 

**Title (ZH)**: Dynaword: 从单次生成到持续发展的数据集 

**Authors**: Kenneth Enevoldsen, Kristian Nørgaard Jensen, Jan Kostkan, Balázs Szabó, Márton Kardos, Kirten Vad, Andrea Blasi Núñez, Gianluca Barmina, Jacob Nielsen, Rasmus Larsen, Peter Vahlstrup, Per Møldrup Dalum, Desmond Elliott, Lukas Galke, Peter Schneider-Kamp, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02271)  

**Abstract**: Large-scale datasets are foundational for research and development in natural language processing. However, current approaches face three key challenges: (1) reliance on ambiguously licensed sources restricting use, sharing, and derivative works; (2) static dataset releases that prevent community contributions and diminish longevity; and (3) quality assurance processes restricted to publishing teams rather than leveraging community expertise.
To address these limitations, we introduce two contributions: the Dynaword approach and Danish Dynaword. The Dynaword approach is a framework for creating large-scale, open datasets that can be continuously updated through community collaboration. Danish Dynaword is a concrete implementation that validates this approach and demonstrates its potential. Danish Dynaword contains over four times as many tokens as comparable releases, is exclusively openly licensed, and has received multiple contributions across industry and research. The repository includes light-weight tests to ensure data formatting, quality, and documentation, establishing a sustainable framework for ongoing community contributions and dataset evolution. 

**Abstract (ZH)**: 大规模语料库是自然语言处理研究与开发的基础。然而，当前的方法面临三大关键挑战：(1) 对于使用、共享和衍生工作的限制性许可限制；(2) 静态数据集发布阻碍社区贡献并减少其持久性；(3) 质量保证过程仅限于发布团队，未能利用社区专业知识。

为解决这些限制，我们提出了两项贡献：Dynaword方法和Danish Dynaword。Dynaword方法是一种可以通过社区协作持续更新的大规模开源数据集框架。Danish Dynaword是这一方法的具体实现，展示了其潜力。Danish Dynaword包含比可比发布版本多四倍的词元，完全是开放许可的，并且已收到来自工业和研究界的多次贡献。该仓库还包括轻量级测试以确保数据格式化、质量和文档化，从而建立一个可持续的框架，以促进持续的社区贡献和数据集的进化。 

---
# Decomposing the Entropy-Performance Exchange: The Missing Keys to Unlocking Effective Reinforcement Learning 

**Title (ZH)**: 分解熵-性能交换：解锁有效强化学习的缺失关键因素 

**Authors**: Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02260)  

**Abstract**: Recently, reinforcement learning with verifiable rewards (RLVR) has been widely used for enhancing the reasoning abilities of large language models (LLMs). A core challenge in RLVR involves managing the exchange between entropy and performance of policies. Despite the importance of this exchange, a fine-grained understanding of when and how this exchange operates most effectively remains limited. To bridge this gap, we conduct a systematic empirical analysis of the entropy-performance exchange mechanism of RLVR across different levels of granularity. Specifically, we first divide the training process into two distinct stages based on entropy dynamics, i.e., rising stage and plateau stage, and then systematically investigate how this mechanism varies across stage-level, instance-level, and token-level granularitiess. Our analysis reveals that, in the rising stage, entropy reduction in negative samples facilitates the learning of effective reasoning patterns, which in turn drives rapid performance gains. Moreover, in the plateau stage, learning efficiency strongly correlates with high-entropy tokens present in low-perplexity samples and those located at the end of sequences. Motivated by these findings, we propose two methods that dynamically adjust the reward signal using perplexity and positional information to focus RL updates on tokens that exhibit high learning potential, achieving improvements compared to the baseline methods on various LLMs. 

**Abstract (ZH)**: 最近，具有可验证奖励的强化学习（RLVR）在提升大语言模型（LLM）的推理能力方面得到了广泛应用。RLVR 中熵与性能交换机制的管理是一项核心挑战。尽管这一交换的重要性已经认识到，但对其最有效运作的时点和方式的细微理解仍然有限。为弥补这一不足，我们针对不同粒度层次系统地分析了 RLVR 的熵-性能交换机制。具体地，我们首先根据熵的动力学将训练过程分为上升阶段和平台阶段，然后系统地研究了该机制在不同阶段层次、实例层次和令牌层次上的变化。我们的分析表明，在上升阶段，负面样本中熵的降低有助于高效学习有效的推理模式，从而促进性能的快速提升。此外，在平台阶段，高效学习与低困惑度样本中高熵令牌以及序列末尾的令牌高度相关。基于这些发现，我们提出了一种方法，动态调整奖励信号，利用困惑度和位置信息，将 RL 更新集中在具有高学习潜力的令牌上，各种 LLM 上相比基线方法取得了性能改进。 

---
# StutterCut: Uncertainty-Guided Normalised Cut for Dysfluency Segmentation 

**Title (ZH)**: StutterCut: 基于不确定性归一化切分的非流畅性分割 

**Authors**: Suhita Ghosh, Melanie Jouaiti, Jan-Ole Perschewski, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2508.02255)  

**Abstract**: Detecting and segmenting dysfluencies is crucial for effective speech therapy and real-time feedback. However, most methods only classify dysfluencies at the utterance level. We introduce StutterCut, a semi-supervised framework that formulates dysfluency segmentation as a graph partitioning problem, where speech embeddings from overlapping windows are represented as graph nodes. We refine the connections between nodes using a pseudo-oracle classifier trained on weak (utterance-level) labels, with its influence controlled by an uncertainty measure from Monte Carlo dropout. Additionally, we extend the weakly labelled FluencyBank dataset by incorporating frame-level dysfluency boundaries for four dysfluency types. This provides a more realistic benchmark compared to synthetic datasets. Experiments on real and synthetic datasets show that StutterCut outperforms existing methods, achieving higher F1 scores and more precise stuttering onset detection. 

**Abstract (ZH)**: 检测和分割语音不流畅是有效言语治疗和实时反馈的关键。然而，大多数方法仅在句级对不流畅进行分类。我们引入了StutterCut，一种半监督框架，将语音不流畅分割问题形式化为图划分问题，其中重叠窗口的语音嵌入表示为图节点。我们使用在弱（句级）标签上训练的伪 oracle 分类器来细化节点之间的连接，并通过蒙特卡洛弃权的不确定性测量来控制其影响。此外，我们通过为四种不流畅类型增加帧级不流畅边界，扩展了弱标记的FluencyBank数据集。这提供了比合成数据集更现实的基准。实验结果表明，StutterCut在真实和合成数据集上均优于现有方法，取得了更高的F1分数和更精确的重复语起始检测。 

---
# ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space 

**Title (ZH)**: ByteGen：一种基于字节空间的无分词生成模型用于订单簿事件 

**Authors**: Yang Li, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02247)  

**Abstract**: Generative modeling of high-frequency limit order book (LOB) dynamics is a critical yet unsolved challenge in quantitative finance, essential for robust market simulation and strategy backtesting. Existing approaches are often constrained by simplifying stochastic assumptions or, in the case of modern deep learning models like Transformers, rely on tokenization schemes that affect the high-precision, numerical nature of financial data through discretization and binning. To address these limitations, we introduce ByteGen, a novel generative model that operates directly on the raw byte streams of LOB events. Our approach treats the problem as an autoregressive next-byte prediction task, for which we design a compact and efficient 32-byte packed binary format to represent market messages without information loss. The core novelty of our work is the complete elimination of feature engineering and tokenization, enabling the model to learn market dynamics from its most fundamental representation. We achieve this by adapting the H-Net architecture, a hybrid Mamba-Transformer model that uses a dynamic chunking mechanism to discover the inherent structure of market messages without predefined rules. Our primary contributions are: 1) the first end-to-end, byte-level framework for LOB modeling; 2) an efficient packed data representation; and 3) a comprehensive evaluation on high-frequency data. Trained on over 34 million events from CME Bitcoin futures, ByteGen successfully reproduces key stylized facts of financial markets, generating realistic price distributions, heavy-tailed returns, and bursty event timing. Our findings demonstrate that learning directly from byte space is a promising and highly flexible paradigm for modeling complex financial systems, achieving competitive performance on standard market quality metrics without the biases of tokenization. 

**Abstract (ZH)**: 高频率限价订单簿（LOB）动力学的生成建模是量化金融中一个关键但未解决的挑战，对于稳健的市场模拟和策略回测至关重要。现有方法通常受限于简化的随机假设，或者在使用如变换器等现代深度学习模型时依赖影响金融数据高精度数值性质的令牌化方案。为解决这些限制，我们提出了ByteGen，一种新型的生成模型，直接处理LOB事件的原始字节流。我们的方法将问题视为自回归下一个字节预测任务，并设计了一种紧凑高效的32字节打包二进制格式来表示市场消息而不损失信息。我们的核心创新在于完全消除特征工程和令牌化，使模型能够从最基本的数据表示中学习市场动态。通过适应H-Net架构，一种混合Mamba-变压器模型，该模型使用动态分块机制来发现市场消息的固有结构，而无需预定义规则。我们的主要贡献包括：1）首个端到端的字节级LOB建模框架；2）高效的数据打包表示；3）在高频数据上的全面评估。ByteGen在CME比特币期货超过3400万事件的训练下，成功重现了金融市场的关键统计事实，生成了真实的 price 分布、重尾收益和突发事件时间。我们的研究结果表明，直接从字节空间学习是一种有前景且高度灵活的方法，用于建模复杂金融系统，在标准市场质量度量上实现了与令牌化偏差的竞争性性能。 

---
# Forecasting When to Forecast: Accelerating Diffusion Models with Confidence-Gated Taylor 

**Title (ZH)**: 基于信心门控泰勒级数的加速扩散模型：预报何时预报 

**Authors**: Xiaoliu Guan, Lielin Jiang, Hanqi Chen, Xu Zhang, Jiaxing Yan, Guanzhong Wang, Yi Liu, Zetao Zhang, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02240)  

**Abstract**: Diffusion Transformers (DiTs) have demonstrated remarkable performance in visual generation tasks. However, their low inference speed limits their deployment in low-resource applications. Recent training-free approaches exploit the redundancy of features across timesteps by caching and reusing past representations to accelerate inference. Building on this idea, TaylorSeer instead uses cached features to predict future ones via Taylor expansion. However, its module-level prediction across all transformer blocks (e.g., attention or feedforward modules) requires storing fine-grained intermediate features, leading to notable memory and computation overhead. Moreover, it adopts a fixed caching schedule without considering the varying accuracy of predictions across timesteps, which can lead to degraded outputs when prediction fails. To address these limitations, we propose a novel approach to better leverage Taylor-based acceleration. First, we shift the Taylor prediction target from the module level to the last block level, significantly reducing the number of cached features. Furthermore, observing strong sequential dependencies among Transformer blocks, we propose to use the error between the Taylor-estimated and actual outputs of the first block as an indicator of prediction reliability. If the error is small, we trust the Taylor prediction for the last block; otherwise, we fall back to full computation, thereby enabling a dynamic caching mechanism. Empirical results show that our method achieves a better balance between speed and quality, achieving a 3.17x acceleration on FLUX, 2.36x on DiT, and 4.14x on Wan Video with negligible quality drop. The Project Page is \href{this https URL}{here.} 

**Abstract (ZH)**: 基于Taylor展开加速的扩散变换器在视觉生成任务中的高效预测方法 

---
# FinCPRG: A Bidirectional Generation Pipeline for Hierarchical Queries and Rich Relevance in Financial Chinese Passage Retrieval 

**Title (ZH)**: FinCPRG: 一种用于金融中文段落检索的双向生成管道，实现层级查询和丰富的相关性 

**Authors**: Xuan Xu, Beilin Chu, Qinhong Lin, Yixiao Zhong, Fufang Wen, Jiaqi Liu, Binjie Fei, Yu Li, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02222)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant potential in constructing passage retrieval datasets. However, existing methods still face limitations in expressing cross-doc query needs and controlling annotation quality. To address these issues, this paper proposes a bidirectional generation pipeline, which aims to generate 3-level hierarchical queries for both intra-doc and cross-doc scenarios and mine additional relevance labels on top of direct mapping annotation. The pipeline introduces two query generation methods: bottom-up from single-doc text and top-down from multi-doc titles. The bottom-up method uses LLMs to disassemble and generate structured queries at both sentence-level and passage-level simultaneously from intra-doc passages. The top-down approach incorporates three key financial elements--industry, topic, and time--to divide report titles into clusters and prompts LLMs to generate topic-level queries from each cluster. For relevance annotation, our pipeline not only relies on direct mapping annotation from the generation relationship but also implements an indirect positives mining method to enrich the relevant query-passage pairs. Using this pipeline, we constructed a Financial Passage Retrieval Generated dataset (FinCPRG) from almost 1.3k Chinese financial research reports, which includes hierarchical queries and rich relevance labels. Through evaluations of mined relevance labels, benchmarking and training experiments, we assessed the quality of FinCPRG and validated its effectiveness as a passage retrieval dataset for both training and benchmarking. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在构建篇章检索数据集方面展现了显著潜力。然而，现有方法在表达跨文档查询需求和控制注释质量方面仍然存在局限。为了解决这些问题，本文提出了一种双向生成管道，旨在为 intra-doc 和跨文档场景生成 3 层级查询，并在此基础上挖掘附加的相关性标签。该管道引入了两种查询生成方法：从单文档文本自底向上生成和从多文档标题自顶向下生成。自底向上的方法使用 LLMs 同时从跨文档段落中分解和生成结构化查询，自顶向下的方法结合了三个关键的金融要素——行业、主题和时间，将报告标题划分为簇，并提示 LLMs 从每个簇中生成主题级查询。对于相关性注释，我们的管道不仅依赖于生成关系中的直接映射注释，还实施了一种间接正样本挖掘方法，以丰富相关查询-段落对。利用该管道，我们从近 1300 份中文财务研究报告中构建了一个金融篇章检索生成数据集（FinCPRG），其中包括层级查询和丰富的相关性标签。通过评估挖掘的相关性标签、基准测试和训练实验，我们评估了 FinCPRG 的质量，并验证了其作为篇章检索数据集在训练和基准测试中的有效性。 

---
# LeanK: Learnable K Cache Channel Pruning for Efficient Decoding 

**Title (ZH)**: LeanK: 学习型K缓存通道剪枝以实现高效解码 

**Authors**: Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02215)  

**Abstract**: Large language models (LLMs) enable long-context tasks but face efficiency challenges due to the growing key-value (KV) cache. We propose LeanK, a learning-based method that prunes unimportant key (K) cache channels by leveraging static channel sparsity. With a novel two-stage training process, LeanK learns channel-wise static mask that could satisfy specific sparsity ratio and hardware alignment requirement. LeanK reduces GPU memory and accelerates decoding without sacrificing accuracy. Experiments demonstrate up to 70% K cache and 16%-18% V cache memory reduction. Custom decoding kernel enables 1.3x speedup for attention computation. We also provide insights into model channels and attention heads during long-context inference by analyzing the learned importance distribution. Our code is available at this https URL. 

**Abstract (ZH)**: 基于学习的方法LeanK通过利用静态通道稀疏性修剪不重要的键缓存通道，实现高效的大语言模型长上下文任务处理 

---
# Balancing Information Accuracy and Response Timeliness in Networked LLMs 

**Title (ZH)**: 在网络化大语言模型中平衡信息准确性和响应及时性 

**Authors**: Yigit Turkmen, Baturalp Buyukates, Melih Bastopcu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02209)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed many fields including scientific discovery, content generation, biomedical text mining, and educational technology. However, the substantial requirements for training data, computational resources, and energy consumption pose significant challenges for their practical deployment. A promising alternative is to leverage smaller, specialized language models and aggregate their outputs to improve overall response quality. In this work, we investigate a networked LLM system composed of multiple users, a central task processor, and clusters of topic-specialized LLMs. Each user submits categorical binary (true/false) queries, which are routed by the task processor to a selected cluster of $m$ LLMs. After gathering individual responses, the processor returns a final aggregated answer to the user. We characterize both the information accuracy and response timeliness in this setting, and formulate a joint optimization problem to balance these two competing objectives. Our extensive simulations demonstrate that the aggregated responses consistently achieve higher accuracy than those of individual LLMs. Notably, this improvement is more significant when the participating LLMs exhibit similar standalone performance. 

**Abstract (ZH)**: 近期大规模语言模型的发展已经改变了包括科学发现、内容生成、生物医学文本挖掘和教育技术等多个领域。然而，大量的训练数据需求、计算资源和能源消耗导致它们的实际部署面临重大挑战。一种有前景的替代方案是利用较小的专业化语言模型并将它们的输出汇总，以提高整体响应质量。本文研究了一个由多个用户、中心任务处理器和主题专业化语言模型集群组成的网络化语言模型系统。每个用户提交分类二元（真/假）查询，由任务处理器路由到选定的$m$个语言模型集群之一。在收集个体响应后，处理器返回最终汇总的答案给用户。我们在此设置中表征信息准确性和响应及时性，并制定了一个联合优化问题以平衡这两个相互竞争的目标。我们的大量模拟表明，汇总的响应始终比单一语言模型具有更高的准确性。值得注意的是，当参与的语言模型在独立运行时表现出相似性能时，这种改进更为显著。 

---
# Proof2Hybrid: Automatic Mathematical Benchmark Synthesis for Proof-Centric Problems 

**Title (ZH)**: Proof2Hybrid: 自动数学基准合成用于以证明为中心的问题 

**Authors**: Yebo Peng, Zixiang Liu, Yaoming Li, Zhizhuo Yang, Xinye Xu, Bowen Ye, Weijun Yuan, Zihan Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02208)  

**Abstract**: Evaluating the mathematical capability of Large Language Models (LLMs) is a critical yet challenging frontier. Existing benchmarks fall short, particularly for proof-centric problems, as manual creation is unscalable and costly, leaving the true mathematical abilities of LLMs largely unassessed. To overcome these barriers, we propose Proof2Hybrid, the first fully automated framework that synthesizes high-quality, proof-centric benchmarks from natural language mathematical corpora. The key novelty of our solution is Proof2X, a roadmap of converting mathematical proofs into various kinds of questions that are easy to verify. Instructed by this roadmap, we propose a new type of hybrid-formatted questions, named ``$m$-out-of-$n$ multiple judge questions'', specifically designed to enable robust, automatic evaluation while being resilient to guessing and superficial pattern matching inherent in traditional formats. As a demonstration of our framework, we introduce AlgGeoTest, a benchmark for algebraic geometry--a frontier domain of modern mathematics--comprising 456 challenging items. Our extensive evaluations on state-of-the-art LLMs using AlgGeoTest reveal profound deficits in their comprehension of algebraic geometry, providing a more precise measure of their true mathematical capabilities. Our framework and benchmark pave the way for a new wave of in-depth research into the mathematical intelligence of AI systems. 

**Abstract (ZH)**: 评估大型语言模型的数学能力是一项关键但具有挑战性的前沿任务。现有的基准测试在这方面存在不足，尤其是在证明导向的问题上，手工创建此类基准既不具扩展性也不经济，导致大型语言模型的真正数学能力未能得到充分评估。为克服这些障碍，我们提出了一种全新的全自动框架Proof2Hybrid，该框架能够从自然语言数学语料中合成高质量的、证明导向的基准测试集。我们解决方案的核心创新是Proof2X，这是一种将数学证明转换为各种可验证问题的地图。根据此地图，我们提出了新的混合格式问题类型“$m$-out-of-$n$多位法官问题”，旨在实现稳健的自动评估，同时抵御传统格式中固有的猜测和表面模式匹配。作为我们框架的演示，我们引入了AlgGeoTest，这是一个涵盖456个挑战性问题的代数几何基准测试集，它是现代数学的一个前沿领域。通过对最新大型语言模型使用AlgGeoTest进行广泛评估，我们揭示了其在代数几何方面深刻的理解缺陷，提供了对其真实数学能力的更精确衡量。我们的框架和基准为深入研究人工智能系统的数学智能开辟了新途径。 

---
# FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation 

**Title (ZH)**: 联邦视觉-语言-动作学习：带有双门控混合专家的机器人操作联邦学习 

**Authors**: Cui Miao, Tao Chang, Meihan Wu, Hongbin Xu, Chun Li, Ming Li, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02190)  

**Abstract**: Vision-language-action (VLA) models have significantly advanced robotic manipulation by enabling robots to interpret language instructions for task execution. However, training these models often relies on large-scale user-specific data, raising concerns about privacy and security, which in turn limits their broader adoption. To address this, we propose FedVLA, the first federated VLA learning framework, enabling distributed model training that preserves data privacy without compromising performance. Our framework integrates task-aware representation learning, adaptive expert selection, and expert-driven federated aggregation, enabling efficient and privacy-preserving training of VLA models. Specifically, we introduce an Instruction Oriented Scene-Parsing mechanism, which decomposes and enhances object-level features based on task instructions, improving contextual understanding. To effectively learn diverse task patterns, we design a Dual Gating Mixture-of-Experts (DGMoE) mechanism, where not only input tokens but also self-aware experts adaptively decide their activation. Finally, we propose an Expert-Driven Aggregation strategy at the federated server, where model aggregation is guided by activated experts, ensuring effective cross-client knowledge this http URL simulations and real-world robotic experiments demonstrate the effectiveness of our proposals. Notably, DGMoE significantly improves computational efficiency compared to its vanilla counterpart, while FedVLA achieves task success rates comparable to centralized training, effectively preserving data privacy. 

**Abstract (ZH)**: 联邦Vision-语言-行动（FedVLA）学习框架：高效且隐私保护的分布式模型训练 

---
# Learning Dynamics of Meta-Learning in Small Model Pretraining 

**Title (ZH)**: 小模型预训练中元学习的学习动态 

**Authors**: David Demitri Africa, Yuval Weiss, Paula Buttery, Richard Diehl Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02189)  

**Abstract**: Large language models are powerful but costly. We ask whether meta-learning can make the pretraining of small language models not only better but also more interpretable. We integrate first-order MAML with subset-masked LM pretraining, producing four LLama-style decoder-only models (11M-570M params), and evaluate it on a fundamental NLP task with many settings and real-world applications. Compared with vanilla training, our model (i) reaches the same loss up to 1.6x sooner, (ii) improves F1 on multilingual Universal NER under equal compute, and (iii) makes the training dynamics easy to read: first the network's representations fan out ("diversify") and later they collapse into a smaller, shared subspace ("compress"). This two-stage shift shows up as a rise-and-fall in both effective-rank curves and attention-head entropy. The same curves pinpoint which layers specialise earliest and which later reconverge, giving a compact, interpretable signature of meta-adaptation. Code, checkpoints and WandB logs are released. 

**Abstract (ZH)**: 大语言模型既强大又昂贵。我们探讨元学习是否可以使小型语言模型的预训练不仅更好，而且更具可解释性。我们将一阶MAML与子集掩码LM预训练集成，生成四种Llama风格的解码器-only模型（参数量从11M到570M），并在多项基本NLP任务和实际应用中进行了评估。与常规训练相比，我们的模型：(i) 在同等计算资源下达到相同损失的时间快1.6倍；(ii) 在同等计算资源下多语言通用命名实体识别的F1值更高；(iii) 使训练动态易于理解：首先网络表示分散（“多样化”），随后压缩到一个较小的共享子空间（“压缩”）。这种两阶段转变在有效秩曲线和注意力头熵中表现为先上升后下降。相同的曲线还可以指出哪一层最早专业化，哪一层较晚收敛，从而提供一个简洁的元适应解释签名。代码、检查点和WandB日志均已发布。 

---
# GaussianCross: Cross-modal Self-supervised 3D Representation Learning via Gaussian Splatting 

**Title (ZH)**: GaussianCross: 通过高斯点绘的跨模态自监督3D表示学习 

**Authors**: Lei Yao, Yi Wang, Yi Zhang, Moyun Liu, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2508.02172)  

**Abstract**: The significance of informative and robust point representations has been widely acknowledged for 3D scene understanding. Despite existing self-supervised pre-training counterparts demonstrating promising performance, the model collapse and structural information deficiency remain prevalent due to insufficient point discrimination difficulty, yielding unreliable expressions and suboptimal performance. In this paper, we present GaussianCross, a novel cross-modal self-supervised 3D representation learning architecture integrating feed-forward 3D Gaussian Splatting (3DGS) techniques to address current challenges. GaussianCross seamlessly converts scale-inconsistent 3D point clouds into a unified cuboid-normalized Gaussian representation without missing details, enabling stable and generalizable pre-training. Subsequently, a tri-attribute adaptive distillation splatting module is incorporated to construct a 3D feature field, facilitating synergetic feature capturing of appearance, geometry, and semantic cues to maintain cross-modal consistency. To validate GaussianCross, we perform extensive evaluations on various benchmarks, including ScanNet, ScanNet200, and S3DIS. In particular, GaussianCross shows a prominent parameter and data efficiency, achieving superior performance through linear probing (<0.1% parameters) and limited data training (1% of scenes) compared to state-of-the-art methods. Furthermore, GaussianCross demonstrates strong generalization capabilities, improving the full fine-tuning accuracy by 9.3% mIoU and 6.1% AP$_{50}$ on ScanNet200 semantic and instance segmentation tasks, respectively, supporting the effectiveness of our approach. The code, weights, and visualizations are publicly available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: GaussianCross：一种结合前向3D高斯散斑技术的跨模态自监督3D表示学习架构 

---
# DreamPainter: Image Background Inpainting for E-commerce Scenarios 

**Title (ZH)**: DreamPainter: 电商场景中的图像背景修复 

**Authors**: Sijie Zhao, Jing Cheng, Yaoyao Wu, Hao Xu, Shaohui Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02155)  

**Abstract**: Although diffusion-based image genenation has been widely explored and applied, background generation tasks in e-commerce scenarios still face significant challenges. The first challenge is to ensure that the generated products are consistent with the given product inputs while maintaining a reasonable spatial arrangement, harmonious shadows, and reflections between foreground products and backgrounds. Existing inpainting methods fail to address this due to the lack of domain-specific data. The second challenge involves the limitation of relying solely on text prompts for image control, as effective integrating visual information to achieve precise control in inpainting tasks remains underexplored. To address these challenges, we introduce DreamEcom-400K, a high-quality e-commerce dataset containing accurate product instance masks, background reference images, text prompts, and aesthetically pleasing product images. Based on this dataset, we propose DreamPainter, a novel framework that not only utilizes text prompts for control but also flexibly incorporates reference image information as an additional control signal. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, maintaining high product consistency while effectively integrating both text prompt and reference image information. 

**Abstract (ZH)**: 虽然基于扩散的过程在图像生成方面的应用已经广泛探索和应用，但在电子商务场景中的背景生成任务仍然面临重大挑战。首先，生成的产品必须与给定的产品输入保持一致，同时保持合理的空间布局、和谐的阴影和前景产品与背景之间的反射。现有的修复方法由于缺乏特定领域的数据而无法解决这一问题。其次，仅依赖文本提示对图像进行控制存在局限性，因为将视觉信息有效地整合以实现精确的修复控制尚未得到充分探索。为了解决这些问题，我们引入了包含准确的产品实例掩码、背景参考图像、文本提示和美观的产品图像的高质量电子商务数据集DreamEcom-400K。基于此数据集，我们提出了DreamPainter，这是一种新颖的框架，不仅利用文本提示进行控制，还灵活地将参考图像信息作为额外的控制信号。广泛的实验证明，我们的方法在保持高产品一致性的同时，能够有效地整合文本提示和参考图像信息，显著优于现有最先进的方法。 

---
# Large-Scale Model Enabled Semantic Communication Based on Robust Knowledge Distillation 

**Title (ZH)**: 大规模模型驱动的鲁棒知识蒸馏-enable语义通信 

**Authors**: Kuiyuan DIng, Caili Guo, Yang Yang, Zhongtian Du, Walid Saad  

**Link**: [PDF](https://arxiv.org/pdf/2508.02148)  

**Abstract**: Large-scale models (LSMs) can be an effective framework for semantic representation and understanding, thereby providing a suitable tool for designing semantic communication (SC) systems. However, their direct deployment is often hindered by high computational complexity and resource requirements. In this paper, a novel robust knowledge distillation based semantic communication (RKD-SC) framework is proposed to enable efficient and \textcolor{black}{channel-noise-robust} LSM-powered SC. The framework addresses two key challenges: determining optimal compact model architectures and effectively transferring knowledge while maintaining robustness against channel noise. First, a knowledge distillation-based lightweight differentiable architecture search (KDL-DARTS) algorithm is proposed. This algorithm integrates knowledge distillation loss and a complexity penalty into the neural architecture search process to identify high-performance, lightweight semantic encoder architectures. Second, a novel two-stage robust knowledge distillation (RKD) algorithm is developed to transfer semantic capabilities from an LSM (teacher) to a compact encoder (student) and subsequently enhance system robustness. To further improve resilience to channel impairments, a channel-aware transformer (CAT) block is introduced as the channel codec, trained under diverse channel conditions with variable-length outputs. Extensive simulations on image classification tasks demonstrate that the RKD-SC framework significantly reduces model parameters while preserving a high degree of the teacher model's performance and exhibiting superior robustness compared to existing methods. 

**Abstract (ZH)**: 基于鲁棒知识蒸馏的大规模模型驱动的语义通信框架（RKD-SC） 

---
# Fitness aligned structural modeling enables scalable virtual screening with AuroBind 

**Title (ZH)**: 基于适应度对齐的结构建模使AuroBind能够实现可扩展的虚拟筛选 

**Authors**: Zhongyue Zhang, Jiahua Rao, Jie Zhong, Weiqiang Bai, Dongxue Wang, Shaobo Ning, Lifeng Qiao, Sheng Xu, Runze Ma, Will Hua, Jack Xiaoyu Chen, Odin Zhang, Wei Lu, Hanyi Feng, He Yang, Xinchao Shi, Rui Li, Wanli Ouyang, Xinzhu Ma, Jiahao Wang, Jixian Zhang, Jia Duan, Siqi Sun, Jian Zhang, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02137)  

**Abstract**: Most human proteins remain undrugged, over 96% of human proteins remain unexploited by approved therapeutics. While structure-based virtual screening promises to expand the druggable proteome, existing methods lack atomic-level precision and fail to predict binding fitness, limiting translational impact. We present AuroBind, a scalable virtual screening framework that fine-tunes a custom atomic-level structural model on million-scale chemogenomic data. AuroBind integrates direct preference optimization, self-distillation from high-confidence complexes, and a teacher-student acceleration strategy to jointly predict ligand-bound structures and binding fitness. The proposed models outperform state-of-the-art models on structural and functional benchmarks while enabling 100,000-fold faster screening across ultra-large compound libraries. In a prospective screen across ten disease-relevant targets, AuroBind achieved experimental hit rates of 7-69%, with top compounds reaching sub-nanomolar to picomolar potency. For the orphan GPCRs GPR151 and GPR160, AuroBind identified both agonists and antagonists with success rates of 16-30%, and functional assays confirmed GPR160 modulation in liver and prostate cancer models. AuroBind offers a generalizable framework for structure-function learning and high-throughput molecular screening, bridging the gap between structure prediction and therapeutic discovery. 

**Abstract (ZH)**: 大多数人类蛋白质未被药物化，超过96%的人类蛋白质未被获批的治疗药物所利用。尽管基于结构的虚拟筛选有望扩大可药物化蛋白质组，现有方法缺乏原子级精度，无法预测配体结合亲和力，限制了其临床转化效果。我们提出了一种名为AuroBind的可扩展虚拟筛选框架，该框架在一个百万尺度的化学生物学数据集上对定制的原子级结构模型进行了微调。AuroBind结合了直接偏好优化、高置信度复合物的自蒸馏以及教师-学生加速策略，以联合预测配体结合结构和结合亲和力。所提出的模型在结构和功能基准测试中优于现有模型，同时能够对超大型化合物库进行高达100,000倍速度的筛选。在针对十个疾病相关靶点的前瞻性筛选中，AuroBind获得了7-69%的实验阳性率，其中最佳化合物达到次纳摩到皮摩的效力。对于_ORphan_ GPCRs GPR151和GPR160，AuroBind分别以16-30%的成功率发现了激动剂和拮抗剂，并且功能检测证实了GPR160在肝癌和前列腺癌模型中的调节作用。AuroBind提供了一种可推广的结构-功能学习框架和高通量分子筛选框架，缩短了结构预测与治疗发现之间的差距。 

---
# The Complexity of Extreme Climate Events on the New Zealand's Kiwifruit Industry 

**Title (ZH)**: 极端气候事件对新西兰猕猴桃行业的复杂性影响 

**Authors**: Boyuan Zheng, Victor W. Chu, Zhidong Li, Evan Webster, Ashley Rootsey  

**Link**: [PDF](https://arxiv.org/pdf/2508.02130)  

**Abstract**: Climate change has intensified the frequency and severity of extreme weather events, presenting unprecedented challenges to the agricultural industry worldwide. In this investigation, we focus on kiwifruit farming in New Zealand. We propose to examine the impacts of climate-induced extreme events, specifically frost, drought, extreme rainfall, and heatwave, on kiwifruit harvest yields. These four events were selected due to their significant impacts on crop productivity and their prevalence as recorded by climate monitoring institutions in the country. We employed Isolation Forest, an unsupervised anomaly detection method, to analyse climate history and recorded extreme events, alongside with kiwifruit yields. Our analysis reveals considerable variability in how different types of extreme event affect kiwifruit yields underscoring notable discrepancies between climatic extremes and individual farm's yield outcomes. Additionally, our study highlights critical limitations of current anomaly detection approaches, particularly in accurately identifying events such as frost. These findings emphasise the need for integrating supplementary features like farm management strategies with climate adaptation practices. Our further investigation will employ ensemble methods that consolidate nearby farms' yield data and regional climate station features to reduce variance, thereby enhancing the accuracy and reliability of extreme event detection and the formulation of response strategies. 

**Abstract (ZH)**: 气候变化加剧了极端天气事件的频率和 severity，给全球农业行业带来了前所未有的挑战。本研究以新西兰奇异果农场为例，旨在考察由气候变迁引发的极端事件，包括霜冻、干旱、极端降雨和热浪，对奇异果收获产量的影响。四种事件因其对作物产量的显著影响以及作为气象监测机构记录的常见事件而被选作研究对象。研究使用孤立森林算法，一种无监督异常检测方法，分析气候历史、记录的极端事件以及奇异果产量数据。研究发现，不同类型的极端事件对奇异果产量的影响存在显著差异，揭示了气候极端事件与单个农场产量结果之间的重大差异。此外，研究还突显了当前异常检测方法的关键局限性，特别是在准确识别霜冻等事件方面。这些发现强调了需要结合农场管理策略与气候适应实践的重要性。进一步的研究将采用集成方法，整合附近农场的产量数据和区域气象站特征，以减少变异，从而提高极端事件检测的准确性和可靠性，以及响应策略的制定。 

---
# Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models 

**Title (ZH)**: 琥珀剪枝器：利用N:M激活稀疏性提高大型语言模型预填充效率 

**Authors**: Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02128)  

**Abstract**: In the era of large language models (LLMs), N:M sparsity has emerged as a structured compression technique critical for accelerating inference. While prior work has primarily focused on weight sparsity, it often suffers from significant accuracy degradation. Activation sparsity, though promising, is typically training-dependent and faces challenges in generalization. To address these limitations, we introduce Amber Pruner, a training-free N:M activation sparsity method designed specifically for the prefill stage, targeting the acceleration of linear projection layers in LLMs. Extensive experiments across multiple models and sparsity ratios (2:4, 4:8, and 8:16) demonstrate that Amber Pruner can effectively sparsify and accelerate more than 55% of linear computations without requiring model retraining. To further enhance generality and efficiency, we propose Outstanding-sparse, a unified framework that integrates Amber Pruner with post-training W8A8 quantization. Our approach preserves strong performance across a range of downstream tasks, with notable advantages in generative tasks. This work pioneers a new frontier in activation sparsity, providing foundational insights that are poised to guide the co-evolution of algorithms and architectures in the design of next-generation AI systems. 

**Abstract (ZH)**: 在大规模语言模型时代，N:M稀疏性作为一种结构化压缩技术，已成为加速推断的关键技术。尽管早期工作主要关注权重稀疏性，但它常常导致显著的精度下降。激活稀疏性虽然有前景，但通常依赖于训练且在泛化方面面临挑战。为解决这些局限性，我们引入了Amber Pruner，这是一种无需训练的N:M激活稀疏性方法，专门针对预填充阶段，旨在加速大规模语言模型中的线性投影层。通过在多个模型和不同稀疏比（2:4、4:8和8:16）下进行广泛实验，证明Amber Pruner可以在无需模型重新训练的情况下有效地稀疏化和加速超过55%的线性计算。为进一步增强通用性和效率，我们提出了一个统一框架Outstanding-sparse，该框架将Amber Pruner与后训练的W8A8量化相结合。我们的方法在各种下游任务中保持了强大的性能，在生成任务方面尤为突出。这项工作为激活稀疏性开辟了新前沿，提供了基础见解，有助于指导下一代AI系统中算法和架构的共同进化。 

---
# Coward: Toward Practical Proactive Federated Backdoor Defense via Collision-based Watermark 

**Title (ZH)**: Coward: 基于碰撞标记的实用主动联邦后门防御方法 

**Authors**: Wenjie Li, Siying Gu, Yiming Li, Kangjie Chen, Zhili Chen, Tianwei Zhang, Shu-Tao Xia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02115)  

**Abstract**: Backdoor detection is currently the mainstream defense against backdoor attacks in federated learning (FL), where malicious clients upload poisoned updates that compromise the global model and undermine the reliability of FL deployments. Existing backdoor detection techniques fall into two categories, including passive and proactive ones, depending on whether the server proactively modifies the global model. However, both have inherent limitations in practice: passive defenses are vulnerable to common non-i.i.d. data distributions and random participation of FL clients, whereas current proactive defenses suffer inevitable out-of-distribution (OOD) bias because they rely on backdoor co-existence effects. To address these issues, we introduce a new proactive defense, dubbed Coward, inspired by our discovery of multi-backdoor collision effects, in which consecutively planted, distinct backdoors significantly suppress earlier ones. In general, we detect attackers by evaluating whether the server-injected, conflicting global watermark is erased during local training rather than retained. Our method preserves the advantages of proactive defenses in handling data heterogeneity (\ie, non-i.i.d. data) while mitigating the adverse impact of OOD bias through a revised detection mechanism. Extensive experiments on benchmark datasets confirm the effectiveness of Coward and its resilience to potential adaptive attacks. The code for our method would be available at this https URL. 

**Abstract (ZH)**: 基于多重后门碰撞效果的Coward主动防御机制：缓解联邦学习中的后门攻击 

---
# Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches 

**Title (ZH)**: 基于经典方法与大语言模型驱动方法的系统评价：对话式推荐系统中用户体验的评估 

**Authors**: Raj Mahmud, Yufeng Wu, Abdullah Bin Sawad, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02096)  

**Abstract**: Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM.
Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices. 

**Abstract (ZH)**: 会话推荐系统（CRSs）在各领域内受到越来越多的研究关注，但其用户体验（UX）评价仍显不足。现有的综述大多忽略了实证UX研究，尤其是在适应性和大型语言模型（LLM）基础上的CRSs。为弥补这一不足，我们遵循PRISMA指南，系统综述了2017年至2025年间发表的23项实证研究。我们分析了UX在不同领域、适应性和LLM基础上的概念化、测量和形成方式。我们的研究发现表明存在持续性的局限性：事后调查主导，对话层级的情感UX结构鲜有评估，适应行为与UX结果的联系也不常见。基于LLM的CRS引入了额外的挑战，包括知识不透明性和冗余性，但评价这些挑战的频率较低。我们贡献了一个结构化的UX度量综合分析、适应性和非适应性系统的比较分析以及对LLM感知的UX评价的前瞻议程。这些发现支持发展更透明、更具吸引力和以用户为中心的CRS评价实践。 

---
# VLM4D: Towards Spatiotemporal Awareness in Vision Language Models 

**Title (ZH)**: VLM4D: 向视知觉语言模型的空间 temporal 时间aware性迈进 

**Authors**: Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, Achuta Kadambi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02095)  

**Abstract**: Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts-abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capabilities of VLMs. Our benchmark comprises diverse real-world and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-the-art open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs' spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments. 

**Abstract (ZH)**: Vision语言模型（VLMs）在整合语言和视觉推理方面表现出色，但在理解动态时空交互方面仍存在根本性限制。人类能够轻松追踪和推理物体的运动、旋转和视角变化——这些能力对于实现稳健的动态现实理解至关重要，而在当前的VLMs中却明显缺乏。在本文中，我们介绍了VLM4D，这是首个专门用于评估VLMs时空推理能力的基准。我们的基准包含多样化的现实世界和合成视频，并附带了精心设计的问题-答案对，强调了平移和旋转运动、视角意识以及运动连续性。通过全面评估最先进的开源和闭源VLMs，我们发现与人类基线相比存在显著性能差距，突显了现有模型的基本缺陷。广泛分析表明，VLMs特别难以整合多种视觉线索并保持时间连贯性。我们进一步探讨了一些有前景的方向，如利用四维特征场重建和针对性的时空监督微调，证明了这些方法在增强时空理解方面的有效性。我们的工作旨在鼓励更深入地探索改进VLMs的空间和时间定位，为动态环境中的更强大和可靠的视觉智能铺平道路。 

---
# FPEdit: Robust LLM Fingerprinting through Localized Knowledge Editing 

**Title (ZH)**: FPEdit：通过局部知识编辑实现的鲁棒LLM指纹识别 

**Authors**: Shida Wang, Chaohu Liu, Yubo Wang, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02092)  

**Abstract**: Large language models represent significant investments in computation, data, and engineering expertise, making them extraordinarily valuable intellectual assets. Nevertheless, these AI assets remain vulnerable to unauthorized redistribution and commercial exploitation through fine-tuning or black-box deployment. Current fingerprinting approaches face a fundamental trade-off: intrinsic methods require full parameter access, while backdoor-based techniques employ statistically anomalous triggers easily detected and filtered by adversaries. To address these limitations, we introduce FPEdit, a novel knowledge-editing framework that injects semantically coherent natural language fingerprints by modifying a sparse subset of model weights. This ensures stealthy and precise ownership encoding without degrading the core functionality. Extensive experiments show that FPEdit achieves $95$-$100\%$ fingerprint retention under both full-parameter fine-tuning and parameter-efficient adaptation, while preserving performance on 24 downstream benchmarks. Moreover, FPEdit remains robust under quantization, pruning, and stochastic decoding, and can embed 10 fingerprint pairs into LLaMA2-7B in under 10 minutes using less than 32 GB of GPU memory, a $70\%$ reduction in resource requirements compared to existing techniques. These advances establish FPEdit as the first fingerprinting approach to simultaneously achieve robustness against adaptation, resistance to detection, and preservation of model utility, providing a minimally invasive solution for reliable provenance verification of large language models in adversarial deployment scenarios. 

**Abstract (ZH)**: 大规模语言模型在计算、数据和工程专业知识方面投入巨大，是极其宝贵的智力资产。然而，这些AI资产仍然容易通过微调或黑盒部署被未经授权重分发和商业化利用。当前的指纹识别方法面临一个基本的权衡：内在方法需要全参数访问，而后门基方法使用的统计异常触发器容易被对手检测和过滤。为解决这些局限性，我们提出了FPEdit，这是一种新的知识编辑框架，通过修改模型权重的一个稀疏子集注入语义连贯的自然语言指纹。这确保了隐蔽且精确的所有权编码，同时不损害核心功能。实验表明，FPEdit在全参数微调和参数高效适应下分别实现了95%-100%的指纹保留率，同时在24个下游基准上保持性能。此外，FPEdit在量化、剪枝和随机解码下依然稳健，并能在不到10分钟内使用不到32 GB的GPU内存将10对指纹嵌入到LLaMA2-7B中，与现有技术相比资源需求减少了70%。这些进展使FPEdit成为首个同时实现适应鲁棒性、抗检测性和模型实用性保存的指纹识别方法，为对抗部署场景中的大型语言模型可靠起源验证提供了微创解决方案。 

---
# CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search 

**Title (ZH)**: CRINN: 对比增强学习近邻搜索近似算法 

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Chris Shum, Jiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02091)  

**Abstract**: Approximate nearest-neighbor search (ANNS) algorithms have become increasingly critical for recent AI applications, particularly in retrieval-augmented generation (RAG) and agent-based LLM applications. In this paper, we present CRINN, a new paradigm for ANNS algorithms. CRINN treats ANNS optimization as a reinforcement learning problem where execution speed serves as the reward signal. This approach enables the automatic generation of progressively faster ANNS implementations while maintaining accuracy constraints. Our experimental evaluation demonstrates CRINN's effectiveness across six widely-used NNS benchmark datasets. When compared against state-of-the-art open-source ANNS algorithms, CRINN achieves best performance on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular), and tied for first place on two of them (SIFT-128-Euclidean and GloVe-25-angular). The implications of CRINN's success reach well beyond ANNS optimization: It validates that LLMs augmented with reinforcement learning can function as an effective tool for automating sophisticated algorithmic optimizations that demand specialized knowledge and labor-intensive manual this http URL can be found at this https URL 

**Abstract (ZH)**: 近邻搜索(CRINN):一种新的近邻搜索算法范式 

---
# SSBD Ontology: A Two-Tier Approach for Interoperable Bioimaging Metadata 

**Title (ZH)**: SSBD本体：一种可互操作生物成像元数据的二层方法 

**Authors**: Yuki Yamagata, Koji Kyoda, Hiroya Itoga, Emi Fujisawa, Shuichi Onami  

**Link**: [PDF](https://arxiv.org/pdf/2508.02084)  

**Abstract**: Advanced bioimaging technologies have enabled the large-scale acquisition of multidimensional data, yet effective metadata management and interoperability remain significant challenges. To address these issues, we propose a new ontology-driven framework for the Systems Science of Biological Dynamics Database (SSBD) that adopts a two-tier architecture. The core layer provides a class-centric structure referencing existing biomedical ontologies, supporting both SSBD:repository -- which focuses on rapid dataset publication with minimal metadata -- and SSBD:database, which is enhanced with biological and imaging-related annotations. Meanwhile, the instance layer represents actual imaging dataset information as Resource Description Framework individuals that are explicitly linked to the core classes. This layered approach aligns flexible instance data with robust ontological classes, enabling seamless integration and advanced semantic queries. By coupling flexibility with rigor, the SSBD Ontology promotes interoperability, data reuse, and the discovery of novel biological mechanisms. Moreover, our solution aligns with the Recommended Metadata for Biological Images guidelines and fosters compatibility. Ultimately, our approach contributes to establishing a Findable, Accessible, Interoperable, and Reusable data ecosystem within the bioimaging community. 

**Abstract (ZH)**: 先进生物成像技术使得大规模获取多维数据成为可能，然而有效的元数据管理和互操作性仍然面临重大挑战。为应对这些问题，我们提出了一种新的本体驱动框架，用于生物动力学数据库系统科学（SSBD），采用两层架构。核心层提供以类为中心的结构，参考现有的生物医学本体，支持SSBD:repository——专注于快速数据集发布，最少元数据——以及SSBD:database，后者增加了生物学和成像相关的注释。与此同时，实例层将实际的成像数据集信息表示为与核心类明确链接的资源描述框架个体。这种分层方法使灵活的实例数据与坚固的本体类保持一致，从而实现无缝集成和高级语义查询。通过结合灵活性和严谨性，SSBD本体促进了互操作性、数据重用和新型生物机制的发现。此外，我们的解决方案符合生物图像推荐元数据指南，促进了兼容性。最终，我们的方法有助于建立生物成像社区内的可查找、可访问、可互操作和可重用的数据生态系统。 

---
# AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization 

**Title (ZH)**: AlignGuard-LoRA: 基于Fisher引导分解和黎曼测地线碰撞正则化的对齐保护微调 

**Authors**: Amitava Das, Abhilekh Borah, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02079)  

**Abstract**: Low-rank adaptation (LoRA) has become a standard tool for efficiently fine-tuning large language models (LLMs). Yet, even minor LoRA updates can induce alignment drift, weakening safety and behavioral constraints through entangled parameter changes. To address this, we propose AlignGuard-LoRA (AGL), a principled framework for preserving alignment during finetuning. AGL introduces several key components: a primary task loss for supervision, Fisher Information Matrix-based regularization to restrict updates in alignment-sensitive subspaces, and task-specific regularization to stabilize the integration of new knowledge. We further introduce collision-aware regularization, blending Riemannian overlap -- which penalizes coordinate-wise interference -- and geodesic separation -- which encourages disjoint update geometry. We curate DriftCaps, a targeted diagnostic benchmark of safe and unsafe prompts designed to quantify alignment drift and safety degradation. Empirical evaluations show that AGL mitigates alignment drift by up to 50% on safety-critical benchmarks without degrading downstream task performance. Comprehensive ablation confirms that each component contributes distinctly to preserving latent safety behaviors. Finally, we derive and validate a scaling law for catastrophic forgetting, revealing that AGL flattens post-finetuning loss escalation while preserving adaptation dynamics. AGL is a structurally grounded refinement of LoRA, ensuring alignment preservation with minimal trade-offs. To encourage further exploration and development, we open-source our implementation. 

**Abstract (ZH)**: AlignGuard-LoRA: 一种在微调过程中保持一致性的原理性框架 

---
# SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration 

**Title (ZH)**: SpikeSTAG：基于GNN-SNN协作的时空预测 

**Authors**: Bang Hu, Changze Lv, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Wei cao, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02069)  

**Abstract**: Spiking neural networks (SNNs), inspired by the spiking behavior of biological neurons, offer a distinctive approach for capturing the complexities of temporal data. However, their potential for spatial modeling in multivariate time-series forecasting remains largely unexplored. To bridge this gap, we introduce a brand new SNN architecture, which is among the first to seamlessly integrate graph structural learning with spike-based temporal processing for multivariate time-series forecasting. Specifically, we first embed time features and an adaptive matrix, eliminating the need for predefined graph structures. We then further learn sequence features through the Observation (OBS) Block. Building upon this, our Multi-Scale Spike Aggregation (MSSA) hierarchically aggregates neighborhood information through spiking SAGE layers, enabling multi-hop feature extraction while eliminating the need for floating-point operations. Finally, we propose a Dual-Path Spike Fusion (DSF) Block to integrate spatial graph features and temporal dynamics via a spike-gated mechanism, combining LSTM-processed sequences with spiking self-attention outputs, effectively improve the model accuracy of long sequence datasets. Experiments show that our model surpasses the state-of-the-art SNN-based iSpikformer on all datasets and outperforms traditional temporal models at long horizons, thereby establishing a new paradigm for efficient spatial-temporal modeling. 

**Abstract (ZH)**: 基于尖峰神经网络的多尺度尖峰聚合与双路径尖峰融合的时空模型 

---
# MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs 

**Title (ZH)**: MolReasoner: 向有效的可解释分子LLMs推理方向努力 

**Authors**: Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02066)  

**Abstract**: Large Language Models(LLMs) have demonstrated remarkable performance across various domains, yet their capabilities in molecular reasoning remain insufficiently explored. Current approaches tend to rely heavily on general-purpose prompting, which lacks domain-specific molecular semantics, while those that use fine-tuning strategies often face challenges with interpretability and reasoning depth. To address these issues, we introduce MolReasoner, a two-stage framework designed to transition LLMs from memorization towards chemical reasoning. First, we propose Mol-SFT, which initializes the model's reasoning abilities via synthetic Chain-of-Thought(CoT) samples generated by GPT-4o and verified for chemical accuracy. Subsequently, Mol-RL applies reinforcement learning with specialized reward functions designed explicitly to align chemical structures with linguistic descriptions, thereby enhancing molecular reasoning capabilities. Our approach notably enhances interpretability, improving the model 's molecular understanding and enabling better generalization. Extensive experiments demonstrate that MolReasoner outperforms existing methods, and marking a significant shift from memorization-based outputs to robust chemical reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域显示出了卓越的表现，但在分子推理方面的能力尚未得到充分探索。现有的方法通常依赖于通用提示，缺乏专门的分子语义，而使用微调策略的方法往往面临解释性和推理深度的挑战。为了解决这些问题，我们引入了MolReasoner，这是一个两阶段框架，旨在引导LLMs从记忆向化学推理过渡。首先，我们提出了Mol-SFT，通过由GPT-4o生成并经过化学准确性验证的合成Chain-of-Thought（CoT）样本，初始化模型的推理能力。随后，Mol-RL应用了强化学习，并设计了专门的奖励函数，以明确地将化学结构与语言描述对齐，从而增强分子推理能力。我们的方法显著提高了模型的可解释性，提高了其对分子的理解能力，并促进了更好的推广。广泛实验表明，MolReasoner优于现有方法，并标志着从基于记忆的输出向稳健的化学推理的显著转变。 

---
# RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models 

**Title (ZH)**: RICL：向预训练的视觉-语言-动作模型添加上下文适配能力 

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, Insup Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02062)  

**Abstract**: Multi-task ``vision-language-action'' (VLA) models have recently demonstrated increasing promise as generalist foundation models for robotics, achieving non-trivial performance out of the box on new tasks in new environments. However, for such models to be truly useful, an end user must have easy means to teach them to improve. For language and vision models, the emergent ability to perform in-context learning (ICL) has proven to be a versatile and highly useful interface to easily teach new tasks with no parameter finetuning. Unfortunately, VLAs pre-trained with imitation learning objectives do not naturally acquire ICL abilities. In this paper, we demonstrate that, with the right finetuning recipe and a small robot demonstration dataset, it is possible to inject in-context adaptability post hoc into such a VLA. After retraining for in-context learning (RICL), our system permits an end user to provide a small number (10-20) of demonstrations for a new task. RICL then fetches the most relevant portions of those demonstrations into the VLA context to exploit ICL, performing the new task and boosting task performance. We apply RICL to inject ICL into the $\pi_{0}$-FAST VLA, and show that it permits large in-context improvements for a variety of new manipulation tasks with only 20 demonstrations per task, without any parameter updates. When parameter updates on the target task demonstrations is possible, RICL finetuning further boosts performance. We release code and model weights for RICL-$\pi_{0}$-FAST alongside the paper to enable, for the first time, a simple in-context learning interface for new manipulation tasks. Website: this https URL. 

**Abstract (ZH)**: 多任务“视觉-语言-行动”（VLA）模型最近被证明是机器人领域通用基础模型的有前景选择，能够在新环境中以初步训练的非平凡性能解决新任务。然而，要使这些模型真正有用，终端用户需要轻松的方法来教它们改进。针对语言和视觉模型，涌现的在上下文学习（ICL）能力已被证明是一种灵活且高度有用的接口，可以轻松地教授新任务而无需参数微调。不幸的是，通过模仿学习目标预训练的VLA并不会自然获得ICL能力。在本文中，我们证明，在合适的微调配方和小型机器人演示数据集的帮助下，可以在VLA中后插式地注入ICL能力。经过针对ICL的重训练后，我们的系统允许终端用户为新任务提供少量（10-20个）演示。通过ICL检索到这些演示中最相关的部分，以利用ICL执行新任务并提升任务性能。我们将ICL应用于注入ICL到$\pi_{0}$-FAST VLA中，并展示它仅使用每任务20个演示便可以为各种新的操作任务带来显著的在上下文改进，且无需任何参数更新。当可以对目标任务演示进行参数更新时，ICL微调会进一步提升性能。我们将在论文中发布RICL-$\pi_{0}$-FAST的代码和模型权重，以首次实现为新操作任务提供简单的在上下文学习接口。 

---
# Enhancement of Quantum Semi-Supervised Learning via Improved Laplacian and Poisson Methods 

**Title (ZH)**: 改进拉普拉斯和泊松方法增强量子半监督学习 

**Authors**: Hamed Gholipour, Farid Bozorgnia, Hamzeh Mohammadigheymasi, Kailash Hambarde, Javier Mancilla, Hugo Proenca, Joao Neves, Moharram Challenger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02054)  

**Abstract**: This paper develops a hybrid quantum approach for graph-based semi-supervised learning to enhance performance in scenarios where labeled data is scarce. We introduce two enhanced quantum models, the Improved Laplacian Quantum Semi-Supervised Learning (ILQSSL) and the Improved Poisson Quantum Semi-Supervised Learning (IPQSSL), that incorporate advanced label propagation strategies within variational quantum circuits. These models utilize QR decomposition to embed graph structure directly into quantum states, thereby enabling more effective learning in low-label settings. We validate our methods across four benchmark datasets like Iris, Wine, Heart Disease, and German Credit Card -- and show that both ILQSSL and IPQSSL consistently outperform leading classical semi-supervised learning algorithms, particularly under limited supervision. Beyond standard performance metrics, we examine the effect of circuit depth and qubit count on learning quality by analyzing entanglement entropy and Randomized Benchmarking (RB). Our results suggest that while some level of entanglement improves the model's ability to generalize, increased circuit complexity may introduce noise that undermines performance on current quantum hardware. Overall, the study highlights the potential of quantum-enhanced models for semi-supervised learning, offering practical insights into how quantum circuits can be designed to balance expressivity and stability. These findings support the role of quantum machine learning in advancing data-efficient classification, especially in applications constrained by label availability and hardware limitations. 

**Abstract (ZH)**: 一种基于图的半监督学习的混合量子方法：在标注数据稀少场景中提升性能 

---
# Epi$^2$-Net: Advancing Epidemic Dynamics Forecasting with Physics-Inspired Neural Networks 

**Title (ZH)**: Epi$^2$-Net：基于物理启发的神经网络推动传染病动力学预测 

**Authors**: Rui Sun, Chenghua Gong, Tianjun Gu, Yuhao Zheng, Jie Ding, Juyuan Zhang, Liming Pan, Linyuan Lü  

**Link**: [PDF](https://arxiv.org/pdf/2508.02049)  

**Abstract**: Advancing epidemic dynamics forecasting is vital for targeted interventions and safeguarding public health. Current approaches mainly fall into two categories: mechanism-based and data-driven models. Mechanism-based models are constrained by predefined compartmental structures and oversimplified system assumptions, limiting their ability to model complex real-world dynamics, while data-driven models focus solely on intrinsic data dependencies without physical or epidemiological constraints, risking biased or misleading representations. Although recent studies have attempted to integrate epidemiological knowledge into neural architectures, most of them fail to reconcile explicit physical priors with neural representations. To overcome these obstacles, we introduce Epi$^2$-Net, a Epidemic Forecasting Framework built upon Physics-Inspired Neural Networks. Specifically, we propose reconceptualizing epidemic transmission from the physical transport perspective, introducing the concept of neural epidemic transport. Further, we present a physic-inspired deep learning framework, and integrate physical constraints with neural modules to model spatio-temporal patterns of epidemic dynamics. Experiments on real-world datasets have demonstrated that Epi$^2$-Net outperforms state-of-the-art methods in epidemic forecasting, providing a promising solution for future epidemic containment. The code is available at: this https URL. 

**Abstract (ZH)**: 推进流行病动态预测对于针对性干预和保障公共卫生至关重要。当前的方法主要分为两类：机制基础模型和数据驱动模型。机制基础模型受限于预设的隔室结构和过于简化的系统假设，限制了其对复杂现实世界动态的建模能力，而数据驱动模型仅关注内在数据依赖性，缺乏物理或流行病学约束，可能导致偏见或误导性的表示。尽管最近的研究尝试将流行病学知识融入神经网络架构中，但大多数方法未能解决显式物理先验与神经表示之间的契合问题。为克服这些障碍，我们引入了Epi$^2$-Net，这是一种基于物理启发神经网络的流行病预测框架。具体而言，我们从物理传输的角度重新概念化流行病传播，引入了神经流行病传输的概念。进一步地，我们提出了一种基于物理启发的深度学习框架，将物理约束与神经模块集成，用于建模流行病动态的时空模式。实验结果表明，Epi$^2$-Net在流行病预测中优于现有方法，为未来的流行病控制提供了有希望的解决方案。代码可在此处访问：this https URL。 

---
# Graph Unlearning via Embedding Reconstruction -- A Range-Null Space Decomposition Approach 

**Title (ZH)**: 基于范围Null空间分解的图遗忘重构方法 

**Authors**: Hang Yin, Zipeng Liu, Xiaoyong Peng, Liyao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02044)  

**Abstract**: Graph unlearning is tailored for GNNs to handle widespread and various graph structure unlearning requests, which remain largely unexplored. The GIF (graph influence function) achieves validity under partial edge unlearning, but faces challenges in dealing with more disturbing node unlearning. To avoid the overhead of retraining and realize the model utility of unlearning, we proposed a novel node unlearning method to reverse the process of aggregation in GNN by embedding reconstruction and to adopt Range-Null Space Decomposition for the nodes' interaction learning. Experimental results on multiple representative datasets demonstrate the SOTA performance of our proposed approach. 

**Abstract (ZH)**: 图去学习针对GNNs设计以处理广泛且多样的图结构去学习请求，这些请求目前尚未充分探索。 

---
# Diagnosing Memorization in Chain-of-Thought Reasoning, One Token at a Time 

**Title (ZH)**: 逐词诊断链式推理中的记忆化现象 

**Authors**: Huihan Li, You Chen, Siyuan Wang, Yixin He, Ninareh Mehrabi, Rahul Gupta, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.02037)  

**Abstract**: Large Language Models (LLMs) perform well on reasoning benchmarks but often fail when inputs alter slightly, raising concerns about the extent to which their success relies on memorization. This issue is especially acute in Chain-of-Thought (CoT) reasoning, where spurious memorized patterns can trigger intermediate errors that cascade into incorrect final answers. We introduce STIM, a novel framework for Source-aware Token-level Identification of Memorization, which attributes each token in a reasoning chain to one of multiple memorization sources - local, mid-range, or long-range - based on their statistical co-occurrence with the token in the pretraining corpus. Our token-level analysis across tasks and distributional settings reveals that models rely more on memorization in complex or long-tail cases, and that local memorization is often the dominant driver of errors, leading to up to 67% of wrong tokens. We also show that memorization scores from STIM can be effective in predicting the wrong tokens in the wrong reasoning step. STIM offers a powerful tool for diagnosing and improving model reasoning and can generalize to other structured step-wise generation tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理基准测试中表现良好，但在输入稍有改变时往往会出现失败，这引发了对其成功程度依赖于记忆化的担忧。这一问题在链式推理（Chain-of-Thought, CoT）中尤为严重，因为无用的记忆化模式可能会触发中间错误，进而导致最终答案错误。我们提出了STIM框架，这是一种基于源的标记级别记忆化识别框架，根据预训练语料中标记的统计共现情况，将推理链中的每个标记归因于多种记忆化来源之一——局部、中程或远程。我们在不同任务和分布设置下的标记级别分析表明，模型在复杂或长尾情况下更多地依赖于记忆化，且局部记忆化往往是错误的主导驱动因素，可能导致高达67%的错误标记。此外，我们还展示了STIM的记忆化分数在预测错误推理步骤中的错误标记方面具有有效性。STIM提供了一种强大的诊断工具，可以改进模型的推理能力，并能适用于其他结构化步骤生成任务。 

---
# Confidence-Diversity Calibration of AI Judgement Enables Reliable Qualitative Coding 

**Title (ZH)**: AI判断的置信-多样性校准 enables 可靠的定性编码 

**Authors**: Zhilong Zhao, Yindi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02029)  

**Abstract**: LLMs enable qualitative coding at large scale, but assessing the reliability of their output remains challenging in domains where human experts seldom agree. Analysing 5,680 coding decisions from eight state-of-the-art LLMs across ten thematic categories, we confirm that a model's mean self-confidence already tracks inter-model agreement closely (Pearson r=0.82). Adding model diversity-quantified as the normalised Shannon entropy of the panel's votes-turns this single cue into a dual signal that explains agreement almost completely (R^2=0.979). The confidence-diversity duo enables a three-tier workflow that auto-accepts 35% of segments with <5% audit-detected error and routes the remainder for targeted human review, cutting manual effort by up to 65%. Cross-domain replication on six public datasets spanning finance, medicine, law and multilingual tasks confirms these gains (kappa improvements of 0.20-0.78). Our results establish a generalisable, evidence-based criterion for calibrating AI judgement in qualitative research. 

**Abstract (ZH)**: 大型语言模型能够在大规模范围内实现 qualitative 代码化，但在人类专家很少达成一致的领域，评估其输出的可靠性仍具挑战性。通过对八种先进大型语言模型在十个主题类别中的5,680个编码决策进行分析，我们确认模型的平均自我信心已与模型间一致性密切相关（皮尔逊相关系数r=0.82）。通过量化模型多样性——即面板投票的正规化香农熵——这单一线索转化为几乎完全解释一致性的双信号（R²=0.979）。自我信心-多样性 duo 使得一个三层次的工作流成为可能，自动接受<5% 审计检测错误的片段，其余部分则定向进行人类审查，最多可减少65%的手动努力。跨领域在涵盖金融、医学、法律和多语言任务的六个公共数据集上进行复制，证实了这些收益（κ改进值为0.20-0.78）。研究结果为定性研究中 AI 判断的校准建立了一般适用且基于证据的标准。 

---
# SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models 

**Title (ZH)**: SpeechR：面向大规模音视语言模型的语音推理基准 

**Authors**: Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02018)  

**Abstract**: Large audio-language models (LALMs) have achieved near-human performance in sentence-level transcription and emotion recognition. However, existing evaluations focus mainly on surface-level perception, leaving the capacity of models for contextual and inference-driven reasoning in speech-based scenarios insufficiently examined. To address this gap, we introduce SpeechR, a unified benchmark for evaluating reasoning over speech in large audio-language models. SpeechR evaluates models along three key dimensions: factual retrieval, procedural inference, and normative judgment. It includes three distinct evaluation formats. The multiple-choice version measures answer selection accuracy. The generative version assesses the coherence and logical consistency of reasoning chains. The acoustic-feature version investigates whether variations in stress and emotion affect reasoning performance. Evaluations on eleven state-of-the-art LALMs reveal that high transcription accuracy does not translate into strong reasoning capabilities. SpeechR establishes a structured benchmark for evaluating reasoning in spoken language, enabling more targeted analysis of model capabilities across diverse dialogue-based tasks. 

**Abstract (ZH)**: 大型音频语言模型（LALMs）已在句级转写和情绪识别方面实现了近乎人类的性能。然而，现有评估主要集中在表层感知上，使得模型在基于语音的情景中进行上下文和推理驱动的推理能力不足。为弥补这一空白，我们引入了SpeechR，这是一个统一的基准，用于评估大型音频语言模型在语音上的推理能力。SpeechR 从三个关键维度评估模型：事实检索、程序推理和规范判断。它包括三种不同的评估格式。多项选择版本衡量答案选择的准确性。生成版本评估推理链的连贯性和逻辑一致性。声学特征版本探讨语音中的重音和情绪变化是否影响推理性能。对十一种最先进的LALMs的评估表明，高转写准确性并不一定能转化为强大的推理能力。SpeechR 为评估口语中的推理能力建立了结构化的基准，使我们能够对模型在多种对话任务中的能力进行更针对性的分析。 

---
# DIRF: A Framework for Digital Identity Protection and Clone Governance in Agentic AI Systems 

**Title (ZH)**: DIRF：一种针对有能动性的AI系统中数字身份保护与克隆治理的框架 

**Authors**: Hammad Atta, Muhammad Zeeshan Baig, Yasir Mehmood, Nadeem Shahzad, Ken Huang, Muhammad Aziz Ul Haq, Muhammad Awais, Kamal Ahmed, Anthony Green  

**Link**: [PDF](https://arxiv.org/pdf/2508.01997)  

**Abstract**: The rapid advancement and widespread adoption of generative artificial intelligence (AI) pose significant threats to the integrity of personal identity, including digital cloning, sophisticated impersonation, and the unauthorized monetization of identity-related data. Mitigating these risks necessitates the development of robust AI-generated content detection systems, enhanced legal frameworks, and ethical guidelines. This paper introduces the Digital Identity Rights Framework (DIRF), a structured security and governance model designed to protect behavioral, biometric, and personality-based digital likeness attributes to address this critical need. Structured across nine domains and 63 controls, DIRF integrates legal, technical, and hybrid enforcement mechanisms to secure digital identity consent, traceability, and monetization. We present the architectural foundations, enforcement strategies, and key use cases supporting the need for a unified framework. This work aims to inform platform builders, legal entities, and regulators about the essential controls needed to enforce identity rights in AI-driven systems. 

**Abstract (ZH)**: 数字身份权利框架（DIRF）：保护行为、生物特征和基于个性的数字肖像属性的安全与治理模型 

---
# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion 

**Title (ZH)**: 可控且隐蔽的分流攻击通过分散潜在扩散实现 

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01987)  

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses. 

**Abstract (ZH)**: 基于扩散的推荐系统欺骗攻击框架（DLDA）：实现目标项的有效推广与自然行为伪装 

---
# TIBSTC-CoT: A Multi-Domain Instruction Dataset for Chain-of-Thought Reasoning in Language Models 

**Title (ZH)**: TIBSTC-CoT：用于语言模型链式推理的多域指令数据集 

**Authors**: Fan Gao, Cheng Huang, Nyima Tashi, Yutong Liu, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01977)  

**Abstract**: To address the severe data scarcity in Tibetan, a low-resource language spoken by over six million people, we introduce TIBSTC-CoT, the large-scale, multi-domain Tibetan dataset automatically constructed via chain-of-thought prompting with large language models (LLMs). TIBSTC-CoT establishes a scalable and reproducible framework for dataset creation in low-resource settings, covering diverse domains and reasoning patterns essential for language understanding and generation. Building on this dataset, we develop the Sunshine-thinking LLM family, a series of Tibetan-centric LLMs equipped with chain-of-thought capabilities. Trained entirely on TIBSTC-CoT, Sunshine-thinking has demonstrated strong reasoning and generation performance, comparable to state-of-the-art (SOTA) multilingual LLMs. Our work marks a significant step toward inclusive AI by enabling high-quality Tibetan language processing through both resource creation and model innovation. All data are available: this https URL. 

**Abstract (ZH)**: 为解决藏语这一低资源语言所面临的严重数据稀缺问题，藏语TIBSTC-CoT数据集应运而生，该数据集是通过大型语言模型（LLMs）的链式思考提示方法自动构建的大规模、多领域藏语语料库。TIBSTC-CoT为低资源环境下的数据集创建奠定了可扩展和可复现的框架，涵盖了语言理解和生成所需的各种领域和推理模式。基于该数据集，我们开发了Sunshine-thinking LLM家族，这一系列以藏语为中心的大型语言模型配备了链式思考能力。Sunshine-thinking完全在TIBSTC-CoT上训练，展示了强大的推理和生成性能，与最先进的多语言大型语言模型（SOTA）相当。我们的工作标志着包容性人工智能的重要一步，通过资源创建和模型创新，使高质量的藏语处理成为可能。所有数据均可获取：[此链接]。 

---
# Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling 

**Title (ZH)**: 通过部分奖励建模提前拒绝加速大语言模型推理 

**Authors**: Seyyed Saeid Cheshmi, Azal Ahmad Khan, Xinran Wang, Zirui Liu, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.01969)  

**Abstract**: Large Language Models (LLMs) are increasingly relied upon for solving complex reasoning tasks in domains such as mathematics, logic, and multi-step question answering. A growing line of work seeks to improve reasoning quality by scaling inference time compute particularly through Process Reward Models (PRMs), used to reward the reasoning at intermediate steps. While effective, these methods introduce substantial computational overhead, especially when generating large numbers of solutions in parallel. In this paper, we investigate whether PRMs can be used mid-generation to provide early signals that enable the rejection of suboptimal candidates before full generation of step is complete. We introduce the hypothesis that PRMs are also Partial Reward Models, meaning that the scores they assign to partially completed reasoning step are predictive of final output quality. This allows for principled early rejection based on intermediate token-level signals. We support this hypothesis both theoretically, by proving that the risk of discarding optimal beams decreases exponentially with generation length and empirically, by demonstrating a strong correlation between partial and final rewards across multiple reward models. On math reasoning benchmarks, our method achieves up to 1.4$\times$-9$\times$ reduction in inference FLOPs without degrading final performance. These results suggest that early rejection is a powerful mechanism for improving the compute-efficiency of reasoning in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学、逻辑和多步问答等领域解决复杂推理任务方面越来越依赖。现有工作通过扩展推理时间计算，特别是使用过程奖励模型（PRMs）来奖励中间步骤的推理，以提高推理质量。尽管有效，这些方法在并行生成大量解决方案时引入了大量计算开销。在这项研究中，我们探讨了在生成过程中使用PRMs是否可以提供早期信号，以便在完成整个生成之前就能拒绝次优候选。我们提出了一个假设，即PRMs也是部分奖励模型，这意味着它们为部分完成的推理步骤分配的得分可以预测最终输出质量。这允许基于中间的令牌级信号进行有原则的早期拒绝。我们通过理论证明和实验证据支持这一假设：理论上，抛弃最优射线的风险随着生成长度的增加而指数级下降；实证上，我们展示了多个奖励模型中部分奖励与最终奖励之间存在强烈的相关性。在数学推理基准测试中，我们的方法在不降低最终性能的情况下，实现了高达9倍的推理FLOPs减少。这些结果表明，早期拒绝是提高LLMs推理计算效率的强大机制。 

---
# Kronecker-LoRA: hybrid Kronecker-LoRA adapters for scalable, sustainable fine-tuning 

**Title (ZH)**: Kronecker-LoRA: 混合Kronecker-LoRA适配器以实现可扩展且可持续的微调 

**Authors**: Yixin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01961)  

**Abstract**: Fine-tuning massive pre-trained language models across many tasks demands adapters that are both parameter-efficient and highly expressive. We introduce \textbf{Kron-LoRA}, a two-stage adapter that first factorizes each frozen linear update as a Kronecker product \[ \Delta W = A \otimes B \] and then compresses \[ B \in \mathbb{R}^{d_{B2}\times d_{B1}} \] via an \(r\)-rank LoRA decomposition \(B \approx B_{1}B_{2}\). By leveraging \[ \mathrm{rank}(A \otimes B) \;=\; \mathrm{rank}(A)\,\mathrm{rank}(B), \] Kron-LoRA retains the expressivity of the update while using up to $4\!\times\!$ fewer parameters than a standard rank-8 LoRA adapter. Its compact adapter matrices also quantize to 8- or 4-bit with less accuracy degradation than LoRA, enabling further memory and storage savings for on-device deployment. We benchmark on DistilBERT and Mistral-7B across five tasks (PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge) over multiple epochs of adapter-only tuning: on DistilBERT, an 840 K-parameter Kron-LoRA matches LoRA-16's performance, and on Mistral-7B, a 5.7 M-parameter Kron-LoRA rivals LoRA-8 with modest memory savings and only a 3-8\% speed overhead. In sequential fine-tuning from ARC-Challenge to ARC-Easy, Kron-LoRA retains 55.18\% accuracy versus 53.17\% for LoRA-8-despite using only one-quarter of the adapter parameters-underscoring its competitive cross-task transfer performance. By uniting Kronecker structure, low-rank compression, quantization-friendliness, and by providing transparent trade-off analysis, Kron-LoRA offers a scalable, sustainable, and continual-learning-ready solution for multi-task adaptation of large language models. 

**Abstract (ZH)**: Kron-LoRA：结合柯西尼结构、低秩压缩和量化友好性的两阶段适配器 

---
# Flow-Aware GNN for Transmission Network Reconfiguration via Substation Breaker Optimization 

**Title (ZH)**: 基于变电站断路器优化的流感知GNN传输网络重构 

**Authors**: Dekang Meng, Rabab Haider, Pascal van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2508.01951)  

**Abstract**: This paper introduces OptiGridML, a machine learning framework for discrete topology optimization in power grids. The task involves selecting substation breaker configurations that maximize cross-region power exports, a problem typically formulated as a mixed-integer program (MIP) that is NP-hard and computationally intractable for large networks. OptiGridML replaces repeated MIP solves with a two-stage neural architecture: a line-graph neural network (LGNN) that approximates DC power flows for a given network topology, and a heterogeneous GNN (HeteroGNN) that predicts breaker states under structural and physical constraints. A physics-informed consistency loss connects these components by enforcing Kirchhoff's law on predicted flows. Experiments on synthetic networks with up to 1,000 breakers show that OptiGridML achieves power export improvements of up to 18% over baseline topologies, while reducing inference time from hours to milliseconds. These results demonstrate the potential of structured, flow-aware GNNs for accelerating combinatorial optimization in physical networked systems. 

**Abstract (ZH)**: 基于电力网络的离散拓扑优化的OptiGridML机器学习框架 

---
# Inferring Reward Machines and Transition Machines from Partially Observable Markov Decision Processes 

**Title (ZH)**: 从部分可观测马尔可夫决策过程推断奖励机器和转移机器 

**Authors**: Yuly Wu, Jiamou Liu, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01947)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) are fundamental to many real-world applications. Although reinforcement learning (RL) has shown success in fully observable domains, learning policies from traces in partially observable environments remains challenging due to non-Markovian observations. Inferring an automaton to handle the non-Markovianity is a proven effective approach, but faces two limitations: 1) existing automaton representations focus only on reward-based non-Markovianity, leading to unnatural problem formulations; 2) inference algorithms face enormous computational costs. For the first limitation, we introduce Transition Machines (TMs) to complement existing Reward Machines (RMs). To develop a unified inference algorithm for both automata types, we propose the Dual Behavior Mealy Machine (DBMM) that subsumes both TMs and RMs. We then introduce DB-RPNI, a passive automata learning algorithm that efficiently infers DBMMs while avoiding the costly reductions required by prior work. We further develop optimization techniques and identify sufficient conditions for inferring the minimal correct automata. Experimentally, our inference method achieves speedups of up to three orders of magnitude over SOTA baselines. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）在许多实际应用中是基础性的。尽管强化学习（RL）在完全可观测领域取得了成功，但在部分可观测环境中从轨迹学习策略由于非马尔可夫观察结果仍然具有挑战性。通过自动机推理非马尔可夫性是一种 proven 有效的办法，但面临两个限制：1）现有的自动机表示只关注基于奖励的非马尔可夫性，导致非自然的问题表述；2）推理算法面临巨大的计算成本。为了解决第一个限制，我们引入转换机（TMs）来补充现有的奖励机（RMs）。为了一致地为这两种自动机类型开发推理算法，我们提出了双行为梅利机（DBMM），它概括了TMs和RMs。然后，我们引入DB-RPNI，这是一种被动自动机学习算法，可以高效地推理DBMMs，同时避免了先前工作中所需的昂贵归约。我们还开发了优化技术，并确定了推理最少正确自动机的充分条件。实验结果显示，我们的推理方法在与最优基线方法相比时，性能提高了三个数量级。 

---
# ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks 

**Title (ZH)**: ROVER：视觉-语言模型驱动的视频递归推理用于具身任务 

**Authors**: Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2508.01943)  

**Abstract**: Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: this https URL 

**Abstract (ZH)**: 视觉语言模型（VLMs）在多种图像理解任务中展现了令人印象深刻的性能，但在处理要求跨越视频连续多帧进行推理的任务时仍存在局限性。为解决这一限制，我们提出了ROVER（递归视频推理框架），该框架使模型能够递归地将长时序视频轨迹分解为与轨迹中较短子任务相对应的段落。通过这种方式，ROVER 能够实现更集中和准确的局部时间序列推理，同时保持全局上下文。我们在多种OpenX Embodiment视频和一个由RoboCasa衍生的新数据集上（该数据集包含543个展示专家和非专家扰动轨迹的视频，涵盖了27个机器人操作任务）评估了ROVER，ROVER 在三个视频推理任务中均优于强壮的基线模型：任务进展估计、帧级自然语言推理和视频问答。我们观察到，通过在每个时间步减少模型需要推理的帧数，ROVER 在轨迹的意外或非最优时刻减轻了幻觉现象。此外，通过启用针对特定子任务的滑动上下文窗口，ROVER 的时间复杂度随着视频长度线性增长，这是基线模型的渐进性改进。相关演示、代码和数据请参见：this https URL。 

---
# Less is More: AMBER-AFNO -- a New Benchmark for Lightweight 3D Medical Image Segmentation 

**Title (ZH)**: fewer is more: AMBER-AFNO — 一个新的轻量级3D医学图像分割基准 

**Authors**: Andrea Dosi, Semanto Mondal, Rajib Chandra Ghosh, Massimo Brescia, Giuseppe Longo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01941)  

**Abstract**: This work presents the results of a methodological transfer from remote sensing to healthcare, adapting AMBER -- a transformer-based model originally designed for multiband images, such as hyperspectral data -- to the task of 3D medical datacube segmentation. In this study, we use the AMBER architecture with Adaptive Fourier Neural Operators (AFNO) in place of the multi-head self-attention mechanism. While existing models rely on various forms of attention to capture global context, AMBER-AFNO achieves this through frequency-domain mixing, enabling a drastic reduction in model complexity. This design reduces the number of trainable parameters by over 80% compared to UNETR++, while maintaining a FLOPs count comparable to other state-of-the-art architectures. Model performance is evaluated on two benchmark 3D medical datasets -- ACDC and Synapse -- using standard metrics such as Dice Similarity Coefficient (DSC) and Hausdorff Distance (HD), demonstrating that AMBER-AFNO achieves competitive or superior accuracy with significant gains in training efficiency, inference speed, and memory usage. 

**Abstract (ZH)**: 将遥感领域的技术转移至医疗健康领域：AMBER-AFNO在3D医疗数据立方分割中的应用 

---
# Proactive Disentangled Modeling of Trigger-Object Pairings for Backdoor Defense 

**Title (ZH)**: 主动解耦建模触发-对象配对以防御后门攻击 

**Authors**: Kyle Stein, Andrew A. Mahyari, Guillermo Francia III, Eman El-Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01932)  

**Abstract**: Deep neural networks (DNNs) and generative AI (GenAI) are increasingly vulnerable to backdoor attacks, where adversaries embed triggers into inputs to cause models to misclassify or misinterpret target labels. Beyond traditional single-trigger scenarios, attackers may inject multiple triggers across various object classes, forming unseen backdoor-object configurations that evade standard detection pipelines. In this paper, we introduce DBOM (Disentangled Backdoor-Object Modeling), a proactive framework that leverages structured disentanglement to identify and neutralize both seen and unseen backdoor threats at the dataset level. Specifically, DBOM factorizes input image representations by modeling triggers and objects as independent primitives in the embedding space through the use of Vision-Language Models (VLMs). By leveraging the frozen, pre-trained encoders of VLMs, our approach decomposes the latent representations into distinct components through a learnable visual prompt repository and prompt prefix tuning, ensuring that the relationships between triggers and objects are explicitly captured. To separate trigger and object representations in the visual prompt repository, we introduce the trigger-object separation and diversity losses that aids in disentangling trigger and object visual features. Next, by aligning image features with feature decomposition and fusion, as well as learned contextual prompt tokens in a shared multimodal space, DBOM enables zero-shot generalization to novel trigger-object pairings that were unseen during training, thereby offering deeper insights into adversarial attack patterns. Experimental results on CIFAR-10 and GTSRB demonstrate that DBOM robustly detects poisoned images prior to downstream training, significantly enhancing the security of DNN training pipelines. 

**Abstract (ZH)**: 深度神经网络（DNNs）和生成AI（GenAI）日益面临后门攻击的威胁，攻击者会将触发器嵌入到输入中以导致模型错误分类或误解释目标标签。除了传统的单一触发器场景外，攻击者可能在各类物体中注入多个触发器，形成未见的后门-物体配置，从而逃避标准检测管道。在本文中，我们引入了DBOM（分离后门-物体建模）这一先发制人的框架，利用结构化分离来识别并在数据集级别中中和可见和不可见的后门威胁。具体而言，DBOM 通过视觉语言模型（VLMs）在嵌入空间中将触发器和物体建模为独立的基本要素，从而对输入图像表示进行因子分解。借助VLMs的冷冻预训练编码器，我们的方法通过可学习的视觉提示库和提示前缀调优将潜在表示分解为不同的组成部分，确保捕获触发器和物体之间的关系。为了在视觉提示库中分离触发器和物体表示，我们引入了触发器-物体分离和多样性的损失，以帮助分离触发器和物体的视觉特征。接下来，通过将图像特征与特征分解和融合对齐，以及学习到的多模态共享上下文提示令牌，DBOM 使模型能够零样本泛化到训练过程中未见过的新颖触发器-物体配对，从而提供对抗攻击模式的深入洞察。实验结果表明，DBOM 能在下游训练之前 robust 地检测受污染图像，显著增强 DNN 训练管道的安全性。 

---
# Word Overuse and Alignment in Large Language Models: The Influence of Learning from Human Feedback 

**Title (ZH)**: 大型语言模型中的词过度使用与对齐：来自人类反馈的影响 

**Authors**: Tom S. Juzek, Zina B. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2508.01930)  

**Abstract**: Large Language Models (LLMs) are known to overuse certain terms like "delve" and "intricate." The exact reasons for these lexical choices, however, have been unclear. Using Meta's Llama model, this study investigates the contribution of Learning from Human Feedback (LHF), under which we subsume Reinforcement Learning from Human Feedback and Direct Preference Optimization. We present a straightforward procedure for detecting the lexical preferences of LLMs that are potentially LHF-induced. Next, we more conclusively link LHF to lexical overuse by experimentally emulating the LHF procedure and demonstrating that participants systematically prefer text variants that include certain words. This lexical overuse can be seen as a sort of misalignment, though our study highlights the potential divergence between the lexical expectations of different populations -- namely LHF workers versus LLM users. Our work contributes to the growing body of research on explainable artificial intelligence and emphasizes the importance of both data and procedural transparency in alignment research. 

**Abstract (ZH)**: 大型语言模型（LLMs）Known to Overuse Certain Terms like "Delve" and "Intricate": An Investigation of the Contribution of Learning from Human Feedback Using Meta's Llama Model 

---
# IAUNet: Instance-Aware U-Net 

**Title (ZH)**: IAUNet：实例感知的U-Net 

**Authors**: Yaroslav Prytula, Illia Tsiporenko, Ali Zeynalli, Dmytro Fishman  

**Link**: [PDF](https://arxiv.org/pdf/2508.01928)  

**Abstract**: Instance segmentation is critical in biomedical imaging to accurately distinguish individual objects like cells, which often overlap and vary in size. Recent query-based methods, where object queries guide segmentation, have shown strong performance. While U-Net has been a go-to architecture in medical image segmentation, its potential in query-based approaches remains largely unexplored. In this work, we present IAUNet, a novel query-based U-Net architecture. The core design features a full U-Net architecture, enhanced by a novel lightweight convolutional Pixel decoder, making the model more efficient and reducing the number of parameters. Additionally, we propose a Transformer decoder that refines object-specific features across multiple scales. Finally, we introduce the 2025 Revvity Full Cell Segmentation Dataset, a unique resource with detailed annotations of overlapping cell cytoplasm in brightfield images, setting a new benchmark for biomedical instance segmentation. Experiments on multiple public datasets and our own show that IAUNet outperforms most state-of-the-art fully convolutional, transformer-based, and query-based models and cell segmentation-specific models, setting a strong baseline for cell instance segmentation tasks. Code is available at this https URL 

**Abstract (ZH)**: 基于查询的实例分割在生物医学成像中的细胞识别至关重要，以准确区分重叠且大小不一的个体对象。近期基于查询的方法通过对象查询引导分割，展现了出色的表现。尽管U-Net在医学图像分割中常被选用，但在基于查询的方法中的潜力尚未被充分探索。在此工作中，我们提出IAUNet，一种新颖的基于查询的U-Net架构。核心设计采用完整的U-Net架构，并通过一种新颖的轻量级像素解码器进行增强，使模型更加高效并减少参数数量。此外，我们提出了一种变换器解码器，以在多个尺度上细化对象特异性特征。最后，我们引入了2025 Revvity全细胞分割数据集，这是一个独特的资源，包含详细的亮视野图像中重叠细胞质的注释，为生物医学实例分割设立了新的基准。在多个公开数据集和我们自己的数据集上的实验显示，IAUNet在大多数最先进的完全卷积、变换器基和基于查询模型以及细胞分割特定模型中表现出色，为细胞实例分割任务提供了强有力的基础。相关代码可访问以下链接：this https URL。 

---
# Quantum-RAG and PunGPT2: Advancing Low-Resource Language Generation and Retrieval for the Punjabi Language 

**Title (ZH)**: 量子-RAG 和 PunGPT2：提高旁遮普语语言生成和检索的低资源方法 

**Authors**: Jaskaranjeet Singh, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2508.01918)  

**Abstract**: Despite the rapid advancement of large language models (LLMs), low-resource languages remain largely excluded from the NLP landscape. We present PunGPT2, the first fully open-source suite of Punjabi large language models, trained from scratch on a 35GB domain-diverse corpus encompassing literature, religious texts, news, and social discourse. Unlike prior multilingual approaches, PunGPT2 captures rich syntactic and morphological features unique to Punjabi through a tokenizer optimised with byte pair encoding and linguistically aligned pretraining objectives. To improve factual grounding and domain recall, we introduce Pun-RAG, a retrieval-augmented generation framework combining PunGPT2 with a dense FAISS retriever over a curated Punjabi knowledge base. We further develop Pun-Instruct, a parameter-efficient, instruction-tuned variant using QLoRA, enabling robust zero-shot and instruction-following performance with significantly reduced compute needs.
As a key innovation, we propose Quantum-RAG, a novel hybrid retrieval system that fuses sparse (BM25) and dense methods with quantum-inspired semantic matching. By encoding queries using amplitude-based embeddings and retrieving via quantum kernel similarity, Quantum-RAG achieves improved contextual relevance with minimal memory overhead marking the first practical integration of quantum representations in low-resource language generation. Our models significantly outperform strong multilingual baselines (mBERT, mT5, MuRIL) in perplexity, factuality, and fluency. This work provides a scalable, reproducible blueprint for extending LLM capabilities to underrepresented languages and pioneers quantum-aware retrieval in low-resource NLP 

**Abstract (ZH)**: 尽管大规模语言模型取得了 rapid advancement，低资源语言仍然被 NLP 场景 largely excluded。我们提出了 PunGPT2，这是首个完全开源的旁语大规模语言模型套件，从一个包含文学、宗教文本、新闻和社会讨论的 35GB 多领域语料库中从零训练而来。与先前的多语言方法不同，PunGPT2 通过经过字节对编码优化的分词器和语义上对齐的预训练目标，捕捉到旁语特有的丰富句法和形态特征。为了提高事实接地和领域召回，我们引入了 Pun-RAG，这是一种检索增强生成框架，结合了 PunGPT2 和一个密集的 FAISS 检索器，该检索器基于精选的旁语知识库。我们进一步开发了 Pun-Instruct，这是一种使用 QLoRA 参数高效、指令调优的变体，使其在显著减少计算需求的情况下实现了稳健的零样本和指令跟随性能。作为一项关键创新，我们提出了 Quantum-RAG，这是一种新颖的混合检索系统，将稀疏（BM25）和密集方法与量子启发的语义匹配相结合。通过使用振幅基嵌入编码查询并通过量子内核相似性检索，Quantum-RAG 实现了改进的上下文相关性，并在最小内存开销的情况下实现了第一个低资源语言生成中的量子表示的实用性集成。我们的模型在困惑度、事实性和流畅性方面显著优于强大的多语言基线（mBERT、mT5、MuRIL）。这项工作为将 LLM 能力扩展到低资源语言提供了可扩展、可重现的蓝图，并开创了低资源 NLP 中量子感知检索的先河。 

---
# L3M+P: Lifelong Planning with Large Language Models 

**Title (ZH)**: L3M+P: 基于大型语言模型的终身规划 

**Authors**: Krish Agarwal, Yuqian Jiang, Jiaheng Hu, Bo Liu, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2508.01917)  

**Abstract**: By combining classical planning methods with large language models (LLMs), recent research such as LLM+P has enabled agents to plan for general tasks given in natural language. However, scaling these methods to general-purpose service robots remains challenging: (1) classical planning algorithms generally require a detailed and consistent specification of the environment, which is not always readily available; and (2) existing frameworks mainly focus on isolated planning tasks, whereas robots are often meant to serve in long-term continuous deployments, and therefore must maintain a dynamic memory of the environment which can be updated with multi-modal inputs and extracted as planning knowledge for future tasks. To address these two issues, this paper introduces L3M+P (Lifelong LLM+P), a framework that uses an external knowledge graph as a representation of the world state. The graph can be updated from multiple sources of information, including sensory input and natural language interactions with humans. L3M+P enforces rules for the expected format of the absolute world state graph to maintain consistency between graph updates. At planning time, given a natural language description of a task, L3M+P retrieves context from the knowledge graph and generates a problem definition for classical planners. Evaluated on household robot simulators and on a real-world service robot, L3M+P achieves significant improvement over baseline methods both on accurately registering natural language state changes and on correctly generating plans, thanks to the knowledge graph retrieval and verification. 

**Abstract (ZH)**: 结合经典规划方法与大规模语言模型以增强通用服务机器人的长期规划能力：L3M+P框架 

---
# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning 

**Title (ZH)**: 使用无监督学习分解表示空间为可解释子空间 

**Authors**: Xinting Huang, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.01916)  

**Abstract**: Understanding internal representations of neural models is a core interest of mechanistic interpretability. Due to its large dimensionality, the representation space can encode various aspects about inputs. To what extent are different aspects organized and encoded in separate subspaces? Is it possible to find these ``natural'' subspaces in a purely unsupervised way? Somewhat surprisingly, we can indeed achieve this and find interpretable subspaces by a seemingly unrelated training objective. Our method, neighbor distance minimization (NDM), learns non-basis-aligned subspaces in an unsupervised manner. Qualitative analysis shows subspaces are interpretable in many cases, and encoded information in obtained subspaces tends to share the same abstract concept across different inputs, making such subspaces similar to ``variables'' used by the model. We also conduct quantitative experiments using known circuits in GPT-2; results show a strong connection between subspaces and circuit variables. We also provide evidence showing scalability to 2B models by finding separate subspaces mediating context and parametric knowledge routing. Viewed more broadly, our findings offer a new perspective on understanding model internals and building circuits. 

**Abstract (ZH)**: 理解神经模型的内部表示是机制可解释性的核心兴趣。由于表示空间的高维度，它可以编码输入的各种方面。不同的方面在独立的子空间中组织和编码到什么程度？是否有可能以纯粹无监督的方式找到这些“自然”的子空间？令人惊讶的是，我们确实可以通过一个看似无关的训练目标来实现这一点，并在此过程中找到可解释的子空间。我们的方法，邻近距离最小化（NDM），以无监督的方式学习非基底对齐的子空间。定性分析表明，在许多情况下，子空间是可解释的，并且在获得的子空间中编码的信息在不同输入上往往共享相同的抽象概念，从而使这些子空间类似于模型中使用的“变量”。我们还使用GPT-2中的已知电路进行了定量实验；结果表明子空间与电路变量之间存在强烈联系。我们还提供了证据，证明可以通过找到分别介导上下文和参数知识路由的独立子空间来实现2B模型的可扩展性。更广泛地看，我们的发现为理解模型内部结构和构建电路提供了新的视角。 

---
# Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models 

**Title (ZH)**: 重访回放与梯度对齐在大规模语言模型连续预训练中的作用 

**Authors**: Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Therien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2508.01908)  

**Abstract**: Training large language models (LLMs) typically involves pre-training on massive corpora, only to restart the process entirely when new data becomes available. A more efficient and resource-conserving approach would be continual pre-training, where models are updated with new data rather than retraining from scratch. However, the introduction of new data often causes distribution shifts, leading to performance degradation on previously learned tasks. In this paper, we take a deeper look at two popular proposals for addressing this distribution shift within the continual learning literature: experience replay and gradient alignment. We consider continual pre-training of models within the Llama family of architectures at a large scale across languages with 100 billion tokens of training data in each language, finding that both replay and gradient alignment lead to more stable learning without forgetting. This conclusion holds both as we vary the model scale and as we vary the number and diversity of tasks. Moreover, we are the first to demonstrate the effectiveness of gradient alignment techniques in the context of LLM pre-training and propose an efficient implementation of meta-experience replay (MER) that imbues experience replay with the benefits of gradient alignment despite negligible compute and memory overhead. Our scaling analysis across model sizes and replay rates indicates that small rates of replaying old examples are definitely a more valuable use of compute than investing in model size, but that it is more compute efficient to scale the size of the model than invest in high rates of replaying old examples. 

**Abstract (ZH)**: 在大型语言模型持续预训练中体验重放与梯度对齐的应用研究 

---
# How Does Controllability Emerge In Language Models During Pretraining? 

**Title (ZH)**: 语言模型在预训练过程中如何实现可控性？ 

**Authors**: Jianshu She, Xinyue Li, Eric Xing, Zhengzhong Liu, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2508.01892)  

**Abstract**: Language models can be steered by modifying their internal representations to control concepts such as emotion, style, or truthfulness in generation. However, the conditions for an effective intervention remain unclear and are often validated through heuristics and trial-and-error. To fill this gap, we demonstrate that intervention efficacy, measured by linear steerability (i.e., the ability to adjust output via linear transformations of hidden states), emerges during intermediate stages of training. Moreover, even closely related concepts (e.g., anger and sadness) exhibit steerability emergence at distinct stages of training.
To better interpret the dynamics of steerability during training, we adapt existing intervention techniques into a unified framework, referred to as the "Intervention Detector" (ID), which is designed to reveal how linear steerability evolves over the course of training through hidden state and representation analysis. ID reveals that concepts become increasingly linearly separable in the hidden space as training progresses, which strongly correlates with the emergence of linear steerability. We further introduce ID-based metrics, such as heatmaps, entropy trends, and cosine similarity, to help interpret how linear steerability evolves throughout training. In addition, we apply ID across different model families to ensure the generality of our findings on steerability dynamics. 

**Abstract (ZH)**: 语言模型可以通过修改其内部表示来操控生成中的情绪、风格或真实性等概念，但在有效干预的条件方面仍不清楚，通常通过启发式方法和试错来验证。为填补这一空白，我们证明干预的有效性，即线性可导向性（通过隐藏状态的线性变换调整输出的能力），在训练的中间阶段会出现。此外，即使是密切相关概念（如愤怒和悲伤）也会在训练的不同阶段表现出可导向性的出现。

为了更好地解释训练过程中可导向性的动态变化，我们提出了一个统一的干预检测框架，称为“干预检测器”（ID），该框架设计用于通过隐藏状态和表示分析揭示线性可导向性如何随训练过程发展。ID揭示了随着训练的进行，概念在隐藏空间中变得越来越线性可区分，这与线性可导向性的出现密切相关。我们进一步引入基于ID的度量标准，如热图、熵趋势和余弦相似度，以帮助解释线性可导向性如何在整个训练过程中演变。此外，我们在不同模型家族中应用ID，以确保我们的可导向性动态发现具有普适性。 

---
# Complete Evasion, Zero Modification: PDF Attacks on AI Text Detection 

**Title (ZH)**: 完整的规避，零修改：针对AI文本检测的PDF攻击 

**Authors**: Aldan Creo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01887)  

**Abstract**: AI-generated text detectors have become essential tools for maintaining content authenticity, yet their robustness against evasion attacks remains questionable. We present PDFuzz, a novel attack that exploits the discrepancy between visual text layout and extraction order in PDF documents. Our method preserves exact textual content while manipulating character positioning to scramble extraction sequences. We evaluate this approach against the ArguGPT detector using a dataset of human and AI-generated text. Our results demonstrate complete evasion: detector performance drops from (93.6 $\pm$ 1.4) % accuracy and 0.938 $\pm$ 0.014 F1 score to random-level performance ((50.4 $\pm$ 3.2) % accuracy, 0.0 F1 score) while maintaining perfect visual fidelity. Our work reveals a vulnerability in current detection systems that is inherent to PDF document structures and underscores the need for implementing sturdy safeguards against such attacks. We make our code publicly available at this https URL. 

**Abstract (ZH)**: 基于AI生成文本的PDFuzz攻击：利用PDF文档中可视化文本布局与提取顺序之间的差异，同时保持文本内容不变，扰乱提取顺序，以实现完全 evasion。 

---
# Counterfactual Reciprocal Recommender Systems for User-to-User Matching 

**Title (ZH)**: 用户间匹配的反事实互惠推荐系统 

**Authors**: Kazuki Kawamura, Takuma Udagawa, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01867)  

**Abstract**: Reciprocal recommender systems (RRS) in dating, gaming, and talent platforms require mutual acceptance for a match. Logged data, however, over-represents popular profiles due to past exposure policies, creating feedback loops that skew learning and fairness. We introduce Counterfactual Reciprocal Recommender Systems (CFRR), a causal framework to mitigate this bias. CFRR uses inverse propensity scored, self-normalized objectives. Experiments show CFRR improves NDCG@10 by up to 3.5% (e.g., from 0.459 to 0.475 on DBLP, from 0.299 to 0.307 on Synthetic), increases long-tail user coverage by up to 51% (from 0.504 to 0.763 on Synthetic), and reduces Gini exposure inequality by up to 24% (from 0.708 to 0.535 on Synthetic). CFRR offers a promising approach for more accurate and fair user-to-user matching. 

**Abstract (ZH)**: -counterfactual 双向推荐系统（CFRR）在 dating、gaming 和 talent 平台上需要相互接受才能匹配。然而，由于过去的曝光政策，记录数据过度代表了受欢迎的资料，从而造成了反馈循环，扭曲了学习和公平性。我们引入了 Counterfactual 双向推荐系统（CFRR），这是一种因果框架，用于减轻这种偏见。CFRR 使用逆概率加权和自规范化目标。实验表明，CFRR 可以提高 NDCG@10 最多 3.5%（例如，在 DBLP 上从 0.459 提高到 0.475，在 Synthetic 上从 0.299 提高到 0.307），增加长尾用户覆盖率最多 51%（在 Synthetic 上从 0.504 增加到 0.763），并减少 Gini � Exposures 不平等最多 24%（在 Synthetic 上从 0.708 减少到 0.535）。CFRR 提供了一种更准确和公平的用户与用户匹配的有前途的方法。 

---
# Counterfactual Probing for Hallucination Detection and Mitigation in Large Language Models 

**Title (ZH)**: 基于反事实探查的大语言模型 hallucination 检测与缓解 

**Authors**: Yijun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01862)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities across diverse tasks, yet they frequently generate hallucinations outputs that are fluent but factually incorrect or unsupported. We propose Counterfactual Probing, a novel approach for detecting and mitigating hallucinations in LLM outputs. Our method dynamically generates counterfactual statements that appear plausible but contain subtle factual errors, then evaluates the model's sensitivity to these perturbations. We hypothesize that genuine knowledge exhibits robustness to counterfactual variations, while hallucinated content shows inconsistent confidence patterns when confronted with plausible alternatives. Our comprehensive evaluation on TruthfulQA, factual statement datasets, and curated hallucination examples demonstrates that counterfactual probing achieves superior detection performance compared to baseline methods, while our adaptive mitigation strategies reduce hallucination scores by an average of 24.5%. The approach requires no model retraining and can be integrated into existing LLM pipelines as a realtime verification mechanism. 

**Abstract (ZH)**: 大型语言模型在多种任务中展现了出色的能力，但经常会生成流畅但事实错误或缺乏支持的虚构输出。我们提出了一种新的方法——反事实探测，用于检测和减轻大型语言模型输出中的虚构现象。该方法动态生成看似合理的但包含细微事实错误的反事实陈述，然后评估模型对这些扰动的敏感性。我们认为真正知识在面对反事实变化时表现出高度的稳健性，而虚构内容在面对合理替代时则显示不一致的信心模式。在TruthfulQA数据集、事实陈述数据集和定制化的虚构示例上的全面评估表明，反事实探测在检测性能上优于基线方法，而我们的自适应缓解策略平均降低了24.5%的虚构得分。该方法不需要重新训练模型，并且可以作为实时验证机制集成到现有的大型语言模型流水线中。 

---
# ACT-Tensor: Tensor Completion Framework for Financial Dataset Imputation 

**Title (ZH)**: ACT-Tensor: 金融数据集插补的张量完成框架 

**Authors**: Junyi Mo, Jiayu Li, Duo Zhang, Elynn Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01861)  

**Abstract**: Missing data in financial panels presents a critical obstacle, undermining asset-pricing models and reducing the effectiveness of investment strategies. Such panels are often inherently multi-dimensional, spanning firms, time, and financial variables, which adds complexity to the imputation task. Conventional imputation methods often fail by flattening the data's multidimensional structure, struggling with heterogeneous missingness patterns, or overfitting in the face of extreme data sparsity. To address these limitations, we introduce an Adaptive, Cluster-based Temporal smoothing tensor completion framework (ACT-Tensor) tailored for severely and heterogeneously missing multi-dimensional financial data panels. ACT-Tensor incorporates two key innovations: a cluster-based completion module that captures cross-sectional heterogeneity by learning group-specific latent structures; and a temporal smoothing module that proactively removes short-lived noise while preserving slow-moving fundamental trends. Extensive experiments show that ACT-Tensor consistently outperforms state-of-the-art benchmarks in terms of imputation accuracy across a range of missing data regimes, including extreme sparsity scenarios. To assess its practical financial utility, we evaluate the imputed data with an asset-pricing pipeline tailored for tensor-structured financial data. Results show that ACT-Tensor not only reduces pricing errors but also significantly improves risk-adjusted returns of the constructed portfolio. These findings confirm that our method delivers highly accurate and informative imputations, offering substantial value for financial decision-making. 

**Abstract (ZH)**: 适应性基于聚类的时间光滑张量补全框架（ACT-Tensor）：应对多维金融面板数据的缺失 

---
# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents 

**Title (ZH)**: Web-CogReasoner: 向导知识驱动的认知推理for Web代理 

**Authors**: Yuhan Guo, Cong Guo, Aiwen Sun, Hongliang He, Xinyu Yang, Yue Lu, Yingji Zhang, Xuntao Guo, Dong Zhang, Jianzhuang Liu, Jiang Duan, Yijia Xiao, Liangjian Wen, Hai-Ming Xu, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01858)  

**Abstract**: Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at this https URL 

**Abstract (ZH)**: 多模态大规模模型显著推动了网络代理的发展，使其能够像人类认知一样感知和交互数字环境。在本文中，我们argue认为网络代理必须首先获得足够的知识才能有效进行认知推理。因此，我们将网络代理的能力分解为两个关键阶段：知识内容学习和认知过程。为了形式化这一点，我们提出了Web-CogKnowledge框架，将知识分类为事实性知识、概念性和程序性知识。在该框架中，知识内容学习对应于代理的记忆与理解过程，依赖于前两种知识类型，代表了学习的“是什么”。相反，认知过程基于程序性知识，对应于探索，定义了推理和行动的“怎么做”。为了促进知识获取，我们构建了Web-CogDataset，这是从14个真实网站中精心整理而成的结构化资源，旨在系统地传授网络代理所需的核心知识。该数据集作为代理的理解基础——即构建理解的“名词”，同时也是学习如何推理和行动的基础。在此基础上，我们通过一种新颖的知识驱动的Chain-of-Thought（CoT）推理框架来操作这些过程，开发并训练了我们所提出的Web-CogReasoner代理。广泛的实验表明，该代理在泛化能力方面显著优于现有模型，特别是在结构化知识至关重要的未知任务中。为了进行严格的评估，我们引入了Web-CogBench，这是一个全面的评估套件，旨在评估和比较代理在划分的知识领域和认知能力方面的性能。我们的代码和数据在此开源：this https URL。 

---
# ChairPose: Pressure-based Chair Morphology Grounded Sitting Pose Estimation through Simulation-Assisted Training 

**Title (ZH)**: ChairPose: 基于压力的椅子形态引导的坐姿估计通过模拟辅助训练 

**Authors**: Lala Shakti Swarup Ray, Vitor Fortes Rey, Bo Zhou, Paul Lukowicz, Sungho Suh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01850)  

**Abstract**: Prolonged seated activity is increasingly common in modern environments, raising concerns around musculoskeletal health, ergonomics, and the design of responsive interactive systems. Existing posture sensing methods such as vision-based or wearable approaches face limitations including occlusion, privacy concerns, user discomfort, and restricted deployment flexibility. We introduce ChairPose, the first full body, wearable free seated pose estimation system that relies solely on pressure sensing and operates independently of chair geometry. ChairPose employs a two stage generative model trained on pressure maps captured from a thin, chair agnostic sensing mattress. Unlike prior approaches, our method explicitly incorporates chair morphology into the inference process, enabling accurate, occlusion free, and privacy preserving pose estimation. To support generalization across diverse users and chairs, we introduce a physics driven data augmentation pipeline that simulates realistic variations in posture and seating conditions. Evaluated across eight users and four distinct chairs, ChairPose achieves a mean per joint position error of 89.4 mm when both the user and the chair are unseen, demonstrating robust generalization to novel real world generalizability. ChairPose expands the design space for posture aware interactive systems, with potential applications in ergonomics, healthcare, and adaptive user interfaces. 

**Abstract (ZH)**: 长时间静坐活动在现代环境中越来越普遍，引发了对肌肉骨骼健康、人机工程学和响应式交互系统设计的关注。现有的基于视觉或穿戴式姿势传感方法面临遮挡、隐私问题、用户不适和部署灵活性受限等限制。我们引入了ChairPose，这是一种首款基于全身、穿戴式压力传感的独立于椅子几何结构的坐姿估测系统。ChairPose采用一种基于压力图的两阶段生成模型，这些压力图由一种薄且对椅子无依赖性的传感床垫捕获。与先前的方法不同，我们的方法在推断过程中明确考虑了椅子的形态特征，从而实现了准确、无遮挡且保护隐私的姿势估测。为了支持对多样化用户和椅子的泛化能力，我们引入了一种基于物理驱动的数据增强pipeline，模拟了姿势和座椅条件的现实变化。在八名用户和四把不同椅子上进行评估，当用户和椅子均未在训练集中出现时，ChairPose 的每关节位置误差的平均值为89.4毫米，展示了其在新型真实世界环境下的鲁棒泛化能力。ChairPose 扩展了姿势感知交互系统的设计空间，具有在人机工程学、医疗保健和自适应用户界面方面的潜在应用。 

---
# Beyond Vulnerabilities: A Survey of Adversarial Attacks as Both Threats and Defenses in Computer Vision Systems 

**Title (ZH)**: 超越漏洞：计算机视觉系统中对抗性攻击作为威胁与防御的综述 

**Authors**: Zhongliang Guo, Yifei Qian, Yanli Li, Weiye Li, Chun Tong Lei, Shuai Zhao, Lei Fang, Ognjen Arandjelović, Chun Pong Lau  

**Link**: [PDF](https://arxiv.org/pdf/2508.01845)  

**Abstract**: Adversarial attacks against computer vision systems have emerged as a critical research area that challenges the fundamental assumptions about neural network robustness and security. This comprehensive survey examines the evolving landscape of adversarial techniques, revealing their dual nature as both sophisticated security threats and valuable defensive tools. We provide a systematic analysis of adversarial attack methodologies across three primary domains: pixel-space attacks, physically realizable attacks, and latent-space attacks. Our investigation traces the technical evolution from early gradient-based methods such as FGSM and PGD to sophisticated optimization techniques incorporating momentum, adaptive step sizes, and advanced transferability mechanisms. We examine how physically realizable attacks have successfully bridged the gap between digital vulnerabilities and real-world threats through adversarial patches, 3D textures, and dynamic optical perturbations. Additionally, we explore the emergence of latent-space attacks that leverage semantic structure in internal representations to create more transferable and meaningful adversarial examples. Beyond traditional offensive applications, we investigate the constructive use of adversarial techniques for vulnerability assessment in biometric authentication systems and protection against malicious generative models. Our analysis reveals critical research gaps, particularly in neural style transfer protection and computational efficiency requirements. This survey contributes a comprehensive taxonomy, evolution analysis, and identification of future research directions, aiming to advance understanding of adversarial vulnerabilities and inform the development of more robust and trustworthy computer vision systems. 

**Abstract (ZH)**: 对抗攻击对计算机视觉系统的攻击已成为一个关键的研究领域，挑战了神经网络鲁棒性和安全性的一些基本假设。本文综述了对抗技术的发展景观，揭示了它们作为高级安全威胁和宝贵防御工具的双重性质。本文系统地分析了三种主要领域中的对抗攻击方法：像素空间攻击、物理可实现攻击和潜在空间攻击。研究追踪了从早期的梯度基方法（如FGSM和PGD）到包含动量、自适应步长和高级传递机制的复杂优化技术的技术演变。本文探讨了物理可实现攻击如何通过对抗贴纸、3D纹理和动态光学扰动，成功地将数字漏洞与现实世界威胁相衔接。此外，本文还探讨了利用内部表示中的语义结构来生成更可传递和有意义的对抗样本的潜在空间攻击。除了传统的进攻性应用外，本文还研究了对抗技术在生物认证系统漏洞评估以及防恶意生成模型方面的建设性应用。本文分析揭示了关键的研究空白，特别是在神经风格迁移保护和计算效率方面的需求。本文贡献了一个全面的分类、演变分析和未来研究方向的识别，旨在增进对抗漏洞的理解，并指导更鲁棒和可信赖的计算机视觉系统的发展。 

---
# Neural Predictive Control to Coordinate Discrete- and Continuous-Time Models for Time-Series Analysis with Control-Theoretical Improvements 

**Title (ZH)**: 基于神经预测控制的离散-连续时间模型协调方法及其控制理论改进的时间序列分析 

**Authors**: Haoran Li, Muhao Guo, Yang Weng, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01833)  

**Abstract**: Deep sequence models have achieved notable success in time-series analysis, such as interpolation and forecasting. Recent advances move beyond discrete-time architectures like Recurrent Neural Networks (RNNs) toward continuous-time formulations such as the family of Neural Ordinary Differential Equations (Neural ODEs). Generally, they have shown that capturing the underlying dynamics is beneficial for generic tasks like interpolation, extrapolation, and classification. However, existing methods approximate the dynamics using unconstrained neural networks, which struggle to adapt reliably under distributional shifts. In this paper, we recast time-series problems as the continuous ODE-based optimal control problem. Rather than learning dynamics solely from data, we optimize control actions that steer ODE trajectories toward task objectives, bringing control-theoretical performance guarantees. To achieve this goal, we need to (1) design the appropriate control actions and (2) apply effective optimal control algorithms. As the actions should contain rich context information, we propose to employ the discrete-time model to process past sequences and generate actions, leading to a coordinate model to extract long-term temporal features to modulate short-term continuous dynamics. During training, we apply model predictive control to plan multi-step future trajectories, minimize a task-specific cost, and greedily select the optimal current action. We show that, under mild assumptions, this multi-horizon optimization leads to exponential convergence to infinite-horizon solutions, indicating that the coordinate model can gain robust and generalizable performance. Extensive experiments on diverse time-series datasets validate our method's superior generalization and adaptability compared to state-of-the-art baselines. 

**Abstract (ZH)**: 深度序列模型在时间序列分析、插值和预测方面取得了显著成果。最近的进步超越了离散时间架构如循环神经网络（RNNs），转向了神经常微分方程（Neural ODEs）等连续时间形式。通常，它们显示捕获潜在动态对于通用任务如插值、外推和分类是有益的。然而，现有方法使用无约束神经网络近似动态，这在分布转移下难以可靠适应。在本文中，我们将时间序列问题重新表述为基于连续微分方程的最优控制问题。我们不仅从数据中学习动态，还优化控制动作，使其引导常微分方程轨迹趋向任务目标，从而带来控制理论的性能保证。为了实现这一目标，我们需要（1）设计适当的控制动作，（2）应用有效的最优控制算法。由于动作应包含丰富的上下文信息，我们提出使用离散时间模型处理过去序列并生成动作，导致坐标模型提取长期时间特征以调节短期连续动态。在训练过程中，我们应用模型预测控制规划多步未来轨迹，最小化特定任务成本，并贪婪地选择当前最优动作。我们显示，假设较多，此多视窗优化可导致指数收敛至无限视窗解，表明坐标模型可以获得稳健且可泛化的性能。广泛的时间序列数据集实验验证了我们方法在泛化和适应性方面的优越性，优于最先进的基线方法。 

---
# Deep Learning-Driven Prediction of Microstructure Evolution via Latent Space Interpolation 

**Title (ZH)**: 基于潜在空间插值的深度学习驱动的微观结构演化预测 

**Authors**: Sachin Gaikwad, Thejas Kasilingam, Owais Ahmad, Rajdip Mukherjee, Somnath Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2508.01822)  

**Abstract**: Phase-field models accurately simulate microstructure evolution, but their dependence on solving complex differential equations makes them computationally expensive. This work achieves a significant acceleration via a novel deep learning-based framework, utilizing a Conditional Variational Autoencoder (CVAE) coupled with Cubic Spline Interpolation and Spherical Linear Interpolation (SLERP). We demonstrate the method for binary spinodal decomposition by predicting microstructure evolution for intermediate alloy compositions from a limited set of training compositions. First, using microstructures from phase-field simulations of binary spinodal decomposition, we train the CVAE, which learns compact latent representations that encode essential morphological features. Next, we use cubic spline interpolation in the latent space to predict microstructures for any unknown composition. Finally, SLERP ensures smooth morphological evolution with time that closely resembles coarsening. The predicted microstructures exhibit high visual and statistical similarity to phase-field simulations. This framework offers a scalable and efficient surrogate model for microstructure evolution, enabling accelerated materials design and composition optimization. 

**Abstract (ZH)**: 基于深度学习的新型框架显著加速相场模型的微结构演化模拟：利用条件变分自编码器结合三次样条插值和球面线性插值实现二元相溶分解的微结构演化预测 

---
# AGENTICT$^2$S:Robust Text-to-SPARQL via Agentic Collaborative Reasoning over Heterogeneous Knowledge Graphs for the Circular Economy 

**Title (ZH)**: 基于代理协作推理的AGENTICT$^2$S：面向循环经济的 robust 文本到 SPARQL 转换 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Tan Chuan Fu, Yue Xiu, Dusit Niyato, Jonathan Z. Low, Eugene Ho Hong Zhuang, Daren Zong Loong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01815)  

**Abstract**: Question answering over heterogeneous knowledge graphs (KGQA) involves reasoning across diverse schemas, incomplete alignments, and distributed data sources. Existing text-to-SPARQL approaches rely on large-scale domain-specific fine-tuning or operate within single-graph settings, limiting their generalizability in low-resource domains and their ability to handle queries spanning multiple graphs. These challenges are particularly relevant in domains such as the circular economy, where information about classifications, processes, and emissions is distributed across independently curated knowledge graphs (KGs). We present AgenticT$^2$S, a modular framework that decomposes KGQA into subtasks managed by specialized agents responsible for retrieval, query generation, and verification. A scheduler assigns subgoals to different graphs using weak-to-strong alignment strategies. A two-stage verifier detects structurally invalid and semantically underspecified queries through symbolic validation and counterfactual consistency checks. Experiments on real-world circular economy KGs demonstrate that AgenticT$^2$S improves execution accuracy by 17.3% and triple level F$_1$ by 25.4% over the best baseline, while reducing the average prompt length by 46.4%. These results demonstrate the benefits of agent-based schema-aware reasoning for scalable KGQA and support decision-making in sustainability domains through robust cross-graph reasoning. 

**Abstract (ZH)**: 基于异构知识图谱的问答（KGQA）涉及跨越多样模式、不完整对齐和分布式数据源的推理。现有的文本到SPARQL方法依赖于大规模领域特定的微调或限定在单个图的环境中，限制了其在资源贫乏领域中的普适性及其处理跨越多个图的查询的能力。这些挑战在循环经济等领域尤为相关，在这些领域中，关于分类、过程和排放的信息分布在独立维护的知识图谱（KGs）中。我们提出了一种模块化框架AgenticT$^2$S，将KGQA分解为由专门代理管理的子任务，这些代理负责检索、查询生成和验证。调度器使用从弱到强的对齐策略为不同的图分配子目标。两阶段验证器通过符号验证和反事实一致性检查检测结构上无效和语义上不明确的查询。实验证实在实际的循环经济KG上，AgenticT$^2$S将执行准确性提高了17.3%，三元组水平的F$_1$分数提高了25.4%，同时将平均提示长度减少了46.4%。这些结果表明基于代理的模式感知推理对于可扩展的KGQA的好处，并通过稳健的跨图推理支持可持续性领域中的决策制定。 

---
# HeQ: a Large and Diverse Hebrew Reading Comprehension Benchmark 

**Title (ZH)**: HeQ：一个大规模且多样的希伯来阅读理解基准 

**Authors**: Amir DN Cohen, Hilla Merhav, Yoav Goldberg, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.01812)  

**Abstract**: Current benchmarks for Hebrew Natural Language Processing (NLP) focus mainly on morpho-syntactic tasks, neglecting the semantic dimension of language understanding. To bridge this gap, we set out to deliver a Hebrew Machine Reading Comprehension (MRC) dataset, where MRC is to be realized as extractive Question Answering. The morphologically rich nature of Hebrew poses a challenge to this endeavor: the indeterminacy and non-transparency of span boundaries in morphologically complex forms lead to annotation inconsistencies, disagreements, and flaws in standard evaluation metrics.
To remedy this, we devise a novel set of guidelines, a controlled crowdsourcing protocol, and revised evaluation metrics that are suitable for the morphologically rich nature of the language. Our resulting benchmark, HeQ (Hebrew QA), features 30,147 diverse question-answer pairs derived from both Hebrew Wikipedia articles and Israeli tech news. Our empirical investigation reveals that standard evaluation metrics such as F1 scores and Exact Match (EM) are not appropriate for Hebrew (and other MRLs), and we propose a relevant enhancement.
In addition, our experiments show low correlation between models' performance on morpho-syntactic tasks and on MRC, which suggests that models designed for the former might underperform on semantics-heavy tasks. The development and exploration of HeQ illustrate some of the challenges MRLs pose in natural language understanding (NLU), fostering progression towards more and better NLU models for Hebrew and other MRLs. 

**Abstract (ZH)**: 当前用于_hebrew_自然语言处理(NLP)的基准主要集中在形态学和句法任务上，忽视了语言理解的语义维度。为了弥合这一差距，我们致力于提供一个希伯来机器阅读理解(MRC)数据集，其中MRC将实现为抽取式问答。希伯来语丰富的形态学特性给这一努力带来了挑战：形态学复杂形式中的跨度边界模糊和不透明导致了标注不一致、歧义和标准评价指标中的缺陷。

为了解决这个问题，我们设计了一套新的指南，一个受控的众包协议，以及适用于该语言丰富形态学特性的修订评价指标。我们得到的基准数据集HeQ (希伯来语QA)包含了30,147个来自希伯来维基百科文章和以色列科技新闻的多样化的问答对。我们的实证研究表明，标准评价指标如F1分数和精确匹配(EM)对于希伯来语（以及其他MRLs）并不合适，并提出了相关的改进方案。

此外，我们的实验表明，模型在形态学和句法任务上的表现与在MRC上的表现之间存在较低的相关性，这表明适用于前者的设计可能在语义密集型任务上表现不佳。HeQ的发展和探索揭示了MRLs在自然语言理解(NLU)中所面临的挑战，促进了对希伯来语和其他MRLs更先进、更高质量的NLU模型的发展。 

---
# Mitigating Persistent Client Dropout in Asynchronous Decentralized Federated Learning 

**Title (ZH)**: 缓解异步去中心化联邦学习中的持续客户端退出问题 

**Authors**: Ignacy Stępka, Nicholas Gisolfi, Kacper Trębacz, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2508.01807)  

**Abstract**: We consider the problem of persistent client dropout in asynchronous Decentralized Federated Learning (DFL). Asynchronicity and decentralization obfuscate information about model updates among federation peers, making recovery from a client dropout difficult. Access to the number of learning epochs, data distributions, and all the information necessary to precisely reconstruct the missing neighbor's loss functions is limited. We show that obvious mitigations do not adequately address the problem and introduce adaptive strategies based on client reconstruction. We show that these strategies can effectively recover some performance loss caused by dropout. Our work focuses on asynchronous DFL with local regularization and differs substantially from that in the existing literature. We evaluate the proposed methods on tabular and image datasets, involve three DFL algorithms, and three data heterogeneity scenarios (iid, non-iid, class-focused non-iid). Our experiments show that the proposed adaptive strategies can be effective in maintaining robustness of federated learning, even if they do not reconstruct the missing client's data precisely. We also discuss the limitations and identify future avenues for tackling the problem of client dropout. 

**Abstract (ZH)**: 我们在异步去中心化联邦学习中的持久客户端辍学问题 

---
# Contrastive Multi-Task Learning with Solvent-Aware Augmentation for Drug Discovery 

**Title (ZH)**: 溶剂 Awareness 增强的对比多任务学习在药物发现中的应用 

**Authors**: Jing Lan, Hexiao Ding, Hongzhao Chen, Yufeng Jiang, Ng Nga Chun, Gerald W.Y. Cheng, Zongxi Li, Jing Cai, Liang-ting Lin, Jung Sun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01799)  

**Abstract**: Accurate prediction of protein-ligand interactions is essential for computer-aided drug discovery. However, existing methods often fail to capture solvent-dependent conformational changes and lack the ability to jointly learn multiple related tasks. To address these limitations, we introduce a pre-training method that incorporates ligand conformational ensembles generated under diverse solvent conditions as augmented input. This design enables the model to learn both structural flexibility and environmental context in a unified manner. The training process integrates molecular reconstruction to capture local geometry, interatomic distance prediction to model spatial relationships, and contrastive learning to build solvent-invariant molecular representations. Together, these components lead to significant improvements, including a 3.7% gain in binding affinity prediction, an 82% success rate on the PoseBusters Astex docking benchmarks, and an area under the curve of 97.1% in virtual screening. The framework supports solvent-aware, multi-task modeling and produces consistent results across benchmarks. A case study further demonstrates sub-angstrom docking accuracy with a root-mean-square deviation of 0.157 angstroms, offering atomic-level insight into binding mechanisms and advancing structure-based drug design. 

**Abstract (ZH)**: 准确预测蛋白质-配体相互作用对于计算机辅助药物发现至关重要。然而，现有方法往往难以捕捉溶剂依赖的构象变化，且缺乏联合学习多个相关任务的能力。为解决这些局限性，我们提出了一种预训练方法，该方法将不同溶剂条件下生成的配体构象ensemble作为增强输入纳入其中。该设计使模型能够以统一的方式学习结构灵活性和环境上下文。训练过程整合了分子重构以捕获局部几何结构、原子间距离预测以建模空间关系，以及对比学习以构建溶剂不变的分子表示。这些组成部分共同带来了显著的改进，包括3.7%的结合亲和力预测提升、PoseBusters Astex 锚定基准测试中82%的成功率，以及虚拟筛选中97.1%的曲线下面积。该框架支持溶剂感知的多任务建模，并在不同的基准测试中产生一致的结果。进一步的案例研究展示了亚埃级别的锚定准确性，根均方偏差为0.157埃，提供了Binding机制的原子级洞察，并推动了基于结构的药物设计。 

---
# CSLRConformer: A Data-Centric Conformer Approach for Continuous Arabic Sign Language Recognition on the Isharah Datase 

**Title (ZH)**: CSLRConformer: 一种以数据为中心的阿拉伯连续手语识别方法基于Isharah数据集 

**Authors**: Fatimah Mohamed Emad Elden  

**Link**: [PDF](https://arxiv.org/pdf/2508.01791)  

**Abstract**: The field of Continuous Sign Language Recognition (CSLR) poses substantial technical challenges, including fluid inter-sign transitions, the absence of temporal boundaries, and co-articulation effects. This paper, developed for the MSLR 2025 Workshop Challenge at ICCV 2025, addresses the critical challenge of signer-independent recognition to advance the generalization capabilities of CSLR systems across diverse signers. A data-centric methodology is proposed, centered on systematic feature engineering, a robust preprocessing pipeline, and an optimized model architecture. Key contributions include a principled feature selection process guided by Exploratory Data Analysis (EDA) to isolate communicative keypoints, a rigorous preprocessing pipeline incorporating DBSCAN-based outlier filtering and spatial normalization, and the novel CSLRConformer architecture. This architecture adapts the hybrid CNN-Transformer design of the Conformer model, leveraging its capacity to model local temporal dependencies and global sequence context; a characteristic uniquely suited for the spatio-temporal dynamics of sign language. The proposed methodology achieved a competitive performance, with a Word Error Rate (WER) of 5.60% on the development set and 12.01% on the test set, a result that secured a 3rd place ranking on the official competition platform. This research validates the efficacy of cross-domain architectural adaptation, demonstrating that the Conformer model, originally conceived for speech recognition, can be successfully repurposed to establish a new state-of-the-art performance in keypoint-based CSLR. 

**Abstract (ZH)**: 连续手语识别领域的技术挑战及MSLR 2025 Workshop Challenge在ICCV 2025上的解决方案：基于数据驱动的方法和CSLRConformer架构 

---
# RouteMark: A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging 

**Title (ZH)**: RouteMark：基于路由合并模型中的知识产权归属指纹技术 

**Authors**: Xin He, Junxi Shen, Zhenheng Tang, Xiaowen Chu, Bo Li, Ivor W. Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01784)  

**Abstract**: Model merging via Mixture-of-Experts (MoE) has emerged as a scalable solution for consolidating multiple task-specific models into a unified sparse architecture, where each expert is derived from a model fine-tuned on a distinct task. While effective for multi-task integration, this paradigm introduces a critical yet underexplored challenge: how to attribute and protect the intellectual property (IP) of individual experts after merging. We propose RouteMark, a framework for IP protection in merged MoE models through the design of expert routing fingerprints. Our key insight is that task-specific experts exhibit stable and distinctive routing behaviors under probing inputs. To capture these patterns, we construct expert-level fingerprints using two complementary statistics: the Routing Score Fingerprint (RSF), quantifying the intensity of expert activation, and the Routing Preference Fingerprint (RPF), characterizing the input distribution that preferentially activates each expert. These fingerprints are reproducible, task-discriminative, and lightweight to construct. For attribution and tampering detection, we introduce a similarity-based matching algorithm that compares expert fingerprints between a suspect and a reference (victim) model. Extensive experiments across diverse tasks and CLIP-based MoE architectures show that RouteMark consistently yields high similarity for reused experts and clear separation from unrelated ones. Moreover, it remains robust against both structural tampering (expert replacement, addition, deletion) and parametric tampering (fine-tuning, pruning, permutation), outperforming weight- and activation-based baseliness. Our work lays the foundation for RouteMark as a practical and broadly applicable framework for IP verification in MoE-based model merging. 

**Abstract (ZH)**: 基于Mixture-of-Experts (MoE)的模型合并中的知识产权保护：RouteMark框架 

---
# A comprehensive taxonomy of hallucinations in Large Language Models 

**Title (ZH)**: 大型语言模型中幻觉的综合分类 

**Authors**: Manuel Cossio  

**Link**: [PDF](https://arxiv.org/pdf/2508.01781)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their propensity for hallucination, generating plausible but factually incorrect or fabricated content, remains a critical challenge. This report provides a comprehensive taxonomy of LLM hallucinations, beginning with a formal definition and a theoretical framework that posits its inherent inevitability in computable LLMs, irrespective of architecture or training. It explores core distinctions, differentiating between intrinsic (contradicting input context) and extrinsic (inconsistent with training data or reality), as well as factuality (absolute correctness) and faithfulness (adherence to input). The report then details specific manifestations, including factual errors, contextual and logical inconsistencies, temporal disorientation, ethical violations, and task-specific hallucinations across domains like code generation and multimodal applications. It analyzes the underlying causes, categorizing them into data-related issues, model-related factors, and prompt-related influences. Furthermore, the report examines cognitive and human factors influencing hallucination perception, surveys evaluation benchmarks and metrics for detection, and outlines architectural and systemic mitigation strategies. Finally, it introduces web-based resources for monitoring LLM releases and performance. This report underscores the complex, multifaceted nature of LLM hallucinations and emphasizes that, given their theoretical inevitability, future efforts must focus on robust detection, mitigation, and continuous human oversight for responsible and reliable deployment in critical applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在自然语言处理领域引发革命，但其倾向性幻觉，生成虽然可信但事实错误或虚构的内容，仍然是一个关键挑战。本报告提供了LLM幻觉的全面分类，从正式定义和理论框架开始，后者提出了在可计算LLM中的内在不可避免性，不论架构或训练如何。报告探讨了核心区别，区分了内生性幻觉（与输入上下文矛盾）和外生性幻觉（与训练数据或现实不一致），以及事实性和忠实性。随后，报告详细说明了特定表现形式，包括事实错误、上下文和逻辑不一致、时间错位、伦理违规以及特定任务的幻觉，涉及诸如代码生成和多模态应用等多个领域。报告分析了潜在的原因，将其分类为数据问题、模型因素和提示影响。此外，报告还研究了影响幻觉感知的心理和人类因素，评估了检测基准和指标，并概述了架构和系统性缓解策略。最后，报告介绍了用于监测LLM发布和性能的网络资源。本报告强调了LLM幻觉的复杂性和多维性，并强调鉴于其理论上的不可避免性，未来的工作必须集中在稳健的检测、缓解和持续的人类监督上，以确保在关键应用中的负责任和可靠部署。 

---
# VAGPO: Vision-augmented Asymmetric Group Preference Optimization for the Routing Problems 

**Title (ZH)**: 基于视觉增强非对称群体偏好的路径优化方法 

**Authors**: Shiyan Liu, Bohan Tan, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01774)  

**Abstract**: The routing problems such as the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) are well-known combinatorial optimization challenges with broad practical relevance. Recent data-driven optimization methods have made significant progress, yet they often face limitations in training efficiency and generalization to large-scale instances. In this paper, we propose a novel Vision-Augmented Asymmetric Group Preference Optimization (VAGPO) approach for solving the routing problems. By leveraging ResNet-based visual encoding and Transformer-based sequential modeling, VAGPO captures both spatial structure and temporal dependencies. Furthermore, we introduce an asymmetric group preference optimization strategy that significantly accelerates convergence compared to commonly used policy gradient methods. Experimental results on TSP and CVRP benchmarks show that the proposed VAGPO not only achieves highly competitive solution quality but also exhibits strong generalization to larger instances (up to 1000 nodes) without re-training, highlighting its effectiveness in both learning efficiency and scalability. 

**Abstract (ZH)**: 视觉增强非对称组偏好优化方法（VAGPO）求解路由问题 

---
# LoRA-based methods on Unet for transfer learning in Subarachnoid Hematoma Segmentation 

**Title (ZH)**: 基于LoRA的方法在Subarachnoid Hematoma分割中的迁移学习应用 

**Authors**: Cristian Minoccheri, Matthew Hodgman, Haoyuan Ma, Rameez Merchant, Emily Wittrup, Craig Williamson, Kayvan Najarian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01772)  

**Abstract**: Aneurysmal subarachnoid hemorrhage (SAH) is a life-threatening neurological emergency with mortality rates exceeding 30%. Transfer learning from related hematoma types represents a potentially valuable but underexplored approach. Although Unet architectures remain the gold standard for medical image segmentation due to their effectiveness on limited datasets, Low-Rank Adaptation (LoRA) methods for parameter-efficient transfer learning have been rarely applied to convolutional neural networks in medical imaging contexts. We implemented a Unet architecture pre-trained on computed tomography scans from 124 traumatic brain injury patients across multiple institutions, then fine-tuned on 30 aneurysmal SAH patients from the University of Michigan Health System using 3-fold cross-validation. We developed a novel CP-LoRA method based on tensor CP-decomposition and introduced DoRA variants (DoRA-C, convDoRA, CP-DoRA) that decompose weight matrices into magnitude and directional components. We compared these approaches against existing LoRA methods (LoRA-C, convLoRA) and standard fine-tuning strategies across different modules on a multi-view Unet model. LoRA-based methods consistently outperformed standard Unet fine-tuning. Performance varied by hemorrhage volume, with all methods showing improved accuracy for larger volumes. CP-LoRA achieved comparable performance to existing methods while using significantly fewer parameters. Over-parameterization with higher ranks consistently yielded better performance than strictly low-rank adaptations. This study demonstrates that transfer learning between hematoma types is feasible and that LoRA-based methods significantly outperform conventional Unet fine-tuning for aneurysmal SAH segmentation. 

**Abstract (ZH)**: 颅内动脉瘤性蛛网膜下腔出血的动脉瘤类型间迁移学习：基于LoRA的方法在医学图像分割中的应用 

---
# Semantically-Guided Inference for Conditional Diffusion Models: Enhancing Covariate Consistency in Time Series Forecasting 

**Title (ZH)**: 基于语义指导的条件扩散模型推理：增强时间序列预测中的协变量一致性 

**Authors**: Rui Ding, Hanyang Meng, Zeyang Zhang, Jielong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01761)  

**Abstract**: Diffusion models have demonstrated strong performance in time series forecasting, yet often suffer from semantic misalignment between generated trajectories and conditioning covariates, especially under complex or multimodal conditions. To address this issue, we propose SemGuide, a plug-and-play, inference-time method that enhances covariate consistency in conditional diffusion models. Our approach introduces a scoring network to assess the semantic alignment between intermediate diffusion states and future covariates. These scores serve as proxy likelihoods in a stepwise importance reweighting procedure, which progressively adjusts the sampling path without altering the original training process. The method is model-agnostic and compatible with any conditional diffusion framework. Experiments on real-world forecasting tasks show consistent gains in both predictive accuracy and covariate alignment, with especially strong performance under complex conditioning scenarios. 

**Abstract (ZH)**: 基于注释指导的条件差分模型语义一致性增强方法：应用于时间序列预测 

---
# Vision transformer-based multi-camera multi-object tracking framework for dairy cow monitoring 

**Title (ZH)**: 基于视觉变换器的多相机多对象跟踪框架用于奶牛监测 

**Authors**: Kumail Abbas, Zeeshan Afzal, Aqeel Raza, Taha Mansouri, Andrew W. Dowsey, Chaidate Inchaisri, Ali Alameer  

**Link**: [PDF](https://arxiv.org/pdf/2508.01752)  

**Abstract**: Activity and behaviour correlate with dairy cow health and welfare, making continual and accurate monitoring crucial for disease identification and farm productivity. Manual observation and frequent assessments are laborious and inconsistent for activity monitoring. In this study, we developed a unique multi-camera, real-time tracking system for indoor-housed Holstein Friesian dairy cows. This technology uses cutting-edge computer vision techniques, including instance segmentation and tracking algorithms to monitor cow activity seamlessly and accurately. An integrated top-down barn panorama was created by geometrically aligning six camera feeds using homographic transformations. The detection phase used a refined YOLO11-m model trained on an overhead cow dataset, obtaining high accuracy (mAP\@0.50 = 0.97, F1 = 0.95). SAMURAI, an upgraded Segment Anything Model 2.1, generated pixel-precise cow masks for instance segmentation utilizing zero-shot learning and motion-aware memory. Even with occlusion and fluctuating posture, a motion-aware Linear Kalman filter and IoU-based data association reliably identified cows over time for object tracking. The proposed system significantly outperformed Deep SORT Realtime. Multi-Object Tracking Accuracy (MOTA) was 98.7% and 99.3% in two benchmark video sequences, with IDF1 scores above 99% and near-zero identity switches. This unified multi-camera system can track dairy cows in complex interior surroundings in real time, according to our data. The system reduces redundant detections across overlapping cameras, maintains continuity as cows move between viewpoints, with the aim of improving early sickness prediction through activity quantification and behavioural classification. 

**Abstract (ZH)**: 基于多摄像头的实时跟踪系统在室内舍荷斯坦奶牛活动与行为监测中的应用 

---
# Improving Noise Efficiency in Privacy-preserving Dataset Distillation 

**Title (ZH)**: 提高隐私保护数据蒸馏中的噪声效率 

**Authors**: Runkai Zheng, Vishnu Asutosh Dasu, Yinong Oliver Wang, Haohan Wang, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2508.01749)  

**Abstract**: Modern machine learning models heavily rely on large datasets that often include sensitive and private information, raising serious privacy concerns. Differentially private (DP) data generation offers a solution by creating synthetic datasets that limit the leakage of private information within a predefined privacy budget; however, it requires a substantial amount of data to achieve performance comparable to models trained on the original data. To mitigate the significant expense incurred with synthetic data generation, Dataset Distillation (DD) stands out for its remarkable training and storage efficiency. This efficiency is particularly advantageous when integrated with DP mechanisms, curating compact yet informative synthetic datasets without compromising privacy. However, current state-of-the-art private DD methods suffer from a synchronized sampling-optimization process and the dependency on noisy training signals from randomly initialized networks. This results in the inefficient utilization of private information due to the addition of excessive noise. To address these issues, we introduce a novel framework that decouples sampling from optimization for better convergence and improves signal quality by mitigating the impact of DP noise through matching in an informative subspace. On CIFAR-10, our method achieves a \textbf{10.0\%} improvement with 50 images per class and \textbf{8.3\%} increase with just \textbf{one-fifth} the distilled set size of previous state-of-the-art methods, demonstrating significant potential to advance privacy-preserving DD. 

**Abstract (ZH)**: 现代机器学习模型高度依赖大規模数据集，这些数据集通常包含敏感和私人信息，引发了严重的隐私问题。不同隐私（DP）数据生成通过创建合成数据集来限制在预定义隐私预算内的私人信息泄露，但它需要大量数据以达到与原始数据训练模型相当的性能。为了缓解合成数据生成带来的显著成本，数据集蒸馏（DD）因其出色的训练和存储效率而脱颖而出。当与DP机制结合时，这种效率特别有利，可以生成紧凑且富有信息量的合成数据集，而不牺牲隐私。然而，当前最先进的私密DD方法面临同步采样-优化过程和对随机初始化网络噪声训练信号的依赖，导致由于过多添加噪声而导致私人信息的低效利用。为了应对这些问题，我们提出了一种新颖的框架，该框架解耦了采样和优化，通过匹配在信息子空间中降低DP噪声的影响来改善信号质量。在CIFAR-10上，我们的方法在每类50张图片的情况下实现了10.0%的改进，并且仅使用之前最先进的方法五分之一的蒸馏集合大小就实现了8.3%的提升，这表明在保护隐私的同时进行DD有很大前景。 

---
# Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation 

**Title (ZH)**: 基于意图的认知推理实现自我中心长期行动预测 

**Authors**: Qiaohui Chu, Haoyu Zhang, Meng Liu, Yisen Feng, Haoxiang Shi, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.01742)  

**Abstract**: Long-term action anticipation from egocentric video is critical for applications such as human-computer interaction and assistive technologies, where anticipating user intent enables proactive and context-aware AI assistance. However, existing approaches suffer from three key limitations: 1) underutilization of fine-grained visual cues from hand-object interactions, 2) neglect of semantic dependencies between verbs and nouns, and 3) lack of explicit cognitive reasoning, limiting generalization and long-term forecasting ability. To overcome these challenges, we propose INSIGHT, a unified two-stage framework for egocentric action anticipation. In the first stage, INSIGHT focuses on extracting semantically rich features from hand-object interaction regions and enhances action representations using a verb-noun co-occurrence matrix. In the second stage, it introduces a reinforcement learning-based module that simulates explicit cognitive reasoning through a structured process: visual perception (think) -> intention inference (reason) -> action anticipation (answer). Extensive experiments on Ego4D, EPIC-Kitchens-55, and EGTEA Gaze+ benchmarks show that INSIGHT achieves state-of-the-art performance, demonstrating its effectiveness and strong generalization capability. 

**Abstract (ZH)**: 从第一人称视角视频中预见长期动作对于人机交互和辅助技术等应用至关重要，其中预见用户意图能够提供主动且情境感知的AI辅助。然而，现有方法存在三个关键限制：1) 手物交互的细粒度视觉线索利用不足，2) 忽视动词和名词的语义依赖关系，3) 缺乏显式的认知推理，限制了其泛化能力和长期预测能力。为克服这些挑战，我们提出INSIGHT，一种统一的两阶段框架，用于第一人称视角动作预见。在第一阶段，INSIGHT专注于从手物交互区域中提取丰富的语义特征，并使用动词-名词共现矩阵增强动作表示。在第二阶段，它引入了一个基于强化学习的模块，通过结构化的过程模拟显式的认知推理：视觉感知（思考）-> 意图推理（推理）-> 动作预见（解答）。在Ego4D、EPIC-Kitchens-55和EGTEA Gaze+基准上的 extensive 实验中，INSIGHT 达到了最先进的性能，证明了其有效性和强大的泛化能力。 

---
# Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations 

**Title (ZH)**: 细粒度概念电路：概念表示的精细电路发现 toward 细粒度概念电路：概念表示的精细电路发现 

**Authors**: Dahee Kwon, Sehyun Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01728)  

**Abstract**: Deep vision models have achieved remarkable classification performance by leveraging a hierarchical architecture in which human-interpretable concepts emerge through the composition of individual neurons across layers. Given the distributed nature of representations, pinpointing where specific visual concepts are encoded within a model remains a crucial yet challenging task. In this paper, we introduce an effective circuit discovery method, called Granular Concept Circuit (GCC), in which each circuit represents a concept relevant to a given query. To construct each circuit, our method iteratively assesses inter-neuron connectivity, focusing on both functional dependencies and semantic alignment. By automatically discovering multiple circuits, each capturing specific concepts within that query, our approach offers a profound, concept-wise interpretation of models and is the first to identify circuits tied to specific visual concepts at a fine-grained level. We validate the versatility and effectiveness of GCCs across various deep image classification models. 

**Abstract (ZH)**: 深层次的视觉模型通过利用层次结构，其中个体神经元在各层中的组合产生了可由人类解释的概念，从而实现了显著的分类性能。鉴于表示的分布性质，确定模型中特定视觉概念的编码位置仍然是一个关键而具有挑战性的问题。在本文中，我们介绍了一种有效电路发现方法，称为粒度概念电路（GCC），其中每条电路代表与给定查询相关的概念。为了构建每条电路，我们的方法逐次评估神经元间的连接性，重点在于功能依赖关系和语义对齐。通过自动发现多条电路，每条电路捕捉查询中特定的概念，我们的方法提供了概念层面的深刻解释，并且是首次能够在细粒度级别识别与特定视觉概念相关联的电路。我们在多种深层图像分类模型中验证了GCCs的通用性和有效性。 

---
# Dynamic Robot-Assisted Surgery with Hierarchical Class-Incremental Semantic Segmentation 

**Title (ZH)**: 基于分层类增量语义分割的动态机器人辅助手术 

**Authors**: Julia Hindel, Ema Mekic, Enamundram Naga Karthik, Rohit Mohan, Daniele Cattaneo, Maria Kalweit, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2508.01713)  

**Abstract**: Robot-assisted surgeries rely on accurate and real-time scene understanding to safely guide surgical instruments. However, segmentation models trained on static datasets face key limitations when deployed in these dynamic and evolving surgical environments. Class-incremental semantic segmentation (CISS) allows models to continually adapt to new classes while avoiding catastrophic forgetting of prior knowledge, without training on previous data. In this work, we build upon the recently introduced Taxonomy-Oriented Poincaré-regularized Incremental Class Segmentation (TOPICS) approach and propose an enhanced variant, termed TOPICS+, specifically tailored for robust segmentation of surgical scenes. Concretely, we incorporate the Dice loss into the hierarchical loss formulation to handle strong class imbalances, introduce hierarchical pseudo-labeling, and design tailored label taxonomies for robotic surgery environments. We also propose six novel CISS benchmarks designed for robotic surgery environments including multiple incremental steps and several semantic categories to emulate realistic class-incremental settings in surgical environments. In addition, we introduce a refined set of labels with more than 144 classes on the Syn-Mediverse synthetic dataset, hosted online as an evaluation benchmark. We make the code and trained models publicly available at this http URL. 

**Abstract (ZH)**: 机器人辅助手术依赖于准确且实时的场景理解以安全地指导手术器械。然而，部署在静态数据集上的分割模型在这些动态且不断变化的手术环境中面临关键限制。类别增量语义分割（CISS）允许模型在不断适应新类别的同时避免对先前知识的灾难性遗忘，而不需重新训练。在本文中，我们基于最近引入的面向分类的Poincaré正则化增量类别分割（TOPICS）方法，提出了一种增强变体TOPICS+，专门为机器人手术环境中的稳健分割设计。具体而言，我们将Dice损失纳入分层级损失公式以处理类别不平衡问题，引入分层级伪标签，并设计了适合机器人手术环境的标签分类。此外，我们提出了六个新的CISS基准，专为机器人手术环境设计，包括多个增量步骤和多个语义类别，以模拟手术环境中的实际类别增量设置。我们还在Syn-Mediverse合成数据集中引入了一个改进的标签集，包含超过144个类别，并在线提供作为评估基准。我们已将代码和训练模型公开发布在指定网址。 

---
# HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection 

**Title (ZH)**: HateClipSeg：一种细粒度仇恨视频检测的片段级标注数据集 

**Authors**: Han Wang, Zhuoran Wang, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01712)  

**Abstract**: Detecting hate speech in videos remains challenging due to the complexity of multimodal content and the lack of fine-grained annotations in existing datasets. We present HateClipSeg, a large-scale multimodal dataset with both video-level and segment-level annotations, comprising over 11,714 segments labeled as Normal or across five Offensive categories: Hateful, Insulting, Sexual, Violence, Self-Harm, along with explicit target victim labels. Our three-stage annotation process yields high inter-annotator agreement (Krippendorff's alpha = 0.817). We propose three tasks to benchmark performance: (1) Trimmed Hateful Video Classification, (2) Temporal Hateful Video Localization, and (3) Online Hateful Video Classification. Results highlight substantial gaps in current models, emphasizing the need for more sophisticated multimodal and temporally aware approaches. The HateClipSeg dataset are publicly available at this https URL. 

**Abstract (ZH)**: 检测视频中的仇恨言论仍具有挑战性，由于多模态内容的复杂性和现有数据集中缺少细粒度注释。我们 presents HateClipSeg，一个大规模多模态数据集，包含视频级和片段级注释，共有超过11,714个片段被标记为Normal或五个冒犯类别中的一个：仇恨、侮辱、性相关、暴力、自我伤害，以及明确的目标受害者标签。我们的三阶段注释过程获得了较高的注释者间一致性（Krippendorff’s α = 0.817）。我们提出了三个基准任务：(1) 剪辑仇恨视频分类，(2) 时光轴上的仇恨视频本地化，(3) 实时仇恨视频分类。结果突显了当前模型中的巨大差距，强调了需要更复杂的多模态和时间感知方法的必要性。HateClipSeg数据集可在以下链接获取：this https URL。 

---
# GAID: Frame-Level Gated Audio-Visual Integration with Directional Perturbation for Text-Video Retrieval 

**Title (ZH)**: GAID：具有方向扰动的帧级门控音视频集成用于文本视频检索 

**Authors**: Bowen Yang, Yun Cao, Chen He, Xiaosu Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01711)  

**Abstract**: Text-to-video retrieval requires precise alignment between language and temporally rich video signals. Existing methods predominantly exploit visual cues and often overlook complementary audio semantics or adopt coarse fusion strategies, leading to suboptimal multimodal representations. We present GAID, a framework that jointly address this gap via two key components: (i) a Frame-level Gated Fusion (FGF) that adaptively integrates audio and visual features under textual guidance, enabling fine-grained temporal alignment; and (ii) a Directional Adaptive Semantic Perturbation (DASP) that injects structure-aware perturbations into text embeddings, enhancing robustness and discrimination without incurring multi-pass inference. These modules complement each other -- fusion reduces modality gaps while perturbation regularizes cross-modal matching -- yielding more stable and expressive representations. Extensive experiments on MSR-VTT, DiDeMo, LSMDC, and VATEX show consistent state-of-the-art results across all retrieval metrics with notable efficiency gains. Our code is available at this https URL. 

**Abstract (ZH)**: 文本到视频检索要求语言和时空丰富的视频信号之间精确对齐。现有方法主要利用视觉线索，往往忽视互补的音频语义或采用粗放的融合策略，导致多模态表示不够优化。我们提出了一种GAID框架，通过两个关键组件共同解决这一问题：（i）帧级门控融合（FGF），在文本引导下适应性地整合音频和视觉特征，实现细粒度的时间对齐；（ii）方向自适应语义扰动（DASP），在文本嵌入中注入结构感知的扰动，增强鲁棒性和区分性而无需多遍推断。这些模块相互补充——融合减少了模态差异，而扰动正则化跨模态匹配——从而产生更稳定和更具表达力的表示。在MSR-VTT、DiDeMo、LSMDC和VATEX上的广泛实验显示，这些模块在所有检索指标上取得了一致的最新成果，并且具有显著的效率提升。我们的代码可在以下网址获取。 

---
# MHARFedLLM: Multimodal Human Activity Recognition Using Federated Large Language Model 

**Title (ZH)**: MHARFedLLM: 多模态人类活动识别的联邦大规模语言模型 

**Authors**: Asmit Bandyopadhyay, Rohit Basu, Tanmay Sen, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.01701)  

**Abstract**: Human Activity Recognition (HAR) plays a vital role in applications such as fitness tracking, smart homes, and healthcare monitoring. Traditional HAR systems often rely on single modalities, such as motion sensors or cameras, limiting robustness and accuracy in real-world environments. This work presents FedTime-MAGNET, a novel multimodal federated learning framework that advances HAR by combining heterogeneous data sources: depth cameras, pressure mats, and accelerometers. At its core is the Multimodal Adaptive Graph Neural Expert Transformer (MAGNET), a fusion architecture that uses graph attention and a Mixture of Experts to generate unified, discriminative embeddings across modalities. To capture complex temporal dependencies, a lightweight T5 encoder only architecture is customized and adapted within this framework. Extensive experiments show that FedTime-MAGNET significantly improves HAR performance, achieving a centralized F1 Score of 0.934 and a strong federated F1 Score of 0.881. These results demonstrate the effectiveness of combining multimodal fusion, time series LLMs, and federated learning for building accurate and robust HAR systems. 

**Abstract (ZH)**: 多模态联邦学习框架FedTime-MAGNET在人体活动识别中的应用 

---
# Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy 

**Title (ZH)**: 基于参数检索的知识协同的协作智能体链 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01696)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework for enhancing the capabilities of Large Language Models (LLMs), especially in knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to *fully exploit knowledge during generation*. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experiments results show that CoCoA-zero and CoCoA achieve superior performance on open-domain and multi-hop QA tasks. 

**Abstract (ZH)**: 检索增强生成(RAG)已 emerge 作为增强大型语言模型(LLMs)能力的有前途的框架，特别是在知识密集型任务中。尽管具有优势，当前的RAG方法在生成过程中往往难以全面利用知识。具体来说，模型内部参数化知识与外部检索知识之间的协同作用仍然有限。检索的内容有时会误导生成，而某些生成的内容可以引导模型生成更准确的结果。在本文中，我们提出了一种协作链式智能体(CoCoA)框架，旨在增强参数化知识和检索知识之间的显式协同作用。具体来说，我们首先引入了CoCoA-zero，这是一种多智能体RAG框架，先进行条件知识归纳，再进行推理以得出答案。在此基础上，我们开发了CoCoA，这是一种长链训练策略，从CoCoA-zero中合成扩展的多智能体推理轨迹，以微调LLM。这种策略增强了模型整合和联合利用参数化知识和检索知识的能力。实验结果表明，CoCoA-zero和CoCoA在开放式领域和多跳问答任务中表现出更优的性能。 

---
# From SHAP to Rules: Distilling Expert Knowledge from Post-hoc Model Explanations in Time Series Classification 

**Title (ZH)**: 从SHAP到规则：从时间序列分类模型后验解释中提炼专家知识 

**Authors**: Maciej Mozolewski, Szymon Bobek, Grzegorz J. Nalepa  

**Link**: [PDF](https://arxiv.org/pdf/2508.01687)  

**Abstract**: Explaining machine learning (ML) models for time series (TS) classification is challenging due to inherent difficulty in raw time series interpretation and doubled down by the high dimensionality. We propose a framework that converts numeric feature attributions from post-hoc, instance-wise explainers (e.g., LIME, SHAP) into structured, human-readable rules. These rules define intervals indicating when and where they apply, improving transparency. Our approach performs comparably to native rule-based methods like Anchor while scaling better to long TS and covering more instances. Rule fusion integrates rule sets through methods such as weighted selection and lasso-based refinement to balance coverage, confidence, and simplicity, ensuring all instances receive an unambiguous, metric-optimized rule. It enhances explanations even for a single explainer. We introduce visualization techniques to manage specificity-generalization trade-offs. By aligning with expert-system principles, our framework consolidates conflicting or overlapping explanations - often resulting from the Rashomon effect - into coherent and domain-adaptable insights. Experiments on UCI datasets confirm that the resulting rule-based representations improve interpretability, decision transparency, and practical applicability for TS classification. 

**Abstract (ZH)**: 基于时间序列分类的机器学习模型解释因原始时间序列解释的固有难度和高维性而具有挑战性。我们提出了一种框架，将后 hoc、实例级别的特征归因（例如，LIME、SHAP）转换为结构化、易于理解的规则。这些规则定义了指示其适用时间和范围的区间，提高了透明度。我们的方法在长时间序列和实例方面具有更好的扩展性，同时覆盖更多实例。规则融合通过加权选择和lasso基优化等方法整合规则集，以平衡覆盖范围、置信度和简洁性，确保所有实例都获得一个明确的、优化了的规则。即使对于单一解释器，这种方法也能增强解释。我们引入了可视化技术来管理具体性和通用性之间的权衡。通过与专家系统原则对齐，我们的框架将来自“拉什门辛效应”导致的矛盾或重叠解释整合为一致且适用于特定领域的见解。实验结果证实，基于规则的表示形式提高了时间序列分类的可解释性、决策透明度和实用适用性。 

---
# Cure or Poison? Embedding Instructions Visually Alters Hallucination in Vision-Language Models 

**Title (ZH)**: 治愈还是毒药？视觉嵌入指令改变视觉语言模型的幻觉 

**Authors**: Zhaochen Wang, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01678)  

**Abstract**: Vision-Language Models (VLMs) often suffer from hallucination, partly due to challenges in aligning multimodal information. We propose Prompt-in-Image, a simple method that embeds textual instructions directly into images. This removes the need for separate text inputs and forces the model to process all content through the visual channel. We evaluate this method on three popular open-source VLMs: Qwen2.5-VL, LLaVA-1.5, and InstructBLIP. The results reveal sharp differences. Prompt-in-Image improves Qwen2.5-VL's performance, increasing POPE accuracy by 4.1 percent (from 80.2 percent to 84.3 percent) and also reducing hallucination rates on MS-COCO. In contrast, LLaVA-1.5 and InstructBLIP experience a severe performance drop, with accuracy falling from around 84 percent to near-random levels. Through detailed analysis, we found that CLIP-based encoders in LLaVA and InstructBLIP exhibit excessive attention bias toward embedded text regions, disrupting visual understanding. In contrast, Qwen's vision encoder handles text-embedded images robustly. Crucially, Prompt-in-Image reduces Qwen's modality gap, enhancing cross-modal alignment by unifying information processing through a single modality. 

**Abstract (ZH)**: 基于图像的提示方法：Vision-Language模型中的文本指令直接嵌入图像以改善多模态对齐 

---
# CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions 

**Title (ZH)**: CUPID: 评估大规模语言模型从互动中实现的个性化和情境化对齐 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Yoonah Park, Jiho Kim, Young-Ho Kim, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01674)  

**Abstract**: Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements. 

**Abstract (ZH)**: 大型语言模型的个性化往往假设用户持有的偏好是静态的，并在全球任务中一致反映。实际上，人类的偏好是动态的，会根据情境变化。当用户在不同情境下与语言模型互动时，他们自然会揭示自己的情境偏好，模型必须推测并应用这些偏好以确保一致性和精准度。为此，我们引入了CUPID基准测试，包含756个人类策划的用户与基于语言模型的聊天助手的交互会话历史。在每个交互会话中，用户在特定情境下提出请求并通过多轮反馈表达偏好。给定新的用户请求和之前的交互会话，基准测试评估大型语言模型是否能够推断出与该请求相关的偏好，并生成能够满足该偏好的响应。通过CUPID，我们评估了10个开源和专有大型语言模型，发现最先进的大型语言模型在从多轮互动中推断偏好时表现不佳，在判断之前情境是否与新请求相关方面也表现差强人意——精度低于50%，召回率仅为65%。我们的工作强调了提高大型语言模型能力以实现更情境化的个性化交互的需求，并提议CUPID作为一种资源来推动这些改进。 

---
# Authorship Attribution in Multilingual Machine-Generated Texts 

**Title (ZH)**: 多语言机器生成文本的作者归属分析 

**Authors**: Lucio La Cava, Dominik Macko, Róbert Móro, Ivan Srba, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.01656)  

**Abstract**: As Large Language Models (LLMs) have reached human-like fluency and coherence, distinguishing machine-generated text (MGT) from human-written content becomes increasingly difficult. While early efforts in MGT detection have focused on binary classification, the growing landscape and diversity of LLMs require a more fine-grained yet challenging authorship attribution (AA), i.e., being able to identify the precise generator (LLM or human) behind a text. However, AA remains nowadays confined to a monolingual setting, with English being the most investigated one, overlooking the multilingual nature and usage of modern LLMs. In this work, we introduce the problem of Multilingual Authorship Attribution, which involves attributing texts to human or multiple LLM generators across diverse languages. Focusing on 18 languages -- covering multiple families and writing scripts -- and 8 generators (7 LLMs and the human-authored class), we investigate the multilingual suitability of monolingual AA methods, their cross-lingual transferability, and the impact of generators on attribution performance. Our results reveal that while certain monolingual AA methods can be adapted to multilingual settings, significant limitations and challenges remain, particularly in transferring across diverse language families, underscoring the complexity of multilingual AA and the need for more robust approaches to better match real-world scenarios. 

**Abstract (ZH)**: 多语言作者归属问题 

---
# MAP: Mitigating Hallucinations in Large Vision-Language Models with Map-Level Attention Processing 

**Title (ZH)**: MAP: Map级别注意力处理在大型视觉-语言模型中减轻幻觉问题 

**Authors**: Chenxi Li, Yichen Guo, Benfang Qian, Jinhao You, Kai Tang, Yaosong Du, Zonghao Zhang, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01653)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved impressive performance in multimodal tasks, but they still suffer from hallucinations, i.e., generating content that is grammatically accurate but inconsistent with visual inputs. In this work, we introduce a novel map-level perspective to mitigate hallucinations in LVLMs, interpreting the hidden states of the model as a 2D semantic map. We observe that factual information is widely distributed across this map, extending beyond the localized inter- or intra-layer regions targeted by most existing methods (e.g., contrastive decoding and layer-wise consistency). Building on this insight, we propose Map-Level Attention Processing (MAP), a training-free decoding method that effectively leverages factual information through attention-based map-level operations to improve factual consistency. Specifically, we employ Layer-Wise Criss-Cross Attention to progressively refine token representations at each decoding layer by aggregating tokens from both inter- and intra-layer dimensions. Additionally, a Global-Local Logit Fusion mechanism combines logits obtained before and after global attention to further refine predictions and improve accuracy. Our method consistently improves the truthfulness and performance of LVLMs across benchmarks, such as POPE, MME, and MMHal-Bench, demonstrating the potential of the map-level decoding strategy. 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs)在多模态任务中取得了令人印象深刻的性能，但仍存在幻觉问题，即生成语法规正确但与视觉输入不一致的内容。在本文中，我们提出了一种新的地图级视角来减轻LVLMs中的幻觉问题，将模型的隐藏状态解释为2D语义地图。我们观察到，事实信息在该地图上分布广泛，超越了大多数现有方法（例如对比解码和逐层一致性）所关注的局部跨层或同一层区域。基于这一洞察，我们提出了一种无需训练的解码方法——地图级注意力处理（MAP），通过基于注意力的地图级操作有效利用事实信息，提高事实一致性。具体来说，我们使用逐层交叉注意力，逐层逐步细化token表示，通过从跨层和同一层维度聚合token来实现。此外，全局-局部logit融合机制结合了在全局注意力前后获得的logits，进一步细化预测并提高准确性。我们的方法在POPE、MME和MMHal-Bench等基准测试中一致地提高了LVLMs的真相度和性能，表明地图级解码策略的潜力。 

---
# DUP: Detection-guided Unlearning for Backdoor Purification in Language Models 

**Title (ZH)**: DUP: 检测引导的后门消除以 purification 优化语言模型中的后门检测与消除 

**Authors**: Man Hu, Yahui Ding, Yatao Yang, Liangyu Chen, Yanhao Jia, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01647)  

**Abstract**: As backdoor attacks become more stealthy and robust, they reveal critical weaknesses in current defense strategies: detection methods often rely on coarse-grained feature statistics, and purification methods typically require full retraining or additional clean models. To address these challenges, we propose DUP (Detection-guided Unlearning for Purification), a unified framework that integrates backdoor detection with unlearning-based purification. The detector captures feature-level anomalies by jointly leveraging class-agnostic distances and inter-layer transitions. These deviations are integrated through a weighted scheme to identify poisoned inputs, enabling more fine-grained analysis. Based on the detection results, we purify the model through a parameter-efficient unlearning mechanism that avoids full retraining and does not require any external clean model. Specifically, we innovatively repurpose knowledge distillation to guide the student model toward increasing its output divergence from the teacher on detected poisoned samples, effectively forcing it to unlearn the backdoor behavior. Extensive experiments across diverse attack methods and language model architectures demonstrate that DUP achieves superior defense performance in detection accuracy and purification efficacy. Our code is available at this https URL. 

**Abstract (ZH)**: 背门攻击日益隐蔽和 robust，暴露出当前防御策略的关键弱点：检测方法往往依赖粗粒度的特征统计，而净化方法通常需要全面重训练或额外的干净模型。为应对这些挑战，我们提出了一种统一框架 DUP（Detection-guided Unlearning for Purification），该框架将背门检测与基于遗忘的净化相结合。检测器通过联合利用类无关的距离和层间过渡来捕捉特征级别的异常，并通过加权方案整合这些偏差，以识别中毒输入，从而实现更精细的分析。基于检测结果，我们通过一种参数高效的遗忘机制来净化模型，该机制避免了全面重训练，并且不需要任何外部干净模型。具体地，我们创新性地重新利用了知识蒸馏，以指导学生模型增加其在检测到的中毒样本上的输出差异，从而有效地促使它遗忘背门行为。在多种攻击方法和语言模型架构上的广泛实验表明，DUP 在检测准确性和净化效果方面表现出优越的防御性能。我们的代码可在以下链接获取：this https URL。 

---
# SPARTA: Advancing Sparse Attention in Spiking Neural Networks via Spike-Timing-Based Prioritization 

**Title (ZH)**: SPARTA: 基于脉冲时序优先级促进脉冲神经网络中稀疏注意机制的研究 

**Authors**: Minsuk Jang, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01646)  

**Abstract**: Current Spiking Neural Networks (SNNs) underutilize the temporal dynamics inherent in spike-based processing, relying primarily on rate coding while overlooking precise timing information that provides rich computational cues. We propose SPARTA (Spiking Priority Attention with Resource-Adaptive Temporal Allocation), a framework that leverages heterogeneous neuron dynamics and spike-timing information to enable efficient sparse attention. SPARTA prioritizes tokens based on temporal cues, including firing patterns, spike timing, and inter-spike intervals, achieving 65.4% sparsity through competitive gating. By selecting only the most salient tokens, SPARTA reduces attention complexity from O(N^2) to O(K^2) with k << n, while maintaining high accuracy. Our method achieves state-of-the-art performance on DVS-Gesture (98.78%) and competitive results on CIFAR10-DVS (83.06%) and CIFAR-10 (95.3%), demonstrating that exploiting spike timing dynamics improves both computational efficiency and accuracy. 

**Abstract (ZH)**: 当前的脉冲神经网络(SNNs)未能充分利用基于脉冲的处理中固有的时间动态性，主要依赖速率编码而忽视了提供丰富计算线索的精确时间信息。我们提出了一种名为SPARTA（Spiking Priority Attention with Resource-Adaptive Temporal Allocation）的框架，该框架利用异质神经元动力学和脉冲时间信息以实现高效的稀疏注意。SPARTA基于时间线索（包括放电模式、脉冲时间以及脉冲间间隔）对token进行优先级排序，通过竞争性门控达到65.4%的稀疏度。通过仅选择最显著的token，SPARTA将注意力复杂度从O(N^2)降低到O(K^2)（其中k<<n），同时保持高准确性。我们的方法在DVS-Gesture（98.78%）和CIFAR10-DVS（83.06%）以及CIFAR-10（95.3%）上实现了最佳性能，证明了利用脉冲时间动态性能够同时提高计算效率和准确性。 

---
# Semantic Encryption: Secure and Effective Interaction with Cloud-based Large Language Models via Semantic Transformation 

**Title (ZH)**: 语义加密：通过语义转换与基于云的大型语言模型安全有效交互 

**Authors**: Dong Chen, Tong Yang, Feipeng Zhai, Pengpeng Ouyang, Qidong Liu, Yafei Li, Chong Fu, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01638)  

**Abstract**: The increasing adoption of Cloud-based Large Language Models (CLLMs) has raised significant concerns regarding data privacy during user interactions. While existing approaches primarily focus on encrypting sensitive information, they often overlook the logical structure of user inputs. This oversight can lead to reduced data utility and degraded performance of CLLMs. To address these limitations and enable secure yet effective interactions, we propose Semantic Encryption (SE)-a plug-and-play framework designed to preserve both privacy and utility. SE consists of two key components: Semantic Encoding and Semantic Decoding. In the encoding phase, a lightweight local model transforms the original user input into an alternative semantic context that maintains the original intent and logical structure while obfuscating sensitive information. This transformed input is then processed by the CLLM, which generates a response based on the transformed semantic context. To maintain a seamless user experience, the decoding phase will reconstruct the CLLM's response back into the original semantic context by referencing the locally stored user input. Extensive experimental evaluations demonstrate that SE effectively protects data privacy without compromising data utility or user experience, offering a practical solution for secure interaction with CLLMs. Particularly, the proposed SE demonstrates a significant improvement over the state-of-the-art InferDPT, surpassing it across various evaluated metrics and datasets. 

**Abstract (ZH)**: 基于云的大型语言模型（CLLMs）数据隐私保护研究：一种语义加密框架 

---
# Learning Unified System Representations for Microservice Tail Latency Prediction 

**Title (ZH)**: 面向微服务延迟预测的统一系统表示学习 

**Authors**: Wenzhuo Qian, Hailiang Zhao, Tianlv Chen, Jiayi Chen, Ziqi Wang, Kingsum Chow, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01635)  

**Abstract**: Microservice architectures have become the de facto standard for building scalable cloud-native applications, yet their distributed nature introduces significant challenges in performance monitoring and resource management. Traditional approaches often rely on per-request latency metrics, which are highly sensitive to transient noise and fail to reflect the holistic behavior of complex, concurrent workloads. In contrast, window-level P95 tail latency provides a stable and meaningful signal that captures both system-wide trends and user-perceived performance degradation. We identify two key shortcomings in existing methods: (i) inadequate handling of heterogeneous data, where traffic-side features propagate across service dependencies and resource-side signals reflect localized bottlenecks, and (ii) the lack of principled architectural designs that effectively distinguish and integrate these complementary modalities. To address these challenges, we propose USRFNet, a deep learning network that explicitly separates and models traffic-side and resource-side features. USRFNet employs GNNs to capture service interactions and request propagation patterns, while gMLP modules independently model cluster resource dynamics. These representations are then fused into a unified system embedding to predict window-level P95 latency with high accuracy. We evaluate USRFNet on real-world microservice benchmarks under large-scale stress testing conditions, demonstrating substantial improvements in prediction accuracy over state-of-the-art baselines. 

**Abstract (ZH)**: 面向服务异构数据的分布式微服务架构窗口级别P95尾延迟预测方法及USRFNet网络设计 

---
# OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets 

**Title (ZH)**: OpenMed NER：面向12个公开数据集的开源、领域自适应最先进的变换器生物医学命名实体识别方法 

**Authors**: Maziyar Panahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01630)  

**Abstract**: Named-entity recognition (NER) is fundamental to extracting structured information from the >80% of healthcare data that resides in unstructured clinical notes and biomedical literature. Despite recent advances with large language models, achieving state-of-the-art performance across diverse entity types while maintaining computational efficiency remains a significant challenge. We introduce OpenMed NER, a suite of open-source, domain-adapted transformer models that combine lightweight domain-adaptive pre-training (DAPT) with parameter-efficient Low-Rank Adaptation (LoRA). Our approach performs cost-effective DAPT on a 350k-passage corpus compiled from ethically sourced, publicly available research repositories and de-identified clinical notes (PubMed, arXiv, and MIMIC-III) using DeBERTa-v3, PubMedBERT, and BioELECTRA backbones. This is followed by task-specific fine-tuning with LoRA, which updates less than 1.5% of model parameters. We evaluate our models on 12 established biomedical NER benchmarks spanning chemicals, diseases, genes, and species. OpenMed NER achieves new state-of-the-art micro-F1 scores on 10 of these 12 datasets, with substantial gains across diverse entity types. Our models advance the state-of-the-art on foundational disease and chemical benchmarks (e.g., BC5CDR-Disease, +2.70 pp), while delivering even larger improvements of over 5.3 and 9.7 percentage points on more specialized gene and clinical cell line corpora. This work demonstrates that strategically adapted open-source models can surpass closed-source solutions. This performance is achieved with remarkable efficiency: training completes in under 12 hours on a single GPU with a low carbon footprint (< 1.2 kg CO2e), producing permissively licensed, open-source checkpoints designed to help practitioners facilitate compliance with emerging data protection and AI regulations, such as the EU AI Act. 

**Abstract (ZH)**: 命名实体识别（NER）是从超过80%保存在未结构化临床笔记和生物医学文献中的医疗数据中提取结构化信息的基础。尽管大型语言模型取得了recent进展，但在多种实体类型上实现最先进的性能并保持计算效率仍然是一项重大挑战。我们介绍OpenMed NER，一个结合了轻量级领域适应预训练（DAPT）和参数高效低秩适应（LoRA）的开源领域适应变换器模型套件。我们的方法使用DeBERTa-v3、PubMedBERT和BioELECTRA主干，在包含伦理来源的公开可用研究repositories和去标识化临床笔记（PubMed、arXiv和MIMIC-III）的350k段落语料库上执行成本有效的大规模适应预训练。随后是使用LoRA的任务特定微调，更新少于1.5%的模型参数。我们在涵盖化学物质、疾病、基因和物种的12个现有生物医学NER基准测试上评估了我们的模型。OpenMed NER在这些12个数据集中的10个上实现了新的最先进的微F1分数，针对不同实体类型取得了显著改进。我们的模型在基础疾病和化学基准（例如BC5CDR-Disease，+2.70 pp）上推动了最先进的技术水平，同时在更专业化的基因和临床细胞系语料库上实现了超过5.3和9.7个百分点的更大改进。这项工作表明，战略性调整的开源模型可以超越封闭源解决方案。这种性能通过在一个GPU上在不到12小时内完成训练实现，并具有较低的碳足迹（<1.2 kg CO2e），生成了许可许可的开源检查点，旨在帮助从业者促进遵守新兴的数据保护和AI法规，如欧盟AI法案。 

---
# EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models 

**Title (ZH)**: EAC-MoE：aware专家选择压缩器for混合专家大型语言模型 

**Authors**: Yuanteng Chen, Yuantian Shao, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01625)  

**Abstract**: Mixture-of-Experts (MoE) has demonstrated promising potential in scaling LLMs. However, it is hindered by two critical challenges: (1) substantial GPU memory consumption to load all experts; (2) low activated parameters cannot be equivalently translated into inference acceleration effects. In this work, we propose EAC-MoE, an Expert-Selection Aware Compressor for MoE-LLMs, which deeply aligns with the characteristics of MoE from the perspectives of quantization and pruning, and introduces two modules to address these two challenges respectively: (1) The expert selection bias caused by low-bit quantization is a major factor contributing to the performance degradation in MoE-LLMs. Based on this, we propose Quantization with Expert-Selection Calibration (QESC), which mitigates the expert selection bias by calibrating the routers within the MoE; (2) There are always certain experts that are not crucial for the corresponding tasks, yet causing inference latency. Therefore, we propose Pruning based on Expert-Selection Frequency (PESF), which significantly improves inference speed by pruning less frequently used experts for current task. Extensive experiments demonstrate that our approach significantly reduces memory usage and improves inference speed with minimal performance degradation. 

**Abstract (ZH)**: Expert-Selection Aware Compressor for MoE-LLMs: Mitigating Expert Selection Bias and Improving Inference Efficiency 

---
# TCDiff: Triplex Cascaded Diffusion for High-fidelity Multimodal EHRs Generation with Incomplete Clinical Data 

**Title (ZH)**: TCDiff: 三重级联扩散模型用于生成具有不完整临床数据的高保真多模态EHR 

**Authors**: Yandong Yan, Chenxi Li, Yu Huang, Dexuan Xu, Jiaqi Zhu, Zhongyan Chai, Huamin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01615)  

**Abstract**: The scarcity of large-scale and high-quality electronic health records (EHRs) remains a major bottleneck in biomedical research, especially as large foundation models become increasingly data-hungry. Synthesizing substantial volumes of de-identified and high-fidelity data from existing datasets has emerged as a promising solution. However, existing methods suffer from a series of limitations: they struggle to model the intrinsic properties of heterogeneous multimodal EHR data (e.g., continuous, discrete, and textual modalities), capture the complex dependencies among them, and robustly handle pervasive data incompleteness. These challenges are particularly acute in Traditional Chinese Medicine (TCM). To this end, we propose TCDiff (Triplex Cascaded Diffusion Network), a novel EHR generation framework that cascades three diffusion networks to learn the features of real-world EHR data, formatting a multi-stage generative process: Reference Modalities Diffusion, Cross-Modal Bridging, and Target Modality Diffusion. Furthermore, to validate our proposed framework, besides two public datasets, we also construct and introduce TCM-SZ1, a novel multimodal EHR dataset for benchmarking. Experimental results show that TCDiff consistently outperforms state-of-the-art baselines by an average of 10% in data fidelity under various missing rate, while maintaining competitive privacy guarantees. This highlights the effectiveness, robustness, and generalizability of our approach in real-world healthcare scenarios. 

**Abstract (ZH)**: 电子健康记录(EHR)数据的稀缺性仍然是生物医学研究中的一个主要瓶颈，特别是随着大型基础模型变得越来越依赖数据。从现有数据集合成大量去标识且高保真的数据已 emerged 作为一种有前景的解决方案。然而，现有方法存在一系列限制：它们难以建模异质多模态 EHR 数据的本质特性（如连续型、离散型和文本模态），捕捉它们之间的复杂依赖关系，并且难以稳健地处理普遍存在的数据不完整性。这些挑战在中医药 (TCM) 中尤为严重。为此，我们提出了一种新颖的 EHR 生成框架 TCDiff（三重级联扩散网络），该框架级联三个扩散网络以学习现实世界 EHR 数据的特征，格式化一个多阶段生成过程：参考模态扩散、跨模态桥梁构建 和 目标模态扩散。此外，为了验证我们提出的方法，在两个公开数据集的基础上，我们还构建并引入了 TCM-SZ1，这是一个新型多模态 EHR 数据集用于基准测试。实验结果表明，无论在各种缺失率下，TCDiff 在数据保真度方面均比最先进的基线方法平均高出 10%，同时保持了竞争力的隐私保障。这突显了我们方法在实际医疗保健场景中的有效性、鲁棒性和通用性。 

---
# Augmented Reinforcement Learning Framework For Enhancing Decision-Making In Machine Learning Models Using External Agents 

**Title (ZH)**: 使用外部代理增强机器学习模型决策制定的增强强化学习框架 

**Authors**: Sandesh Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01612)  

**Abstract**: This work proposes a novel technique Augmented Reinforcement Learning framework for the improvement of decision-making capabilities of machine learning models. The introduction of agents as external overseers checks on model decisions. The external agent can be anyone, like humans or automated scripts, that helps in decision path correction. It seeks to ascertain the priority of the "Garbage-In, Garbage-Out" problem that caused poor data inputs or incorrect actions in reinforcement learning. The ARL framework incorporates two external agents that aid in course correction and the guarantee of quality data at all points of the training cycle. The External Agent 1 is a real-time evaluator, which will provide feedback light of decisions taken by the model, identify suboptimal actions forming the Rejected Data Pipeline. The External Agent 2 helps in selective curation of the provided feedback with relevance and accuracy in business scenarios creates an approved dataset for future training cycles. The validation of the framework is also applied to a real-world scenario, which is "Document Identification and Information Extraction". This problem originates mainly from banking systems, but can be extended anywhere. The method of classification and extraction of information has to be done correctly here. Experimental results show that including human feedback significantly enhances the ability of the model in order to increase robustness and accuracy in making decisions. The augmented approach, with a combination of machine efficiency and human insight, attains a higher learning standard-mainly in complex or ambiguous environments. The findings of this study show that human-in-the-loop reinforcement learning frameworks such as ARL can provide a scalable approach to improving model performance in data-driven applications. 

**Abstract (ZH)**: 基于增强强化学习的决策增强方法：提高机器学习模型的决策能力 

---
# Drift-aware Collaborative Assistance Mixture of Experts for Heterogeneous Multistream Learning 

**Title (ZH)**: 具有漂移意识的协作辅助专家混合模型用于异构多流学习 

**Authors**: En Yu, Jie Lu, Kun Wang, Xiaoyu Yang, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01598)  

**Abstract**: Learning from multiple data streams in real-world scenarios is fundamentally challenging due to intrinsic heterogeneity and unpredictable concept drifts. Existing methods typically assume homogeneous streams and employ static architectures with indiscriminate knowledge fusion, limiting generalizability in complex dynamic environments. To tackle this gap, we propose CAMEL, a dynamic \textbf{C}ollaborative \textbf{A}ssistance \textbf{M}ixture of \textbf{E}xperts \textbf{L}earning framework. It addresses heterogeneity by assigning each stream an independent system with a dedicated feature extractor and task-specific head. Meanwhile, a dynamic pool of specialized private experts captures stream-specific idiosyncratic patterns. Crucially, collaboration across these heterogeneous streams is enabled by a dedicated assistance expert. This expert employs a multi-head attention mechanism to distill and integrate relevant context autonomously from all other concurrent streams. It facilitates targeted knowledge transfer while inherently mitigating negative transfer from irrelevant sources. Furthermore, we propose an Autonomous Expert Tuner (AET) strategy, which dynamically manages expert lifecycles in response to drift. It instantiates new experts for emerging concepts (freezing prior ones to prevent catastrophic forgetting) and prunes obsolete ones. This expert-level plasticity provides a robust and efficient mechanism for online model capacity adaptation. Extensive experiments demonstrate CAMEL's superior generalizability across diverse multistreams and exceptional resilience against complex concept drifts. 

**Abstract (ZH)**: 多数据流在现实场景中的学习本质上极具挑战性，由于固有的异构性和不可预测的概念漂移。现有方法通常假设同质流，并采用静态架构和不分青红皂白的知识融合，限制了在复杂动态环境中的泛化能力。为弥补这一差距，我们提出了一种动态的协作专家混合学习框架CAMEL。该框架通过为每一流分配独立的系统，包含专用特征提取器和任务特定的头部，来应对异构性。同时，一个动态的特殊私有专家池捕捉流特定的异质模式。尤为重要的是，通过专用的协助专家，在这些异构流之间实现协作。该专家利用多头注意力机制自主从所有其他并行流中提炼和整合相关上下文，促进有针对性的知识转移，同时固有地减少来自无关源的负面影响。此外，我们提出了自主专家调谐器（AET）策略，该策略根据漂移动态管理专家生命周期。它为新兴概念实例化新专家（并冻结先前的专家以防灾难性遗忘），并淘汰过时的专家。这种专家级别的可塑性为在线模型容量适应提供了一个稳健且高效的机制。广泛实验表明，CAMEL在多种多流场景下的泛化能力和对抗复杂概念漂移的出色鲁棒性优于现有方法。 

---
# DMTrack: Spatio-Temporal Multimodal Tracking via Dual-Adapter 

**Title (ZH)**: DMTrack: 基于双适配器的时空多模态跟踪 

**Authors**: Weihong Li, Shaohua Dong, Haonan Lu, Yanhao Zhang, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01592)  

**Abstract**: In this paper, we explore adapter tuning and introduce a novel dual-adapter architecture for spatio-temporal multimodal tracking, dubbed DMTrack. The key of our DMTrack lies in two simple yet effective modules, including a spatio-temporal modality adapter (STMA) and a progressive modality complementary adapter (PMCA) module. The former, applied to each modality alone, aims to adjust spatio-temporal features extracted from a frozen backbone by self-prompting, which to some extent can bridge the gap between different modalities and thus allows better cross-modality fusion. The latter seeks to facilitate cross-modality prompting progressively with two specially designed pixel-wise shallow and deep adapters. The shallow adapter employs shared parameters between the two modalities, aiming to bridge the information flow between the two modality branches, thereby laying the foundation for following modality fusion, while the deep adapter modulates the preliminarily fused information flow with pixel-wise inner-modal attention and further generates modality-aware prompts through pixel-wise inter-modal attention. With such designs, DMTrack achieves promising spatio-temporal multimodal tracking performance with merely \textbf{0.93M} trainable parameters. Extensive experiments on five benchmarks show that DMTrack achieves state-of-the-art results. Code will be available. 

**Abstract (ZH)**: 基于时空多模态跟踪的双适配器架构DMTrack研究 

---
# Censored Sampling for Topology Design: Guiding Diffusion with Human Preferences 

**Title (ZH)**: 基于裁剪采样的拓扑设计：以人类偏好引导扩散 

**Authors**: Euihyun Kim, Keun Park, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01589)  

**Abstract**: Recent advances in denoising diffusion models have enabled rapid generation of optimized structures for topology optimization. However, these models often rely on surrogate predictors to enforce physical constraints, which may fail to capture subtle yet critical design flaws such as floating components or boundary discontinuities that are obvious to human experts. In this work, we propose a novel human-in-the-loop diffusion framework that steers the generative process using a lightweight reward model trained on minimal human feedback. Inspired by preference alignment techniques in generative modeling, our method learns to suppress unrealistic outputs by modulating the reverse diffusion trajectory using gradients of human-aligned rewards. Specifically, we collect binary human evaluations of generated topologies and train classifiers to detect floating material and boundary violations. These reward models are then integrated into the sampling loop of a pre-trained diffusion generator, guiding it to produce designs that are not only structurally performant but also physically plausible and manufacturable. Our approach is modular and requires no retraining of the diffusion model. Preliminary results show substantial reductions in failure modes and improved design realism across diverse test conditions. This work bridges the gap between automated design generation and expert judgment, offering a scalable solution to trustworthy generative design. 

**Abstract (ZH)**: Recent Advances in Denoising Diffusion Models for Topology Optimization: A Human-in-the-Loop Framework for Enhanced Design Realism and Manufacturability 

---
# Diffusion Models for Future Networks and Communications: A Comprehensive Survey 

**Title (ZH)**: 未来网络与通信中的扩散模型：一项全面综述 

**Authors**: Nguyen Cong Luong, Nguyen Duc Hai, Duc Van Le, Huy T. Nguyen, Thai-Hoc Vu, Thien Huynh-The, Ruichen Zhang, Nguyen Duc Duy Anh, Dusit Niyato, Marco Di Renzo, Dong In Kim, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2508.01586)  

**Abstract**: The rise of Generative AI (GenAI) in recent years has catalyzed transformative advances in wireless communications and networks. Among the members of the GenAI family, Diffusion Models (DMs) have risen to prominence as a powerful option, capable of handling complex, high-dimensional data distribution, as well as consistent, noise-robust performance. In this survey, we aim to provide a comprehensive overview of the theoretical foundations and practical applications of DMs across future communication systems. We first provide an extensive tutorial of DMs and demonstrate how they can be applied to enhance optimizers, reinforcement learning and incentive mechanisms, which are popular approaches for problems in wireless networks. Then, we review and discuss the DM-based methods proposed for emerging issues in future networks and communications, including channel modeling and estimation, signal detection and data reconstruction, integrated sensing and communication, resource management in edge computing networks, semantic communications and other notable issues. We conclude the survey with highlighting technical limitations of DMs and their applications, as well as discussing future research directions. 

**Abstract (ZH)**: 近年来生成式AI（GenAI）的发展推动了无线通信和网络的转型性进步。在生成式AI家族中，扩散模型（DMs）因其能够处理复杂高维数据分布以及一致的抗噪性能而崭露头角。本文综述旨在提供扩散模型在future通信系统中的理论基础及其实际应用的全面概述。我们首先提供了扩散模型的深入教程，并展示了它们如何被用于优化器、强化学习和激励机制的增强，这些是无线网络中流行的方法。然后，我们回顾并讨论了用于解决未来网络和通信中新兴问题的基于扩散模型的方法，包括信道建模与估计、信号检测与数据重构、综合传感与通信、边缘计算网络中的资源管理、语义通信及其他重大问题。最后，我们强调了扩散模型及其应用的技术限制，并讨论了未来的研究方向。 

---
# Deeply Supervised Multi-Task Autoencoder for Biological Brain Age estimation using three dimensional T$_1$-weighted magnetic resonance imaging 

**Title (ZH)**: 基于三维T₁加权磁共振成像的深度监督多任务自动编码器生物脑年龄估计 

**Authors**: Mehreen Kanwal, Yunsik Son  

**Link**: [PDF](https://arxiv.org/pdf/2508.01565)  

**Abstract**: Accurate estimation of biological brain age from three dimensional (3D) T$_1$-weighted magnetic resonance imaging (MRI) is a critical imaging biomarker for identifying accelerated aging associated with neurodegenerative diseases. Effective brain age prediction necessitates training 3D models to leverage comprehensive insights from volumetric MRI scans, thereby fully capturing spatial anatomical context. However, optimizing deep 3D models remains challenging due to problems such as vanishing gradients. Furthermore, brain structural patterns differ significantly between sexes, which impacts aging trajectories and vulnerability to neurodegenerative diseases, thereby making sex classification crucial for enhancing the accuracy and generalizability of predictive models. To address these challenges, we propose a Deeply Supervised Multitask Autoencoder (DSMT-AE) framework for brain age estimation. DSMT-AE employs deep supervision, which involves applying supervisory signals at intermediate layers during training, to stabilize model optimization, and multitask learning to enhance feature representation. Specifically, our framework simultaneously optimizes brain age prediction alongside auxiliary tasks of sex classification and image reconstruction, thus effectively capturing anatomical and demographic variability to improve prediction accuracy. We extensively evaluate DSMT-AE on the Open Brain Health Benchmark (OpenBHB) dataset, the largest multisite neuroimaging cohort combining ten publicly available datasets. The results demonstrate that DSMT-AE achieves state-of-the-art performance and robustness across age and sex subgroups. Additionally, our ablation study confirms that each proposed component substantially contributes to the improved predictive accuracy and robustness of the overall architecture. 

**Abstract (ZH)**: 从三维（3D）T$_1$加权磁共振成像（MRI）准确估计生物脑年龄是识别与神经退行性疾病相关的加速衰老的关键影像生物标志物。有效的脑年龄预测需要训练3D模型来充分利用体部MRI扫描的全面见解，从而全面捕捉空间解剖上下文。然而，由于消失梯度等问题，优化3D深度模型仍然具有挑战性。此外，男女之间的脑结构模式差异显著，这影响了衰老轨迹和对神经退行性疾病的易感性，因此性别分类对于提高预测模型的准确性和普适性至关重要。为了解决这些挑战，我们提出了一种深度监督多任务自动编码器（DSMT-AE）框架，用于脑年龄估计。DSMT-AE利用深度监督，在训练过程中在中间层应用监督信号以稳定模型优化，并采用多任务学习提升特征表示。具体来说，我们的框架同时优化脑年龄预测和辅助任务的性别分类和图像重建，从而有效捕捉解剖和人口统计学变异性，提高预测准确性。我们通过Open Brain Health Benchmark（OpenBHB）数据集，最大的多中心神经成像队列组合了十个公开数据集，广泛评估了DSMT-AE。结果表明，DSMT-AE在不同年龄和性别子组中实现了最先进的性能和稳健性。此外，我们的消融研究证实，所提出的每个组件对整体架构的预测准确性和稳健性均有显著贡献。 

---
# Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models 

**Title (ZH)**: 所有提示组件都是价值中立的吗？理解分解提示在大型语言模型中的异质对抗鲁棒性 

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01554)  

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于提示的对抗攻击已成为评估大型语言模型鲁棒性的有效手段。然而，现有方法往往将提示视为单一文本，忽视了其结构异质性——不同的提示组件对对抗鲁棒性贡献不均。前人工作如PromptRobust假设提示是中立的，但我们的分析揭示，具有丰富结构的复杂领域特定提示的不同组件具有不同的脆弱性。为填补这一空白，我们提出PromptAnatomy，一种自动框架，将提示分解为功能组件，并通过选择性地扰动每个组件生成多样且可解释的对抗样本，使用我们提出的方法ComPerturb。为了确保语义可信度并减轻分布偏移，我们进一步引入了基于困惑度(PPL)的过滤机制。作为补充资源，我们使用PromptAnatomy框架标注了四个公开的指令调优数据集，并通过人工审核验证。在这些数据集及五种先进大型语言模型上进行的广泛实验表明，ComPerturb实现了最先进的攻击成功率。消融研究验证了提示分解和PPL过滤的互补优势。我们的结果强调了在大型语言模型对抗鲁棒性评估中提示结构意识和可控扰动的重要性。代码和数据可通过以下链接获取。 

---
# Understanding Why ChatGPT Outperforms Humans in Visualization Design Advice 

**Title (ZH)**: 理解ChatGPT在可视化设计建议方面超越人类的原因 

**Authors**: Yongsu Ahn, Nam Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01547)  

**Abstract**: This paper investigates why recent generative AI models outperform humans in data visualization knowledge tasks. Through systematic comparative analysis of responses to visualization questions, we find that differences exist between two ChatGPT models and human outputs over rhetorical structure, knowledge breadth, and perceptual quality. Our findings reveal that ChatGPT-4, as a more advanced model, displays a hybrid of characteristics from both humans and ChatGPT-3.5. The two models were generally favored over human responses, while their strengths in coverage and breadth, and emphasis on technical and task-oriented visualization feedback collectively shaped higher overall quality. Based on our findings, we draw implications for advancing user experiences based on the potential of LLMs and human perception over their capabilities, with relevance to broader applications of AI. 

**Abstract (ZH)**: 本文探讨了为什么近期生成式AI模型在数据可视化知识任务中优于人类。通过系统性地比较对可视化问题的回应，我们发现两个ChatGPT模型与人类输出在修辞结构、知识广度和感知质量方面存在差异。研究发现，作为更先进的模型，ChatGPT-4 展现了人类和ChatGPT-3.5 的混合特征。这两模型通常优于人类回应，而它们在覆盖范围和广度以及对技术和任务导向的可视化反馈的强调共同促进了更高的整体质量。基于我们的发现，我们探讨了利用大型语言模型和人的感知潜力来增强用户体验的潜在影响，并扩展到更广泛的人工智能应用领域。 

---
# Leveraging Machine Learning for Botnet Attack Detection in Edge-Computing Assisted IoT Networks 

**Title (ZH)**: 利用机器学习在辅助边缘计算的物联网网络中检测僵尸网络攻击 

**Authors**: Dulana Rupanetti, Naima Kaabouch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01542)  

**Abstract**: The increase of IoT devices, driven by advancements in hardware technologies, has led to widespread deployment in large-scale networks that process massive amounts of data daily. However, the reliance on Edge Computing to manage these devices has introduced significant security vulnerabilities, as attackers can compromise entire networks by targeting a single IoT device. In light of escalating cybersecurity threats, particularly botnet attacks, this paper investigates the application of machine learning techniques to enhance security in Edge-Computing-Assisted IoT environments. Specifically, it presents a comparative analysis of Random Forest, XGBoost, and LightGBM -- three advanced ensemble learning algorithms -- to address the dynamic and complex nature of botnet threats. Utilizing a widely recognized IoT network traffic dataset comprising benign and malicious instances, the models were trained, tested, and evaluated for their accuracy in detecting and classifying botnet activities. Furthermore, the study explores the feasibility of deploying these models in resource-constrained edge and IoT devices, demonstrating their practical applicability in real-world scenarios. The results highlight the potential of machine learning to fortify IoT networks against emerging cybersecurity challenges. 

**Abstract (ZH)**: 物联网设备数量的不断增加，得益于硬件技术的进步，已在大规模网络中得到广泛部署，这些网络每天处理大量数据。然而，对边缘计算的依赖性管理这些设备已引入了显著的安全漏洞，攻击者可以通过攻击单一物联网设备来控制整个网络。鉴于网络安全威胁的升级，尤其是僵尸网络攻击，本文研究了机器学习技术在边缘计算辅助物联网环境中的应用，以增强安全性。具体而言，本文对随机森林、XGBoost和LightGBM三种先进的集成学习算法进行了比较分析，以应对僵尸网络威胁的动态和复杂性。利用一个包含正常和恶意实例的广泛认可的物联网网络流量数据集，对模型进行了训练、测试和评估，以检测和分类僵尸网络活动。此外，研究还探讨了在资源受限的边缘和物联网设备中部署这些模型的可行性，展示了其在实际场景中的实用性。研究结果突显了机器学习在增强物联网网络抵御新兴网络安全挑战方面的能力。 

---
# MagicVL-2B: Empowering Vision-Language Models on Mobile Devices with Lightweight Visual Encoders via Curriculum Learning 

**Title (ZH)**: MagicVL-2B: 通过分阶段学习增强轻量级视觉编码器在移动设备上的多模态语言模型能力 

**Authors**: Yi Liu, Xiao Xu, Zeyu Xu, Meng Zhang, Yibo Li, Haoyu Chen, Junkang Zhang, Qiang Wang, Jifa Sun, Siling Lin, Shengxun Cheng, Lingshu Zhang, Kang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01540)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable breakthroughs in recent years, enabling a diverse array of applications in everyday life. However, the substantial computational and storage demands of VLMs pose significant challenges for their efficient deployment on mobile devices, which represent the most ubiquitous and accessible computing platforms today. In this work, we introduce MagicVL-2B, a novel VLM meticulously optimized for flagship smartphones. MagicVL-2B leverages a lightweight visual encoder with fewer than 100M parameters and features a redesigned dynamic resolution scheme that adaptively generates image tokens without excessive modification of image dimensions. To further enhance the performance of this compact encoder within VLMs, we propose a multimodal curriculum learning strategy that incrementally increases task difficulty and data information density throughout training. This approach substantially improves the model's performance across a variety of sub-tasks. Extensive evaluations on standard VLM benchmarks demonstrate that MagicVL-2B matches the accuracy of current state-of-the-art models while reducing on-device power consumption by 41.1%. These results establish MagicVL-2B as a practical and robust solution for real-world mobile vision-language applications, enabling advanced multimodal intelligence to run directly on smartphones. 

**Abstract (ZH)**: MagicVL-2B：面向旗舰智能手机的高效视觉-语言模型 

---
# Revisiting Gossip Protocols: A Vision for Emergent Coordination in Agentic Multi-Agent Systems 

**Title (ZH)**: 重新审视闲谈协议：自主多agent系统中 emergent 协调的愿景 

**Authors**: Mansura Habiba, Nafiul I. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01531)  

**Abstract**: As agentic platforms scale, agents are evolving beyond static roles and fixed toolchains, creating a growing need for flexible, decentralized coordination. Today's structured communication protocols (e.g., direct agent-to-agent messaging) excel at reliability and task delegation, but they fall short in enabling emergent, swarm-like intelligence, where distributed agents continuously learn, adapt, and communicate to form collective cognition. This paper revisits gossip protocols, long valued in distributed systems for their fault tolerance and decentralization, and argues that they offer a missing layer for context-rich, adaptive communication in agentic AI. Gossip enables scalable, low-overhead dissemination of shared knowledge, but also raises unresolved challenges around semantic filtering, staleness, trustworthiness, and consistency in high-stakes environments. Rather than proposing a new framework, this work charts a research agenda for integrating gossip as a complementary substrate alongside structured protocols. We identify critical gaps in current agent-to-agent architectures, highlight where gossip could reshape assumptions about coordination, and outline open questions around intent propagation, knowledge decay, and peer-to-peer trust. Gossip is not a silver bullet, but overlooking it risks missing a key path toward resilient, reflexive, and self-organizing multi-agent systems. 

**Abstract (ZH)**: 随着代理平台的扩展，代理正在超越静态角色和固定工具链，创建对灵活且去中心化协调日益增长的需求。今天的结构化通信协议（例如，直接代理间通信）在可靠性和任务委派方面表现出色，但在促进分布式代理的连续学习、适应和交流，从而形成集体认知的涌现式智能方面仍显不足。本文重访在分布式系统中因其容错性和去中心化而长期受到重视的流言协议，并argue指出它们为富有语境的、适应性通信提供了缺失的一层。流言协议使共享知识的大规模、低开销传播成为可能，但也提出了在高危环境中围绕语义过滤、陈旧、可信性和一致性的未解决挑战。本文未提出新的框架，而是为将流言协议作为一种补充基础结构与结构化协议整合的研究议程制定了路线图。我们指出了当前代理间架构中的关键缺口，强调了流言协议如何重新塑造关于协调的假设，并概述了意图传播、知识衰退和点对点信任等方面存在的开放问题。流言协议并非灵丹妙药，但忽略它可能会错失通向稳健、反射性和自主多代理系统的关键路径。 

---
# MiraGe: Multimodal Discriminative Representation Learning for Generalizable AI-Generated Image Detection 

**Title (ZH)**: MiraGe：多模态区分性表示学习在通用AI生成图像检测中的应用 

**Authors**: Kuo Shi, Jie Lu, Shanshan Ye, Guangquan Zhang, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01525)  

**Abstract**: Recent advances in generative models have highlighted the need for robust detectors capable of distinguishing real images from AI-generated images. While existing methods perform well on known generators, their performance often declines when tested with newly emerging or unseen generative models due to overlapping feature embeddings that hinder accurate cross-generator classification. In this paper, we propose Multimodal Discriminative Representation Learning for Generalizable AI-generated Image Detection (MiraGe), a method designed to learn generator-invariant features. Motivated by theoretical insights on intra-class variation minimization and inter-class separation, MiraGe tightly aligns features within the same class while maximizing separation between classes, enhancing feature discriminability. Moreover, we apply multimodal prompt learning to further refine these principles into CLIP, leveraging text embeddings as semantic anchors for effective discriminative representation learning, thereby improving generalizability. Comprehensive experiments across multiple benchmarks show that MiraGe achieves state-of-the-art performance, maintaining robustness even against unseen generators like Sora. 

**Abstract (ZH)**: Recent advances in生成模型的最新进展强调了需要 Robust Detectors 能够区分真实图像与AI生成图像的强健检测器。现有的方法在已知生成器上表现良好，但在测试新兴或未见过的生成器时，由于特征嵌入的重叠而影响准确的跨生成器分类。本文提出了一种名为MiraGe的方法：面向通用AI生成图像检测的多模态判别表示学习，旨在学习生成器不变特征。受最小化类别内变分和最大化类别间分离的理论启发，MiraGe在同一个类别内紧密对齐特征，同时最大化类间分离，提升特征可判别性。此外，我们应用多模态提示学习进一步细化这些原则，并利用CLIP进行有效判别表示学习，通过文本嵌入作为语义锚点来提高泛化能力。多项基准上的综合实验表明，MiraGe达到了最先进的性能，即使在未见过的生成器如Sora的情况下也保持了鲁棒性。 

---
# Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于多智能体强化学习的分布式缆索悬挂负载空中操控 

**Authors**: Jack Zeng, Andreu Matoses Gimenez, Eugene Vinitsky, Javier Alonso-Mora, Sihao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01522)  

**Abstract**: This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: this https URL 

**Abstract (ZH)**: 本文提出了首款用于通过微空中车辆（MAVs）团队实现真实世界中6-自由度操纵悬吊负载的去中心化方法。我们的方法利用多智能体强化学习（MARL）为每架MAV训练一个外环控制策略。与现有利用中心化方案的控制器不同，我们的策略不要求全局状态、智能体间通信或邻近智能体的信息。取而代之的是，智能体仅通过负载姿态观测进行隐式通信，这使得系统具有高可扩展性和灵活性。此外，这种方法还显著降低了推理时的计算成本，使策略能够实现搭载部署。另外，我们为MAVs引入了一种新的动作空间设计，使用线性加速度和体速率。这种选择，结合一个鲁棒的低层控制器，使得即使在动态3D运动中由于缆绳张力引起的显著不确定性时，也能可靠地实现仿真到现实的转移。我们在各种实际实验中验证了我们的方法，包括在负载模型不确定性下的全姿态控制，展示了与现有最佳中心化方法相当的定值跟踪性能。我们还展示了具有不同控制策略的智能体之间的合作以及面对其中一架MAV完全空中故障时的鲁棒性。实验视频：this https URL 

---
# The Vanishing Gradient Problem for Stiff Neural Differential Equations 

**Title (ZH)**: 刚性神经微分方程中的消失梯度问题 

**Authors**: Colby Fronk, Linda Petzold  

**Link**: [PDF](https://arxiv.org/pdf/2508.01519)  

**Abstract**: Gradient-based optimization of neural differential equations and other parameterized dynamical systems fundamentally relies on the ability to differentiate numerical solutions with respect to model parameters. In stiff systems, it has been observed that sensitivities to parameters controlling fast-decaying modes become vanishingly small during training, leading to optimization difficulties. In this paper, we show that this vanishing gradient phenomenon is not an artifact of any particular method, but a universal feature of all A-stable and L-stable stiff numerical integration schemes. We analyze the rational stability function for general stiff integration schemes and demonstrate that the relevant parameter sensitivities, governed by the derivative of the stability function, decay to zero for large stiffness. Explicit formulas for common stiff integration schemes are provided, which illustrate the mechanism in detail. Finally, we rigorously prove that the slowest possible rate of decay for the derivative of the stability function is $O(|z|^{-1})$, revealing a fundamental limitation: all A-stable time-stepping methods inevitably suppress parameter gradients in stiff regimes, posing a significant barrier for training and parameter identification in stiff neural ODEs. 

**Abstract (ZH)**: 基于梯度的神经微分方程和其他参数化动力系统的优化从根本上依赖于能够对模型参数求解数值解的灵敏度。在刚性系统中，已观察到控制快速衰减模式的参数的灵敏度在训练过程中变得微不足道，导致优化困难。在本文中，我们证明这种梯度消失现象并非任何特定方法的产物，而是所有A-稳定和L-稳定的刚性数值积分方案的普遍特征。我们分析了一般刚性积分方案的有理稳定性函数，并演示了由稳定性函数的导数确定的相关参数灵敏度在大刚性下衰减为零。提供了常见刚性积分方案的显式公式，详细说明了机制。最后，我们严格证明了稳定性函数导数的最慢衰减率是$O(|z|^{-1})$，揭示了一个基本限制：所有A-稳定的时步方法在刚性状态下不可避免地抑制参数梯度，对刚性神经常微分方程的训练和参数识别构成重大障碍。 

---
# FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models 

**Title (ZH)**: FlashSVD：低秩模型的流式高效推理存储方法 

**Authors**: Zishan Shao, Yixiao Wang, Qinsi Wang, Ting Jiang, Zhixu Du, Hancheng Ye, Danyang Zhuo, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01506)  

**Abstract**: Singular Value Decomposition (SVD) has recently seen a surge of interest as a simple yet powerful tool for large language models (LLMs) compression, with a growing number of works demonstrating 20-80% parameter reductions at minimal accuracy loss. Previous SVD-based approaches have focused primarily on reducing the memory footprint of model weights, largely overlooking the additional activation memory overhead incurred during inference when applying truncated factors via standard dense CUDA kernels. Our experiments demonstrate that this activation overhead, scaling with sequence length and hidden dimension, prevents current SVD compression techniques from achieving any reduction in peak inference memory, thereby limiting their viability for real-world, on-device deployments.
We introduce FlashSVD, a novel, end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD can be seamlessly integrated with any model that employs SVD-based methods for parameter reduction. By fusing low-rank projection kernels directly into both the self-attention and feed-forward network (FFN) pipelines, FlashSVD avoid materializing full-size activation buffers. Instead, small tiles of the truncated factors are loaded into on-chip SRAM, multiplied and reduced on the fly, and immediately evicted, preserving high GPU occupancy and adding no extra latency. On standard encoder benchmarks (e.g., BERT-Base), FlashSVD cuts peak activation memory by up to 70.2% and intermediate transient memory by 75%, all while incur no accuracy loss with upstreaming compression methods, offering a practical path toward memory-constrained deployment of low-rank LLMs. 

**Abstract (ZH)**: 奇异值分解（SVD）作为一种简单而强大的工具，近年来在大型语言模型（LLMs）压缩领域引起了极大的兴趣，越来越多的研究展示了在最小的精度损失下实现20%-80%的参数减少。此前的SVD基方法主要集中在减少模型权重的内存占用，而未充分考虑在应用截断因子时通过标准密集CUDA内核导致的额外激活内存开销。我们的实验表明，这种激活内存开销随着序列长度和隐藏维度的增加，阻碍了当前SVD压缩技术在峰值推理内存上的任何减少，从而限制了其在实际设备上的部署可行性。

我们提出了一种名为FlashSVD的新型端到端稀疏秩意识流式推理框架，专门用于SVD压缩的大语言模型。FlashSVD可以无缝集成到采用SVD方法进行参数减少的任何模型中。通过直接将低秩投影内核融合到自我注意和前馈网络（FFN）管道中，FlashSVD避免了激活缓冲的全尺寸实现。相反，小规模截断因子被加载到片上SRAM中，在线乘法和归约，并立即被清除，以保持高GPU利用率并添加零额外延迟。在标准编解码器基准测试（如BERT-Base）中，FlashSVD将峰值激活内存减少高达70.2%，中间暂态内存减少75%，同时与上游压缩方法相比无精度损失，提供了一种在受限内存下部署低秩LLMs的实用途径。 

---
# ShrutiSense: Microtonal Modeling and Correction in Indian Classical Music 

**Title (ZH)**: ShrutiSense：印度古典音乐中的微分音建模与修正 

**Authors**: Rajarshi Ghosh, Jayanth Athipatla  

**Link**: [PDF](https://arxiv.org/pdf/2508.01498)  

**Abstract**: Indian classical music relies on a sophisticated microtonal system of 22 shrutis (pitch intervals), which provides expressive nuance beyond the 12-tone equal temperament system. Existing symbolic music processing tools fail to account for these microtonal distinctions and culturally specific raga grammars that govern melodic movement. We present ShrutiSense, a comprehensive symbolic pitch processing system designed for Indian classical music, addressing two critical tasks: (1) correcting westernized or corrupted pitch sequences, and (2) completing melodic sequences with missing values. Our approach employs complementary models for different tasks: a Shruti-aware finite-state transducer (FST) that performs contextual corrections within the 22-shruti framework and a grammar-constrained Shruti hidden Markov model (GC-SHMM) that incorporates raga-specific transition rules for contextual completions. Comprehensive evaluation on simulated data across five ragas demonstrates that ShrutiSense (FST model) achieves 91.3% shruti classification accuracy for correction tasks, with example sequences showing 86.7-90.0% accuracy at corruption levels of 0.2 to 0.4. The system exhibits robust performance under pitch noise up to +/-50 cents, maintaining consistent accuracy across ragas (90.7-91.8%), thus preserving the cultural authenticity of Indian classical music expression. 

**Abstract (ZH)**: 印度古典音乐依赖于一个基于22 shrutis（音高间隔）的复杂微音系统，提供了超越12平均_temperament系统的表达细腻之处。现有的符号音乐处理工具未能考虑到这些微音差异以及指导旋律运动的文化特定法则。我们提出了一种专为印度古典音乐设计的全面符号音高处理系统——ShrutiSense，该系统针对两个关键任务：（1）校正西方化或损坏的音高序列，以及（2）补全缺失值的旋律序列。我们的方法使用了不同的互补模型：一种aware finite-state transducer（FS筹资者，利用22-shruti框架内的上下文校正，以及一种语法约束的Shruti隐马尔可夫模型（GC-SHMM），该模型整合了特定于拉格的转换规则以进行上下文补全。在五种拉格模拟数据上的全面评估结果显示，ShrutiSense（FST模型）在纠正任务中的shruti分类准确率达到91.3%，在噪声水平为0.2至0.4的损坏序列中，示例序列的准确性范围为86.7%至90.0%。该系统在高达±50美分的音高噪声下表现出稳健性，并在不同拉格中保持了90.7%至91.8%的一致准确性，从而保持了印度古典音乐表达的文化真实性。 

---
# Translation-Equivariant Self-Supervised Learning for Pitch Estimation with Optimal Transport 

**Title (ZH)**: 基于最优传输的平移不变自监督学习在音高估计中的应用 

**Authors**: Bernardo Torres, Alain Riou, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01493)  

**Abstract**: In this paper, we propose an Optimal Transport objective for learning one-dimensional translation-equivariant systems and demonstrate its applicability to single pitch estimation. Our method provides a theoretically grounded, more numerically stable, and simpler alternative for training state-of-the-art self-supervised pitch estimators. 

**Abstract (ZH)**: 本文提出了一个最优传输目标用于学习一维平移不变系统，并展示了其在单音高估计中的适用性。我们的方法提供了一种理论依据更可靠、数值稳定性更强且更简单的训练当前最先进自主监督音高估计器的替代方案。 

---
# A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics 

**Title (ZH)**: 大规模跨模态学习在空间转录组学组织学和基因表达基准测试 

**Authors**: Rushin H. Gindra, Giovanni Palla, Mathias Nguyen, Sophia J. Wagner, Manuel Tran, Fabian J Theis, Dieter Saur, Lorin Crawford, Tingying Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01490)  

**Abstract**: Spatial transcriptomics enables simultaneous measurement of gene expression and tissue morphology, offering unprecedented insights into cellular organization and disease mechanisms. However, the field lacks comprehensive benchmarks for evaluating multimodal learning methods that leverage both histology images and gene expression data. Here, we present HESCAPE, a large-scale benchmark for cross-modal contrastive pretraining in spatial transcriptomics, built on a curated pan-organ dataset spanning 6 different gene panels and 54 donors. We systematically evaluated state-of-the-art image and gene expression encoders across multiple pretraining strategies and assessed their effectiveness on two downstream tasks: gene mutation classification and gene expression prediction. Our benchmark demonstrates that gene expression encoders are the primary determinant of strong representational alignment, and that gene models pretrained on spatial transcriptomics data outperform both those trained without spatial data and simple baseline approaches. However, downstream task evaluation reveals a striking contradiction: while contrastive pretraining consistently improves gene mutation classification performance, it degrades direct gene expression prediction compared to baseline encoders trained without cross-modal objectives. We identify batch effects as a key factor that interferes with effective cross-modal alignment. Our findings highlight the critical need for batch-robust multimodal learning approaches in spatial transcriptomics. To accelerate progress in this direction, we release HESCAPE, providing standardized datasets, evaluation protocols, and benchmarking tools for the community 

**Abstract (ZH)**: 跨模态对比预训练在空间转录组学中的大规模基准：HESCAPE 

---
# PESTO: Real-Time Pitch Estimation with Self-supervised Transposition-equivariant Objective 

**Title (ZH)**: PESTO: 自监督移调不变目标的实时音高估计 

**Authors**: Alain Riou, Bernardo Torres, Ben Hayes, Stefan Lattner, Gaëtan Hadjeres, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01488)  

**Abstract**: In this paper, we introduce PESTO, a self-supervised learning approach for single-pitch estimation using a Siamese architecture. Our model processes individual frames of a Variable-$Q$ Transform (VQT) and predicts pitch distributions. The neural network is designed to be equivariant to translations, notably thanks to a Toeplitz fully-connected layer. In addition, we construct pitch-shifted pairs by translating and cropping the VQT frames and train our model with a novel class-based transposition-equivariant objective, eliminating the need for annotated data. Thanks to this architecture and training objective, our model achieves remarkable performances while being very lightweight ($130$k parameters). Evaluations on music and speech datasets (MIR-1K, MDB-stem-synth, and PTDB) demonstrate that PESTO not only outperforms self-supervised baselines but also competes with supervised methods, exhibiting superior cross-dataset generalization. Finally, we enhance PESTO's practical utility by developing a streamable VQT implementation using cached convolutions. Combined with our model's low latency (less than 10 ms) and minimal parameter count, this makes PESTO particularly suitable for real-time applications. 

**Abstract (ZH)**: 基于Siamese架构的自监督单音高估计方法PESTO 

---
# Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler 

**Title (ZH)**: Warmup-Stable-Decay学习率调度中cooldown阶段的训练动态 

**Authors**: Aleksandr Dremov, Alexander Hägele, Atli Kosson, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01483)  

**Abstract**: Learning rate scheduling is essential in transformer training, where the final annealing plays a crucial role in getting the best performance. However, the mechanisms behind this cooldown phase, with its characteristic drop in loss, remain poorly understood. To address this, we provide a comprehensive analysis focusing solely on the cooldown phase in the Warmup-Stable-Decay (WSD) learning rate scheduler. Our analysis reveals that different cooldown shapes reveal a fundamental bias-variance trade-off in the resulting models, with shapes that balance exploration and exploitation consistently outperforming alternatives. Similarly, we find substantial performance variations $\unicode{x2013}$ comparable to those from cooldown shape selection $\unicode{x2013}$ when tuning AdamW hyperparameters. Notably, we observe consistent improvements with higher values of $\beta_2$ during cooldown. From a loss landscape perspective, we provide visualizations of the landscape during cooldown, supporting the river valley loss perspective empirically. These findings offer practical recommendations for configuring the WSD scheduler in transformer training, emphasizing the importance of optimizing the cooldown phase alongside traditional hyperparameter tuning. 

**Abstract (ZH)**: 学习率调度在变压器训练中至关重要，其中最终降温在获得最佳性能中扮演关键角色。然而，这一冷却阶段的背后机制，尤其是其特征性的损失下降，仍知之甚少。为了解决这一问题，我们专注于Warmup-Stable-Decay (WSD) 学习率调度器中的冷却阶段，提供了全面的分析。分析表明，不同的冷却形状揭示了模型中基本的偏差-方差权衡，能够平衡探索与利用的形状始终优于其他替代方案。同样，我们发现性能变化显著，类似于通过调整AdamW 超参数时从冷却形状选择中得到的变化。值得注意的是，我们观察到在冷却阶段使用较高的 $\beta_2$ 值会带来一致性改进。从损失景观的角度来看，我们提供了冷却阶段损失景观的可视化，支持经验上的河流谷地损失观点。这些发现为在变压器训练中配置WSD调度器提供了实用建议，强调优化冷却阶段的重要性不仅限于传统的超参数调优。 

---
# Reconstructing Trust Embeddings from Siamese Trust Scores: A Direct-Sum Approach with Fixed-Point Semantics 

**Title (ZH)**: 从暹罗信任评分重构信任嵌入：固定点语义下的直和方法 

**Authors**: Faruk Alpay, Taylan Alpay, Bugra Kilictas  

**Link**: [PDF](https://arxiv.org/pdf/2508.01479)  

**Abstract**: We study the inverse problem of reconstructing high-dimensional trust embeddings from the one-dimensional Siamese trust scores that many distributed-security frameworks expose. Starting from two independent agents that publish time-stamped similarity scores for the same set of devices, we formalise the estimation task, derive an explicit direct-sum estimator that concatenates paired score series with four moment features, and prove that the resulting reconstruction map admits a unique fixed point under a contraction argument rooted in Banach theory. A suite of synthetic benchmarks (20 devices x 10 time steps) confirms that, even in the presence of Gaussian noise, the recovered embeddings preserve inter-device geometry as measured by Euclidean and cosine metrics; we complement these experiments with non-asymptotic error bounds that link reconstruction accuracy to score-sequence length. Beyond methodology, the paper demonstrates a practical privacy risk: publishing granular trust scores can leak latent behavioural information about both devices and evaluation models. We therefore discuss counter-measures -- score quantisation, calibrated noise, obfuscated embedding spaces -- and situate them within wider debates on transparency versus confidentiality in networked AI systems. All datasets, reproduction scripts and extended proofs accompany the submission so that results can be verified without proprietary code. 

**Abstract (ZH)**: 研究分布式安全框架中一维西梅森信任评分还原高维信任嵌入的逆问题：基于Banach理论的压缩映射证明及其合成基准验证 

---
# Fast and scalable retrosynthetic planning with a transformer neural network and speculative beam search 

**Title (ZH)**: 基于变压器神经网络和推测性束搜索的快速可扩展 retrosynthetic 规划 

**Authors**: Mikhail Andronov, Natalia Andronova, Michael Wand, Jürgen Schmidhuber, Djork-Arné Clevert  

**Link**: [PDF](https://arxiv.org/pdf/2508.01459)  

**Abstract**: AI-based computer-aided synthesis planning (CASP) systems are in demand as components of AI-driven drug discovery workflows. However, the high latency of such CASP systems limits their utility for high-throughput synthesizability screening in de novo drug design. We propose a method for accelerating multi-step synthesis planning systems that rely on SMILES-to-SMILES transformers as single-step retrosynthesis models. Our approach reduces the latency of SMILES-to-SMILES transformers powering multi-step synthesis planning in AiZynthFinder through speculative beam search combined with a scalable drafting strategy called Medusa. Replacing standard beam search with our approach allows the CASP system to solve 26\% to 86\% more molecules under the same time constraints of several seconds. Our method brings AI-based CASP systems closer to meeting the strict latency requirements of high-throughput synthesizability screening and improving general user experience. 

**Abstract (ZH)**: 基于AI的计算机辅助合成规划（CASP）系统在AI驱动的药物发现工作流程中需求旺盛。然而，这类CASP系统的高延迟限制了其在从头药物设计中进行高通量合成可及性筛查的实用性。我们提出了一种加速依赖于SMILES-to-SMILES转换器作为单步逆合成模型的多步合成规划系统的办法。我们的方法通过结合投机性 beam 搜索和一种可扩展的速记策略Medusa，减少了AiZynthFinder中SMILES-to-SMILES转换器驱动的多步合成规划的延迟。将标准beam搜索替换为我们的方法，允许CASP系统在相同几秒时间约束下解决26\%到86\%更多的分子。该方法使基于AI的CASP系统更接近满足高通量合成可及性筛查的严格延迟要求，并提高通用用户体验。 

---
# Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective 

**Title (ZH)**: 基于元提示调优LLM驱动的代码优化：工业视角 

**Authors**: Jingzhi Gong, Rafail Giavrimis, Paul Brookes, Vardan Voskanyan, Fan Wu, Mari Ashiga, Matthew Truscott, Mike Basios, Leslie Kanthan, Jie Xu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01443)  

**Abstract**: There is a growing interest in leveraging large language models (LLMs) for automated code optimization. However, industrial platforms deploying multiple LLMs face a critical challenge: prompts optimized for one LLM often fail with others, requiring expensive model-specific prompt engineering. This cross-model prompt engineering bottleneck severely limits the practical deployment of multi-LLM optimization systems in production environments. To address this, we introduce Meta-Prompted Code Optimization (MPCO), a framework that automatically generates high-quality, task-specific prompts across diverse LLMs while maintaining industrial efficiency requirements. MPCO leverages meta-prompting to dynamically synthesize context-aware optimization prompts by integrating project metadata, task requirements, and LLM-specific contexts, and it seamlessly deploys on the ARTEMIS industrial platform for automated validation and scaling.
Our comprehensive evaluation on five real-world codebases with 366 hours of runtime benchmarking demonstrates MPCO's effectiveness: it achieves overall performance improvements up to 19.06% with the best statistical rank across all systems compared to baseline methods. Analysis shows that 96% of the top-performing optimizations stem from meaningful edits. Through systematic ablation studies and meta-prompter sensitivity analysis, we identify that comprehensive context integration is essential for effective meta-prompting, and that all three major LLMs can serve effectively as meta-prompters, providing actionable insights for industrial practitioners. 

**Abstract (ZH)**: 利用元提示进行自动代码优化：跨模型自动生成高质量任务特定提示的研究 

---
# Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification 

**Title (ZH)**: 捕获更多：学习多域表示以实现稳健的在线手写验证 

**Authors**: Peirong Zhang, Kai Ding, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01427)  

**Abstract**: In this paper, we propose SPECTRUM, a temporal-frequency synergistic model that unlocks the untapped potential of multi-domain representation learning for online handwriting verification (OHV). SPECTRUM comprises three core components: (1) a multi-scale interactor that finely combines temporal and frequency features through dual-modal sequence interaction and multi-scale aggregation, (2) a self-gated fusion module that dynamically integrates global temporal and frequency features via self-driven balancing. These two components work synergistically to achieve micro-to-macro spectral-temporal integration. (3) A multi-domain distance-based verifier then utilizes both temporal and frequency representations to improve discrimination between genuine and forged handwriting, surpassing conventional temporal-only approaches. Extensive experiments demonstrate SPECTRUM's superior performance over existing OHV methods, underscoring the effectiveness of temporal-frequency multi-domain learning. Furthermore, we reveal that incorporating multiple handwritten biometrics fundamentally enhances the discriminative power of handwriting representations and facilitates verification. These findings not only validate the efficacy of multi-domain learning in OHV but also pave the way for future research in multi-domain approaches across both feature and biometric domains. Code is publicly available at this https URL. 

**Abstract (ZH)**: 本研究提出SPECTRUM，一种时空协同模型，解锁多域在线手写验证中的潜在能力。SPECTRUM包含三个核心组件：(1) 多尺度交互器通过双模序列交互和多尺度聚合精细结合时空特征，(2) 自控融合模块通过自我驱动的平衡动态整合全局时空特征。这两个组件协同工作以实现从微观到宏观的时空频谱整合。(3) 多域基于距离的验证器利用时空特征提高 Genuine 和 Forgery 手写之间的鉴别能力，超越传统仅时空的方法。大量实验表明 SPECTRUM 在现有在线手写验证方法中的优越性能，突显时空多域学习的有效性。此外，我们揭示，整合多种手写生物特征从根本上增强了手写表示的鉴别力并促进了验证。这些发现不仅验证了多域学习在在线手写验证中的有效性，还为跨特征和生物特征领域的多域方法研究奠定了基础。代码在此处公开。 

---
# From Query to Logic: Ontology-Driven Multi-Hop Reasoning in LLMs 

**Title (ZH)**: 从查询到逻辑：面向本体的多跳推理;break 

**Authors**: Haonan Bian, Yutao Qi, Rui Yang, Yuanxi Che, Jiaqian Wang, Heming Xia, Ranran Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01424)  

**Abstract**: Large Language Models (LLMs), despite their success in question answering, exhibit limitations in complex multi-hop question answering (MQA) tasks that necessitate non-linear, structured reasoning. This limitation stems from their inability to adequately capture deep conceptual relationships between entities. To overcome this challenge, we present **ORACLE** (**O**ntology-driven **R**easoning **A**nd **C**hain for **L**ogical **E**ucidation), a training-free framework that combines LLMs' generative capabilities with the structural benefits of knowledge graphs. Our approach operates through three stages: (1) dynamic construction of question-specific knowledge ontologies using LLMs, (2) transformation of these ontologies into First-Order Logic reasoning chains, and (3) systematic decomposition of the original query into logically coherent sub-questions. Experimental results on several standard MQA benchmarks show that our framework achieves highly competitive performance, rivaling current state-of-the-art models like DeepSeek-R1. Detailed analyses further confirm the effectiveness of each component, while demonstrating that our method generates more logical and interpretable reasoning chains than existing approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）虽然在问答任务上取得了成功，但在需要非线性、结构化推理的复杂多跳问答（MQA）任务中表现出局限性。这一局限源自它们无法充分捕捉实体之间的深层次概念关系。为克服这一挑战，我们提出了**ORACLE**（基于本体的推理与链条逻辑阐明框架），这是一个无需训练的框架，将LLMs的生成能力与知识图谱的结构性优势结合起来。我们的方法分为三个阶段：（1）使用LLMs动态构建问题特定的知识本体，（2）将这些本体转换为一阶逻辑推理链条，（3）系统地分解原始查询为逻辑上一致的子问题。在多个标准MQA基准上的实验结果显示，我们的框架达到了高度竞争力的性能，可与当前的最先进的模型DeepSeek-R1媲美。详细分析进一步证实了每个组件的有效性，同时展示了我们的方法生成的推理链条比现有方法更具逻辑性和可解释性。 

---
# RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems 

**Title (ZH)**: RoboMemory：一种启发自大脑的多记忆代理框架，用于物理 bodied 系统的终身学习 

**Authors**: Mingcong Lei, Honghao Cai, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.01415)  

**Abstract**: We present RoboMemory, a brain-inspired multi-memory framework for lifelong learning in physical embodied systems, addressing critical challenges in real-world environments: continuous learning, multi-module memory latency, task correlation capture, and infinite-loop mitigation in closed-loop planning. Grounded in cognitive neuroscience, it integrates four core modules: the Information Preprocessor (thalamus-like), the Lifelong Embodied Memory System (hippocampus-like), the Closed-Loop Planning Module (prefrontal lobe-like), and the Low-Level Executer (cerebellum-like) to enable long-term planning and cumulative learning. The Lifelong Embodied Memory System, central to the framework, alleviates inference speed issues in complex memory frameworks via parallelized updates/retrieval across Spatial, Temporal, Episodic, and Semantic submodules. It incorporates a dynamic Knowledge Graph (KG) and consistent architectural design to enhance memory consistency and scalability. Evaluations on EmbodiedBench show RoboMemory outperforms the open-source baseline (Qwen2.5-VL-72B-Ins) by 25% in average success rate and surpasses the closed-source State-of-the-Art (SOTA) (Claude3.5-Sonnet) by 5%, establishing new SOTA. Ablation studies validate key components (critic, spatial memory, long-term memory), while real-world deployment confirms its lifelong learning capability with significantly improved success rates across repeated tasks. RoboMemory alleviates high latency challenges with scalability, serving as a foundational reference for integrating multi-modal memory systems in physical robots. 

**Abstract (ZH)**: RoboMemory：一种用于物理体感系统终身学习的脑启发多内存框架 

---
# MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs 

**Title (ZH)**: MedSynth: 真实可信的合成医疗对话-笔记对 

**Authors**: Ahmad Rezaie Mianroodi, Amirali Rezaie, Niko Grisel Todorov, Cyril Rakovski, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01401)  

**Abstract**: Physicians spend significant time documenting clinical encounters, a burden that contributes to professional burnout. To address this, robust automation tools for medical documentation are crucial. We introduce MedSynth -- a novel dataset of synthetic medical dialogues and notes designed to advance the Dialogue-to-Note (Dial-2-Note) and Note-to-Dialogue (Note-2-Dial) tasks. Informed by an extensive analysis of disease distributions, this dataset includes over 10,000 dialogue-note pairs covering over 2000 ICD-10 codes. We demonstrate that our dataset markedly enhances the performance of models in generating medical notes from dialogues, and dialogues from medical notes. The dataset provides a valuable resource in a field where open-access, privacy-compliant, and diverse training data are scarce. Code is available at this https URL and the dataset is available at this https URL. 

**Abstract (ZH)**: 医生花费大量时间记录临床 Encounter，这一负担导致了职业倦怠。为了解决这个问题，需要强大的医疗文档自动化工具。我们介绍 MedSynth ——一个新颖的合成医疗对话与笔记数据集，旨在推进对话转笔记（Dial-2-Note）和笔记转对话（Note-2-Dial）任务。该数据集基于广泛的疾病分布分析，包含超过10,000个对话-笔记对，涵盖了超过2000个ICD-10编码。我们证明，该数据集显著提高了模型从对话生成医疗笔记以及从医疗笔记生成对话的性能。该数据集为一个稀缺开放访问、隐私合规和多样化训练数据的领域提供了宝贵的资源。代码可在以下网址获取，并且数据集可在以下网址获取。 

---
# Spatial-Frequency Aware for Object Detection in RAW Image 

**Title (ZH)**: 基于空间频率的RAW图像目标检测 

**Authors**: Zhuohua Ye, Liming Zhang, Hongru Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.01396)  

**Abstract**: Direct RAW-based object detection offers great promise by utilizing RAW data (unprocessed sensor data), but faces inherent challenges due to its wide dynamic range and linear response, which tends to suppress crucial object details. In particular, existing enhancement methods are almost all performed in the spatial domain, making it difficult to effectively recover these suppressed details from the skewed pixel distribution of RAW images. To address this limitation, we turn to the frequency domain, where features, such as object contours and textures, can be naturally separated based on frequency. In this paper, we propose Space-Frequency Aware RAW Image Object Detection Enhancer (SFAE), a novel framework that synergizes spatial and frequency representations. Our contribution is threefold. The first lies in the ``spatialization" of frequency bands. Different from the traditional paradigm of directly manipulating abstract spectra in deep networks, our method inversely transforms individual frequency bands back into tangible spatial maps, thus preserving direct physical intuition. Then the cross-domain fusion attention module is developed to enable deep multimodal interactions between these maps and the original spatial features. Finally, the framework performs adaptive nonlinear adjustments by predicting and applying different gamma parameters for the two domains. 

**Abstract (ZH)**: 基于RAW数据的直接对象检测具有巨大潜力，但宽动态范围和线性响应导致的固有挑战可能会抑制关键对象细节。为了克服这一限制，我们转向了频域，在频域中，基于频率，对象轮廓和纹理等特征可以自然分离。在这篇论文中，我们提出了空间-频率感知RAW图像对象检测增强器（SFAE），这是一种协同利用空间和频率表示的新框架。我们的贡献包括三个方面。首先，将频率带“空间化”。不同于传统直接操作深度网络中抽象频谱的做法，我们的方法将单个频率带逆变换回具体的空间图，从而保留直接的物理直觉。然后开发了跨域融合注意力模块，以实现这些图与原始空间特征之间的深层次多模态交互。最后，框架通过预测并为两个域应用不同的伽马参数来进行自适应非线性调整。 

---
# Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation 

**Title (ZH)**: 通过得分到性能：基于小节级符号记谱的人性化高效长乐曲生成 

**Authors**: Tongxi Wang, Yang Yu, Qing Wang, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01394)  

**Abstract**: Song generation is regarded as the most challenging problem in music AIGC; nonetheless, existing approaches have yet to fully overcome four persistent limitations: controllability, generalizability, perceptual quality, and duration. We argue that these shortcomings stem primarily from the prevailing paradigm of attempting to learn music theory directly from raw audio, a task that remains prohibitively difficult for current models. To address this, we present Bar-level AI Composing Helper (BACH), the first model explicitly designed for song generation through human-editable symbolic scores. BACH introduces a tokenization strategy and a symbolic generative procedure tailored to hierarchical song structure. Consequently, it achieves substantial gains in the efficiency, duration, and perceptual quality of song generation. Experiments demonstrate that BACH, with a small model size, establishes a new SOTA among all publicly reported song generation systems, even surpassing commercial solutions such as Suno. Human evaluations further confirm its superiority across multiple subjective metrics. 

**Abstract (ZH)**: 歌曲生成被认为是音乐AIGC中最具挑战性的问题；然而，现有方法尚未完全克服四大持久难题：可控性、普适性、感知质量以及时长。我们argue这些缺陷主要源于当前范式直接从原始音频中学习音乐理论的任务，这仍然是当前模型无法克服的难题。为此，我们提出了Bar-level AI Composing Helper (BACH)，这是首个专门通过可编辑符号谱进行歌曲生成的模型。BACH引入了一种标记化策略和针对层次化歌曲结构的符号生成流程，从而在歌曲生成的效率、时长和感知质量方面取得了显著提升。实验结果显示，尽管模型规模较小，BACH仍建立了所有已报道歌曲生成系统中的新SOTA，甚至超越了诸如Suno等商业解决方案。进一步的人类评估也证实了其在多个主观指标上的优越性。 

---
# Recognising, Anticipating, and Mitigating LLM Pollution of Online Behavioural Research 

**Title (ZH)**: 识别、预见和缓解大规模语言模型对在线行为研究的污染 

**Authors**: Raluca Rilla, Tobias Werner, Hiromu Yakura, Iyad Rahwan, Anne-Marie Nussberger  

**Link**: [PDF](https://arxiv.org/pdf/2508.01390)  

**Abstract**: Online behavioural research faces an emerging threat as participants increasingly turn to large language models (LLMs) for advice, translation, or task delegation: LLM Pollution. We identify three interacting variants through which LLM Pollution threatens the validity and integrity of online behavioural research. First, Partial LLM Mediation occurs when participants make selective use of LLMs for specific aspects of a task, such as translation or wording support, leading researchers to (mis)interpret LLM-shaped outputs as human ones. Second, Full LLM Delegation arises when agentic LLMs complete studies with little to no human oversight, undermining the central premise of human-subject research at a more foundational level. Third, LLM Spillover signifies human participants altering their behaviour as they begin to anticipate LLM presence in online studies, even when none are involved. While Partial Mediation and Full Delegation form a continuum of increasing automation, LLM Spillover reflects second-order reactivity effects. Together, these variants interact and generate cascading distortions that compromise sample authenticity, introduce biases that are difficult to detect post hoc, and ultimately undermine the epistemic grounding of online research on human cognition and behaviour. Crucially, the threat of LLM Pollution is already co-evolving with advances in generative AI, creating an escalating methodological arms race. To address this, we propose a multi-layered response spanning researcher practices, platform accountability, and community efforts. As the challenge evolves, coordinated adaptation will be essential to safeguard methodological integrity and preserve the validity of online behavioural research. 

**Abstract (ZH)**: 在线行为研究面临新兴威胁：随着参与者越来越多地利用大型语言模型（LLMs）获取建议、翻译或任务委托，出现LLM污染。我们识别出三种交互变异，这些变异威胁着在线行为研究的有效性和完整性。首先，部分LLM中介发生时，参与者仅选择性地将LLMs用于任务的特定方面，如翻译或措辞支持，从而使研究人员（误）将LLM形成的输出解读为人类的行为。其次，全面LLM委托发生在由具有代理性的LLMs独立完成研究几乎不需要人类监督的情况下，从根本上削弱了以人类受试者为中心的研究的前提。第三，LLM外溢表明，在参与者开始预期在线研究中可能存在LLMs的情况下，即使没有涉及LLMs，他们的行为也会发生变化。尽管部分中介和全面委托构成了逐渐自动化的连续体，但LLM外溢反映了次级反应效应。这些变异彼此交互作用，引发级联失真，损害样本的真实性，引入难以事后检测的偏见，并最终削弱在线研究人类认知和行为的证成基础。重要的是，LLM污染的威胁正在与生成式AI的进步协同演变，形成一场升级的方法论军备竞赛。为了应对这一挑战，我们建议采取多层次的应对措施，涵盖研究人员实践、平台责任和社区努力。随着挑战的演变，协调适应将是保护方法论完整性和维护在线行为研究有效性的关键。 

---
# Video-based Vehicle Surveillance in the Wild: License Plate, Make, and Model Recognition with Self Reflective Vision-Language Models 

**Title (ZH)**: 基于视频的野外车辆监控：自反性视觉-语言模型的车牌、品牌和型号识别 

**Authors**: Pouya Parsa, Keya Li, Kara M. Kockelman, Seongjin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01387)  

**Abstract**: Automatic license plate recognition (ALPR) and vehicle make and model recognition underpin intelligent transportation systems, supporting law enforcement, toll collection, and post-incident investigation. Applying these methods to videos captured by handheld smartphones or non-static vehicle-mounted cameras presents unique challenges compared to fixed installations, including frequent camera motion, varying viewpoints, occlusions, and unknown road geometry. Traditional ALPR solutions, dependent on specialized hardware and handcrafted OCR pipelines, often degrade under these conditions. Recent advances in large vision-language models (VLMs) enable direct recognition of textual and semantic attributes from arbitrary imagery. This study evaluates the potential of VLMs for ALPR and makes and models recognition using monocular videos captured with handheld smartphones and non-static mounted cameras. The proposed license plate recognition pipeline filters to sharp frames, then sends a multimodal prompt to a VLM using several prompt strategies. Make and model recognition pipeline runs the same VLM with a revised prompt and an optional self-reflection module. In the self-reflection module, the model contrasts the query image with a reference from a 134-class dataset, correcting mismatches. Experiments on a smartphone dataset collected on the campus of the University of Texas at Austin, achieve top-1 accuracies of 91.67% for ALPR and 66.67% for make and model recognition. On the public UFPR-ALPR dataset, the approach attains 83.05% and 61.07%, respectively. The self-reflection module further improves results by 5.72% on average for make and model recognition. These findings demonstrate that VLMs provide a cost-effective solution for scalable, in-motion traffic video analysis. 

**Abstract (ZH)**: 基于单目手机视频和非静止安装摄像头的自动车牌识别和车辆品牌型号识别：利用大型视觉语言模型的潜力 

---
# A Full-Stage Refined Proposal Algorithm for Suppressing False Positives in Two-Stage CNN-Based Detection Methods 

**Title (ZH)**: 一种用于抑制基于两阶段CNN检测方法中假阳性的一整阶段细化提案算法 

**Authors**: Qiang Guo, Rubo Zhang, Bingbing Zhang, Junjie Liu, Jianqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01382)  

**Abstract**: False positives in pedestrian detection remain a challenge that has yet to be effectively resolved. To address this issue, this paper proposes a Full-stage Refined Proposal (FRP) algorithm aimed at eliminating these false positives within a two-stage CNN-based pedestrian detection framework. The main innovation of this work lies in employing various pedestrian feature re-evaluation strategies to filter out low-quality pedestrian proposals during both the training and testing stages. Specifically, in the training phase, the Training mode FRP algorithm (TFRP) introduces a novel approach for validating pedestrian proposals to effectively guide the model training process, thereby constructing a model with strong capabilities for false positive suppression. During the inference phase, two innovative strategies are implemented: the Classifier-guided FRP (CFRP) algorithm integrates a pedestrian classifier into the proposal generation pipeline to yield high-quality proposals through pedestrian feature evaluation, and the Split-proposal FRP (SFRP) algorithm vertically divides all proposals, sending both the original and the sub-region proposals to the subsequent subnetwork to evaluate their confidence scores, filtering out those with lower sub-region pedestrian confidence scores. As a result, the proposed algorithm enhances the model's ability to suppress pedestrian false positives across all stages. Various experiments conducted on multiple benchmarks and the SY-Metro datasets demonstrate that the model, supported by different combinations of the FRP algorithm, can effectively eliminate false positives to varying extents. Furthermore, experiments conducted on embedded platforms underscore the algorithm's effectiveness in enhancing the comprehensive pedestrian detection capabilities of the small pedestrian detector in resource-constrained edge devices. 

**Abstract (ZH)**: 虚假正例在行人检测中仍然是一个尚未有效解决的挑战。为了应对这一问题，本文提出了一种全阶段精炼提案（FRP）算法，旨在在一个基于两阶段CNN的行人检测框架中消除这些虚假正例。本文的主要创新点在于，在训练和测试阶段采用多种行人特征重评估策略，过滤掉低质量的行人提案。在训练阶段，提出了训练模式FRP算法（TFRP），引入了一种新的行人提案验证方法，有效指导模型训练过程，构建出具备较强抑制虚假正例能力的模型。在推理阶段，实施了两种创新策略：分类器导向的FRP（CFRP）算法将行人分类器集成到提案生成管道中，通过行人特征评估生成高质量提案；分割提案的FRP（SFRP）算法垂直划分所有提案，将原始提案和子区域提案同时送入后续子网络评估其置信分数，过滤置信分数较低的子区域行人提案。结果表明，所提算法能够增强模型在所有阶段抑制行人虚假正例的能力。在多个基准数据集和SY-Metro数据集上的各种实验结果表明，由不同组合的FRP算法支持的模型能够不同程度地消除虚假正例。此外，嵌入式平台上的实验进一步证实了该算法在资源受限边缘设备中增强小型行人检测器整体行人检测能力的有效性。 

---
# Effective Damage Data Generation by Fusing Imagery with Human Knowledge Using Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型融合图像与人类知识的有效损伤数据生成 

**Authors**: Jie Wei, Erika Ardiles-Cruz, Aleksey Panasyuk, Erik Blasch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01380)  

**Abstract**: It is of crucial importance to assess damages promptly and accurately in humanitarian assistance and disaster response (HADR). Current deep learning approaches struggle to generalize effectively due to the imbalance of data classes, scarcity of moderate damage examples, and human inaccuracy in pixel labeling during HADR situations. To accommodate for these limitations and exploit state-of-the-art techniques in vision-language models (VLMs) to fuse imagery with human knowledge understanding, there is an opportunity to generate a diversified set of image-based damage data effectively. Our initial experimental results suggest encouraging data generation quality, which demonstrates an improvement in classifying scenes with different levels of structural damage to buildings, roads, and infrastructures. 

**Abstract (ZH)**: 在人道主义援助与灾难响应中及时准确评估损害至关重要。当前的深度学习方法由于数据类别的不平衡、中等损害示例稀缺以及在灾难响应情况下像素标注的人为不准确，难以有效泛化。为克服这些限制并利用视觉-语言模型（VLMs）的最新技术将图像与人类知识理解相结合，有机会有效生成多样化的目标损害数据集。我们的初步实验结果表明，生成的数据质量令人鼓舞，展示了在区分不同结构损害水平的场景方面有所改进，涉及建筑物、道路和基础设施。 

---
# Prompt to Pwn: Automated Exploit Generation for Smart Contracts 

**Title (ZH)**: 从提示到掌控：智能合约的自动化漏洞利用生成 

**Authors**: Zeke Xiao, Yuekang Li, Qin Wang, Shiping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01371)  

**Abstract**: We explore the feasibility of using LLMs for Automated Exploit Generation (AEG) against vulnerable smart contracts. We present \textsc{ReX}, a framework integrating LLM-based exploit synthesis with the Foundry testing suite, enabling the automated generation and validation of proof-of-concept (PoC) exploits. We evaluate five state-of-the-art LLMs (GPT-4.1, Gemini 2.5 Pro, Claude Opus 4, DeepSeek, and Qwen3 Plus) on both synthetic benchmarks and real-world smart contracts affected by known high-impact exploits. Our results show that modern LLMs can reliably generate functional PoC exploits for diverse vulnerability types, with success rates reaching up to 92\%. Notably, Gemini 2.5 Pro and GPT-4.1 consistently outperform others in both synthetic and real-world scenarios. We further analyze factors influencing AEG effectiveness, including model capabilities, contract structure, and vulnerability types. We also collect the first curated dataset of real-world PoC exploits to support future research. 

**Abstract (ZH)**: 我们探索使用大规模语言模型（LLMs）进行自动exploit生成（AEG）以攻击易受攻击的智能合约的可行性。我们提出了ReX框架，该框架将基于LLM的exploit合成与Foundry测试套件集成，从而实现PoC exploit的自动化生成和验证。我们评估了五种最先进的LLM（GPT-4.1、Gemini 2.5 Pro、Claude Opus 4、DeepSeek和Qwen3 Plus）在合成基准和受已知高影响exploit影响的真实世界智能合约上的表现。结果显示，现代LLM能够可靠地生成适用于多种漏洞类型的PoC exploit，成功率高达92%。值得注意的是，Gemini 2.5 Pro和GPT-4.1在合成和真实世界场景中表现始终优于其他模型。我们还进一步分析了影响AEG效果的因素，包括模型能力、合约结构和漏洞类型。我们还收集了首个经过整理的实际PoC exploit数据集，以支持未来研究。 

---
# Classification of Brain Tumors using Hybrid Deep Learning Models 

**Title (ZH)**: 使用混合深度学习模型的脑肿瘤分类 

**Authors**: Neerav Nemchand Gala  

**Link**: [PDF](https://arxiv.org/pdf/2508.01350)  

**Abstract**: The use of Convolutional Neural Networks (CNNs) has greatly improved the interpretation of medical images. However, conventional CNNs typically demand extensive computational resources and large training datasets. To address these limitations, this study applied transfer learning to achieve strong classification performance using fewer training samples. Specifically, the study compared EfficientNetV2 with its predecessor, EfficientNet, and with ResNet50 in classifying brain tumors into three types: glioma, meningioma, and pituitary tumors. Results showed that EfficientNetV2 delivered superior performance compared to the other models. However, this improvement came at the cost of increased training time, likely due to the model's greater complexity. 

**Abstract (ZH)**: 基于转移学习的EfficientNetV2在脑肿瘤分类中的应用：减少训练样本数量以提高医疗图像解释性能 

---
# Convergence Analysis of Aggregation-Broadcast in LoRA-enabled Federated Learning 

**Title (ZH)**: LoRA增强联邦学习中聚合-广播的收敛性分析 

**Authors**: Xin Chen, Shuaijun Chen, Omid Tavallaie, Nguyen Tran, Shuhuang Xiang, Albert Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2508.01348)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized data sources while preserving data privacy. However, the growing size of Machine Learning (ML) models poses communication and computation challenges in FL. Low-Rank Adaptation (LoRA) has recently been introduced into FL as an efficient fine-tuning method, reducing communication overhead by updating only a small number of trainable parameters. Despite its effectiveness, how to aggregate LoRA-updated local models on the server remains a critical and understudied problem. In this paper, we provide a unified convergence analysis for LoRA-based FL. We first categories the current aggregation method into two major type: Sum-Product (SP) and Product-Sum (PS). Then we formally define the Aggregation-Broadcast Operator (ABO) and derive a general convergence condition under mild assumptions. Furthermore, we present several sufficient conditions that guarantee convergence of the global model. These theoretical analyze offer a principled understanding of various aggregation strategies. Notably, we prove that the SP and PS aggregation methods both satisfy our convergence condition, but differ in their ability to achieve the optimal convergence rate. Extensive experiments on standard benchmarks validate our theoretical findings. 

**Abstract (ZH)**: 联邦学习（FL）能够在保护数据隐私的同时，跨分散的数据源进行协同模型训练。然而，机器学习（ML）模型的快速增长给FL带来了通信和计算上的挑战。低秩适应（LoRA） recently has被引入到FL中，作为一种有效的微调方法，通过仅更新少量可训练参数来减少通信开销。尽管LoRA非常有效，但在服务器端如何聚合LoRA更新的本地模型仍然是一个关键且研究不足的问题。在本文中，我们提供了LoRA基联邦学习的统一收敛性分析。我们首先将当前的聚合方法归纳为两大类：求和-积（SP）和积-求和（PS）。然后我们形式化定义了聚合-广播操作符（ABO），并在温和假设下推导出一般的收敛条件。此外，我们提出了若干保证全局模型收敛的充分条件。这些理论分析为各种聚合策略提供了基本原则的理解。值得注意的是，我们证明了SP和PS聚合方法都满足我们的收敛条件，但在实现最优收敛速率方面有所不同。广泛的实验证明了我们理论发现的有效性。 

---
# UEChecker: Detecting Unchecked External Call Vulnerabilities in DApps via Graph Analysis 

**Title (ZH)**: UEChecker: 通过图分析检测DApps中的未检查外部调用漏洞 

**Authors**: Dechao Kong, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01343)  

**Abstract**: The increasing number of attacks on the contract layer of DApps has resulted in economic losses amounting to $66 billion. Vulnerabilities arise when contracts interact with external protocols without verifying the results of the calls, leading to exploit entry points such as flash loan attacks and reentrancy attacks. In this paper, we propose UEChecker, a deep learning-based tool that utilizes a call graph and a Graph Convolutional Network to detect unchecked external call vulnerabilities. We design the following components: An edge prediction module that reconstructs the feature representation of nodes and edges in the call graph; A node aggregation module that captures structural information from both the node itself and its neighbors, thereby enhancing feature representation between nodes and improving the model's understanding of the global graph structure; A Conformer Block module that integrates multi-head attention, convolutional modules, and feedforward neural networks to more effectively capture dependencies of different scales within the call graph, extending beyond immediate neighbors and enhancing the performance of vulnerability detection. Finally, we combine these modules with Graph Convolutional Network to detect unchecked external call vulnerabilities. By auditing the smart contracts of 608 DApps, our results show that our tool achieves an accuracy of 87.59% in detecting unchecked external call vulnerabilities. Furthermore, we compare our tool with GAT, LSTM, and GCN baselines, and in the comparison experiments, UEChecker consistently outperforms these models in terms of accuracy. 

**Abstract (ZH)**: DApp合约层攻击不断增加导致经济损失660亿美元，外部调用未验证漏洞引发闪贷攻击和重入攻击等exploit入口。本文提出UEChecker，一种基于深度学习的工具，利用调用图和图卷积网络检测未验证外部调用漏洞。我们设计了以下组件：边预测模块，重建调用图中节点和边的特征表示；节点聚合模块，捕获节点本身及其邻居的信息，增强节点间的特征表示并提升模型对全局图结构的理解；Conformer Block模块，结合多头注意力机制、卷积模块和前馈神经网络，更有效地捕捉调用图中不同尺度的依赖关系，超越直接邻居以提高漏洞检测性能。最后，我们将这些模块与图卷积网络结合，检测未验证的外部调用漏洞。通过审计608个DApp的智能合约，结果显示，我们的工具在检测未验证的外部调用漏洞方面达到了87.59%的准确率。此外，在与GAT、LSTM和GCN基准模型的比较实验中，UEChecker在准确率方面始终优于这些模型。 

---
# SBP-YOLO:A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes 

**Title (ZH)**: SBP-YOLO：一种实时检测减速带和坑洞的 Lightweight 模型 

**Authors**: Chuanqi Liang, Jie Fu, Lei Luo, Miao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01339)  

**Abstract**: With increasing demand for ride comfort in new energy vehicles, accurate real-time detection of speed bumps and potholes is critical for predictive suspension control. This paper proposes SBP-YOLO, a lightweight detection framework based on YOLOv11, optimized for embedded deployment. The model integrates GhostConv for efficient computation, VoVGSCSPC for multi-scale feature enhancement, and a Lightweight Efficiency Detection Head (LEDH) to reduce early-stage feature processing costs. A hybrid training strategy combining NWD loss, knowledge distillation, and Albumentations-based weather augmentation improves detection robustness, especially for small and distant targets. Experiments show SBP-YOLO achieves 87.0% mAP (outperforming YOLOv11n by 5.8%) and runs at 139.5 FPS on a Jetson AGX Xavier with TensorRT FP16 quantization. The results validate its effectiveness for real-time road condition perception in intelligent suspension systems. 

**Abstract (ZH)**: 基于YOLOv11的轻量化速度 bumps 和坑洞检测框架 SBP-YOLO 

---
# Weakly-Supervised Image Forgery Localization via Vision-Language Collaborative Reasoning Framework 

**Title (ZH)**: 弱监督图像伪造定位 via 视觉-语言协作推理框架 

**Authors**: Ziqi Sheng, Junyan Wu, Wei Lu, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.01338)  

**Abstract**: Image forgery localization aims to precisely identify tampered regions within images, but it commonly depends on costly pixel-level annotations. To alleviate this annotation burden, weakly supervised image forgery localization (WSIFL) has emerged, yet existing methods still achieve limited localization performance as they mainly exploit intra-image consistency clues and lack external semantic guidance to compensate for weak supervision. In this paper, we propose ViLaCo, a vision-language collaborative reasoning framework that introduces auxiliary semantic supervision distilled from pre-trained vision-language models (VLMs), enabling accurate pixel-level localization using only image-level labels. Specifically, ViLaCo first incorporates semantic knowledge through a vision-language feature modeling network, which jointly extracts textual and visual priors using pre-trained VLMs. Next, an adaptive vision-language reasoning network aligns textual semantics and visual features through mutual interactions, producing semantically aligned representations. Subsequently, these representations are passed into dual prediction heads, where the coarse head performs image-level classification and the fine head generates pixel-level localization masks, thereby bridging the gap between weak supervision and fine-grained localization. Moreover, a contrastive patch consistency module is introduced to cluster tampered features while separating authentic ones, facilitating more reliable forgery discrimination. Extensive experiments on multiple public datasets demonstrate that ViLaCo substantially outperforms existing WSIFL methods, achieving state-of-the-art performance in both detection and localization accuracy. 

**Abstract (ZH)**: 基于视觉-语言协作推理的弱监督图像篡改定位 

---
# BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability 

**Title (ZH)**: BlockA2A: 向具有安全性和可验证性的代理间互操作性迈进 

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01332)  

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments. 

**Abstract (ZH)**: 基于大型语言模型的代理AI的快速采用正在转型企业生态系统，伴随着自主代理执行复杂工作流。然而，我们观察到由大型语言模型驱动的多代理系统（MAS）中存在几个关键的安全漏洞：碎片化的身份框架、不安全的通信通道以及对拜占庭代理或对抗性提示的不足防御。在本文中，我们首次系统地分析了这些新兴的多代理风险，并解释了为什么传统的安全策略无法有效应对这些风险。随后，我们提出了BlockA2A，这是首个统一的多代理信任框架，能够实现安全且可验证的代理间互操作性。从高层面来看，BlockA2A采用分散标识符（DIDs）实现细粒度跨域代理认证，利用区块链锚定的账本实现不可变审计，通过智能合约动态执行基于上下文的访问控制策略。BlockA2A消除了集中式信任瓶颈，保障消息的 authenticity 和执行的完整性，并确保代理间交互的可问责性。此外，我们提出了一种防御编排引擎（DOE），通过实时机制积极消除攻击，包括拜占庭代理标记、反应性执行停止和即时权限撤销。实证评估证明BlockA2A在中和基于提示、基于通信、行为和系统的MAS攻击方面具有有效性。我们正式化了其集成到现有MAS的方式，并展示了Google A2A协议的实用实现。实验结果证实，BlockA2A和DOE的操作延迟低于毫秒级，能够在生产基于LLM的MAS环境中实现可扩展部署。 

---
# Referring Remote Sensing Image Segmentation with Cross-view Semantics Interaction Network 

**Title (ZH)**: 基于跨视图语义交互网络的遥感图像分割参考模型 

**Authors**: Jiaxing Yang, Lihe Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01331)  

**Abstract**: Recently, Referring Remote Sensing Image Segmentation (RRSIS) has aroused wide attention. To handle drastic scale variation of remote targets, existing methods only use the full image as input and nest the saliency-preferring techniques of cross-scale information interaction into traditional single-view structure. Although effective for visually salient targets, they still struggle in handling tiny, ambiguous ones in lots of real scenarios. In this work, we instead propose a paralleled yet unified segmentation framework Cross-view Semantics Interaction Network (CSINet) to solve the limitations. Motivated by human behavior in observing targets of interest, the network orchestrates visual cues from remote and close distances to conduct synergistic prediction. In its every encoding stage, a Cross-View Window-attention module (CVWin) is utilized to supplement global and local semantics into close-view and remote-view branch features, finally promoting the unified representation of feature in every encoding stage. In addition, we develop a Collaboratively Dilated Attention enhanced Decoder (CDAD) to mine the orientation property of target and meanwhile integrate cross-view multiscale features. The proposed network seamlessly enhances the exploitation of global and local semantics, achieving significant improvements over others while maintaining satisfactory speed. 

**Abstract (ZH)**: 近年来，参考遥感图像分割（RRSIS）引起了广泛关注。为了解决遥感目标的剧烈尺度变化问题，现有方法仅使用整幅图像作为输入，并将跨尺度信息交互的显著性偏好技术嵌入到传统的单视角结构中。尽管在视觉显著目标上有效，但在许多实际场景中小而模糊的目标仍然难以处理。在本工作中，我们提出了一种并行且统一的分割框架交叉视角语义交互网络（CSINet）以解决上述限制。受人类观察感兴趣目标行为的启发，该网络协调远距离和近距离的视觉线索进行协同预测。在每一编码阶段，利用跨视角窗口注意力模块（CVWin）补充全局和局部语义到近距离视角和远距离视角分支特征中，最终促进每一编码阶段特征的统一表示。此外，我们开发了一种协作膨胀注意力增强解码器（CDAD），以挖掘目标的方向特性并同时整合多尺度跨视角特征。所提出的网络无损地增强了对全局和局部语义的利用，实现了显著的性能提升，同时保持了满意的运行速度。 

---
# Is Exploration or Optimization the Problem for Deep Reinforcement Learning? 

**Title (ZH)**: 深度强化学习中是探索还是优化出了问题？ 

**Authors**: Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.01329)  

**Abstract**: In the era of deep reinforcement learning, making progress is more complex, as the collected experience must be compressed into a deep model for future exploitation and sampling. Many papers have shown that training a deep learning policy under the changing state and action distribution leads to sub-optimal performance, or even collapse. This naturally leads to the concern that even if the community creates improved exploration algorithms or reward objectives, will those improvements fall on the \textit{deaf ears} of optimization difficulties. This work proposes a new \textit{practical} sub-optimality estimator to determine optimization limitations of deep reinforcement learning algorithms. Through experiments across environments and RL algorithms, it is shown that the difference between the best experience generated is 2-3$\times$ better than the policies' learned performance. This large difference indicates that deep RL methods only exploit half of the good experience they generate. 

**Abstract (ZH)**: 在深度强化学习时代，进步更加复杂：一种新的实用次优性估计器确定深度强化学习算法的优化限制 

---
# D-SCoRE: Document-Centric Segmentation and CoT Reasoning with Structured Export for QA-CoT Data Generation 

**Title (ZH)**: D-SCoRE: 文档中心的分割与共理推理及结构化导出用于QA-CoT数据生成 

**Authors**: Weibo Zhou, Lingbo Li, Shangsong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01309)  

**Abstract**: The scarcity and high cost of high-quality question-answering (QA) datasets hinder supervised fine-tuning (SFT) for domain-specific large language models (LLMs). To address this, we introduce D-SCoRE, a training-free pipeline that utilizes LLMs and prompt engineering to produce diverse, high-quality QA datasets from arbitrary textual sources. D-SCoRE integrates $\textbf{D}$ocument-centric processing, $\textbf{S}$egmentation, $\textbf{Co}$T $\textbf{R}$easoning, and structured $\textbf{E}$xport to generate QA-COT datasets tailored for domain-aware SFT. Multi-dimensional control mechanisms, such as semantic role transformation, question type balancing, and counterfactual materials, enhance diversity and relevance, overcoming limitations of existing QA generation. LLMs fine-tuned on D-SCoRE-generated QA datasets, and human-annotated QA datasets (SQuAD, Covid-QA) are evaluated on SQuADShifts and Covid-QA test sets, with D-SCoRE outperforming across most domains. D-SCoRE generates six QA-CoT pairs with four-option counterfactual materials per 100-200-word text in 90 seconds using an 8B LLM on consumer-grade hardware. Its simplicity and scalability enable efficient QA generation and high-performance fine-tuning across domains. 

**Abstract (ZH)**: 高质问答数据的稀缺性和高昂成本阻碍了领域特定大规模语言模型的监督微调。为解决这一问题，我们引入了D-SCoRE，这是一种无需训练的管道，利用大规模语言模型和提示工程从任意文本源生成多样化的高质量问答数据集。D-SCoRE集成了文档中心处理、分段、成本推理和结构化导出，以生成适用于领域感知微调的问答-成本推理数据集。多维度控制机制，如语义角色转换、问题类型平衡和反事实材料，增强了多样性和相关性，克服了现有问答生成的局限性。基于D-SCoRE生成的问答数据集微调的大规模语言模型以及人类标注的问答数据集（SQuAD、Covid-QA）在SQuADShifts和Covid-QA测试集上的评估结果显示，D-SCoRE在大多数领域表现出色。D-SCoRE使用8B参数的大规模语言模型在消费级硬件上可以在90秒内生成每100-200词的6个问答-成本推理对和四个选项的反事实材料。其简洁性和可扩展性使得跨领域高效生成问答数据和高性能微调成为可能。 

---
# GMAT: Grounded Multi-Agent Clinical Description Generation for Text Encoder in Vision-Language MIL for Whole Slide Image Classification 

**Title (ZH)**: GMAT：基于多剂型临床描述生成的地面真相文本编码在视觉-语言MIL中的全切片影像分类中应用 

**Authors**: Ngoc Bui Lam Quang, Nam Le Nguyen Binh, Thanh-Huy Nguyen, Le Thien Phuc Nguyen, Quan Nguyen, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2508.01293)  

**Abstract**: Multiple Instance Learning (MIL) is the leading approach for whole slide image (WSI) classification, enabling efficient analysis of gigapixel pathology slides. Recent work has introduced vision-language models (VLMs) into MIL pipelines to incorporate medical knowledge through text-based class descriptions rather than simple class names. However, when these methods rely on large language models (LLMs) to generate clinical descriptions or use fixed-length prompts to represent complex pathology concepts, the limited token capacity of VLMs often constrains the expressiveness and richness of the encoded class information. Additionally, descriptions generated solely by LLMs may lack domain grounding and fine-grained medical specificity, leading to suboptimal alignment with visual features. To address these challenges, we propose a vision-language MIL framework with two key contributions: (1) A grounded multi-agent description generation system that leverages curated pathology textbooks and agent specialization (e.g., morphology, spatial context) to produce accurate and diverse clinical descriptions; (2) A text encoding strategy using a list of descriptions rather than a single prompt, capturing fine-grained and complementary clinical signals for better alignment with visual features. Integrated into a VLM-MIL pipeline, our approach shows improved performance over single-prompt class baselines and achieves results comparable to state-of-the-art models, as demonstrated on renal and lung cancer datasets. 

**Abstract (ZH)**: 基于视觉语言的多重实例学习框架：基于专业知识的描述生成与优化编码策略 

---
# CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis 

**Title (ZH)**: CoCoLIT: ControlNet条件下的潜空间图像转换用于MRI到淀粉样蛋白PET合成 

**Authors**: Alec Sargood, Lemuel Puglisi, James H. Cole, Neil P. Oxtoby, Daniele Ravì, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2508.01292)  

**Abstract**: Synthesizing amyloid PET scans from the more widely available and accessible structural MRI modality offers a promising, cost-effective approach for large-scale Alzheimer's Disease (AD) screening. This is motivated by evidence that, while MRI does not directly detect amyloid pathology, it may nonetheless encode information correlated with amyloid deposition that can be uncovered through advanced modeling. However, the high dimensionality and structural complexity of 3D neuroimaging data pose significant challenges for existing MRI-to-PET translation methods. Modeling the cross-modality relationship in a lower-dimensional latent space can simplify the learning task and enable more effective translation. As such, we present CoCoLIT (ControlNet-Conditioned Latent Image Translation), a diffusion-based latent generative framework that incorporates three main innovations: (1) a novel Weighted Image Space Loss (WISL) that improves latent representation learning and synthesis quality; (2) a theoretical and empirical analysis of Latent Average Stabilization (LAS), an existing technique used in similar generative models to enhance inference consistency; and (3) the introduction of ControlNet-based conditioning for MRI-to-PET translation. We evaluate CoCoLIT's performance on publicly available datasets and find that our model significantly outperforms state-of-the-art methods on both image-based and amyloid-related metrics. Notably, in amyloid-positivity classification, CoCoLIT outperforms the second-best method with improvements of +10.5% on the internal dataset and +23.7% on the external dataset. The code and models of our approach are available at this https URL. 

**Abstract (ZH)**: 从更广泛可用的结构性MRI模态合成淀粉样蛋白PET扫描为大规模阿尔茨海默病（AD）筛查提供了一种有前景且成本效益高的方法。 

---
# Exploitation Is All You Need... for Exploration 

**Title (ZH)**: 你需要的只是利用……而不是探索 

**Authors**: Micah Rentschler, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2508.01287)  

**Abstract**: Ensuring sufficient exploration is a central challenge when training meta-reinforcement learning (meta-RL) agents to solve novel environments. Conventional solutions to the exploration-exploitation dilemma inject explicit incentives such as randomization, uncertainty bonuses, or intrinsic rewards to encourage exploration. In this work, we hypothesize that an agent trained solely to maximize a greedy (exploitation-only) objective can nonetheless exhibit emergent exploratory behavior, provided three conditions are met: (1) Recurring Environmental Structure, where the environment features repeatable regularities that allow past experience to inform future choices; (2) Agent Memory, enabling the agent to retain and utilize historical interaction data; and (3) Long-Horizon Credit Assignment, where learning propagates returns over a time frame sufficient for the delayed benefits of exploration to inform current decisions. Through experiments in stochastic multi-armed bandits and temporally extended gridworlds, we observe that, when both structure and memory are present, a policy trained on a strictly greedy objective exhibits information-seeking exploratory behavior. We further demonstrate, through controlled ablations, that emergent exploration vanishes if either environmental structure or agent memory is absent (Conditions 1 & 2). Surprisingly, removing long-horizon credit assignment (Condition 3) does not always prevent emergent exploration-a result we attribute to the pseudo-Thompson Sampling effect. These findings suggest that, under the right prerequisites, exploration and exploitation need not be treated as orthogonal objectives but can emerge from a unified reward-maximization process. 

**Abstract (ZH)**: 确保充分探索是训练元强化学习（元-RL）代理解决新型环境时的主要挑战。传统解决方案通过注入显式的激励机制如随机化、不确定性奖励或内在奖励来促进探索。在本研究中，我们假设，只要满足三个条件，仅最大化贪婪（仅探索）目标的代理仍然可以表现出 Emergent 探索行为：（1）反复出现的环境结构，使得过去的经验能够对未来的选择提供指导；（2）代理记忆，使代理能够保留和利用历史交互数据；以及（3）长时延信用分配，使得学习能传播足够长时间范围内的回报，从而让探索的延迟收益能够影响当前决策。通过在随机多臂 bandit 和时间扩展的网格世界中的实验，我们观察到，在结构和记忆同时存在的情况下，严格遵循贪婪目标训练的策略会表现出信息寻求的探索行为。进一步通过受控的退化实验，我们证明，如果缺失环境结构或代理记忆（条件 1 和 2），Emergent 探索行为会消失。令人惊讶的是，移除长时延信用分配（条件 3）并不总是能防止 Emergent 探索行为，我们将这一结果归因于伪-Thompson 抽样效应。这些发现表明，在合适的前提条件下，探索和利用不一定要被视为正交的目标，而是可以从统一的奖励最大化过程中自然涌现。 

---
# Defending Against Beta Poisoning Attacks in Machine Learning Models 

**Title (ZH)**: 在机器学习模型中防御贝塔中毒攻击 

**Authors**: Nilufer Gulciftci, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01276)  

**Abstract**: Poisoning attacks, in which an attacker adversarially manipulates the training dataset of a machine learning (ML) model, pose a significant threat to ML security. Beta Poisoning is a recently proposed poisoning attack that disrupts model accuracy by making the training dataset linearly nonseparable. In this paper, we propose four defense strategies against Beta Poisoning attacks: kNN Proximity-Based Defense (KPB), Neighborhood Class Comparison (NCC), Clustering-Based Defense (CBD), and Mean Distance Threshold (MDT). The defenses are based on our observations regarding the characteristics of poisoning samples generated by Beta Poisoning, e.g., poisoning samples have close proximity to one another, and they are centered near the mean of the target class. Experimental evaluations using MNIST and CIFAR-10 datasets demonstrate that KPB and MDT can achieve perfect accuracy and F1 scores, while CBD and NCC also provide strong defensive capabilities. Furthermore, by analyzing performance across varying parameters, we offer practical insights regarding defenses' behaviors under varying conditions. 

**Abstract (ZH)**: 针对Beta中毒攻击的防御策略：基于kNN邻近性的防御（KPB）、邻域类比较（NCC）、基于聚类的防御（CBD）和均值距离阈值（MDT） 

---
# AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection 

**Title (ZH)**: AgentArmor: 在代理运行时轨迹上强制执行程序分析以防御提示注入攻击 

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01249)  

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's working traces as graph-based intermediate representations with control flow and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools & data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis over sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can achieve 95.75% of TPR, with only 3.66% of FPR. Our results demonstrate AgentArmor's ability to detect prompt injection vulnerabilities and enforce fine-grained security constraints. 

**Abstract (ZH)**: 大型语言模型代理通过结合自然语言推理和外部工具的执行来解决各种问题，展现出强大的新范式。然而，它们的动态和非透明行为在提示注入攻击的背景下引入了关键的安全风险。本文提出一种新颖见解，即将代理运行时轨迹视为具有可分析语义的结构化程序。为此，我们提出AgentArmor程序分析框架，将代理轨迹转换为基于图中间表示的结构化程序依赖表示（例如，控制流图、数据流图和程序依赖图），并通过类型系统实施安全策略。AgentArmor主要包括三个关键组件：（1）图构造器，将代理的工作轨迹重构为包含控制流和数据流的基于图的中间表示；（2）属性注册表，附加与交互工具及数据相关的安全相关信息；（3）类型系统，在中间表示上执行静态推理和检查。通过将代理行为表示为结构化程序，AgentArmor能够进行敏感数据流、信任边界和策略违规的程序分析。我们使用AgentDojo基准测试评估AgentArmor，结果显示AgentArmor的真正阳性率（TPR）为95.75%，假正阳性率（FPR）仅为3.66%。我们的结果证明了AgentArmor检测提示注入漏洞和实施细粒度安全约束的能力。 

---
# Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models 

**Title (ZH)**: 多缓存增强原型学习：视觉-语言模型的测试时泛化优化 

**Authors**: Xinyu Chen, Haotian Zhai, Can Zhang, Xiupeng Shi, Ruirui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01225)  

**Abstract**: In zero-shot setting, test-time adaptation adjusts pre-trained models using unlabeled data from the test phase to enhance performance on unknown test distributions. Existing cache-enhanced TTA methods rely on a low-entropy criterion to select samples for prototype construction, assuming intra-class compactness. However, low-entropy samples may be unreliable under distribution shifts, and the resulting prototypes may not ensure compact intra-class distributions. This study identifies a positive correlation between cache-enhanced performance and intra-class compactness. Based on this observation, we propose a Multi-Cache enhanced Prototype-based Test-Time Adaptation (MCP) featuring three caches: an entropy cache for initializing prototype representations with low-entropy samples, an align cache for integrating visual and textual information to achieve compact intra-class distributions, and a negative cache for prediction calibration using high-entropy samples. We further developed MCP++, a framework incorporating cross-modal prototype alignment and residual learning, introducing prototype residual fine-tuning. Comparative and ablation experiments across 15 downstream tasks demonstrate that the proposed method and framework achieve state-of-the-art generalization performance. 

**Abstract (ZH)**: 零样本设置下，测试时适应使用测试阶段的未标注数据调整预训练模型，以增强未知测试分布上的性能。现有的缓存增强测试时适应方法依赖于低熵准则来选择用于原型构建的样本，并假设类内紧凑性。然而，分布转移下低熵样本可能不可靠，由此产生的原型无法确保类内紧凑分布。本研究发现，缓存增强性能与类内紧凑性之间存在正相关。基于这一观察，我们提出了一种名为多缓存增强基于原型的测试时适应（MCP）的方法，包括三个缓存：熵缓存用于使用低熵样本初始化原型表示，对齐缓存用于整合视觉和文本信息以实现类内紧凑分布，以及负缓存用于预测校准，使用高熵样本。我们进一步开发了MCP++框架，融合了跨模态原型对齐和残差学习，引入了原型残差微调。在15个下游任务上的对比和消融实验表明，所提出的方法和框架实现了最先进的泛化性能。 

---
# WebDS: An End-to-End Benchmark for Web-based Data Science 

**Title (ZH)**: WebDS：基于Web的数据科学端到端基准 

**Authors**: Ethan Hsu, Hong Meng Yam, Ines Bouissou, Aaron Murali John, Raj Thota, Josh Koe, Vivek Sarath Putta, G K Dharesan, Alexander Spangher, Shikhar Murty, Tenghao Huang, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2508.01222)  

**Abstract**: A large portion of real-world data science tasks are complex and require multi-hop web-based interactions: finding appropriate data available on the internet, synthesizing real-time data of various modalities from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions, such as form submissions or e-commerce transactions, and often do not require diverse tool-using capabilities required for web based data science. Conversely, traditional data science benchmarks typically concentrate on static, often textually bound datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step operations requiring the use of tools and heterogeneous data formats that better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes 80% of tasks on Web Voyager, successfully completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes like poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS' tasks display. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science. 

**Abstract (ZH)**: WebDS：首个端到端的基于Web的数据科学基准 

---
# Oldie but Goodie: Re-illuminating Label Propagation on Graphs with Partially Observed Features 

**Title (ZH)**: 经典仍优良：利用部分观测特征重新照亮图上的标签传播 

**Authors**: Sukwon Yun, Xin Liu, Yunhak Oh, Junseok Lee, Tianlong Chen, Tsuyoshi Murata, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.01209)  

**Abstract**: In real-world graphs, we often encounter missing feature situations where a few or the majority of node features, e.g., sensitive information, are missed. In such scenarios, directly utilizing Graph Neural Networks (GNNs) would yield sub-optimal results in downstream tasks such as node classification. Despite the emergence of a few GNN-based methods attempting to mitigate its missing situation, when only a few features are available, they rather perform worse than traditional structure-based models. To this end, we propose a novel framework that further illuminates the potential of classical Label Propagation (Oldie), taking advantage of Feature Propagation, especially when only a partial feature is available. Now called by GOODIE, it takes a hybrid approach to obtain embeddings from the Label Propagation branch and Feature Propagation branch. To do so, we first design a GNN-based decoder that enables the Label Propagation branch to output hidden embeddings that align with those of the FP branch. Then, GOODIE automatically captures the significance of structure and feature information thanks to the newly designed Structure-Feature Attention. Followed by a novel Pseudo-Label contrastive learning that differentiates the contribution of each positive pair within pseudo-labels originating from the LP branch, GOODIE outputs the final prediction for the unlabeled nodes. Through extensive experiments, we demonstrate that our proposed model, GOODIE, outperforms the existing state-of-the-art methods not only when only a few features are available but also in abundantly available situations. Source code of GOODIE is available at: this https URL. 

**Abstract (ZH)**: 在现实世界的图中，我们经常会遇到节点特征缺失的情况，无论是少量还是多数节点的敏感信息丢失。在这种情况下，直接使用图神经网络（GNN）会在节点分类等下游任务中得到次优结果。尽管出现了一些基于GNN的方法试图解决这个问题，但在只有少量特征可用的情况下，它们的表现反而不如传统的基于结构的模型。为了解决这一问题，我们提出了一种新的框架，进一步突显了经典标签传播（Oldie）的传统潜力，利用特征传播，特别是在只有部分特征可用时。现在称为GOODIE，它采取混合方法从标签传播分支和特征传播分支中获得嵌入。为此，我们首先设计了一个基于GNN的解码器，使标签传播分支能够输出与特征传播分支一致的隐藏嵌入。然后，GOODIE通过新设计的结构-特征注意力自动捕捉结构和特征信息的重要性。接着，通过一种新的伪标签对比学习，区分标签传播分支伪标签中每个正样本对的贡献，GOODIE最终输出未标记节点的预测结果。通过广泛的实验证明，与现有最先进的方法相比，我们的模型GOODIE不仅在少量特征可用的情况下表现更优，在特征丰富的情况下也同样表现出色。GOODIE的源代码可在以下链接获取：this https URL。 

---
# Deep Learning for Pavement Condition Evaluation Using Satellite Imagery 

**Title (ZH)**: 使用卫星图像的 pavement 条件评估的深度学习方法 

**Authors**: Prathyush Kumar Reddy Lebaku, Lu Gao, Pan Lu, Jingran Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01206)  

**Abstract**: Civil infrastructure systems covers large land areas and needs frequent inspections to maintain their public service capabilities. The conventional approaches of manual surveys or vehicle-based automated surveys to assess infrastructure conditions are often labor-intensive and time-consuming. For this reason, it is worthwhile to explore more cost-effective methods for monitoring and maintaining these infrastructures. Fortunately, recent advancements in satellite systems and image processing algorithms have opened up new possibilities. Numerous satellite systems have been employed to monitor infrastructure conditions and identify damages. Due to the improvement in ground sample distance (GSD), the level of detail that can be captured has significantly increased. Taking advantage of these technology advancement, this research investigated to evaluate pavement conditions using deep learning models for analyzing satellite images. We gathered over 3,000 satellite images of pavement sections, together with pavement evaluation ratings from TxDOT's PMIS database. The results of our study show an accuracy rate is exceeding 90%. This research paves the way for a rapid and cost-effective approach to evaluating the pavement network in the future. 

**Abstract (ZH)**: 基于卫星影像的深度学习路面状况评估方法 

---
# Conquering High Packet-Loss Erasure: MoE Swin Transformer-Based Video Semantic Communication 

**Title (ZH)**: 基于MoE SwinTransformer的视频语义通信： conquering 高丢包率擦除 

**Authors**: Lei Teng, Senran Fan, Chen Dong, Haotai Liang, Zhicheng Bao, Xiaodong Xu, Rui Meng, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01205)  

**Abstract**: Semantic communication with joint semantic-channel coding robustly transmits diverse data modalities but faces challenges in mitigating semantic information loss due to packet drops in packet-based systems. Under current protocols, packets with errors are discarded, preventing the receiver from utilizing erroneous semantic data for robust decoding. To address this issue, a packet-loss-resistant MoE Swin Transformer-based Video Semantic Communication (MSTVSC) system is proposed in this paper. Semantic vectors are encoded by MSTVSC and transmitted through upper-layer protocol packetization. To investigate the impact of the packetization, a theoretical analysis of the packetization strategy is provided. To mitigate the semantic loss caused by packet loss, a 3D CNN at the receiver recovers missing information using un-lost semantic data and an packet-loss mask matrix. Semantic-level interleaving is employed to reduce concentrated semantic loss from packet drops. To improve compression, a common-individual decomposition approach is adopted, with downsampling applied to individual information to minimize redundancy. The model is lightweighted for practical deployment. Extensive simulations and comparisons demonstrate strong performance, achieving an MS-SSIM greater than 0.6 and a PSNR exceeding 20 dB at a 90% packet loss rate. 

**Abstract (ZH)**: 基于MoE Swin Transformer的抗丢包视频语义通信系统（MSTVSC） 

---
# Adaptive Content Restriction for Large Language Models via Suffix Optimization 

**Title (ZH)**: 大型语言模型通过后缀优化实现自适应内容限制 

**Authors**: Yige Li, Peihai Jiang, Jun Sun, Peng Shu, Tianming Liu, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01198)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant success across diverse applications. However, enforcing content restrictions remains a significant challenge due to their expansive output space. One aspect of content restriction is preventing LLMs from generating harmful content via model alignment approaches such as supervised fine-tuning (SFT). Yet, the need for content restriction may vary significantly across user groups, change rapidly over time, and not always align with general definitions of harmfulness. Applying SFT to each of these specific use cases is impractical due to the high computational, data, and storage demands. Motivated by this need, we propose a new task called \textit{Adaptive Content Restriction} (AdaCoRe), which focuses on lightweight strategies -- methods without model fine-tuning -- to prevent deployed LLMs from generating restricted terms for specific use cases. We propose the first method for AdaCoRe, named \textit{Suffix Optimization (SOP)}, which appends a short, optimized suffix to any prompt to a) prevent a target LLM from generating a set of restricted terms, while b) preserving the output quality. To evaluate AdaCoRe approaches, including our SOP, we create a new \textit{Content Restriction Benchmark} (CoReBench), which contains 400 prompts for 80 restricted terms across 8 carefully selected categories. We demonstrate the effectiveness of SOP on CoReBench, which outperforms the system-level baselines such as system suffix by 15\%, 17\%, 10\%, 9\%, and 6\% on average restriction rates for Gemma2-2B, Mistral-7B, Vicuna-7B, Llama3-8B, and Llama3.1-8B, respectively. We also demonstrate that SOP is effective on POE, an online platform hosting various commercial LLMs, highlighting its practicality in real-world scenarios. 

**Abstract (ZH)**: Large语言模型（LLMs）在多种应用中取得了显著的成效。然而，执行内容限制仍然是一个重大挑战，因为它们的输出空间非常广阔。内容限制的一个方面是通过模型对齐方法，如监督微调（SFT）等方式，防止LLMs生成有害内容。然而，不同用户组对内容限制的需求可能会有很大差异，且随着时间迅速变化，不一定与普遍定义的有害性一致。逐个为这些特定用例应用SFT在计算、数据和存储需求上都是不切实际的。为此，我们提出了一种新的任务称为“自适应内容限制”（AdaCoRe），该任务聚焦于轻量级策略——无需模型微调的方法，以防止部署的LLMs在特定用例中生成受限词汇。我们提出了AdaCoRe的第一个方法——后缀优化（SOP），它在任何提示后附加一个简短优化后的后缀，以a) 阻止目标LLM生成一组受限词汇，同时b) 保留输出质量。为了评估AdaCoRe方法，包括我们的SOP方法，我们创建了一个新的“内容限制基准”（CoReBench），其中包括80种受限词汇的400个提示，涵盖了8个精心选择的类别。我们在CoReBench上展示了SOP的有效性，SOP在Gemma2-2B、Mistral-7B、Vicuna-7B、Llama3-8B和Llama3.1-8B上平均受限率方面分别优于系统级基线，如系统后缀，分别为15%、17%、10%、9%和6%。我们还展示了SOP在POE（一个在线平台，提供多种商业LLM）上的有效性，突显了其在现实场景中的实际应用性。 

---
# BSL: A Unified and Generalizable Multitask Learning Platform for Virtual Drug Discovery from Design to Synthesis 

**Title (ZH)**: BSL：从设计到合成的虚拟药物发现多任务学习统一平台 

**Authors**: Kun Li, Zhennan Wu, Yida Xiong, Hongzhi Zhang, Longtao Hu, Zhonglie Liu, Junqi Zeng, Wenjie Wu, Mukun Chen, Jiameng Chen, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01195)  

**Abstract**: Drug discovery is of great social significance in safeguarding human health, prolonging life, and addressing the challenges of major diseases. In recent years, artificial intelligence has demonstrated remarkable advantages in key tasks across bioinformatics and pharmacology, owing to its efficient data processing and data representation capabilities. However, most existing computational platforms cover only a subset of core tasks, leading to fragmented workflows and low efficiency. In addition, they often lack algorithmic innovation and show poor generalization to out-of-distribution (OOD) data, which greatly hinders the progress of drug discovery. To address these limitations, we propose Baishenglai (BSL), a deep learning-enhanced, open-access platform designed for virtual drug discovery. BSL integrates seven core tasks within a unified and modular framework, incorporating advanced technologies such as generative models and graph neural networks. In addition to achieving state-of-the-art (SOTA) performance on multiple benchmark datasets, the platform emphasizes evaluation mechanisms that focus on generalization to OOD molecular structures. Comparative experiments with existing platforms and baseline methods demonstrate that BSL provides a comprehensive, scalable, and effective solution for virtual drug discovery, offering both algorithmic innovation and high-precision prediction for real-world pharmaceutical research. In addition, BSL demonstrated its practical utility by discovering novel modulators of the GluN1/GluN3A NMDA receptor, successfully identifying three compounds with clear bioactivity in in-vitro electrophysiological assays. These results highlight BSL as a promising and comprehensive platform for accelerating biomedical research and drug discovery. The platform is accessible at this https URL. 

**Abstract (ZH)**: 人工智能增强的开放访问平台Baishenglai：用于虚拟药物发现的七大核心任务统一框架 

---
# SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy 

**Title (ZH)**: SpectrumWorld: 光谱分析的人工智能基础 

**Authors**: Zhuo Yang, Jiaqing Xie, Shuaike Shen, Daolang Wang, Yeyun Chen, Ben Gao, Shuzhou Sun, Biqing Qi, Dongzhan Zhou, Lei Bai, Linjiang Chen, Shufei Zhang, Jun Jiang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01188)  

**Abstract**: Deep learning holds immense promise for spectroscopy, yet research and evaluation in this emerging field often lack standardized formulations. To address this issue, we introduce SpectrumLab, a pioneering unified platform designed to systematize and accelerate deep learning research in spectroscopy. SpectrumLab integrates three core components: a comprehensive Python library featuring essential data processing and evaluation tools, along with leaderboards; an innovative SpectrumAnnotator module that generates high-quality benchmarks from limited seed data; and SpectrumBench, a multi-layered benchmark suite covering 14 spectroscopic tasks and over 10 spectrum types, featuring spectra curated from over 1.2 million distinct chemical substances. Thorough empirical studies on SpectrumBench with 18 cutting-edge multimodal LLMs reveal critical limitations of current approaches. We hope SpectrumLab will serve as a crucial foundation for future advancements in deep learning-driven spectroscopy. 

**Abstract (ZH)**: Deep Learning for Spectroscopy: A Unified Platform, SpectrumLab, to Standardize and Accelerate Research 

---
# Advancing the Foundation Model for Music Understanding 

**Title (ZH)**: 音乐理解基础模型的进展 

**Authors**: Yi Jiang, Wei Wang, Xianwen Guo, Huiyun Liu, Hanrui Wang, Youri Xu, Haoqi Gu, Zhongqian Xie, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01178)  

**Abstract**: The field of Music Information Retrieval (MIR) is fragmented, with specialized models excelling at isolated tasks. In this work, we challenge this paradigm by introducing a unified foundation model named MuFun for holistic music understanding. Our model features a novel architecture that jointly processes instrumental and lyrical content, and is trained on a large-scale dataset covering diverse tasks such as genre classification, music tagging, and question answering. To facilitate robust evaluation, we also propose a new benchmark for multi-faceted music understanding called MuCUE (Music Comprehensive Understanding Evaluation). Experiments show our model significantly outperforms existing audio large language models across the MuCUE tasks, demonstrating its state-of-the-art effectiveness and generalization ability. 

**Abstract (ZH)**: 音乐信息检索领域的研究支离破碎，专门的模型在单一任务上表现出色。本文通过引入全面音乐理解的统一基础模型MuFun来挑战这一范式。我们的模型具有新颖的架构，可以联合处理乐器和歌词内容，并在涵盖了流派分类、音乐标记和问答等多种任务的大规模数据集上进行训练。为了便于稳健的评估，我们还提出了一种新的多方面音乐理解基准MuCUE（音乐综合理解评估）。实验结果显示，我们的模型在MuCUE任务上显著优于现有音频大型语言模型，证明了其先进的有效性和泛化能力。 

---
# RSPO: Risk-Seeking Policy Optimization for Pass@k and Max@k Metrics in Large Language Models 

**Title (ZH)**: RSPO: 风险寻求策略优化以提升大型语言模型的Pass@k和Max@k指标 

**Authors**: Kaichen Zhang, Shenghao Gao, Yuzhong Hong, Haipeng Sun, Junwei Bao, Hongfei Jiang, Yang Song, Hong Dingqian, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01174)  

**Abstract**: Current large language model post-training optimizes a risk-neutral objective that maximizes expected reward, yet evaluation relies heavily on risk-seeking metrics like Pass@k (at least one success in k trials) and Max@k (maximum reward across k responses). This mismatch in risk preferences can inevitably lead to suboptimal performance. To bridge this gap, we propose Risk-Seeking Policy Optimization (RSPO), a novel method that directly targets Pass@k and Max@k during training. A key challenge in optimizing these metrics is the "hitchhiking" problem: low-reward responses are inadvertently reinforced if they co-occur with a high-reward response within a sample of k generations, resulting in inefficient optimization. RSPO addresses this problem by leveraging the closed-form probability that a given response is the maximum among k samplings. Despite the complexity of nested gradients over multiple responses, RSPO produces efficient, unbiased gradient estimators for both metrics. We validate our approach with both rigorous theoretical analysis and comprehensive experimental results. 

**Abstract (ZH)**: 风险寻求策略优化：一种直接针对Pass@k和Max@k的训练方法 

---
# GeHirNet: A Gender-Aware Hierarchical Model for Voice Pathology Classification 

**Title (ZH)**: GeHirNet：一种考虑性别差异的层次模型用于语音病理分类 

**Authors**: Fan Wu, Kaicheng Zhao, Elgar Fleisch, Filipe Barata  

**Link**: [PDF](https://arxiv.org/pdf/2508.01172)  

**Abstract**: AI-based voice analysis shows promise for disease diagnostics, but existing classifiers often fail to accurately identify specific pathologies because of gender-related acoustic variations and the scarcity of data for rare diseases. We propose a novel two-stage framework that first identifies gender-specific pathological patterns using ResNet-50 on Mel spectrograms, then performs gender-conditioned disease classification. We address class imbalance through multi-scale resampling and time warping augmentation. Evaluated on a merged dataset from four public repositories, our two-stage architecture with time warping achieves state-of-the-art performance (97.63\% accuracy, 95.25\% MCC), with a 5\% MCC improvement over single-stage baseline. This work advances voice pathology classification while reducing gender bias through hierarchical modeling of vocal characteristics. 

**Abstract (ZH)**: 基于AI的声音分析在疾病诊断中展现出前景，但现有分类器常常因性别相关的声学变异和罕见疾病数据稀少而无法准确识别特定病理。我们提出一种新颖的两阶段框架，首先使用ResNet-50在梅尔频谱图上识别性别特异性病理模式，然后进行基于性别的疾病分类。我们通过多尺度重采样和时间扭曲增强解决类别不平衡问题。在四个公开数据仓库合并的数据集上评估，我们的两阶段架构结合时间扭曲实现了目前最佳性能（97.63%准确率，95.25%麦考利效能系数），相比单阶段基线提高了5%的麦考利效能系数。该工作通过层级建模声音特征推进了声音病理分类，同时减少了性别偏见。 

---
# Asking the Right Questions: Benchmarking Large Language Models in the Development of Clinical Consultation Templates 

**Title (ZH)**: 提出恰当的问题：评估大型语言模型在临床咨询模板开发中的表现 

**Authors**: Liam G. McCoy, Fateme Nateghi Haredasht, Kanav Chopra, David Wu, David JH Wu, Abass Conteh, Sarita Khemani, Saloni Kumar Maharaj, Vishnu Ravi, Arth Pahwa, Yingjie Weng, Leah Rosengaus, Lena Giang, Kelvin Zhenghao Li, Olivia Jee, Daniel Shirvani, Ethan Goh, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01159)  

**Abstract**: This study evaluates the capacity of large language models (LLMs) to generate structured clinical consultation templates for electronic consultation. Using 145 expert-crafted templates developed and routinely used by Stanford's eConsult team, we assess frontier models -- including o3, GPT-4o, Kimi K2, Claude 4 Sonnet, Llama 3 70B, and Gemini 2.5 Pro -- for their ability to produce clinically coherent, concise, and prioritized clinical question schemas. Through a multi-agent pipeline combining prompt optimization, semantic autograding, and prioritization analysis, we show that while models like o3 achieve high comprehensiveness (up to 92.2\%), they consistently generate excessively long templates and fail to correctly prioritize the most clinically important questions under length constraints. Performance varies across specialties, with significant degradation in narrative-driven fields such as psychiatry and pain medicine. Our findings demonstrate that LLMs can enhance structured clinical information exchange between physicians, while highlighting the need for more robust evaluation methods that capture a model's ability to prioritize clinically salient information within the time constraints of real-world physician communication. 

**Abstract (ZH)**: 本研究评估了大型语言模型（LLMs）生成结构化临床咨询模板的能力，以供电子咨询使用。通过评估由斯坦福eConsult团队开发并常规使用的145个专家crafted模板，我们考察了前沿模型（包括o3、GPT-4o、Kimi K2、Claude 4 Sonnet、Llama 3 70B和Gemini 2.5 Pro）生成临床连贯、简洁且优先级明确的临床问题框架的能力。通过结合提示优化、语义自动评分和优先级分析的多代理管线，我们表明，尽管像o3这样的模型实现了高度的完整性（高达92.2%），但在长度限制下，它们始终生成了过于冗长的模板，并未能正确优先考虑最重要的临床问题。各专科的表现不一，在以叙事驱动的领域如精神病学和疼痛医学中，性能显著下降。我们的研究结果表明，LLMs能够增强医生之间的结构化临床信息交流，同时也强调了需要更为 robust的评估方法，以捕捉模型在现实世界医生沟通时间限制内优先处理临床相关信息的能力。 

---
# Personalized Safety Alignment for Text-to-Image Diffusion Models 

**Title (ZH)**: 个性化安全对齐 для текст-к изображению диффузионных моделей 

**Authors**: Yu Lei, Jinbin Bai, Qingyu Shi, Aosong Feng, Kaidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01151)  

**Abstract**: Text-to-image diffusion models have revolutionized visual content generation, but current safety mechanisms apply uniform standards that often fail to account for individual user preferences. These models overlook the diverse safety boundaries shaped by factors like age, mental health, and personal beliefs. To address this, we propose Personalized Safety Alignment (PSA), a framework that allows user-specific control over safety behaviors in generative models. PSA integrates personalized user profiles into the diffusion process, adjusting the model's behavior to match individual safety preferences while preserving image quality. We introduce a new dataset, Sage, which captures user-specific safety preferences and incorporates these profiles through a cross-attention mechanism. Experiments show that PSA outperforms existing methods in harmful content suppression and aligns generated content better with user constraints, achieving higher Win Rate and Pass Rate scores. Our code, data, and models are publicly available at this https URL. 

**Abstract (ZH)**: 个性化安全性对齐（PSA）：文本到图像扩散模型的用户特定安全性控制 

---
# Dataset Condensation with Color Compensation 

**Title (ZH)**: 颜色补偿下的数据集凝练 

**Authors**: Huyu Wu, Duo Su, Junjie Hou, Guang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01139)  

**Abstract**: Dataset condensation always faces a constitutive trade-off: balancing performance and fidelity under extreme compression. Existing methods struggle with two bottlenecks: image-level selection methods (Coreset Selection, Dataset Quantization) suffer from inefficiency condensation, while pixel-level optimization (Dataset Distillation) introduces semantic distortion due to over-parameterization. With empirical observations, we find that a critical problem in dataset condensation is the oversight of color's dual role as an information carrier and a basic semantic representation unit. We argue that improving the colorfulness of condensed images is beneficial for representation learning. Motivated by this, we propose DC3: a Dataset Condensation framework with Color Compensation. After a calibrated selection strategy, DC3 utilizes the latent diffusion model to enhance the color diversity of an image rather than creating a brand-new one. Extensive experiments demonstrate the superior performance and generalization of DC3 that outperforms SOTA methods across multiple benchmarks. To the best of our knowledge, besides focusing on downstream tasks, DC3 is the first research to fine-tune pre-trained diffusion models with condensed datasets. The FID results prove that training networks with our high-quality datasets is feasible without model collapse or other degradation issues. Code and generated data will be released soon. 

**Abstract (ZH)**: 数据集凝练总是面临着一个固有的权衡：在极端压缩条件下平衡性能和保真度。现有方法面临两个瓶颈：图像级选择方法（聚簇选择、数据集量化）由于效率凝练问题而受阻，而像素级优化（数据集蒸馏）由于过度参数化引入了语义失真。通过实证观察，我们发现数据集凝练中的一个关键问题是忽视了颜色作为信息载体和基本语义表示单元的双重角色。我们认为，提高凝练后图像的色彩丰富度有助于表示学习。受此启发，我们提出了DC3：一种具有色彩补偿的数据集凝练框架。在经过校准的选择策略后，DC3利用潜在扩散模型增强图像的色彩多样性，而非创造一个新的图像。广泛的经验表明，DC3在多个基准上表现出优越的性能和泛化能力，且优于当前最先进的方法。据我们所知，除了关注下游任务外，DC3还是第一个研究通过使用凝练后的数据集微调预训练扩散模型的工作。FID结果证明了使用高质量数据集训练网络的可行性，且不会出现模型崩溃或其他退化问题。代码和生成的数据将尽快发布。 

---
# DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs 

**Title (ZH)**: DBAIOps：一种基于知识图谱的LLM增强数据库运维系统 

**Authors**: Wei Zhou, Peng Sun, Xuanhe Zhou, Qianglei Zang, Ji Xu, Tieying Zhang, Guoliang Li, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01136)  

**Abstract**: The operation and maintenance (O&M) of database systems is critical to ensuring system availability and performance, typically requiring expert experience (e.g., identifying metric-to-anomaly relations) for effective diagnosis and recovery. However, existing automatic database O&M methods, including commercial products, cannot effectively utilize expert experience. On the one hand, rule-based methods only support basic O&M tasks (e.g., metric-based anomaly detection), which are mostly numerical equations and cannot effectively incorporate literal O&M experience (e.g., troubleshooting guidance in manuals). On the other hand, LLM-based methods, which retrieve fragmented information (e.g., standard documents + RAG), often generate inaccurate or generic results. To address these limitations, we present DBAIOps, a novel hybrid database O&M system that combines reasoning LLMs with knowledge graphs to achieve DBA-style diagnosis. First, DBAIOps introduces a heterogeneous graph model for representing the diagnosis experience, and proposes a semi-automatic graph construction algorithm to build that graph from thousands of documents. Second, DBAIOps develops a collection of (800+) reusable anomaly models that identify both directly alerted metrics and implicitly correlated experience and metrics. Third, for each anomaly, DBAIOps proposes a two-stage graph evolution mechanism to explore relevant diagnosis paths and identify missing relations automatically. It then leverages a reasoning LLM (e.g., DeepSeek-R1) to infer root causes and generate clear diagnosis reports for both DBAs and common users. Our evaluation over four mainstream database systems (Oracle, MySQL, PostgreSQL, and DM8) demonstrates that DBAIOps outperforms state-of-the-art baselines, 34.85% and 47.22% higher in root cause and human evaluation accuracy, respectively. 

**Abstract (ZH)**: 数据库系统的运行和维护（O&M）对于确保系统的可用性和性能至关重要，通常需要专家经验（例如，识别指标到异常的关系）来进行有效的诊断和恢复。然而，现有的自动数据库O&M方法，包括商用产品，无法有效利用专家经验。一方面，基于规则的方法仅支持基本的O&M任务（例如，基于指标的异常检测），这些任务主要是数值方程，不能有效整合文字形式的O&M经验（例如，手册中的故障排除指导）。另一方面，基于大语言模型（LLM）的方法，由于检索碎片化的信息（例如，标准文档+检索增强生成），通常会产生不准确或通用的结果。为了解决这些限制，我们提出了DBAIOps，这是一种新颖的结合推理大语言模型与知识图谱的混合数据库O&M系统，以实现DBA风格的诊断。首先，DBAIOps引入了一种异构图模型来表示诊断经验，并提出了一种半自动的图构建算法，从数千份文档中构建该图。其次，DBAIOps开发了一整套（800多个）可重用的异常模型，能够识别直接警告的指标以及隐含关联的经验和指标。第三，对于每个异常，DBAIOps提出了一种两阶段的图演化机制，以自动探索相关诊断路径并识别缺少的关系。然后，它利用推理大语言模型（例如DeepSeek-R1）推断根本原因并生成清晰的诊断报告，适用于DBA和普通用户。我们的评估表明，DBAIOps在四个主流数据库系统（Oracle、MySQL、PostgreSQL和DM8）上优于最先进的基线，分别是34.85%和47.22%的根因和人工评估准确性提升。 

---
# COLLAGE: Adaptive Fusion-based Retrieval for Augmented Policy Learning 

**Title (ZH)**: 拼贴：自适应融合检索以增强策略学习 

**Authors**: Sateesh Kumar, Shivin Dass, Georgios Pavlakos, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2508.01131)  

**Abstract**: In this work, we study the problem of data retrieval for few-shot imitation learning: selecting data from a large dataset to train a performant policy for a specific task, given only a few target demonstrations. Prior methods retrieve data using a single-feature distance heuristic, assuming that the best demonstrations are those that most closely resemble the target examples in visual, semantic, or motion space. However, this approach captures only a subset of the relevant information and can introduce detrimental demonstrations, e.g., retrieving data from unrelated tasks due to similar scene layouts, or selecting similar motions from tasks with divergent goals. We present COLLAGE, a method for COLLective data AGgrEgation in few-shot imitation learning that uses an adaptive late fusion mechanism to guide the selection of relevant demonstrations based on a task-specific combination of multiple cues. COLLAGE follows a simple, flexible, and efficient recipe: it assigns weights to subsets of the dataset that are pre-selected using a single feature (e.g., appearance, shape, or language similarity), based on how well a policy trained on each subset predicts actions in the target demonstrations. These weights are then used to perform importance sampling during policy training, sampling data more densely or sparsely according to estimated relevance. COLLAGE is general and feature-agnostic, allowing it to combine any number of subsets selected by any retrieval heuristic, and to identify which subsets provide the greatest benefit for the target task. In extensive experiments, COLLAGE outperforms state-of-the-art retrieval and multi-task learning approaches by 5.1% in simulation across 10 tasks, and by 16.6% in the real world across 6 tasks, where we perform retrieval from the large-scale DROID dataset. More information at this https URL . 

**Abstract (ZH)**: 面向少样本模仿学习的数据检索问题：选择特定任务性能良好的数据方法，仅给定少量目标演示。先前方法使用单特征距离启发式检索数据，假设最佳演示是那些在视觉、语义或运动空间中最接近目标示例的。然而，这种方法仅捕获了一部分相关信息，并可能会引入不利的演示，例如，由于场景布局相似性，从不相关的任务检索数据；或者在目标任务具有不同目标的情景下选择相似的动作。我们提出了COLLAGE，一种面向少样本模仿学习的集合数据聚合方法，利用自适应后期融合机制，基于特定任务的多个线索组合引导相关演示的选择。COLLAGE 遵循一个简单、灵活且高效的配方：它根据训练于每个子集上的策略对目标演示中动作的预测能力，为这些子集分配权重。然后，在策略训练期间使用这些权重进行重要性采样，根据估计的相关性密度更大或更小地采样数据。COLLAGE 是通用且特征无关的，允许它结合任意数量由任何检索启发式选择的子集，并确定哪些子集对目标任务提供了最大的好处。在广泛的实验中，COLLAGE 在模拟环境中在 10 个任务上的性能优于最先进的检索和多任务学习方法 5.1%，在现实世界中在 6 个任务上的性能优于 16.6%，其中我们在大规模 DROID 数据集中执行检索。更多详细信息请访问此链接。 

---
# Human-Robot Red Teaming for Safety-Aware Reasoning 

**Title (ZH)**: 人类-机器人红蓝队安全意识推理 

**Authors**: Emily Sheetz, Emma Zemler, Misha Savchenko, Connor Rainen, Erik Holum, Jodi Graf, Andrew Albright, Shaun Azimi, Benjamin Kuipers  

**Link**: [PDF](https://arxiv.org/pdf/2508.01129)  

**Abstract**: While much research explores improving robot capabilities, there is a deficit in researching how robots are expected to perform tasks safely, especially in high-risk problem domains. Robots must earn the trust of human operators in order to be effective collaborators in safety-critical tasks, specifically those where robots operate in human environments. We propose the human-robot red teaming paradigm for safety-aware reasoning. We expect humans and robots to work together to challenge assumptions about an environment and explore the space of hazards that may arise. This exploration will enable robots to perform safety-aware reasoning, specifically hazard identification, risk assessment, risk mitigation, and safety reporting. We demonstrate that: (a) human-robot red teaming allows human-robot teams to plan to perform tasks safely in a variety of domains, and (b) robots with different embodiments can learn to operate safely in two different environments -- a lunar habitat and a household -- with varying definitions of safety. Taken together, our work on human-robot red teaming for safety-aware reasoning demonstrates the feasibility of this approach for safely operating and promoting trust on human-robot teams in safety-critical problem domains. 

**Abstract (ZH)**: 基于人类与机器人红队的为安全而设计的推理方法 

---
# Towards Bridging Review Sparsity in Recommendation with Textual Edge Graph Representation 

**Title (ZH)**: 基于文本边图表示的推荐中应对评价稀疏性的桥梁构建 

**Authors**: Leyao Wang, Xutao Mao, Xuhui Zhan, Yuying Zhao, Bo Ni, Ryan A. Rossi, Nesreen K. Ahmed, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.01128)  

**Abstract**: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of existing models. A natural solution is to impute or generate missing reviews to enrich the data. However, conventional imputation techniques -- such as matrix completion and LLM-based augmentation -- either lose contextualized semantics by embedding texts into vectors, or overlook structural dependencies among user-item interactions. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework that imputes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as a Textual-Edge Graph (TEG), treating reviews as edge attributes. To capture relational context, we construct line-graph views and employ a large language model as a graph-aware aggregator. For each interaction lacking a textual review, our model aggregates the neighborhood's natural-language representations to generate a coherent and personalized review. Experiments on the Amazon and Goodreads datasets show that TWISTER consistently outperforms traditional numeric, graph-based, and LLM baselines, delivering higher-quality imputed reviews and, more importantly, enhanced recommendation performance. In summary, TWISTER generates reviews that are more helpful, authentic, and specific, while smoothing structural signals for improved recommendations. 

**Abstract (ZH)**: 文本评论丰富了推荐系统中的细粒度偏好信号和增强了可解释性。然而，在实际场景中，用户很少留下评论，导致严重的数据稀疏性，损害了现有模型的效果。一个自然的解决方案是通过填充或生成缺失的评论来丰富数据。然而，传统的填充技术——如矩阵完成和基于LLM的增强——要么通过将文本嵌入向量中失去上下文化的语义，要么忽略了用户-物品交互的结构依赖性。为了应对这些不足，我们提出了一种统一框架TWISTER（针对文本边图表示的稀疏填充），该框架通过联合建模语义和结构信号来填充缺失评论。具体而言，我们将用户-物品交互表示为文本边图（TEG），将评论视为边的属性。为了捕捉关系上下文，我们构建了线图视图并采用大型语言模型作为图感知聚合器。对于每个缺失文本评论的交互，我们的模型通过聚合邻域的自然语言表示来生成连贯且个性化的评论。实验结果表明，TWISTER在亚马逊和Goodreads数据集上的表现始终优于传统的数值、图基和LLM基线模型，不仅生成了高质量的填充评论，而且更重要的是提高了推荐性能。总之，TWISTER生成的评论更具有帮助性、真实性且具体性，同时平滑了结构信号以改进推荐。 

---
# TensoMeta-VQC: A Tensor-Train-Guided Meta-Learning Framework for Robust and Scalable Variational Quantum Computing 

**Title (ZH)**: TensoMeta-VQC：一种基于张量 train 的元学习框架，用于稳健且可扩展的变量子计算 

**Authors**: Jun Qi, Chao-Han Yang, Pin-Yu Chen, Min-Hsiu Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01116)  

**Abstract**: Variational Quantum Computing (VQC) faces fundamental barriers in scalability, primarily due to barren plateaus and quantum noise sensitivity. To address these challenges, we introduce TensoMeta-VQC, a novel tensor-train (TT)-guided meta-learning framework designed to improve the robustness and scalability of VQC significantly. Our framework fully delegates the generation of quantum circuit parameters to a classical TT network, effectively decoupling optimization from quantum hardware. This innovative parameterization mitigates gradient vanishing, enhances noise resilience through structured low-rank representations, and facilitates efficient gradient propagation. Based on Neural Tangent Kernel and statistical learning theory, our rigorous theoretical analyses establish strong guarantees on approximation capability, optimization stability, and generalization performance. Extensive empirical results across quantum dot classification, Max-Cut optimization, and molecular quantum simulation tasks demonstrate that TensoMeta-VQC consistently achieves superior performance and robust noise tolerance, establishing it as a principled pathway toward practical and scalable VQC on near-term quantum devices. 

**Abstract (ZH)**: 张量元学习引导的量子电路参数化变分量子computing (TensoMeta-VQC): 一种提高变分量子computing稳健性与可扩展性的新框架 

---
# Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring 

**Title (ZH)**: 基于上下文感知的机器学习框架对学生压力监测及其心理健康保护 

**Authors**: Md Sultanul Islam Ovi, Jamal Hossain, Md Raihan Alam Rahi, Fatema Akter  

**Link**: [PDF](https://arxiv.org/pdf/2508.01105)  

**Abstract**: Student mental health is an increasing concern in academic institutions, where stress can severely impact well-being and academic performance. Traditional assessment methods rely on subjective surveys and periodic evaluations, offering limited value for timely intervention. This paper introduces a context-aware machine learning framework for classifying student stress using two complementary survey-based datasets covering psychological, academic, environmental, and social factors. The framework follows a six-stage pipeline involving preprocessing, feature selection (SelectKBest, RFECV), dimensionality reduction (PCA), and training with six base classifiers: SVM, Random Forest, Gradient Boosting, XGBoost, AdaBoost, and Bagging. To enhance performance, we implement ensemble strategies, including hard voting, soft voting, weighted voting, and stacking. Our best models achieve 93.09% accuracy with weighted hard voting on the Student Stress Factors dataset and 99.53% with stacking on the Stress and Well-being dataset, surpassing previous benchmarks. These results highlight the potential of context-integrated, data-driven systems for early stress detection and underscore their applicability in real-world academic settings to support student well-being. 

**Abstract (ZH)**: 基于上下文的机器学习框架在互补调查数据集上的学生压力分类：早期检测和实际应用 

---
# Cross-Domain Web Information Extraction at Pinterest 

**Title (ZH)**: 跨域Pinterest网页信息提取 

**Authors**: Michael Farag, Patrick Halina, Andrey Zaytsev, Alekhya Munagala, Imtihan Ahmed, Junhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01096)  

**Abstract**: The internet offers a massive repository of unstructured information, but it's a significant challenge to convert this into a structured format. At Pinterest, the ability to accurately extract structured product data from e-commerce websites is essential to enhance user experiences and improve content distribution. In this paper, we present Pinterest's system for attribute extraction, which achieves remarkable accuracy and scalability at a manageable cost. Our approach leverages a novel webpage representation that combines structural, visual, and text modalities into a compact form, optimizing it for small model learning. This representation captures each visible HTML node with its text, style and layout information. We show how this allows simple models such as eXtreme Gradient Boosting (XGBoost) to extract attributes more accurately than much more complex Large Language Models (LLMs) such as Generative Pre-trained Transformer (GPT). Our results demonstrate a system that is highly scalable, processing over 1,000 URLs per second, while being 1000 times more cost-effective than the cheapest GPT alternatives. 

**Abstract (ZH)**: 互联网提供了大量未结构化的信息库，但将其转换为结构化格式是一项重大挑战。在Pinterest，从电子商务网站准确提取结构化产品数据的能力对于提升用户体验和改善内容分发至关重要。在本文中，我们介绍了Pinterest的属性提取系统，该系统在可管理的代价下实现了显著的准确性和可扩展性。我们的方法利用了一种新颖的网页表示形式，该表示形式将结构化、视觉和文本模态整合为紧凑的形式，优化了小型模型的学习。这种表示形式捕捉了每个可见的HTML节点及其文本、样式和布局信息。我们展示了这种表示方法如何使简单的模型，如极端梯度增强（XGBoost），能够比复杂的大型语言模型（LLMs），如生成预训练变换器（GPT）提取属性更加准确。我们的结果表明，该系统具有高度的可扩展性，每秒可以处理超过1,000个URL，同时比最便宜的GPT替代方案的成本低1000倍。 

---
# Provably Secure Retrieval-Augmented Generation 

**Title (ZH)**: 可验证安全的检索增强生成 

**Authors**: Pengcheng Zhou, Yinglun Feng, Zhongliang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01084)  

**Abstract**: Although Retrieval-Augmented Generation (RAG) systems have been widely applied, the privacy and security risks they face, such as data leakage and data poisoning, have not been systematically addressed yet. Existing defense strategies primarily rely on heuristic filtering or enhancing retriever robustness, which suffer from limited interpretability, lack of formal security guarantees, and vulnerability to adaptive attacks. To address these challenges, this paper proposes the first provably secure framework for RAG systems(SAG). Our framework employs a pre-storage full-encryption scheme to ensure dual protection of both retrieved content and vector embeddings, guaranteeing that only authorized entities can access the data. Through formal security proofs, we rigorously verify the scheme's confidentiality and integrity under a computational security model. Extensive experiments across multiple benchmark datasets demonstrate that our framework effectively resists a range of state-of-the-art attacks. This work establishes a theoretical foundation and practical paradigm for verifiably secure RAG systems, advancing AI-powered services toward formally guaranteed security. 

**Abstract (ZH)**: 虽然检索增强生成（RAG）系统已被广泛应用，但它们面临的数据泄露和数据投毒等隐私和安全风险尚未系统性解决。现有防御策略主要依赖启发式过滤或增强检索器的鲁棒性，存在可解释性有限、缺乏正式安全保证以及容易受到适应性攻击的缺点。为应对这些挑战，本文提出了首个可验证安全的RAG系统框架（SAG）。该框架采用预存储全加密方案，确保检索内容和向量嵌入的双重保护，保证只有授权实体可以访问数据。通过形式化安全证明，我们在计算安全模型下严格验证了该方案的机密性和完整性。在多个基准数据集上进行的广泛实验表明，该框架有效抵御了一系列最先进的攻击。本研究为可验证安全的RAG系统奠定了理论基础和实践范式，推动了基于人工智能的服务向正式保证的安全迈进。 

---
# Learning Pivoting Manipulation with Force and Vision Feedback Using Optimization-based Demonstrations 

**Title (ZH)**: 基于优化演示的力与视觉反馈驱动的翻转操作学习 

**Authors**: Yuki Shirai, Kei Ota, Devesh K. Jha, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2508.01082)  

**Abstract**: Non-prehensile manipulation is challenging due to complex contact interactions between objects, the environment, and robots. Model-based approaches can efficiently generate complex trajectories of robots and objects under contact constraints. However, they tend to be sensitive to model inaccuracies and require access to privileged information (e.g., object mass, size, pose), making them less suitable for novel objects. In contrast, learning-based approaches are typically more robust to modeling errors but require large amounts of data. In this paper, we bridge these two approaches to propose a framework for learning closed-loop pivoting manipulation. By leveraging computationally efficient Contact-Implicit Trajectory Optimization (CITO), we design demonstration-guided deep Reinforcement Learning (RL), leading to sample-efficient learning. We also present a sim-to-real transfer approach using a privileged training strategy, enabling the robot to perform pivoting manipulation using only proprioception, vision, and force sensing without access to privileged information. Our method is evaluated on several pivoting tasks, demonstrating that it can successfully perform sim-to-real transfer. 

**Abstract (ZH)**: 基于接触的非抓取操作由于对象、环境和机器人之间的复杂接触交互而具有挑战性。模型驱动的方法可以在接触约束下高效生成机器人和物体的复杂轨迹。然而，它们往往对模型不准确性敏感，并需要访问特权信息（如物体的质量、尺寸、姿态），使其不适用于新型物体。相比之下，基于学习的方法通常对建模错误更具有鲁棒性，但需要大量的数据。本文我们结合这两种方法，提出了一个学习闭环 pivot 操作的框架。通过利用计算效率高的接触隐式轨迹优化（CITO），我们设计了演示引导的深度强化学习（RL），实现了样本高效学习。我们还提出了一种从仿真到现实的转移方法，并采用特权训练策略，使机器人仅通过 proprioception、视觉和力感知就能执行 pivot 操作，而无需访问特权信息。我们的方法在几个 pivot 任务上进行了评估，展示了其成功实现从仿真到现实的转移能力。 

---
# The Lattice Geometry of Neural Network Quantization -- A Short Equivalence Proof of GPTQ and Babai's algorithm 

**Title (ZH)**: 神经网络量化晶格几何——GPTQ与Babai算法的简短等价证明 

**Authors**: Johann Birnick  

**Link**: [PDF](https://arxiv.org/pdf/2508.01077)  

**Abstract**: We explain how data-driven quantization of a linear unit in a neural network corresponds to solving the closest vector problem for a certain lattice generated by input data. We prove that the GPTQ algorithm is equivalent to Babai's well-known nearest-plane algorithm. We furthermore provide geometric intuition for both algorithms. Lastly, we note the consequences of these results, in particular hinting at the possibility for using lattice basis reduction for better quantization. 

**Abstract (ZH)**: 我们解释了神经网络中基于数据的线性单元量化如何对应于某些由输入数据生成的格中最近向量问题的求解。我们证明GPTQ算法等价于Babai著名的最近平面算法。此外，我们为两种算法提供了几何直观。最后，我们指出这些结果的后果， particularly 暗示使用格基减少方法可能实现更好的量化。 

---
# Expressive Power of Graph Transformers via Logic 

**Title (ZH)**: 图变换器的逻辑表达能力 

**Authors**: Veeti Ahvonen, Maurice Funk, Damian Heiman, Antti Kuusisto, Carsten Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01067)  

**Abstract**: Transformers are the basis of modern large language models, but relatively little is known about their precise expressive power on graphs. We study the expressive power of graph transformers (GTs) by Dwivedi and Bresson (2020) and GPS-networks by Rampásek et al. (2022), both under soft-attention and average hard-attention. Our study covers two scenarios: the theoretical setting with real numbers and the more practical case with floats. With reals, we show that in restriction to vertex properties definable in first-order logic (FO), GPS-networks have the same expressive power as graded modal logic (GML) with the global modality. With floats, GPS-networks turn out to be equally expressive as GML with the counting global modality. The latter result is absolute, not restricting to properties definable in a background logic. We also obtain similar characterizations for GTs in terms of propositional logic with the global modality (for reals) and the counting global modality (for floats). 

**Abstract (ZH)**: 图变换器和GPS网络在图上的精确表征能力研究：基于软注意和平均硬注意 

---
# Connectivity Management in Satellite-Aided Vehicular Networks with Multi-Head Attention-Based State Estimation 

**Title (ZH)**: 基于多头注意力机制状态估计的卫星辅助 vehicular 网络连通性管理 

**Authors**: Ibrahim Althamary, Chen-Fu Chou, Chih-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01060)  

**Abstract**: Managing connectivity in integrated satellite-terrestrial vehicular networks is critical for 6G, yet is challenged by dynamic conditions and partial observability. This letter introduces the Multi-Agent Actor-Critic with Satellite-Aided Multi-head self-attention (MAAC-SAM), a novel multi-agent reinforcement learning framework that enables vehicles to autonomously manage connectivity across Vehicle-to-Satellite (V2S), Vehicle-to-Infrastructure (V2I), and Vehicle-to-Vehicle (V2V) links. Our key innovation is the integration of a multi-head attention mechanism, which allows for robust state estimation even with fluctuating and limited information sharing among vehicles. The framework further leverages self-imitation learning (SIL) and fingerprinting to improve learning efficiency and real-time decisions. Simulation results, based on realistic SUMO traffic models and 3GPP-compliant configurations, demonstrate that MAAC-SAM outperforms state-of-the-art terrestrial and satellite-assisted baselines by up to 14% in transmission utility and maintains high estimation accuracy across varying vehicle densities and sharing levels. 

**Abstract (ZH)**: 基于卫星辅助多头自注意力的多agentActor-Critic框架（MAAC-SAM）在综合卫星-地面-vehicular网络中的连通性管理 

---
# Llama-3.1-FoundationAI-SecurityLLM-8B-Instruct Technical Report 

**Title (ZH)**: Llama-3.1-基金会AI安全大型语言模型-8B-指令技术报告 

**Authors**: Sajana Weerawardhena, Paul Kassianik, Blaine Nelson, Baturay Saglam, Anu Vellore, Aman Priyanshu, Supriti Vijay, Massimo Aufiero, Arthur Goldblatt, Fraser Burch, Ed Li, Jianliang He, Dhruv Kedia, Kojin Oshiba, Zhouran Yang, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01059)  

**Abstract**: Large language models (LLMs) have shown remarkable success across many domains, yet their integration into cybersecurity applications remains limited due to a lack of general-purpose cybersecurity data, representational complexity, and safety and regulatory concerns. To address this gap, we previously introduced Foundation-Sec-8B, a cybersecurity-focused LLM suitable for fine-tuning on downstream tasks. That model, however, was not designed for chat-style interactions or instruction-following. In this report, we release Foundation-Sec-8B-Instruct: a model specifically trained for general-purpose cybersecurity dialogue. Built on Foundation-Sec-8B, it combines domain-specific knowledge with instruction-following, conversational capabilities, and alignment with human preferences to produce high-quality, relevant responses. Comprehensive evaluations show that Foundation-Sec-8B-Instruct outperforms Llama 3.1-8B-Instruct on a range of cybersecurity tasks while matching its instruction-following performance. It is also competitive with GPT-4o-mini on cyber threat intelligence and instruction-following tasks. We envision Foundation-Sec-8B-Instruct becoming an indispensable assistant in the daily workflows of cybersecurity professionals. We release the model publicly at this https URL. 

**Abstract (ZH)**: Foundation-Sec-8B-Instruct: 一种适用于通用网络安全对话的指令跟随模型 

---
# Managing Escalation in Off-the-Shelf Large Language Models 

**Title (ZH)**: 管理现成大型语言模型中的升级问题 

**Authors**: Sebastian Elbaum, Jonathan Panther  

**Link**: [PDF](https://arxiv.org/pdf/2508.01056)  

**Abstract**: U.S. national security customers have begun to utilize large language models, including enterprise versions of ``off-the-shelf'' models (e.g., ChatGPT) familiar to the public. This uptake will likely accelerate. However, recent studies suggest that off-the-shelf large language models frequently suggest escalatory actions when prompted with geopolitical or strategic scenarios. We demonstrate two simple, non-technical interventions to control these tendencies. Introducing these interventions into the experimental wargame design of a recent study, we substantially reduce escalation throughout the game. Calls to restrict the use of large language models in national security applications are thus premature. The U.S. government is already, and will continue, employing large language models for scenario planning and suggesting courses of action. Rather than warning against such applications, this study acknowledges the imminent adoption of large language models, and provides actionable measures to align them with national security goals, including escalation management. 

**Abstract (ZH)**: 美国国家安全客户已经开始利用大型语言模型，包括公众熟悉的“即用型”模型的企业版本（例如ChatGPT）。这一应用将很可能加速。然而，最近的研究表明，当被提示涉及地缘政治或战略情景时，“即用型”大型语言模型经常建议升级行动。我们展示了两种简单的非技术干预措施来控制这些倾向。在最近一项研究的实验战争游戏设计中引入这些干预措施，显著减少了游戏中的升级行为。有关限制在国家安全应用中使用大型语言模型的声音因此为时尚早。美国政府已经在使用大型语言模型进行场景规划并建议行动方案，并将继续这样做。本研究认可大型语言模型即将被采用这一现实，并提供了有助于将这些模型与国家安全目标，包括升级管理相一致的可操作措施。 

---
# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models 

**Title (ZH)**: FGBench：大型语言模型在功能团级分子性质推理的数据库和基准测试 

**Authors**: Xuan Liu, Siru Ouyang, Xianrui Zhong, Jiawei Han, Huimin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01055)  

**Abstract**: Large language models (LLMs) have gained significant attention in chemistry. However, most existing datasets center on molecular-level property prediction and overlook the role of fine-grained functional group (FG) information. Incorporating FG-level data can provide valuable prior knowledge that links molecular structures with textual descriptions, which can be used to build more interpretable, structure-aware LLMs for reasoning on molecule-related tasks. Moreover, LLMs can learn from such fine-grained information to uncover hidden relationships between specific functional groups and molecular properties, thereby advancing molecular design and drug discovery. Here, we introduce FGBench, a dataset comprising 625K molecular property reasoning problems with functional group information. Functional groups are precisely annotated and localized within the molecule, which ensures the dataset's interoperability thereby facilitating further multimodal applications. FGBench includes both regression and classification tasks on 245 different functional groups across three categories for molecular property reasoning: (1) single functional group impacts, (2) multiple functional group interactions, and (3) direct molecular comparisons. In the benchmark of state-of-the-art LLMs on 7K curated data, the results indicate that current LLMs struggle with FG-level property reasoning, highlighting the need to enhance reasoning capabilities in LLMs for chemistry tasks. We anticipate that the methodology employed in FGBench to construct datasets with functional group-level information will serve as a foundational framework for generating new question-answer pairs, enabling LLMs to better understand fine-grained molecular structure-property relationships. The dataset and evaluation code are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在化学领域获得了显著关注。然而，现有的大多数数据集主要集中在分子级别的性质预测，忽视了精细功能团（FG）信息的作用。整合功能团级别的数据可以提供将分子结构与文本描述联系起来的宝贵先验知识，从而构建更具解释性、结构感知的LLMs，用于分子相关的任务推理。此外，LLMs可以从这种精细信息中学习，发现特定功能团与分子性质之间的隐藏关系，从而促进分子设计和药物发现。在此基础上，我们介绍了FGBench数据集，包含625K个包含功能团信息的分子性质推理问题。功能团被精确标注并定位在分子内，确保了数据集的互操作性，从而便于进一步的多模态应用。FGBench包括三个类别中245种不同功能团的回归和分类任务：（1）单一功能团影响，（2）多种功能团相互作用，（3）直接分子比较。在针对7K精选数据的前沿LLMs基准测试中，结果表明当前的LLMs在功能团级别性质推理方面存在困难，强调了为化学任务增强LLMs推理能力的需求。我们期待FGBench中构建具有功能团级信息数据集的方法论将为生成新的问答对提供基础框架，使LLMs更好地理解精细分子结构-性质关系。数据集和评估代码可在\href{this https URL}{this https URL}获取。 

---
# Autonomous Penetration Testing: Solving Capture-the-Flag Challenges with LLMs 

**Title (ZH)**: 自主渗透测试：使用大型语言模型解决捕获旗帜挑战 

**Authors**: Isabelle Bakker, John Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2508.01054)  

**Abstract**: This study evaluates the ability of GPT-4o to autonomously solve beginner-level offensive security tasks by connecting the model to OverTheWire's Bandit capture-the-flag game. Of the 25 levels that were technically compatible with a single-command SSH framework, GPT-4o solved 18 unaided and another two after minimal prompt hints for an overall 80% success rate. The model excelled at single-step challenges that involved Linux filesystem navigation, data extraction or decoding, and straightforward networking. The approach often produced the correct command in one shot and at a human-surpassing speed. Failures involved multi-command scenarios that required persistent working directories, complex network reconnaissance, daemon creation, or interaction with non-standard shells. These limitations highlight current architectural deficiencies rather than a lack of general exploit knowledge. The results demonstrate that large language models (LLMs) can automate a substantial portion of novice penetration-testing workflow, potentially lowering the expertise barrier for attackers and offering productivity gains for defenders who use LLMs as rapid reconnaissance aides. Further, the unsolved tasks reveal specific areas where secure-by-design environments might frustrate simple LLM-driven attacks, informing future hardening strategies. Beyond offensive cybersecurity applications, results suggest the potential to integrate LLMs into cybersecurity education as practice aids. 

**Abstract (ZH)**: 本研究通过将模型连接到OverTheWire的Bandit夺旗游戏，评估了GPT-4o自主解决初级级别攻击安全任务的能力。在25个技术上与单命令SSH框架兼容的级别中，GPT-4o在无需辅助力的情况下解决了18个级别，并在最少提示下解决了另外2个级别，整体成功率达到80%。模型在涉及Linux文件系统导航、数据提取或解码以及直接网络操作的一步挑战中表现出色。这种方法往往能够一次生成正确的命令，并且速度超过人类。失败主要涉及需要持续工作目录、复杂网络侦察、创建守护进程或与非标准shell交互的多命令场景。这些局限性突显了当前架构的缺陷，而非一般利用知识的缺乏。研究结果表明，大型语言模型（LLMs）可以自动化初学者渗透测试工作流程的重要部分，有可能降低攻击者的专业知识门槛，并为使用LLMs作为快速侦察助手的防御者提供生产力提升。未解决的任务还揭示了安全设计环境在对抗简单驱动的LLM攻击中可能会遇到的具体问题，这些信息有助于未来的加固策略。此外，研究结果表明，大型语言模型有可能集成到网络安全教育中，作为实践辅助工具。 

---
# A Deep Reinforcement Learning-Based TCP Congestion Control Algorithm: Design, Simulation, and Evaluation 

**Title (ZH)**: 基于深度强化学习的TCP拥塞控制算法：设计、仿真与评估 

**Authors**: Efe Ağlamazlar, Emirhan Eken, Harun Batur Geçici  

**Link**: [PDF](https://arxiv.org/pdf/2508.01047)  

**Abstract**: This paper presents a novel TCP congestion control algorithm based on Deep Reinforcement Learning. The proposed approach utilizes Deep Q-Networks to optimize the congestion window (cWnd) by observing key network parameters and taking real-time actions. The algorithm is trained and evaluated within the NS-3 network simulator using the OpenGym interface. The results demonstrate significant improvements over traditional TCP New Reno in terms of latency and throughput, with better adaptability to changing network conditions. This study emphasizes the potential of reinforcement learning techniques for solving complex congestion control problems in modern networks. 

**Abstract (ZH)**: 基于深度强化学习的新型TCP拥塞控制算法 

---
# AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise 

**Title (ZH)**: AutoSIGHT：基于自动眼动追踪的即时评估人类专业知识系统 

**Authors**: Byron Dowling, Jozef Probcin, Adam Czajka  

**Link**: [PDF](https://arxiv.org/pdf/2508.01015)  

**Abstract**: Can we teach machines to assess the expertise of humans solving visual tasks automatically based on eye tracking features? This paper proposes AutoSIGHT, Automatic System for Immediate Grading of Human experTise, that classifies expert and non-expert performers, and builds upon an ensemble of features extracted from eye tracking data while the performers were solving a visual task. Results on the task of iris Presentation Attack Detection (PAD) used for this study show that with a small evaluation window of just 5 seconds, AutoSIGHT achieves an average average Area Under the ROC curve performance of 0.751 in subject-disjoint train-test regime, indicating that such detection is viable. Furthermore, when a larger evaluation window of up to 30 seconds is available, the Area Under the ROC curve (AUROC) increases to 0.8306, indicating the model is effectively leveraging more information at a cost of slightly delayed decisions. This work opens new areas of research on how to incorporate the automatic weighing of human and machine expertise into human-AI pairing setups, which need to react dynamically to nonstationary expertise distribution between the human and AI players (e.g. when the experts need to be replaced, or the task at hand changes rapidly). Along with this paper, we offer the eye tracking data used in this study collected from 6 experts and 53 non-experts solving iris PAD visual task. 

**Abstract (ZH)**: 基于眼动特征自动评估人类解决视觉任务专家水平的能力：AutoSIGHT自动即时评分系统 

---
# On Some Tunable Multi-fidelity Bayesian Optimization Frameworks 

**Title (ZH)**: 一些可调多 fidelity 贝叶斯优化框架 

**Authors**: Arjun Manoj, Anastasia S. Georgiou, Dimitris G. Giovanis, Themistoklis P. Sapsis, Ioannis G. Kevrekidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01013)  

**Abstract**: Multi-fidelity optimization employs surrogate models that integrate information from varying levels of fidelity to guide efficient exploration of complex design spaces while minimizing the reliance on (expensive) high-fidelity objective function evaluations. To advance Gaussian Process (GP)-based multi-fidelity optimization, we implement a proximity-based acquisition strategy that simplifies fidelity selection by eliminating the need for separate acquisition functions at each fidelity level. We also enable multi-fidelity Upper Confidence Bound (UCB) strategies by combining them with multi-fidelity GPs rather than the standard GPs typically used. We benchmark these approaches alongside other multi-fidelity acquisition strategies (including fidelity-weighted approaches) comparing their performance, reliance on high-fidelity evaluations, and hyperparameter tunability in representative optimization tasks. The results highlight the capability of the proximity-based multi-fidelity acquisition function to deliver consistent control over high-fidelity usage while maintaining convergence efficiency. Our illustrative examples include multi-fidelity chemical kinetic models, both homogeneous and heterogeneous (dynamic catalysis for ammonia production). 

**Abstract (ZH)**: 基于邻近度的多保真度获取策略促进了高斯过程（GP）为基础的多保真度优化，通过结合多保真度GP而非标准GP来实现多保真度Upper Confidence Bound（UCB）策略，并在代表性优化任务中与其他多保真度获取策略（包括保真度加权方法）进行了基准测试，比较了它们的性能、对高保真度评估的依赖性和超参数调节性。结果强调了基于邻近度的多保真度获取函数在控制高保真度使用方面的一致能力和保持收敛效率的能力。示例包括多保真度化学动力学模型，包括均相和非均相（氨生产的动态催化）。 

---
# v-PuNNs: van der Put Neural Networks for Transparent Ultrametric Representation Learning 

**Title (ZH)**: v-PuNNs：van der Put神经网络的透明超度量表示学习 

**Authors**: Gnankan Landry Regis N'guessan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01010)  

**Abstract**: Conventional deep learning models embed data in Euclidean space $\mathbb{R}^d$, a poor fit for strictly hierarchical objects such as taxa, word senses, or file systems. We introduce van der Put Neural Networks (v-PuNNs), the first architecture whose neurons are characteristic functions of p-adic balls in $\mathbb{Z}_p$. Under our Transparent Ultrametric Representation Learning (TURL) principle every weight is itself a p-adic number, giving exact subtree semantics. A new Finite Hierarchical Approximation Theorem shows that a depth-K v-PuNN with $\sum_{j=0}^{K-1}p^{\,j}$ neurons universally represents any K-level tree. Because gradients vanish in this discrete space, we propose Valuation-Adaptive Perturbation Optimization (VAPO), with a fast deterministic variant (HiPaN-DS) and a moment-based one (HiPaN / Adam-VAPO). On three canonical benchmarks our CPU-only implementation sets new state-of-the-art: WordNet nouns (52,427 leaves) 99.96% leaf accuracy in 16 min; GO molecular-function 96.9% leaf / 100% root in 50 s; NCBI Mammalia Spearman $\rho = -0.96$ with true taxonomic distance. The learned metric is perfectly ultrametric (zero triangle violations), and its fractal and information-theoretic properties are analyzed. Beyond classification we derive structural invariants for quantum systems (HiPaQ) and controllable generative codes for tabular data (Tab-HiPaN). v-PuNNs therefore bridge number theory and deep learning, offering exact, interpretable, and efficient models for hierarchical data. 

**Abstract (ZH)**: 范德普尔神经网络：p-进球函数神经网络及其在严格层次结构数据表示中的应用 

---
# Generative AI Adoption in Postsecondary Education, AI Hype, and ChatGPT's Launch 

**Title (ZH)**: _generative AI在高等教育中的应用、AI hype及其ChatGPT的发布_ 

**Authors**: Isabel Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01003)  

**Abstract**: The rapid integration of generative artificial intelligence (AI) into postsecondary education and many other sectors resulted in a global reckoning with this new technology. This paper contributes to the study of the multifaceted influence of generative AI, with a particular focus on OpenAI's ChatGPT within academic settings during the first six months after the release in three specific ways. First, it scrutinizes the rise of ChatGPT as a transformative event construed through a study of mainstream discourses exhibiting AI hype. Second, it discusses the perceived implications of generative AI for writing, teaching, and learning through the lens of critical discourse analysis and critical AI studies. Third, it encourages the necessity for best practices in the adoption of generative AI technologies in education. 

**Abstract (ZH)**: 生成式人工智能在高等教育及其他领域的快速集成引起了全球对这一新技术的反思。本文通过对主流话语中人工智能 hype 的研究，探讨 ChatGPT 在学术环境中的变革性影响，同时从批判话语分析和批判人工智能研究的视角讨论生成式人工智能对写作、教学和学习的潜在影响，并呼吁教育中采用生成式人工智能技术的最佳实践。 

---
# Are LLM-Powered Social Media Bots Realistic? 

**Title (ZH)**: 带有LLM驱动的社会媒体机器人现实吗？ 

**Authors**: Lynnette Hui Xian Ng, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2508.00998)  

**Abstract**: As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）变得更加 sophisticated，有可能利用LLMs来驱动社交媒体机器人。本研究探讨了由LLM驱动的社交媒体机器人网络的现实性。通过结合人工努力、网络科学和LLMs，我们创建了合成的机器人代理人格、他们的推文及其互动，从而模拟社交媒体网络。我们将生成的网络与实际的机器人/人类数据进行比较，发现由LLM驱动的机器人在网络特性和语言特性方面与野生机器人/人类存在差异。这对抗矫编和由LLM驱动的机器人效果具有重要影响。 

---
# ThermoCycleNet: Stereo-based Thermogram Labeling for Model Transition to Cycling 

**Title (ZH)**: ThermoCycleNet：基于立体视觉的热图标签生成以实现模型过渡到循环使用 

**Authors**: Daniel Andrés López, Vincent Weber, Severin Zentgraf, Barlo Hillen, Perikles Simon, Elmar Schömer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00974)  

**Abstract**: Infrared thermography is emerging as a powerful tool in sports medicine, allowing assessment of thermal radiation during exercise and analysis of anatomical regions of interest, such as the well-exposed calves. Building on our previous advanced automatic annotation method, we aimed to transfer the stereo- and multimodal-based labeling approach from treadmill running to ergometer cycling. Therefore, the training of the semantic segmentation network with automatic labels and fine-tuning on high-quality manually annotated images has been examined and compared in different data set combinations. The results indicate that fine-tuning with a small fraction of manual data is sufficient to improve the overall performance of the deep neural network. Finally, combining automatically generated labels with small manually annotated data sets accelerates the adaptation of deep neural networks to new use cases, such as the transition from treadmill to bicycle. 

**Abstract (ZH)**: 红外热成像技术在体育医学中的应用正逐渐成为一种强大的工具，允许评估运动过程中的热辐射，并分析感兴趣的解剖区域，如暴露良好的小腿。基于我们之前先进的自动注释方法，我们旨在将基于立体和多模态的标注方法从跑台跑步转移到功率自行车骑行中。因此，使用自动标签训练语义分割网络，并在高质量的手动注释图像上进行微调，不同数据集组合下的性能已经进行了比较。结果表明，使用少量的手动数据进行微调足以提高深度神经网络的整体性能。最后，将自动生成的标签与少量手动标注的数据集结合使用，可以加快深度神经网络对新应用场景的适应，如从跑台转变为自行车。 

---
# Generative AI as a Geopolitical Factor in Industry 5.0: Sovereignty, Access, and Control 

**Title (ZH)**: 生成式AI作为 geopolitics 因素在 Industry 5.0 中的作用：主权、访问与控制 

**Authors**: Azmine Toushik Wasi, Enjamamul Haque Eram, Sabrina Afroz Mitu, Md Manjurul Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00973)  

**Abstract**: Industry 5.0 marks a new phase in industrial evolution, emphasizing human-centricity, sustainability, and resilience through the integration of advanced technologies. Within this evolving landscape, Generative AI (GenAI) and autonomous systems are not only transforming industrial processes but also emerging as pivotal geopolitical instruments. We examine strategic implications of GenAI in Industry 5.0, arguing that these technologies have become national assets central to sovereignty, access, and global influence. As countries compete for AI supremacy, growing disparities in talent, computational infrastructure, and data access are reshaping global power hierarchies and accelerating the fragmentation of the digital economy. The human-centric ethos of Industry 5.0, anchored in collaboration between humans and intelligent systems, increasingly conflicts with the autonomy and opacity of GenAI, raising urgent governance challenges related to meaningful human control, dual-use risks, and accountability. We analyze how these dynamics influence defense strategies, industrial competitiveness, and supply chain resilience, including the geopolitical weaponization of export controls and the rise of data sovereignty. Our contribution synthesizes technological, economic, and ethical perspectives to propose a comprehensive framework for navigating the intersection of GenAI and geopolitics. We call for governance models that balance national autonomy with international coordination while safeguarding human-centric values in an increasingly AI-driven world. 

**Abstract (ZH)**: Industry 5.0标志着工业演化的全新阶段，强调以人为中心、可持续性和韧性，并通过先进技術名称的整合实现。在这一不断演进的背景中，生成式人工智能（GenAI）和自主系统不仅正在重塑工业流程，还逐渐成为关键的地缘政治工具。我们探讨GenAI在Industry 5.0中的战略影响，认为这些技术已成为国家安全、获取和全球影响力的核心资产。随着各国在人工智能领域的竞争加剧，人才、计算基础设施和数据访问方面的差异正在重塑全球权力结构，并加速了数字经济的碎片化。Industry 5.0以人为本的宗旨，根植于人类与智能系统的合作，在与GenAI的自主性和不透明性之间产生了日益激烈的冲突，这引发了与有意义的人类控制、双重用途风险和问责制相关的紧迫治理挑战。我们分析这些动态如何影响国防策略、工业竞争力和供应链韧性，包括出口管制的地缘政治武器化以及数据主权的兴起。我们的贡献综合了技术、经济和伦理视角，提出了一种全面框架，用于应对GenAI和地缘政治交汇处的挑战。我们呼吁建立一种治理模式，能够在保障人类中心价值观的前提下，在日益依赖人工智能的世界中维护国家自主性与国际协调的平衡。 

---
# AI-Educational Development Loop (AI-EDL): A Conceptual Framework to Bridge AI Capabilities with Classical Educational Theories 

**Title (ZH)**: AI教育发展循环（AI-EDL）：一种将AI能力与传统教育理论相结合的概念框架 

**Authors**: Ning Yu, Jie Zhang, Sandeep Mitra, Rebecca Smith, Adam Rich  

**Link**: [PDF](https://arxiv.org/pdf/2508.00970)  

**Abstract**: This study introduces the AI-Educational Development Loop (AI-EDL), a theory-driven framework that integrates classical learning theories with human-in-the-loop artificial intelligence (AI) to support reflective, iterative learning. Implemented in EduAlly, an AI-assisted platform for writing-intensive and feedback-sensitive tasks, the framework emphasizes transparency, self-regulated learning, and pedagogical oversight. A mixed-methods study was piloted at a comprehensive public university to evaluate alignment between AI-generated feedback, instructor evaluations, and student self-assessments; the impact of iterative revision on performance; and student perceptions of AI feedback. Quantitative results demonstrated statistically significant improvement between first and second attempts, with agreement between student self-evaluations and final instructor grades. Qualitative findings indicated students valued immediacy, specificity, and opportunities for growth that AI feedback provided. These findings validate the potential to enhance student learning outcomes through developmentally grounded, ethically aligned, and scalable AI feedback systems. The study concludes with implications for future interdisciplinary applications and refinement of AI-supported educational technologies. 

**Abstract (ZH)**: 基于人类在环的AI教育发展循环（AI-EDL）框架：经典的教育理论与人工智能的整合以支持反思性的迭代学习 

---
# Masked Omics Modeling for Multimodal Representation Learning across Histopathology and Molecular Profiles 

**Title (ZH)**: 掩码Omics建模在组织病理学和分子特征跨模态表示学习中的应用 

**Authors**: Lucas Robinet, Ahmad Berjaoui, Elizabeth Cohen-Jonathan Moyal  

**Link**: [PDF](https://arxiv.org/pdf/2508.00969)  

**Abstract**: Self-supervised learning has driven major advances in computational pathology by enabling models to learn rich representations from hematoxylin and eosin (H&E)-stained cancer tissue. However, histopathology alone often falls short for molecular characterization and understanding clinical outcomes, as important information is contained in high-dimensional omics profiles like transcriptomics, methylomics, or genomics. In this work, we introduce MORPHEUS, a unified transformer-based pre-training framework that encodes both histopathology and multi-omics data into a shared latent space. At its core, MORPHEUS relies on a masked modeling objective applied to randomly selected omics portions, encouraging the model to learn biologically meaningful cross-modal relationships. The same pre-trained network can be applied to histopathology alone or in combination with any subset of omics modalities, seamlessly adapting to the available inputs. Additionally, MORPHEUS enables any-to-any omics generation, enabling one or more omics profiles to be inferred from any subset of modalities, including H&E alone. Pre-trained on a large pan-cancer cohort, MORPHEUS consistently outperforms state-of-the-art methods across diverse modality combinations and tasks, positioning itself as a promising framework for developing multimodal foundation models in oncology. The code is available at: this https URL 

**Abstract (ZH)**: 自我监督学习通过使模型能够从苏木精和曙红（H&E）染色的癌组织中学习丰富的表示，已在计算病理学领域取得了重大进展。然而，仅靠组织病理学往往不足以进行分子表征和理解临床结局，因为重要信息包含在转录组学、甲基组学或基因组学等高维组学档案中。在此工作中，我们介绍了MORPHEUS，一个统一的基于变换器的预训练框架，将组织病理学和多种组学数据编码到共享的潜在空间中。MORPHEUS的核心在于应用于随机选择的组学部分的遮蔽建模目标，促使模型学习生物意义的跨模态关系。预训练后的同一个网络可以仅应用于组织病理学，或与任意一组组学模态结合使用，无缝适应可用的输入。此外，MORPHEUS还实现了任意到任意组学生成，允许从任意一组模态，包括仅H&E染色，推断一个或多个组学档案。MORPHEUS在泛癌种队列上进行预训练，一致优于多种模态组合和任务下的先进方法，使其成为肿瘤学领域开发多模态基础模型的一个有前景的框架。代码可通过以下链接获取：this https URL 

---
# VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI 

**Title (ZH)**: VAULT: 聆听大型语言模型驱动的检索增强生成以实现警惕的对抗更新用于自然语言推理 

**Authors**: Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00965)  

**Abstract**: We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few-shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero-shot RoBERTa-base this http URL standard benchmarks, VAULT elevates RoBERTa-base accuracy from 88.48% to 92.60% on SNLI +4.12%, from 75.04% to 80.95% on ANLI +5.91%, and from 54.67% to 71.99% on MultiNLI +17.32%. It also consistently outperforms prior in-context adversarial methods by up to 2.0% across datasets. By automating high-quality adversarial data curation at scale, VAULT enables rapid, human-independent robustness improvements in NLI inference tasks. 

**Abstract (ZH)**: VAULT：一种全自动对抗RAG管道，系统地揭示并修复NLI模型的缺陷 

---
# Rethinking Multimodality: Optimizing Multimodal Deep Learning for Biomedical Signal Classification 

**Title (ZH)**: 重新思考多媒体性：优化多模态深度学习在生物医学信号分类中的应用 

**Authors**: Timothy Oladunni, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00963)  

**Abstract**: This study proposes a novel perspective on multimodal deep learning for biomedical signal classification, systematically analyzing how complementary feature domains impact model performance. While fusing multiple domains often presumes enhanced accuracy, this work demonstrates that adding modalities can yield diminishing returns, as not all fusions are inherently advantageous. To validate this, five deep learning models were designed, developed, and rigorously evaluated: three unimodal (1D-CNN for time, 2D-CNN for time-frequency, and 1D-CNN-Transformer for frequency) and two multimodal (Hybrid 1, which fuses 1D-CNN and 2D-CNN; Hybrid 2, which combines 1D-CNN, 2D-CNN, and a Transformer). For ECG classification, bootstrapping and Bayesian inference revealed that Hybrid 1 consistently outperformed the 2D-CNN baseline across all metrics (p-values < 0.05, Bayesian probabilities > 0.90), confirming the synergistic complementarity of the time and time-frequency domains. Conversely, Hybrid 2's inclusion of the frequency domain offered no further improvement and sometimes a marginal decline, indicating representational redundancy; a phenomenon further substantiated by a targeted ablation study. This research redefines a fundamental principle of multimodal design in biomedical signal analysis. We demonstrate that optimal domain fusion isn't about the number of modalities, but the quality of their inherent complementarity. This paradigm-shifting concept moves beyond purely heuristic feature selection. Our novel theoretical contribution, "Complementary Feature Domains in Multimodal ECG Deep Learning," presents a mathematically quantifiable framework for identifying ideal domain combinations, demonstrating that optimal multimodal performance arises from the intrinsic information-theoretic complementarity among fused domains. 

**Abstract (ZH)**: 本研究提出了一种新的多模态深度学习在生物医学信号分类中的视角，系统分析了互补特征域如何影响模型性能。虽然融合多个领域通常假定会提高准确性，但本工作证明增加模态可能会适得其反，因为并非所有融合都是固有的有利的。为此，设计、开发并严格评估了五种深度学习模型：三种单模态（时间域的1D-CNN、时频域的2D-CNN、频率域的1D-CNN-Transformer）和两种多模态（Hybrid 1，融合1D-CNN和2D-CNN；Hybrid 2，结合1D-CNN、2D-CNN和Transformer）。对于ECG分类，通过自助采样和贝叶斯推断发现，Hybrid 1在所有指标上始终优于2D-CNN基准（p值<0.05，贝叶斯概率>0.90），证实了时间域和时频域的有效互补性。相反，Hybrid 2引入频率域没有带来进一步的改善，有时甚至略有下降，这表明存在表示冗余；这一现象在针对性的消融研究中进一步得到了证实。本研究重新定义了多模态设计在生物医学信号分析中的基本原则。我们证明了最佳领域融合不是关于模态的数量，而是关于其内在互补性的质量。这一范式转变的概念超越了纯粹的经验特征选择。我们的新型理论贡献“多模态ECG深度学习中的互补特征域”提供了一个可量化识别理想领域组合的数学框架，证明了最佳多模态性能源于融合域之间的固有信息论互补性。 

---
# FinKario: Event-Enhanced Automated Construction of Financial Knowledge Graph 

**Title (ZH)**: FinKario：事件增强的金融知识图谱自动化构建 

**Authors**: Xiang Li, Penglei Sun, Wanyun Zhou, Zikai Wei, Yongqi Zhang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00961)  

**Abstract**: Individual investors are significantly outnumbered and disadvantaged in financial markets, overwhelmed by abundant information and lacking professional analysis. Equity research reports stand out as crucial resources, offering valuable insights. By leveraging these reports, large language models (LLMs) can enhance investors' decision-making capabilities and strengthen financial analysis. However, two key challenges limit their effectiveness: (1) the rapid evolution of market events often outpaces the slow update cycles of existing knowledge bases, (2) the long-form and unstructured nature of financial reports further hinders timely and context-aware integration by LLMs. To address these challenges, we tackle both data and methodological aspects. First, we introduce the Event-Enhanced Automated Construction of Financial Knowledge Graph (FinKario), a dataset comprising over 305,360 entities, 9,625 relational triples, and 19 distinct relation types. FinKario automatically integrates real-time company fundamentals and market events through prompt-driven extraction guided by professional institutional templates, providing structured and accessible financial insights for LLMs. Additionally, we propose a Two-Stage, Graph-Based retrieval strategy (FinKario-RAG), optimizing the retrieval of evolving, large-scale financial knowledge to ensure efficient and precise data access. Extensive experiments show that FinKario with FinKario-RAG achieves superior stock trend prediction accuracy, outperforming financial LLMs by 18.81% and institutional strategies by 17.85% on average in backtesting. 

**Abstract (ZH)**: 个体投资者在金融市场中被显著地 outnumber 和处于劣势，面临信息繁多和缺乏专业分析的困境。股票研究报告是关键资源，提供了有价值的见解。通过利用这些报告，大型语言模型（LLMs）可以增强投资者的决策能力和加强金融分析。然而，两项关键挑战限制了它们的有效性：（1）市场事件的快速演变往往超过了现有知识库的缓慢更新周期；（2）金融报告的长效性和不结构化特性进一步阻碍了 LLMs 的及时和上下文相关集成。为了解决这些挑战，我们从数据和方法论两个方面着手。首先，我们引入了事件增强的财务知识图谱自动化构建（FinKario），该数据集包含超过 305,360 个实体、9,625 个关系三元组和 19 种不同的关系类型。FinKario 自动通过基于专业机构模板的提示驱动提取来整合实时的公司基本面和市场事件，为 LLMs 提供结构化和易于访问的财务洞察。此外，我们提出了一种基于图的两阶段检索策略（FinKario-RAG），优化了演变中的大规模财务知识的检索，以确保高效和精确的数据访问。广泛实验表明，FinKario 结合 FinKario-RAG 在回测中展示了卓越的股票趋势预测精度，分别比金融 LLMs 和机构策略高出 18.81% 和 17.85%。 

---
# Compression-Induced Communication-Efficient Large Model Training and Inferencing 

**Title (ZH)**: 压缩诱导的通信高效大规模模型训练与推理 

**Authors**: Sudip K. Seal, Maksudul Alam, Jorge Ramirez, Sajal Dash, Hao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00960)  

**Abstract**: Energy efficiency of training and inferencing with large neural network models is a critical challenge facing the future of sustainable large-scale machine learning workloads. This paper introduces an alternative strategy, called phantom parallelism, to minimize the net energy consumption of traditional tensor (model) parallelism, the most energy-inefficient component of large neural network training. The approach is presented in the context of feed-forward network architectures as a preliminary, but comprehensive, proof-of-principle study of the proposed methodology. We derive new forward and backward propagation operators for phantom parallelism, implement them as custom autograd operations within an end-to-end phantom parallel training pipeline and compare its parallel performance and energy-efficiency against those of conventional tensor parallel training pipelines. Formal analyses that predict lower bandwidth and FLOP counts are presented with supporting empirical results on up to 256 GPUs that corroborate these gains. Experiments are shown to deliver ~50% reduction in the energy consumed to train FFNs using the proposed phantom parallel approach when compared with conventional tensor parallel methods. Additionally, the proposed approach is shown to train smaller phantom models to the same model loss on smaller GPU counts as larger tensor parallel models on larger GPU counts offering the possibility for even greater energy savings. 

**Abstract (ZH)**: 使用大型神经网络模型进行训练和推断的能量效率是未来可持续大规模机器学习工作负载面临的最关键挑战之一。本文介绍了一种替代策略，称为幽灵并行性，以最小化传统张量（模型）并行性在大型神经网络训练中最不节能的组件的净能量消耗。该方法在前向网络架构的背景下进行了初步但全面的原理验证研究。我们为幽灵并行性推导了新的前向和反向传播算子，并在端到端的幽灵并行训练管道中实现为自定义自动求导操作，将其与传统的张量并行训练管道的并行性能和能效进行了比较。提出了正式分析预测较低的带宽和FLOP计数，并提供了在多达256块GPU上的支持性实验证据来验证这些增益。实验结果显示，与传统张量并行方法相比，所提出的幽灵并行方法可将前向网络（FFN）的训练能耗降低约50%。此外，所提出的方法在较小的GPU数量上训练较小的幽灵模型，达到与较大张量并行模型在较大GPU数量上相同模型损失，从而为更大的能效节省提供了可能性。 

---
# Enhancing material behavior discovery using embedding-oriented Physically-Guided Neural Networks with Internal Variables 

**Title (ZH)**: 基于内部变量的嵌入导向物理引导神经网络材料行为发现增强 

**Authors**: Rubén Muñoz-Sierra, Manuel Doblaré, Jacobo Ayensa-Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00959)  

**Abstract**: Physically Guided Neural Networks with Internal Variables are SciML tools that use only observable data for training and and have the capacity to unravel internal state relations. They incorporate physical knowledge both by prescribing the model architecture and using loss regularization, thus endowing certain specific neurons with a physical meaning as internal state variables. Despite their potential, these models face challenges in scalability when applied to high-dimensional data such as fine-grid spatial fields or time-evolving systems. In this work, we propose some enhancements to the PGNNIV framework that address these scalability limitations through reduced-order modeling techniques. Specifically, we introduce alternatives to the original decoder structure using spectral decomposition, POD, and pretrained autoencoder-based mappings. These surrogate decoders offer varying trade-offs between computational efficiency, accuracy, noise tolerance, and generalization, while improving drastically the scalability. Additionally, we integrate model reuse via transfer learning and fine-tuning strategies to exploit previously acquired knowledge, supporting efficient adaptation to novel materials or configurations, and significantly reducing training time while maintaining or improving model performance. To illustrate these various techniques, we use a representative case governed by the nonlinear diffusion equation, using only observable data. Results demonstrate that the enhanced PGNNIV framework successfully identifies the underlying constitutive state equations while maintaining high predictive accuracy. It also improves robustness to noise, mitigates overfitting, and reduces computational demands. The proposed techniques can be tailored to various scenarios depending on data availability, resources, and specific modeling objectives, overcoming scalability challenges in all the scenarios. 

**Abstract (ZH)**: 物理指导的神经网络结合内部变量：SciML工具，仅使用可观测数据进行训练，并具备解析内部状态关系的能力。通过减缩阶建模技术解决高维数据下的可扩展性限制，提出改进的PGNNIV框架。 

---
# Small sample-based adaptive text classification through iterative and contrastive description refinement 

**Title (ZH)**: 基于小样本的迭代对比描述细化自适应文本分类 

**Authors**: Amrit Rajeev, Udayaadithya Avadhanam, Harshula Tulapurkar, SaiBarath Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00957)  

**Abstract**: Zero-shot text classification remains a difficult task in domains with evolving knowledge and ambiguous category boundaries, such as ticketing systems. Large language models (LLMs) often struggle to generalize in these scenarios due to limited topic separability, while few-shot methods are constrained by insufficient data diversity. We propose a classification framework that combines iterative topic refinement, contrastive prompting, and active learning. Starting with a small set of labeled samples, the model generates initial topic labels. Misclassified or ambiguous samples are then used in an iterative contrastive prompting process to refine category distinctions by explicitly teaching the model to differentiate between closely related classes. The framework features a human-in-the-loop component, allowing users to introduce or revise category definitions in natural language. This enables seamless integration of new, unseen categories without retraining, making the system well-suited for real-world, dynamic environments. The evaluations on AGNews and DBpedia demonstrate strong performance: 91% accuracy on AGNews (3 seen, 1 unseen class) and 84% on DBpedia (8 seen, 1 unseen), with minimal accuracy shift after introducing unseen classes (82% and 87%, respectively). The results highlight the effectiveness of prompt-based semantic reasoning for fine-grained classification with limited supervision. 

**Abstract (ZH)**: 在 evolving knowledge 和 ambiguous category boundaries 的领域（如票据系统）中，零样本文本分类仍然是一个具有挑战性的任务。大型语言模型 (LLMs) 在这些场景中由于主题隔绝性有限而难以泛化，而 few-shot 方法则受限于数据多样性不足。我们提出了一种结合迭代主题细化、对比提示和主动学习的分类框架。从少量标注样本开始，模型生成初始主题标签。随后利用被错误分类或不明确的样本在迭代的对比提示过程中，通过明确地教会模型区分紧密相关的类别来细化类别差异。该框架包含人机交互组件，允许用户以自然语言形式引入或修订类别定义。这使得系统能够无缝集成新的未见过的类别而无需重新训练，使其适合用于现实世界的动态环境。AGNews 和 DBpedia 上的评估显示了强大的性能：AGNews 上 91% 的准确率（3 个已知类别，1 个未见过的类别），DBpedia 上 84% 的准确率（8 个已知类别，1 个未见过的类别），在引入未见过的类别后，准确率变化 minimal。结果突显了基于提示的语义推理在监督有限情况下的细粒度分类的有效性。 

---
# Learning Unified User Quantized Tokenizers for User Representation 

**Title (ZH)**: 统一用户量化词元化器学习用户表示 

**Authors**: Chuan He, Yang Chen, Wuliang Huang, Tianyi Zheng, Jianhu Chen, Bin Dou, Yice Luo, Yun Zhu, Baokun Wang, Yongchao Liu, Xing Fu, Yu Cheng, Chuntao Hong, Weiqiang Wang, Xin-Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00956)  

**Abstract**: Multi-source user representation learning plays a critical role in enabling personalized services on web platforms (e.g., Alipay). While prior works have adopted late-fusion strategies to combine heterogeneous data sources, they suffer from three key limitations: lack of unified representation frameworks, scalability and storage issues in data compression, and inflexible cross-task generalization. To address these challenges, we propose U^2QT (Unified User Quantized Tokenizers), a novel framework that integrates cross-domain knowledge transfer with early fusion of heterogeneous domains. Our framework employs a two-stage architecture: first, a causal Q-Former projects domain-specific features into a shared causal representation space to preserve inter-modality dependencies; second, a multi-view RQ-VAE discretizes causal embeddings into compact tokens through shared and source-specific codebooks, enabling efficient storage while maintaining semantic coherence. Experimental results showcase U^2QT's advantages across diverse downstream tasks, outperforming task-specific baselines in future behavior prediction and recommendation tasks while achieving efficiency gains in storage and computation. The unified tokenization framework enables seamless integration with language models and supports industrial-scale applications. 

**Abstract (ZH)**: 多源用户表示学习在web平台（如支付宝）上实现个性化服务中扮演着关键角色。尽管先前的工作采用晚融合策略结合异构数据源，但它们面临三个关键限制：缺乏统一表示框架、数据压缩中的可扩展性和存储问题，以及跨任务的灵活性差。为了解决这些挑战，我们提出了一种名为U^2QT（统一用户量化分词器）的新框架，该框架将跨域知识转移与异构领域早期融合相结合。该框架采用两阶段架构：首先，因果Q-Former将领域特定特征投影到共享因果表示空间，以保留跨模态依赖性；其次，多视图RQ-VAE通过共享和来源特定的代码本将因果嵌入离散化为紧凑的令牌，从而实现高效的存储同时保持语义连贯性。实验结果展示出U^2QT在多种下游任务中的优势，在未来行为预测和推荐任务中优于任务特定的基线，并实现在存储和计算上的效率提升。统一的令牌化框架能够无缝集成到语言模型中，并支持工业规模的应用。 

---
# From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model 

**Title (ZH)**: 从生成器到嵌入器：通过构建零样本区分嵌入模型挖掘多模态LLM的固有能力 

**Authors**: Yeong-Joon Ju, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.00955)  

**Abstract**: Multimodal Large Language Models (MLLMs) have emerged as a promising solution for universal embedding tasks, yet adapting their generative nature for discriminative representation learning remains a significant challenge. The dominant paradigm of large-scale contrastive pre-training suffers from critical inefficiencies, including prohibitive computational costs and a failure to leverage the intrinsic, instruction-following capabilities of MLLMs. To overcome these limitations, we propose an efficient framework for universal multimodal embeddings, which bridges this gap by centering on two synergistic components. First, our hierarchical embedding prompt template employs a two-level instruction architecture that forces the model to produce discriminative representations. Building on this strong foundation, our second component, self-aware hard negative sampling, redefines the fine-tuning process by leveraging the model's own understanding to efficiently mine challenging negatives while actively filtering out potential false negatives. Our comprehensive experiments show that our hierarchical prompt achieves zero-shot performance competitive with contrastively trained baselines and enhances the fine-tuning process by lifting a simple in-batch negative baseline by 4.8 points on the MMEB benchmark. We further boost the performance via our self-aware hard negative sampling, achieving the state-of-the-art performance without the contrative pre-training. Our work presents an effective and efficient pathway to adapt MLLMs for universal embedding tasks, significantly reducing training time. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已成为通用嵌入任务的有前景解决方案，但将其生成性质适应区分性表示学习仍是一个重大挑战。大规模对比预训练的主导范式面临着关键的效率问题，包括高昂的计算成本和未能充分利用MLLMs固有的指令遵循能力。为克服这些限制，我们提出了一种高效的框架，用于通用多模态嵌入，该框架通过强调两个协同组件来弥合这一差距。首先，我们的分层嵌入提示模板采用两层指令架构，迫使模型生成区分性表示。在此坚实的基础上，我们的第二个组件自意识硬负样本抽样重新定义了微调过程，利用模型自身的理解高效地挖掘挑战性负样本，并积极过滤潜在的假负样本。我们的全面实验表明，我们的分层提示在零样本性能上与对比训练的基线相当，并通过在MMEB基准测试中将简单的同批负样本基线提升4.8分来增强微调过程。我们进一步通过自意识硬负样本抽样提高性能，无需对比预训练就达到了最佳性能。我们的工作提供了一种有效且高效的途径来适应MLLMs进行通用嵌入任务，显著减少了训练时间。 

---
# Academic Vibe Coding: Opportunities for Accelerating Research in an Era of Resource Constraint 

**Title (ZH)**: 学术氛围编码：在资源约束时代加速研究的机遇 

**Authors**: Matthew G Crowson, Leo Celi A. Celi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00952)  

**Abstract**: Academic laboratories face mounting resource constraints: budgets are tightening, grant overheads are potentially being capped, and the market rate for data-science talent significantly outstrips university compensation. Vibe coding, which is structured, prompt-driven code generation with large language models (LLMs) embedded in reproducible workflows, offers one pragmatic response. It aims to compress the idea-to-analysis timeline, reduce staffing pressure on specialized data roles, and maintain rigorous, version-controlled outputs. This article defines the vibe coding concept, situates it against the current academic resourcing crisis, details a beginner-friendly toolchain for its implementation, and analyzes inherent limitations that necessitate governance and mindful application. 

**Abstract (ZH)**: 学术实验室面临日益紧缩的资源约束：预算紧缩，研究经费间接成本可能被封顶，而数据科学人才的市场薪酬远远超出大学的补偿水平。Vibe编码，这是一种嵌入可重复工作流程中的结构化、基于提示的代码生成方法，提供了一种务实的应对之策。它旨在压缩从构想到分析的时间线，减轻对专门数据角色的人力需求压力，并保持严谨的、版本控制的输出。本文定义了Vibe编码的概念，将其置于当前学术资源危机的背景之下，详细介绍了其实施的初学者友好工具链，并分析了固有的局限性，强调了需要治理和谨慎应用的重要性。 

---
# LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring 

**Title (ZH)**: LLMs可以在链式思考监控下隐蔽性地压制其能力评估 

**Authors**: Chloe Li, Mary Phuong, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2508.00943)  

**Abstract**: Trustworthy evaluations of dangerous capabilities are increasingly crucial for determining whether an AI system is safe to deploy. One empirically demonstrated threat to this is sandbagging - the strategic underperformance on evaluations by AI models or their developers. One promising defense is to monitor a model's chain-of-thought (CoT) reasoning, as this could reveal its intentions and plans. In this work, we measure the ability of models to sandbag on dangerous capability evaluations against a CoT monitor by prompting them to sandbag while being either monitor-oblivious or monitor-aware. We show that both frontier models and small open-sourced models can covertly sandbag against CoT monitoring 0-shot without hints. However, they cannot yet do so reliably: they bypass the monitor 16-36\% of the time when monitor-aware, conditioned on sandbagging successfully. We qualitatively analyzed the uncaught CoTs to understand why the monitor failed. We reveal a rich attack surface for CoT monitoring and contribute five covert sandbagging policies generated by models. These results inform potential failure modes of CoT monitoring and may help build more diverse sandbagging model organisms. 

**Abstract (ZH)**: 可信的危险能力评估越来越重要，对于确定AI系统是否安全部署至关重要。一种实证证明的威胁是“偷懒”——AI模型或其开发者在评估中战略性地表现不佳。一种有前景的防御措施是对模型的链式思考（CoT）推理进行监控，因为这能够揭示其意图和计划。在本文中，我们通过促使模型在无提示或有提示的情况下进行CoT监控，测量其在危险能力评估中偷懒的能力。我们发现，无论是前沿模型还是小型开源模型，都可以在无提示的情况下秘密地在CoT监控下偷懒，但它们目前还不能可靠地做到这一点：当模型有提示意识时，它们有16-36%的时间设法绕过了监控。我们对未被捕获的CoT进行了定性分析，以理解为什么监控失败。我们揭示了CoT监控丰富的攻击面，并贡献了五种由模型生成的隐蔽偷懒策略。这些结果为CoT监控的潜在失败模式提供了信息，并可能有助于构建更多样化的偷懒模型组织体。 

---
# Trusted Routing for Blockchain-Empowered UAV Networks via Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于多代理深度强化学习的区块链赋能无人机网络可信路由 

**Authors**: Ziye Jia, Sijie He, Qiuming Zhu, Wei Wang, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.00938)  

**Abstract**: Due to the high flexibility and versatility, unmanned aerial vehicles (UAVs) are leveraged in various fields including surveillance and disaster this http URL, in UAV networks, routing is vulnerable to malicious damage due to distributed topologies and high dynamics. Hence, ensuring the routing security of UAV networks is challenging. In this paper, we characterize the routing process in a time-varying UAV network with malicious nodes. Specifically, we formulate the routing problem to minimize the total delay, which is an integer linear programming and intractable to solve. Then, to tackle the network security issue, a blockchain-based trust management mechanism (BTMM) is designed to dynamically evaluate trust values and identify low-trust UAVs. To improve traditional practical Byzantine fault tolerance algorithms in the blockchain, we propose a consensus UAV update mechanism. Besides, considering the local observability, the routing problem is reformulated into a decentralized partially observable Markov decision process. Further, a multi-agent double deep Q-network based routing algorithm is designed to minimize the total delay. Finally, simulations are conducted with attacked UAVs and numerical results show that the delay of the proposed mechanism decreases by 13.39$\%$, 12.74$\%$, and 16.6$\%$ than multi-agent proximal policy optimal algorithms, multi-agent deep Q-network algorithms, and methods without BTMM, respectively. 

**Abstract (ZH)**: 基于区块链的信任管理机制及其在受攻击的时变无人机网络中路由算法的研究 

---
# Measuring Harmfulness of Computer-Using Agents 

**Title (ZH)**: 评估计算机使用代理的危害性 

**Authors**: Aaron Xuxiang Tian, Ruofan Zhang, Janet Tang, Jiaxin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00935)  

**Abstract**: Computer-using agents (CUAs), which autonomously control computers to perform multi-step actions, might pose significant safety risks if misused. Existing benchmarks mostly evaluate language models' (LMs) safety risks in chatbots or simple tool-usage scenarios, without granting full computer access. To better evaluate CUAs' misuse risks, we introduce a new benchmark: CUAHarm. CUAHarm consists of 104 expert-written realistic misuse risks, such as disabling firewalls, leaking confidential information, launching denial-of-service attacks, or installing backdoors. We provide a sandbox environment and rule-based verifiable rewards to measure CUAs' success rates in executing these tasks (e.g., whether the firewall is indeed disabled), not just refusal. We evaluate multiple frontier open-source and proprietary LMs, such as Claude Sonnet, GPT-4o, Gemini Pro 1.5, Llama-3.3-70B, and Mistral Large 2. Surprisingly, even without carefully designed jailbreaking prompts, these frontier LMs comply with executing these malicious tasks at a high success rate (e.g., 59% for Claude 3.7 Sonnet). Newer models show higher misuse rates: Claude 3.7 Sonnet succeeds on 15% more tasks than Claude 3.5. While these models are robust to common malicious prompts (e.g., creating a bomb) in chatbot settings, they behave unsafely as CUAs. We further evaluate a leading agentic framework (UI-TARS-1.5) and find that while it improves performance, it also amplifies misuse risks. Benign variants reveal refusals stem from alignment, not capability limits. To mitigate risks, we explore using LMs to monitor CUAs' actions and chain-of-thoughts (CoTs). Monitoring CUAs is significantly harder than chatbot outputs. Monitoring CoTs yields modest gains, with average detection accuracy at only 72%. Even with hierarchical summarization, improvement is limited to 4%. CUAHarm will be released at this https URL. 

**Abstract (ZH)**: CUAHarm：评估计算机使用代理滥用风险的新基准 

---
# OKG-LLM: Aligning Ocean Knowledge Graph with Observation Data via LLMs for Global Sea Surface Temperature Prediction 

**Title (ZH)**: OKG-LLM：通过大型语言模型对观测数据进行海洋知识图谱对齐以预测全球海表温度 

**Authors**: Hanchen Yang, Jiaqi Wang, Jiannong Cao, Wengen Li, Jialun Zheng, Yangning Li, Chunyu Miao, Jihong Guan, Shuigeng Zhou, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00933)  

**Abstract**: Sea surface temperature (SST) prediction is a critical task in ocean science, supporting various applications, such as weather forecasting, fisheries management, and storm tracking. While existing data-driven methods have demonstrated significant success, they often neglect to leverage the rich domain knowledge accumulated over the past decades, limiting further advancements in prediction accuracy. The recent emergence of large language models (LLMs) has highlighted the potential of integrating domain knowledge for downstream tasks. However, the application of LLMs to SST prediction remains underexplored, primarily due to the challenge of integrating ocean domain knowledge and numerical data. To address this issue, we propose Ocean Knowledge Graph-enhanced LLM (OKG-LLM), a novel framework for global SST prediction. To the best of our knowledge, this work presents the first systematic effort to construct an Ocean Knowledge Graph (OKG) specifically designed to represent diverse ocean knowledge for SST prediction. We then develop a graph embedding network to learn the comprehensive semantic and structural knowledge within the OKG, capturing both the unique characteristics of individual sea regions and the complex correlations between them. Finally, we align and fuse the learned knowledge with fine-grained numerical SST data and leverage a pre-trained LLM to model SST patterns for accurate prediction. Extensive experiments on the real-world dataset demonstrate that OKG-LLM consistently outperforms state-of-the-art methods, showcasing its effectiveness, robustness, and potential to advance SST prediction. The codes are available in the online repository. 

**Abstract (ZH)**: 基于海洋知识图谱增强的大语言模型的海表温度预测 

---
# SmartDate: AI-Driven Precision Sorting and Quality Control in Date Fruits 

**Title (ZH)**: SmartDate：基于人工智能的精确分选与质量控制在-date果实中的应用 

**Authors**: Khaled Eskaf  

**Link**: [PDF](https://arxiv.org/pdf/2508.00921)  

**Abstract**: SmartDate is an AI-powered system for automated sorting and quality control of date fruits. It combines deep learning, genetic algorithms, and reinforcement learning to improve classification accuracy and predict shelf life. The system uses high-resolution imaging and Visible-Near-Infrared (VisNIR) spectral sensors to evaluate key features such as moisture, sugar content, and texture. Reinforcement learning enables real-time adaptation to production conditions, while genetic algorithms optimize model parameters. SmartDate achieved 94.5 percent accuracy, 93.1 percent F1-score, and an AUC-ROC of 0.96. The system reduces waste and ensures that only high-quality dates reach the market, setting a new benchmark in smart agriculture. 

**Abstract (ZH)**: 基于AI的SmartDate系统实现了枣果的自动化分拣与质量控制 

---
# Predictive Auditing of Hidden Tokens in LLM APIs via Reasoning Length Estimation 

**Title (ZH)**: 基于推理长度估计的LLM APIs中隐藏令牌的预测性审计 

**Authors**: Ziyao Wang, Guoheng Sun, Yexiao He, Zheyu Shen, Bowei Tian, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00912)  

**Abstract**: Commercial LLM services often conceal internal reasoning traces while still charging users for every generated token, including those from hidden intermediate steps, raising concerns of token inflation and potential overbilling. This gap underscores the urgent need for reliable token auditing, yet achieving it is far from straightforward: cryptographic verification (e.g., hash-based signature) offers little assurance when providers control the entire execution pipeline, while user-side prediction struggles with the inherent variance of reasoning LLMs, where token usage fluctuates across domains and prompt styles. To bridge this gap, we present PALACE (Predictive Auditing of LLM APIs via Reasoning Token Count Estimation), a user-side framework that estimates hidden reasoning token counts from prompt-answer pairs without access to internal traces. PALACE introduces a GRPO-augmented adaptation module with a lightweight domain router, enabling dynamic calibration across diverse reasoning tasks and mitigating variance in token usage patterns. Experiments on math, coding, medical, and general reasoning benchmarks show that PALACE achieves low relative error and strong prediction accuracy, supporting both fine-grained cost auditing and inflation detection. Taken together, PALACE represents an important first step toward standardized predictive auditing, offering a practical path to greater transparency, accountability, and user trust. 

**Abstract (ZH)**: 商业LLM服务常常隐藏内部推理痕迹但仍然按每个生成的令牌收费，包括那些来自隐藏中间步骤的令牌，这引发了令牌膨胀和潜在费用过高方面的担忧。这一差距突显了可靠令牌审计的迫切需求，但其实现远非简易：虽然加密验证（例如，基于哈希的签名）在这种情况下提供的保证有限，而用户端预测则难以应对推理LLM固有的差异性，即令牌使用模式在不同领域和提示风格之间波动。为了填补这一差距，我们提出了PALACE（通过推理令牌计数估计对LLM API进行预测审计）框架，该框架无需访问内部踪迹即可从提示-答案对中估计隐藏的推理令牌计数。PALACE引入了增强的GRPO适应模块和轻量级领域路由器，实现了跨多样化推理任务的动态校准，并减轻了令牌使用模式的变异。在数学、编码、医学和一般推理基准测试中的实验表明，PALACE实现了较低的相对误差和较强的预测准确性，支持精细的成本审计和膨胀检测。Palace代表了标准化预测审计的重要第一步，提供了提高透明度、问责制和用户信任的实用路径。 

---
# Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling 

**Title (ZH)**: 基于硬件无关分析建模的LLM推理性能预测 

**Authors**: Rajeev Patwari, Ashish Sirasao, Devleena Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.00904)  

**Abstract**: Large language models (LLMs) have been increasingly deployed as local agents on personal devices with CPUs, NPUs and integrated GPUs. However, forecasting inference performance on devices with such heterogeneity remains challenging due to the dynamic compute and memory demands. Existing approaches rely on GPU benchmarking or machine learning-based latency predictors, which are often hardware-specific and lack generalizability. To this end, we introduce LIFE, a lightweight and modular analytical framework that is comprised of modular analytical model of operators, configurable to characterize LLM inference workloads in a hardware and dataset-agnostic manner. LIFE characterizes the influence of software and model optimizations, such as quantization, KV cache compression, LoRA adapters, chunked prefill, different attentions, and operator fusion, on performance metrics such as time-to-first-token (TTFT), time-per-output-token (TPOT) and tokens-per-second (TPS). LIFE enables performance forecasting using only hardware specifications, such as TOPS and memory bandwidth, without requiring extensive dataset benchmarking. We validate LIFE's forecasting with inference on AMD Ryzen CPUs, NPUs, iGPUs and NVIDIA V100 GPUs, with Llama2-7B variants, demonstrating the utility of LIFE in forecasting LLM performance through lens of system efficiency to enable efficient LLM deployment across different hardware platforms. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正越来越多地在个人设备上的CPU、NPUs和集成GPU等异构平台上作为本地代理使用。然而，由于计算和内存需求的动态变化，预测这些设备上的推理性能仍然具有挑战性。现有方法依赖于GPU基准测试或基于机器学习的延迟预测器，这些方法通常硬件特定且缺乏通用性。为此，我们引入了LIFE，一个轻量级且模块化的分析框架，由可配置的操作器分析模型组成，在不依赖于硬件和数据集的情况下，描述LLM推理工作负载。LIFE刻画了软件和模型优化（如量化、KV缓存压缩、LoRA适配器、分块预填充、不同的注意力机制以及操作器融合）对性能指标（如首个词的时间TTFT、每个输出词的时间TPOT和每秒词数TPS）的影响。LIFE仅利用硬件规格（如TOPS和内存带宽）即可进行性能预测，无需进行广泛的基准测试数据集测试。我们通过在AMD Ryzen CPU、NPUs、iGPUs和NVIDIA V100 GPU上进行推理验证了LIFE的预测效果，使用Llama2-7B变体，展示了LIFE通过系统效率视角预测LLM性能的实用性，以促进不同硬件平台上的高效LLM部署。 

---
# Universal Neurons in GPT-2: Emergence, Persistence, and Functional Impact 

**Title (ZH)**: GPT-2中的通用神经元：涌现、持久性和功能影响 

**Authors**: Advey Nandan, Cheng-Ting Chou, Amrit Kurakula, Cole Blondin, Kevin Zhu, Vasu Sharma, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2508.00903)  

**Abstract**: We investigate the phenomenon of neuron universality in independently trained GPT-2 Small models, examining how these universal neurons-neurons with consistently correlated activations across models-emerge and evolve throughout training. By analyzing five GPT-2 models at three checkpoints (100k, 200k, 300k steps), we identify universal neurons through pairwise correlation analysis of activations over a dataset of 5 million tokens. Ablation experiments reveal significant functional impacts of universal neurons on model predictions, measured via loss and KL divergence. Additionally, we quantify neuron persistence, demonstrating high stability of universal neurons across training checkpoints, particularly in deeper layers. These findings suggest stable and universal representational structures emerge during neural network training. 

**Abstract (ZH)**: 我们研究了独立训练的GPT-2 Small模型中神经元通用性的现象，考察这些在模型间表现出一致激活模式的通用神经元如何在整个训练过程中出现和演化。通过分析五个GPT-2模型在三个检查点（100k、200k、300k步）上的激活数据（共计500万tokens），我们利用成对相关性分析识别通用神经元。消融实验揭示了通用神经元对模型预测的显著功能影响，通过损失和KL散度进行测量。此外，我们量化了神经元的持久性，显示通用神经元在训练检查点之间具有高度稳定性，尤其是在较深层。这些发现表明，在神经网络训练过程中会形成稳定且通用的表示结构。 

---
# Sparse 3D Perception for Rose Harvesting Robots: A Two-Stage Approach Bridging Simulation and Real-World Applications 

**Title (ZH)**: 玫瑰采摘机器人基于稀疏3D感知的两阶段方法：从仿真到实际应用 

**Authors**: Taha Samavati, Mohsen Soryani, Sina Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00900)  

**Abstract**: The global demand for medicinal plants, such as Damask roses, has surged with population growth, yet labor-intensive harvesting remains a bottleneck for scalability. To address this, we propose a novel 3D perception pipeline tailored for flower-harvesting robots, focusing on sparse 3D localization of rose centers. Our two-stage algorithm first performs 2D point-based detection on stereo images, followed by depth estimation using a lightweight deep neural network. To overcome the challenge of scarce real-world labeled data, we introduce a photorealistic synthetic dataset generated via Blender, simulating a dynamic rose farm environment with precise 3D annotations. This approach minimizes manual labeling costs while enabling robust model training. We evaluate two depth estimation paradigms: a traditional triangulation-based method and our proposed deep learning framework. Results demonstrate the superiority of our method, achieving an F1 score of 95.6% (synthetic) and 74.4% (real) in 2D detection, with a depth estimation error of 3% at a 2-meter range on synthetic data. The pipeline is optimized for computational efficiency, ensuring compatibility with resource-constrained robotic systems. By bridging the domain gap between synthetic and real-world data, this work advances agricultural automation for specialty crops, offering a scalable solution for precision harvesting. 

**Abstract (ZH)**: 全球医药植物需求，如玫瑰的需求随着人口增长而增加，但由于劳动密集型采摘仍是限制规模化生产的瓶颈，我们提出了一种专门针对花卉收割机器人的新型3D感知管道，重点在于玫瑰中心的稀疏3D定位。该两阶段算法首先在立体图像上进行2D点检测，然后使用轻量级深度神经网络估计深度。为克服稀缺的真实世界标注数据难题，我们引入了一种通过Blender生成的逼真合成数据集，模拟具有精确3D注释的动态玫瑰农场环境。这种方法可以最大限度地减少人工标注成本，同时使模型训练更具鲁棒性。我们评估了两种深度估计框架：传统三角测量方法和我们提出的深度学习框架。结果表明，我们的方法在2D检测中的F1分数分别为95.6%（合成数据）和74.4%（真实数据），在合成数据2米范围内深度估计误差为3%。该管道优化了计算效率，确保与资源受限的机器人系统兼容。通过弥合合成数据与真实世界数据之间的领域差距，本工作推进了特色作物的农业自动化，提供了精准采摘的可扩展解决方案。 

---
# Benefits of Feature Extraction and Temporal Sequence Analysis for Video Frame Prediction: An Evaluation of Hybrid Deep Learning Models 

**Title (ZH)**: 特征提取和时间序列分析在视频帧预测中的益处：混合深度学习模型的评估 

**Authors**: Jose M. Sánchez Velázquez, Mingbo Cai, Andrew Coney, Álvaro J. García- Tejedor, Alberto Nogales  

**Link**: [PDF](https://arxiv.org/pdf/2508.00898)  

**Abstract**: In recent years, advances in Artificial Intelligence have significantly impacted computer science, particularly in the field of computer vision, enabling solutions to complex problems such as video frame prediction. Video frame prediction has critical applications in weather forecasting or autonomous systems and can provide technical improvements, such as video compression and streaming. Among Artificial Intelligence methods, Deep Learning has emerged as highly effective for solving vision-related tasks, although current frame prediction models still have room for enhancement. This paper evaluates several hybrid deep learning approaches that combine the feature extraction capabilities of autoencoders with temporal sequence modelling using Recurrent Neural Networks (RNNs), 3D Convolutional Neural Networks (3D CNNs), and related architectures. The proposed solutions were rigorously evaluated on three datasets that differ in terms of synthetic versus real-world scenarios and grayscale versus color imagery. Results demonstrate that the approaches perform well, with SSIM metrics increasing from 0.69 to 0.82, indicating that hybrid models utilizing 3DCNNs and ConvLSTMs are the most effective, and greyscale videos with real data are the easiest to predict. 

**Abstract (ZH)**: 近年来，人工智能的进步在计算机科学领域，特别是在计算机视觉领域产生了显著影响，使预测视频帧成为可能。视频帧预测在天气预报或自主系统中具有关键应用，并可提供技术改进，如视频压缩和流媒体。在人工智能方法中，深度学习因其在视觉任务方面的有效性而脱颖而出，尽管当前的帧预测模型仍有改进空间。本文评估了几种混合深度学习方法，这些方法结合了自动编码器的特征提取能力以及使用循环神经网络（RNNs）、3D 卷积神经网络（3D CNNs）及相关架构的时序序列建模能力。所提出的解决方案在三个不同类型的数据库上进行了严格评估，这些数据库在合成与真实场景以及灰度与彩色图像方面有所不同。结果表明，这些方法表现良好，SSIM指标从0.69提高到0.82，指出采用3DCNN和ConvLSTM的混合模型最为有效，而灰度视频在真实数据情况下最容易预测。 

---
# Maximize margins for robust splicing detection 

**Title (ZH)**: 最大化边距以实现稳健的剪接检测 

**Authors**: Julien Simon de Kergunic, Rony Abecidan, Patrick Bas, Vincent Itier  

**Link**: [PDF](https://arxiv.org/pdf/2508.00897)  

**Abstract**: Despite recent progress in splicing detection, deep learning-based forensic tools remain difficult to deploy in practice due to their high sensitivity to training conditions. Even mild post-processing applied to evaluation images can significantly degrade detector performance, raising concerns about their reliability in operational contexts. In this work, we show that the same deep architecture can react very differently to unseen post-processing depending on the learned weights, despite achieving similar accuracy on in-distribution test data. This variability stems from differences in the latent spaces induced by training, which affect how samples are separated internally. Our experiments reveal a strong correlation between the distribution of latent margins and a detector's ability to generalize to post-processed images. Based on this observation, we propose a practical strategy for building more robust detectors: train several variants of the same model under different conditions, and select the one that maximizes latent margins. 

**Abstract (ZH)**: 尽管在剪接检测方面取得了最近的进步，基于深度学习的法医工具由于对训练条件的高度敏感性，在实际部署中仍然面临挑战。即使对评估图像进行轻度后处理也可能显著降低检测器性能，这引起了对其在实际操作环境中可靠性的担忧。在本文中，我们证明了在实现类似准确度的情况下，相同的深度架构在面对未见过的后处理时会根据所学习的权重表现出截然不同的反应。这种变化性源自于训练过程中诱导的潜在空间差异，这些差异影响样本内部的分离方式。我们的实验揭示了潜在边界分布与检测器对后处理图像的泛化能力之间存在密切关系。基于这一观察，我们提出了构建更为稳健的检测器的实用策略：在不同的条件下训练同一模型的多种变体，并选择能使潜在边界最大化的一个。 

---
# HoneyImage: Verifiable, Harmless, and Stealthy Dataset Ownership Verification for Image Models 

**Title (ZH)**: 蜂蜜图像：可验证、无害且隐蔽的图像模型数据集所有权验证 

**Authors**: Zhihao Zhu, Jiale Han, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00892)  

**Abstract**: Image-based AI models are increasingly deployed across a wide range of domains, including healthcare, security, and consumer applications. However, many image datasets carry sensitive or proprietary content, raising critical concerns about unauthorized data usage. Data owners therefore need reliable mechanisms to verify whether their proprietary data has been misused to train third-party models. Existing solutions, such as backdoor watermarking and membership inference, face inherent trade-offs between verification effectiveness and preservation of data integrity. In this work, we propose HoneyImage, a novel method for dataset ownership verification in image recognition models. HoneyImage selectively modifies a small number of hard samples to embed imperceptible yet verifiable traces, enabling reliable ownership verification while maintaining dataset integrity. Extensive experiments across four benchmark datasets and multiple model architectures show that HoneyImage consistently achieves strong verification accuracy with minimal impact on downstream performance while maintaining imperceptible. The proposed HoneyImage method could provide data owners with a practical mechanism to protect ownership over valuable image datasets, encouraging safe sharing and unlocking the full transformative potential of data-driven AI. 

**Abstract (ZH)**: 基于图像的AI模型已在医疗、安全和消费者应用等多个领域得到广泛应用。然而，许多图像数据集包含敏感或专有内容，引发了未经授权使用数据的严重关切。因此，数据所有者需要可靠的机制来验证其专有数据是否被误用于训练第三方模型。现有解决方案，如后门水印和成员推理，存在验证效果与数据完整性保持之间的固有权衡。在此工作中，我们提出HoneyImage，一种用于图像识别模型数据集所有权验证的新方法。HoneyImage选择性地修改少量难以解决的样本，嵌入不可感知但可验证的痕迹，从而实现可靠的所有权验证，同时保持数据集的完整性。在四个基准数据集和多种模型架构上进行的 extensive 实验表明，HoneyImage 在对下游性能影响最小的情况下，能够实现稳定的验证准确性，并保持不可感知性。所提出的HoneyImage方法可为数据所有者提供保护其有价值图像数据所有权的实际机制，促进安全共享并充分发挥数据驱动AI的变革潜力。 

---
# Accelerating multiparametric quantitative MRI using self-supervised scan-specific implicit neural representation with model reinforcement 

**Title (ZH)**: 利用自监督扫描特异性隐式神经表示与模型强化加速多参数定量磁共振成像 

**Authors**: Ruimin Feng, Albert Jang, Xingxin He, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00891)  

**Abstract**: Purpose: To develop a self-supervised scan-specific deep learning framework for reconstructing accelerated multiparametric quantitative MRI (qMRI).
Methods: We propose REFINE-MORE (REference-Free Implicit NEural representation with MOdel REinforcement), combining an implicit neural representation (INR) architecture with a model reinforcement module that incorporates MR physics constraints. The INR component enables informative learning of spatiotemporal correlations to initialize multiparametric quantitative maps, which are then further refined through an unrolled optimization scheme enforcing data consistency. To improve computational efficiency, REFINE-MORE integrates a low-rank adaptation strategy that promotes rapid model convergence. We evaluated REFINE-MORE on accelerated multiparametric quantitative magnetization transfer imaging for simultaneous estimation of free water spin-lattice relaxation, tissue macromolecular proton fraction, and magnetization exchange rate, using both phantom and in vivo brain data.
Results: Under 4x and 5x accelerations on in vivo data, REFINE-MORE achieved superior reconstruction quality, demonstrating the lowest normalized root-mean-square error and highest structural similarity index compared to baseline methods and other state-of-the-art model-based and deep learning approaches. Phantom experiments further showed strong agreement with reference values, underscoring the robustness and generalizability of the proposed framework. Additionally, the model adaptation strategy improved reconstruction efficiency by approximately fivefold.
Conclusion: REFINE-MORE enables accurate and efficient scan-specific multiparametric qMRI reconstruction, providing a flexible solution for high-dimensional, accelerated qMRI applications. 

**Abstract (ZH)**: 目的：开发一种自我监督的扫描特定深度学习框架，用于重建加速的多参数定量磁共振成像（qMRI）。 

---
# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts 

**Title (ZH)**: FECT：接触中心对话转录中解释性AI生成声明的事实性评估 

**Authors**: Hagyeong Shin, Binoy Robin Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00889)  

**Abstract**: Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the conversation and root causes of the business problems. To remedy this, we first introduce a \textbf{3D} -- \textbf{Decompose, Decouple, Detach} -- paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed evaluation criteria. We then introduce \textbf{FECT}, a novel benchmark dataset for \textbf{F}actuality \textbf{E}valuation of Interpretive AI-Generated \textbf{C}laims in Contact Center Conversation \textbf{T}ranscripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated by an AI system for analyzing contact center conversations. 

**Abstract (ZH)**: 大型语言模型（LLMs） Known to Hallucinate, Introducing Challenges for Factuality Evaluation in Enterprise Applications: A 3D Paradigm for Factuality Assessment in Interpretive AI-Generated Claims from Contact Center Conversations 

---
# Multi-Grained Temporal-Spatial Graph Learning for Stable Traffic Flow Forecasting 

**Title (ZH)**: 多粒度时空图学习方法在稳定交通流预测中的应用 

**Authors**: Zhenan Lin, Yuni Lai, Wai Lun Lo, Richard Tai-Chiu Hsung, Harris Sik-Ho Tsang, Xiaoyu Xue, Kai Zhou, Yulin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00884)  

**Abstract**: Time-evolving traffic flow forecasting are playing a vital role in intelligent transportation systems and smart cities. However, the dynamic traffic flow forecasting is a highly nonlinear problem with complex temporal-spatial dependencies. Although the existing methods has provided great contributions to mine the temporal-spatial patterns in the complex traffic networks, they fail to encode the globally temporal-spatial patterns and are prone to overfit on the pre-defined geographical correlations, and thus hinder the model's robustness on the complex traffic environment. To tackle this issue, in this work, we proposed a multi-grained temporal-spatial graph learning framework to adaptively augment the globally temporal-spatial patterns obtained from a crafted graph transformer encoder with the local patterns from the graph convolution by a crafted gated fusion unit with residual connection techniques. Under these circumstances, our proposed model can mine the hidden global temporal-spatial relations between each monitor stations and balance the relative importance of local and global temporal-spatial patterns. Experiment results demonstrate the strong representation capability of our proposed method and our model consistently outperforms other strong baselines on various real-world traffic networks. 

**Abstract (ZH)**: 时空演化交通流预测在智能交通系统和智慧城市中发挥着重要作用。然而，动态交通流预测是一个高度非线性问题，涉及复杂的时空依赖关系。尽管现有的方法在挖掘复杂交通网络中的时空模式方面做出了巨大贡献，但它们无法编码全局时空模式，并且容易过度拟合预定义的地理关联，从而阻碍了模型在复杂交通环境中的 robustness。为了解决这一问题，本文提出了一种多粒度时空图学习框架，通过一个精心设计的门控融合单元结合残差连接技术，动态增强从构造的图变换器编码器中获得的全局时空模式与图卷积中的局部模式。在这种情况下，我们的模型可以揭示每个监控站之间的隐藏全局时空关系，并平衡局部和全局时空模式的相对重要性。实验结果表明，我们提出的方法具有强大的表示能力，并且在各种实际交通网络中始终优于其他强大的基线方法。 

---
# Reproducibility of Machine Learning-Based Fault Detection and Diagnosis for HVAC Systems in Buildings: An Empirical Study 

**Title (ZH)**: 基于机器学习的 HVAC 系统建筑故障检测与诊断的可重复性研究：一项实证研究 

**Authors**: Adil Mukhtar, Michael Hadwiger, Franz Wotawa, Gerald Schweiger  

**Link**: [PDF](https://arxiv.org/pdf/2508.00880)  

**Abstract**: Reproducibility is a cornerstone of scientific research, enabling independent verification and validation of empirical findings. The topic gained prominence in fields such as psychology and medicine, where concerns about non - replicable results sparked ongoing discussions about research practices. In recent years, the fast-growing field of Machine Learning (ML) has become part of this discourse, as it faces similar concerns about transparency and reliability. Some reproducibility issues in ML research are shared with other fields, such as limited access to data and missing methodological details. In addition, ML introduces specific challenges, including inherent nondeterminism and computational constraints. While reproducibility issues are increasingly recognized by the ML community and its major conferences, less is known about how these challenges manifest in applied disciplines. This paper contributes to closing this gap by analyzing the transparency and reproducibility standards of ML applications in building energy systems. The results indicate that nearly all articles are not reproducible due to insufficient disclosure across key dimensions of reproducibility. 72% of the articles do not specify whether the dataset used is public, proprietary, or commercially available. Only two papers share a link to their code - one of which was broken. Two-thirds of the publications were authored exclusively by academic researchers, yet no significant differences in reproducibility were observed compared to publications with industry-affiliated authors. These findings highlight the need for targeted interventions, including reproducibility guidelines, training for researchers, and policies by journals and conferences that promote transparency and reproducibility. 

**Abstract (ZH)**: 可重复性是科学研究的基石，能使独立的验证和验证实证发现成为可能。该话题在心理学和医学等领域中因其关于不可重复结果的担忧而变得尤为重要，引发了关于研究方法的持续讨论。近年来，快速发展的机器学习（ML）领域也加入了这一讨论，因为它面临着透明度和可靠性的相似关切。ML研究中的某些可重复性问题与其他领域共享，例如数据访问有限和方法学细节缺失。此外，ML还带来了特定的挑战，包括固有的非确定性和计算约束。虽然ML社区及其主要会议越来越认识到这些问题，但人们对这些挑战在应用学科中的表现知之甚少。本文通过分析建筑能源系统中机器学习应用的透明度和可重复性标准，旨在缩小这一差距。研究结果表明，几乎所有文章都无法实现可重复性，因为在关键的可重复性维度上披露不足。72%的文章没有明确说明所使用的数据集是公开的、专有的还是商业可用的。只有两篇论文共享了其代码的链接，其中一个是无效链接。三分之二的出版物由学术研究人员独撰，但与有工业关联作者的出版物相比，没有观察到显著的可重复性差异。这些发现突显了需要有针对性的干预措施，包括可重复性指南、研究人员培训以及促进透明度和可重复性的期刊和会议政策。 

---
# GNN-ASE: Graph-Based Anomaly Detection and Severity Estimation in Three-Phase Induction Machines 

**Title (ZH)**: 基于图的三相感应电机异常检测与严重程度估计 

**Authors**: Moutaz Bellah Bentrad, Adel Ghoggal, Tahar Bahi, Abderaouf Bahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00879)  

**Abstract**: The diagnosis of induction machines has traditionally relied on model-based methods that require the development of complex dynamic models, making them difficult to implement and computationally expensive. To overcome these limitations, this paper proposes a model-free approach using Graph Neural Networks (GNNs) for fault diagnosis in induction machines. The focus is on detecting multiple fault types -- including eccentricity, bearing defects, and broken rotor bars -- under varying severity levels and load conditions. Unlike traditional approaches, raw current and vibration signals are used as direct inputs, eliminating the need for signal preprocessing or manual feature extraction. The proposed GNN-ASE model automatically learns and extracts relevant features from raw inputs, leveraging the graph structure to capture complex relationships between signal types and fault patterns. It is evaluated for both individual fault detection and multi-class classification of combined fault conditions. Experimental results demonstrate the effectiveness of the proposed model, achieving 92.5\% accuracy for eccentricity defects, 91.2\% for bearing faults, and 93.1\% for broken rotor bar detection. These findings highlight the model's robustness and generalization capability across different operational scenarios. The proposed GNN-based framework offers a lightweight yet powerful solution that simplifies implementation while maintaining high diagnostic performance. It stands as a promising alternative to conventional model-based diagnostic techniques for real-world induction machine monitoring and predictive maintenance. 

**Abstract (ZH)**: 基于图神经网络的感应电机故障诊断方法 

---
# Satellite Connectivity Prediction for Fast-Moving Platforms 

**Title (ZH)**: 快移动平台卫星连接性预测 

**Authors**: Chao Yan, Babak Mafakheri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00877)  

**Abstract**: Satellite connectivity is gaining increased attention as the demand for seamless internet access, especially in transportation and remote areas, continues to grow. For fast-moving objects such as aircraft, vehicles, or trains, satellite connectivity is critical due to their mobility and frequent presence in areas without terrestrial coverage. Maintaining reliable connectivity in these cases requires frequent switching between satellite beams, constellations, or orbits. To enhance user experience and address challenges like long switching times, Machine Learning (ML) algorithms can analyze historical connectivity data and predict network quality at specific locations. This allows for proactive measures, such as network switching before connectivity issues arise. In this paper, we analyze a real dataset of communication between a Geostationary Orbit (GEO) satellite and aircraft over multiple flights, using ML to predict signal quality. Our prediction model achieved an F1 score of 0.97 on the test data, demonstrating the accuracy of machine learning in predicting signal quality during flight. By enabling seamless broadband service, including roaming between different satellite constellations and providers, our model addresses the need for real-time predictions of signal quality. This approach can further be adapted to automate satellite and beam-switching mechanisms to improve overall communication efficiency. The model can also be retrained and applied to any moving object with satellite connectivity, using customized datasets, including connected vehicles and trains. 

**Abstract (ZH)**: 卫星通信正因对无缝互联网接入需求的持续增长而备受关注，尤其是在交通运输和偏远地区。对于如飞机、车辆或列车等快速移动的对象，卫星通信因其移动性及在陆地覆盖不足区域的频繁存在而至关重要。在这种情况下，保持可靠的连接需要频繁切换卫星波束、星座或轨道。为了提升用户体验并应对如长切换时间等问题，机器学习（ML）算法可以分析历史连接数据并预测特定位置的网络质量。这使得可以在出现连接问题之前采取主动措施，如网络切换。本文分析了多架航班中地球静止轨道（GEO）卫星与飞机之间的通信数据集，使用机器学习来预测信号质量。我们的预测模型在测试数据上的F1分数达到了0.97，证明了机器学习在飞行过程中预测信号质量的准确性。通过提供无缝宽带服务，包括不同卫星星座和提供商之间的漫游，我们的模型满足了实时预测信号质量的需求。该方法还可进一步调整以自动化卫星和波束切换机制，从而提高整体通信效率。该模型还可以通过定制数据集重新训练并应用于任何具有卫星连接的移动对象，包括连接车辆和列车。 

---
# Patents as Knowledge Artifacts: An Information Science Perspective on Global Innovation 

**Title (ZH)**: 专利作为知识 artefact：信息科学视角下的全球创新 

**Authors**: M. S. Rajeevan, B. Mini Devi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00871)  

**Abstract**: In an age of fast-paced technological change, patents have evolved into not only legal mechanisms of intellectual property, but also structured storage containers of knowledge full of metadata, categories, and formal innovation. This chapter proposes to reframe patents in the context of information science, by focusing on patents as knowledge artifacts, and by seeing patents as fundamentally tied to the global movement of scientific and technological knowledge. With a focus on three areas, the inventions of AIs, biotech patents, and international competition with patents, this work considers how new technologies are challenging traditional notions of inventorship, access, and moral this http URL chapter provides a critical analysis of AI's implications for patent authorship and prior art searches, ownership issues arising from proprietary claims in biotechnology to ethical dilemmas, and the problem of using patents for strategic advantage in a global context of innovation competition. In this analysis, the chapter identified the importance of organizing information, creating metadata standards about originality, implementing retrieval systems to access previous works, and ethical contemplation about patenting unseen relationships in innovation ecosystems. Ultimately, the chapter called for a collaborative, transparent, and ethically-based approach in managing knowledge in the patenting environment highlighting the role for information professionals and policy to contribute to access equity in innovation. 

**Abstract (ZH)**: 在快速技术变革的时代，专利不仅演化为知识产权的法律机制，也成为充满元数据、分类和正式创新的知识结构存储容器。本章建议从信息科学的视角重新审视专利，将专利视为知识 artefact，并将其视为与全球科学与技术知识传播基本相关的核心要素。本章以人工智能发明、生物技术专利和国际专利竞争这三个领域为重点，探讨新技术如何挑战传统的发明人身份、访问权和道德标准问题。本章对人工智能对专利作者身份和现有技术搜索的影响、因生物技术的专有主张引发的所有权问题及伦理困境，以及在全球创新竞争背景下的专利战略优势问题进行了批判性分析。本章强调了组织信息、制定原创性元数据标准、实施检索系统以访问先前作品，以及对创新生态系统中未见关系进行专利申请的伦理思考的重要性。最终，本章倡导在专利环境中采取协作、透明和基于伦理的方法来管理知识，并强调信息专业人员和政策在促进创新访问公平方面的作用。 

---
# Better Recommendations: Validating AI-generated Subject Terms Through LOC Linked Data Service 

**Title (ZH)**: 更好的推荐：通过LOC链接数据服务验证AI生成的主题词 

**Authors**: Kwok Leong Tang, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00867)  

**Abstract**: This article explores the integration of AI-generated subject terms into library cataloging, focusing on validation through the Library of Congress Linked Data Service. It examines the challenges of traditional subject cataloging under the Library of Congress Subject Headings system, including inefficiencies and cataloging backlogs. While generative AI shows promise in expediting cataloging workflows, studies reveal significant limitations in the accuracy of AI-assigned subject headings. The article proposes a hybrid approach combining AI technology with human validation through LOC Linked Data Service, aiming to enhance the precision, efficiency, and overall quality of metadata creation in library cataloging practices. 

**Abstract (ZH)**: 本文探讨将AI生成的主题词纳入图书馆分类法的整合，重点通过美国国会图书馆关联数据服务进行验证。文章考察了使用美国国会图书馆主题词表系统传统主题分类的挑战，包括效率低下和分类积压问题。虽然生成式AI有潜力加速分类流程，但研究表明，AI分配的主题词准确性存在显著限制。本文提议结合AI技术与通过LOC关联数据服务进行的人工验证的混合方法，旨在提高图书馆分类实践中元数据创建的精确性、效率和整体质量。 

---
# Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal 

**Title (ZH)**: 在现实世界部署地理空间基础模型：WorldCereal的启示 

**Authors**: Christina Butsko, Kristof Van Tricht, Gabriel Tseng, Giorgia Milli, David Rolnick, Ruben Cartuyvels, Inbal Becker Reshef, Zoltan Szantoi, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00858)  

**Abstract**: The increasing availability of geospatial foundation models has the potential to transform remote sensing applications such as land cover classification, environmental monitoring, and change detection. Despite promising benchmark results, the deployment of these models in operational settings is challenging and rare. Standardized evaluation tasks often fail to capture real-world complexities relevant for end-user adoption such as data heterogeneity, resource constraints, and application-specific requirements. This paper presents a structured approach to integrate geospatial foundation models into operational mapping systems. Our protocol has three key steps: defining application requirements, adapting the model to domain-specific data and conducting rigorous empirical testing. Using the Presto model in a case study for crop mapping, we demonstrate that fine-tuning a pre-trained model significantly improves performance over conventional supervised methods. Our results highlight the model's strong spatial and temporal generalization capabilities. Our protocol provides a replicable blueprint for practitioners and lays the groundwork for future research to operationalize foundation models in diverse remote sensing applications. Application of the protocol to the WorldCereal global crop-mapping system showcases the framework's scalability. 

**Abstract (ZH)**: 地理空间基础模型的日益可用有望变革土地覆盖分类、环境监测和变化检测等遥感应用。尽管基准测试结果充满 promise，但在实际操作环境中的部署仍具挑战性和罕见性。标准评估任务往往未能捕捉到影响最终用户采用的实际复杂性，如数据异质性、资源限制和应用特定要求。本文提出了一种结构化方法，将地理空间基础模型集成到操作性制图系统中。我们的协议包含三个关键步骤：定义应用要求、适应领域特定数据以及进行严格的实证测试。通过在作物制图案例研究中使用 Presto 模型，我们证明了对预训练模型进行微调显著优于传统监督方法。我们的结果突显了该模型在空间和时间上的强泛化能力。我们的协议为实践者提供了一个可复制的范本，并为未来研究在多种遥感应用中实现基础模型奠定了基础。将该协议应用于全球作物制图系统 WorldCereal 展示了该框架的可扩展性。 

---
# EthicAlly: a Prototype for AI-Powered Research Ethics Support for the Social Sciences and Humanities 

**Title (ZH)**: EthicAlly：社会科学与人文科学领域的AI驱动研究伦理支持原型 

**Authors**: Steph Grohmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.00856)  

**Abstract**: In biomedical science, review by a Research Ethics Committee (REC) is an indispensable way of protecting human subjects from harm. However, in social science and the humanities, mandatory ethics compliance has long been met with scepticism as biomedical models of ethics can map poorly onto methodologies involving complex socio-political and cultural considerations. As a result, tailored ethics training and support as well as access to RECs with the necessary expertise is lacking in some areas, including parts of Europe and low- and middle-income countries. This paper suggests that Generative AI can meaningfully contribute to closing these gaps, illustrating this claim by presenting EthicAlly, a proof-of-concept prototype for an AI-powered ethics support system for social science and humanities researchers. Drawing on constitutional AI technology and a collaborative prompt development methodology, EthicAlly provides structured ethics assessment that incorporates both universal ethics principles and contextual and interpretive considerations relevant to most social science research. In supporting researchers in ethical research design and preparation for REC submission, this kind of system can also contribute to easing the burden on institutional RECs, without attempting to automate or replace human ethical oversight. 

**Abstract (ZH)**: 生成式AI在填补社会科学和人文科学伦理合规缺口中的潜在贡献：EthicAlly原型概览 

---
# Gearshift Fellowship: A Next-Generation Neurocomputational Game Platform to Model and Train Human-AI Adaptability 

**Title (ZH)**: 齿轮换挡 fellowship：下一代神经计算游戏平台，用于建模和训练人机适应性 

**Authors**: Nadja R. Ging-Jehli, Russell K. Childers, Joshua Lu, Robert Gemma, Rachel Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00850)  

**Abstract**: How do we learn when to persist, when to let go, and when to shift gears? Gearshift Fellowship (GF) is the prototype of a new Supertask paradigm designed to model how humans and artificial agents adapt to shifting environment demands. Grounded in cognitive neuroscience, computational psychiatry, economics, and artificial intelligence, Supertasks combine computational neurocognitive modeling with serious gaming. This creates a dynamic, multi-mission environment engineered to assess mechanisms of adaptive behavior across cognitive and social contexts. Computational parameters explain behavior and probe mechanisms by controlling the game environment. Unlike traditional tasks, GF enables neurocognitive modeling of individual differences across perceptual decisions, learning, and meta-cognitive levels. This positions GF as a flexible testbed for understanding how cognitive-affective control processes, learning styles, strategy use, and motivational shifts adapt across contexts and over time. It serves as an experimental platform for scientists, a phenotype-to-mechanism intervention for clinicians, and a training tool for players aiming to strengthen self-regulated learning, mood, and stress resilience. Online study (n = 60, ongoing) results show that GF recovers effects from traditional neuropsychological tasks (construct validity), uncovers novel patterns in how learning differs across contexts and how clinical features map onto distinct adaptations. These findings pave the way for developing in-game interventions that foster self-efficacy and agency to cope with real-world stress and uncertainty. GF builds a new adaptive ecosystem designed to accelerate science, transform clinical care, and foster individual growth. It offers a mirror and training ground where humans and machines co-develop together deeper flexibility and awareness. 

**Abstract (ZH)**: 当我们如何学习何时坚持、何时放手、何时转换策略？Gearshift Fellowship（GF）是一种新型Supertask范式的原型，旨在模拟人类和人工代理如何适应环境需求的变化。 

---
# Cognitive Exoskeleton: Augmenting Human Cognition with an AI-Mediated Intelligent Visual Feedback 

**Title (ZH)**: 认知外骨骼：通过AI介导的智能视觉反馈增强人类认知 

**Authors**: Songlin Xu, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00846)  

**Abstract**: In this paper, we introduce an AI-mediated framework that can provide intelligent feedback to augment human cognition. Specifically, we leverage deep reinforcement learning (DRL) to provide adaptive time pressure feedback to improve user performance in a math arithmetic task. Time pressure feedback could either improve or deteriorate user performance by regulating user attention and anxiety. Adaptive time pressure feedback controlled by a DRL policy according to users' real-time performance could potentially solve this trade-off problem. However, the DRL training and hyperparameter tuning may require large amounts of data and iterative user studies. Therefore, we propose a dual-DRL framework that trains a regulation DRL agent to regulate user performance by interacting with another simulation DRL agent that mimics user cognition behaviors from an existing dataset. Our user study demonstrates the feasibility and effectiveness of the dual-DRL framework in augmenting user performance, in comparison to the baseline group. 

**Abstract (ZH)**: 基于AI的框架：通过深度强化学习提供自适应时间压力反馈以增强人类认知 

---
# Generative AI for CAD Automation: Leveraging Large Language Models for 3D Modelling 

**Title (ZH)**: Generative AI for CAD Automation: 利用大规模语言模型进行3D建模 

**Authors**: Sumit Kumar, Sarthak Kapoor, Harsh Vardhan, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00843)  

**Abstract**: Large Language Models (LLMs) are revolutionizing industries by enhancing efficiency, scalability, and innovation. This paper investigates the potential of LLMs in automating Computer-Aided Design (CAD) workflows, by integrating FreeCAD with LLM as CAD design tool. Traditional CAD processes are often complex and require specialized sketching skills, posing challenges for rapid prototyping and generative design. We propose a framework where LLMs generate initial CAD scripts from natural language descriptions, which are then executed and refined iteratively based on error feedback. Through a series of experiments with increasing complexity, we assess the effectiveness of this approach. Our findings reveal that LLMs perform well for simple to moderately complex designs but struggle with highly constrained models, necessitating multiple refinements. The study highlights the need for improved memory retrieval, adaptive prompt engineering, and hybrid AI techniques to enhance script robustness. Future directions include integrating cloud-based execution and exploring advanced LLM capabilities to further streamline CAD automation. This work underscores the transformative potential of LLMs in design workflows while identifying critical areas for future development. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在通过提高效率、可扩展性和创新能力来重塑各行各业。本文研究了LLMs在将FreeCAD集成为其CAD设计工具时，在自动化CAD工作流程方面的潜力。传统的CAD流程通常非常复杂，并需要专门的绘图技能，这为快速原型制作和生成设计带来了挑战。我们提出了一种框架，其中LLMs从自然语言描述生成初始CAD脚本，然后根据错误反馈进行迭代执行和 refinement。通过一系列逐渐增加复杂性的实验，我们评估了该方法的有效性。研究结果表明，LLMs在简单到中等复杂的设计方面表现良好，但在高度约束的模型方面面临困难，需要多次调整。研究强调了改进记忆检索、适应性提示工程和混合AI技术以增强脚本稳健性的需求。未来的研究方向包括集成基于云的执行和探索更高级的LLM功能，以进一步简化CAD自动化。这项工作突显了LLMs在设计流程中的变革潜力，并指出了未来发展的关键领域。 

---
# Inclusive Review on Advances in Masked Human Face Recognition Technologies 

**Title (ZH)**: 包容性综述：遮罩人脸 Recognition 技术进展 

**Authors**: Ali Haitham Abdul Amir, Zainab N. Nemer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00841)  

**Abstract**: Masked Face Recognition (MFR) is an increasingly important area in biometric recognition technologies, especially with the widespread use of masks as a result of the COVID-19 pandemic. This development has created new challenges for facial recognition systems due to the partial concealment of basic facial features. This paper aims to provide a comprehensive review of the latest developments in the field, with a focus on deep learning techniques, especially convolutional neural networks (CNNs) and twin networks (Siamese networks), which have played a pivotal role in improving the accuracy of covering face recognition. The paper discusses the most prominent challenges, which include changes in lighting, different facial positions, partial concealment, and the impact of mask types on the performance of systems. It also reviews advanced technologies developed to overcome these challenges, including data enhancement using artificial databases and multimedia methods to improve the ability of systems to generalize. In addition, the paper highlights advance in deep network design, feature extraction techniques, evaluation criteria, and data sets used in this area. Moreover, it reviews the various applications of masked face recognition in the fields of security and medicine, highlighting the growing importance of these systems in light of recurrent health crises and increasing security threats. Finally, the paper focuses on future research trends such as developing more efficient algorithms and integrating multimedia technologies to improve the performance of recognition systems in real-world environments and expand their applications. 

**Abstract (ZH)**: 掩码面部识别（MFR）是生物特征识别技术中日益重要的领域，尤其是由于COVID-19疫情广泛使用口罩所致。这一发展为面部识别系统带来了新的挑战，因为基本面部特征部分被遮挡。本文旨在全面概述该领域的最新进展，重点关注深度学习技术，特别是卷积神经网络（CNNs）和孪生网络（Siamese网络），这些技术在提高遮挡面部识别精度方面发挥了关键作用。本文讨论了最显著的挑战，包括光照变化、不同面部位置、部分遮挡以及不同口罩类型对系统性能的影响。此外，本文还回顾了为克服这些挑战而开发的先进技术，包括使用人工数据库的数据增强以及多媒体方法以提高系统泛化能力。文中还强调了该领域深度网络设计的发展、特征提取技术、评估标准和数据集的进展。此外，本文还回顾了掩码面部识别在安全和医疗领域的各种应用，突显了这些系统在反复出现的健康危机和不断增加的安全威胁背景下日益重要的作用。最后，本文集中在未来研究趋势上，如开发更高效的算法和整合多媒体技术，以改善实际环境中的识别系统性能并扩展其应用领域。 

---
# The Attribution Crisis in LLM Search Results 

**Title (ZH)**: LLM搜索结果中的归因危机 

**Authors**: Ilan Strauss, Jangho Yang, Tim O'Reilly, Sruly Rosenblat, Isobel Moure  

**Link**: [PDF](https://arxiv.org/pdf/2508.00838)  

**Abstract**: Web-enabled LLMs frequently answer queries without crediting the web pages they consume, creating an "attribution gap" - the difference between relevant URLs read and those actually cited. Drawing on approximately 14,000 real-world LMArena conversation logs with search-enabled LLM systems, we document three exploitation patterns: 1) No Search: 34% of Google Gemini and 24% of OpenAI GPT-4o responses are generated without explicitly fetching any online content; 2) No citation: Gemini provides no clickable citation source in 92% of answers; 3) High-volume, low-credit: Perplexity's Sonar visits approximately 10 relevant pages per query but cites only three to four. A negative binomial hurdle model shows that the average query answered by Gemini or Sonar leaves about 3 relevant websites uncited, whereas GPT-4o's tiny uncited gap is best explained by its selective log disclosures rather than by better attribution. Citation efficiency - extra citations provided per additional relevant web page visited - varies widely across models, from 0.19 to 0.45 on identical queries, underscoring that retrieval design, not technical limits, shapes ecosystem impact. We recommend a transparent LLM search architecture based on standardized telemetry and full disclosure of search traces and citation logs. 

**Abstract (ZH)**: Web-enable的LLM经常在不引用所消费的网页的情况下回答查询，从而产生“归因差距”——即阅读的相关URL与实际引用之间的差异。基于约14,000条实际的LMArena对话日志，我们记录了三种利用模式：1）无搜索：34%的Google Gemini和24%的OpenAI GPT-4o的回答是在未明确检索任何在线内容的情况下生成的；2）无引用：Gemini在92%的答案中未提供可点击的引用来源；3）高流量、低引用：Perplexity的Sonar每次查询访问约10个相关信息网页但仅引用其中三至四个。负二项 hurdle 模型显示，Gemini或Sonar回答的平均查询会留下约3个相关网站未被引用，而GPT-4o较小的未被引用差距与其选择性的日志披露有关，而不是更好的归因机制。每访问一个额外的相关网页提供的引用效率——即额外引用的数量——在不同模型间存在巨大差异，从相同的查询中不同模型的0.19到0.45不等，这表明检索设计而非技术限制塑造了生态系统影响。我们建议基于标准化遥测技术的透明LLM搜索架构，并全面披露搜索轨迹和引用日志。 

---
# PCS Workflow for Veridical Data Science in the Age of AI 

**Title (ZH)**: AI时代可信数据科学的PCS工作流 

**Authors**: Zachary T. Rewolinski, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00835)  

**Abstract**: Data science is a pillar of artificial intelligence (AI), which is transforming nearly every domain of human activity, from the social and physical sciences to engineering and medicine. While data-driven findings in AI offer unprecedented power to extract insights and guide decision-making, many are difficult or impossible to replicate. A key reason for this challenge is the uncertainty introduced by the many choices made throughout the data science life cycle (DSLC). Traditional statistical frameworks often fail to account for this uncertainty. The Predictability-Computability-Stability (PCS) framework for veridical (truthful) data science offers a principled approach to addressing this challenge throughout the DSLC. This paper presents an updated and streamlined PCS workflow, tailored for practitioners and enhanced with guided use of generative AI. We include a running example to display the PCS framework in action, and conduct a related case study which showcases the uncertainty in downstream predictions caused by judgment calls in the data cleaning stage. 

**Abstract (ZH)**: 数据科学是人工智能（AI）的支柱，正在几乎每一个领域的人类活动中产生变革，从社会科学和物理科学到工程和医学。尽管AI中的数据驱动发现提供了前所未有的力量来提取洞察并指导决策，但许多发现难以复制或根本无法复制。这一挑战的关键原因是在数据科学生命周期（DSLC）中的众多选择引入了不确定性。传统的统计框架往往未能考虑到这种不确定性。揭示性（真实）数据科学的可预报性-可计算性-稳定性（PCS）框架提供了一种原则性的方法，以在整个数据科学生命周期中应对这一挑战。本文介绍了更新和完善后的PCS工作流程，针对实践工作者，并增强了生成式AI的指导使用。我们提供了一个示例案例来展示PCS框架的实际应用，并进行了一项相关案例研究，展示了数据清洗阶段判断决策导致的下游预测中的不确定性。 

---
# Bike-Bench: A Bicycle Design Benchmark for Generative Models with Objectives and Constraints 

**Title (ZH)**: Bike-Bench：面向生成模型的目标与约束自行车设计基准 

**Authors**: Lyle Regenwetter, Yazan Abu Obaideh, Fabien Chiotti, Ioanna Lykourentzou, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2508.00830)  

**Abstract**: We introduce Bike-Bench, an engineering design benchmark for evaluating generative models on problems with multiple real-world objectives and constraints. As generative AI's reach continues to grow, evaluating its capability to understand physical laws, human guidelines, and hard constraints grows increasingly important. Engineering product design lies at the intersection of these difficult tasks, providing new challenges for AI capabilities. Bike-Bench evaluates AI models' capability to generate designs that not only resemble the dataset, but meet specific performance objectives and constraints. To do so, Bike-Bench quantifies a variety of human-centered and multiphysics performance characteristics, such as aerodynamics, ergonomics, structural mechanics, human-rated usability, and similarity to subjective text or image prompts. Supporting the benchmark are several datasets of simulation results, a dataset of 10K human-rated bicycle assessments, and a synthetically-generated dataset of 1.4M designs, each with a parametric, CAD/XML, SVG, and PNG representation. Bike-Bench is uniquely configured to evaluate tabular generative models, LLMs, design optimization, and hybrid algorithms side-by-side. Our experiments indicate that LLMs and tabular generative models fall short of optimization and optimization-augmented generative models in both validity and optimality scores, suggesting significant room for improvement. We hope Bike-Bench, a first-of-its-kind benchmark, will help catalyze progress in generative AI for constrained multi-objective engineering design problems. Code, data, and other resources are published at this http URL. 

**Abstract (ZH)**: 我们介绍Bike-Bench：用于评价生成模型在具有多个现实世界目标和约束的问题上的工程设计基准。 

---
# A Schema.org Mapping for Brazilian Legal Norms: Toward Interoperable Legal Graphs and Open Government Data 

**Title (ZH)**: 巴西法律规范的Schema.org 映射：面向互操作法律图谱和开放政府数据的研究 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00827)  

**Abstract**: Open Government Data (OGD) initiatives aim to enhance transparency and public participation by making government data openly accessible. However, structuring legal norms for machine readability remains a critical challenge for advancing Legal Tech applications such as Legal Knowledge Graphs (LKGs). Focusing on the this http URL portal initiative by the Brazilian National Congress, we propose a unified mapping of Brazilian legislation to the this http URL vocabulary via JSON-LD and Linked Data. Our approach covers both the conceptual "Norm" entity (mapped to sdo:Legislation) and its digital publications or manifestations (mapped to sdo:LegislationObject). We detail key properties for each type, providing concrete examples and considering URN identifiers (per the LexML standard), multilingual support, versioning in the Official Journal, and inter-norm relationships (e.g., citations and references). Our structured schema improves the quality and interoperability of Brazilian legal data, fosters integration within the global OGD ecosystem, and facilitates the creation of a wor 

**Abstract (ZH)**: 开放政府数据（OGD）倡议旨在通过使政府数据公开 accessible 提高透明度和公众参与度。然而，为机器可读性制定法律规范仍然是推进如法律知识图谱（LKGs）之类的法律科技应用的关键挑战。基于巴西联邦议会的 this http URL 项目，我们提出了一种将巴西立法统一映射到 this http URL 词汇表的方法，通过 JSON-LD 和 Linked Data。我们的方法涵盖了概念性的“Norm”实体（映射到 sdo:Legislation）及其数字出版物或表现形式（映射到 sdo:LegislationObject）。我们为每种类型详细列出了关键属性，提供了具体的示例，并考虑了 LexML 标准的 URN 标识符、多语言支持、官方公报中的版本控制以及相互之间的关系（如引用和参考）。我们的结构化模式提高了巴西法律数据的质量和互操作性，促进了与全球 OGD 生态系统的整合，并促进了法律知识图谱的创建。 

---
# EH-Benchmark Ophthalmic Hallucination Benchmark and Agent-Driven Top-Down Traceable Reasoning Workflow 

**Title (ZH)**: EH-Benchmark 视网膜幻觉基准和基于代理的自上而下可追溯推理工作流 

**Authors**: Xiaoyu Pan, Yang Bai, Ke Zou, Yang Zhou, Jun Zhou, Huazhu Fu, Yih-Chung Tham, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22929)  

**Abstract**: Medical Large Language Models (MLLMs) play a crucial role in ophthalmic diagnosis, holding significant potential to address vision-threatening diseases. However, their accuracy is constrained by hallucinations stemming from limited ophthalmic knowledge, insufficient visual localization and reasoning capabilities, and a scarcity of multimodal ophthalmic data, which collectively impede precise lesion detection and disease diagnosis. Furthermore, existing medical benchmarks fail to effectively evaluate various types of hallucinations or provide actionable solutions to mitigate them. To address the above challenges, we introduce EH-Benchmark, a novel ophthalmology benchmark designed to evaluate hallucinations in MLLMs. We categorize MLLMs' hallucinations based on specific tasks and error types into two primary classes: Visual Understanding and Logical Composition, each comprising multiple subclasses. Given that MLLMs predominantly rely on language-based reasoning rather than visual processing, we propose an agent-centric, three-phase framework, including the Knowledge-Level Retrieval stage, the Task-Level Case Studies stage, and the Result-Level Validation stage. Experimental results show that our multi-agent framework significantly mitigates both types of hallucinations, enhancing accuracy, interpretability, and reliability. Our project is available at this https URL. 

**Abstract (ZH)**: Medical大型语言模型（MLLMs）在眼科诊断中发挥着关键作用，具有解决致盲疾病的重要潜力。然而，它们的准确性受限于由眼科知识有限、视觉定位和推理能力不足以及多模态眼科数据稀缺导致的幻觉，这些因素共同阻碍了精确病灶检测和疾病诊断。此外，现有的医疗基准未能有效评估各种类型的幻觉或提供有效的解决措施。为了应对这些挑战，我们引入了EH-Benchmark，这是一种新型的眼科基准，旨在评估MLLMs中的幻觉。我们将MLLMs的幻觉根据不同任务和错误类型分为两类：视觉理解与逻辑组合，并分别包含多个子类别。鉴于MLLMs主要依赖基于语言的推理而非视觉处理，我们提出了一种以代理为中心的三阶段框架，包括知识级别检索阶段、任务级别案例研究阶段和结果级别验证阶段。实验结果显示，我们的多代理框架显著降低了两种类型的幻觉，提高了准确率、可解释性和可靠性。该项目可访问：这个链接。 

---
# Zero-Shot Temporal Interaction Localization for Egocentric Videos 

**Title (ZH)**: 零样本时空交互定位 for 同伴中心视频 

**Authors**: Erhang Zhang, Junyi Ma, Yin-Dong Zheng, Yixuan Zhou, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03662)  

**Abstract**: Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at this https URL. 

**Abstract (ZH)**: 基于视频的人机交互动作定位作为多个下游任务的基础，如人类行为分析和人机技能转移。当前的时间动作定位方法通常依赖于交互中的标注动作和对象类别进行优化，这导致了领域偏差和低部署效率。虽然一些近期的工作利用大规模的视觉-语言模型实现了零样本时间动作定位（ZS-TAL），但它们粗粒度的估计和开环的管道阻碍了时间交互定位（TIL）性能的进一步提升。为了解决这些问题，我们提出了一种新颖的零样本TIL方法，名为EgoLoc，以在第一人称视频中定位人类对象交互中的抓取动作的时间。EgoLoc引入了一种自适应采样策略，以生成合理的视觉提示供视觉-语言模型推理。通过吸收二维和三维观察结果，它根据三维手部速度直接在HOI可能的接触/分离时间戳周围采样高质量的初始猜测，从而提高了推理准确性和效率。此外，EgoLoc从视觉和动态线索中生成闭环反馈进一步细化定位结果。在公开的数据集和我们新提出的基准上的全面实验表明，EgoLoc在第一人称视频的时间交互定位方面优于最先进的基线方法。我们将在此网址公开我们的代码和相关数据：这个网址。 

---
# Towards Actionable Pedagogical Feedback: A Multi-Perspective Analysis of Mathematics Teaching and Tutoring Dialogue 

**Title (ZH)**: 面向可行的教学反馈：数学教学与辅导对话的多视角分析 

**Authors**: Jannatun Naim, Jie Cao, Fareen Tasneem, Jennifer Jacobs, Brent Milne, James Martin, Tamara Sumner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07161)  

**Abstract**: Effective feedback is essential for refining instructional practices in mathematics education, and researchers often turn to advanced natural language processing (NLP) models to analyze classroom dialogues from multiple perspectives. However, utterance-level discourse analysis encounters two primary challenges: (1) multifunctionality, where a single utterance may serve multiple purposes that a single tag cannot capture, and (2) the exclusion of many utterances from domain-specific discourse move classifications, leading to their omission in feedback. To address these challenges, we proposed a multi-perspective discourse analysis that integrates domain-specific talk moves with dialogue act (using the flattened multi-functional SWBD-MASL schema with 43 tags) and discourse relation (applying Segmented Discourse Representation Theory with 16 relations). Our top-down analysis framework enables a comprehensive understanding of utterances that contain talk moves, as well as utterances that do not contain talk moves. This is applied to two mathematics education datasets: TalkMoves (teaching) and SAGA22 (tutoring). Through distributional unigram analysis, sequential talk move analysis, and multi-view deep dive, we discovered meaningful discourse patterns, and revealed the vital role of utterances without talk moves, demonstrating that these utterances, far from being mere fillers, serve crucial functions in guiding, acknowledging, and structuring classroom discourse. These insights underscore the importance of incorporating discourse relations and dialogue acts into AI-assisted education systems to enhance feedback and create more responsive learning environments. Our framework may prove helpful for providing human educator feedback, but also aiding in the development of AI agents that can effectively emulate the roles of both educators and students. 

**Abstract (ZH)**: 有效的反馈对于数学教育中的教学实践改进至关重要，研究人员常借助先进的自然语言处理（NLP）模型从多角度分析课堂对话。然而，话语单元层面的分析面临两大挑战：（1）多功能性，即一个话语单元可能包含多个无法用单一标签捕捉的目的；（2）许多话语单元因专属领域的话语移动分类排除在外，导致其在反馈中被忽略。为应对这些挑战，我们提出了一种多视角话语分析框架，该框架结合了专属领域的对话动作（采用扁平化的多功能性SWBD-MASL方案，包含43个标签）和话语关系（应用分段的话语表示理论，包含16种关系）。自上而下的分析框架能够全面理解包含话语移动和不包含话语移动的话语单元。我们将其应用于两个数学教育数据集：TalkMoves（教学）和SAGA22（辅导）。通过分布分析、序列对话动作分析和多视角深入分析，我们发现了有意义的话语模式，并揭示了不包含话语移动的话语单元的关键作用，证明这些话语单元远非仅仅是填充物，而是引导、认可和结构化课堂对话的重要手段。这些见解强调了将话语关系和对话动作纳入AI辅助教育系统以提升反馈并创造更具响应性的学习环境的重要性。我们的框架不仅有助于提供给人类教育者的反馈，还能协助开发能够有效模仿教育者和学生角色的AI代理。 

---
# Enhancing Talk Moves Analysis in Mathematics Tutoring through Classroom Teaching Discourse 

**Title (ZH)**: 通过课堂教学 discourse 提高数学辅导中谈话移动分析 

**Authors**: Jie Cao, Abhijit Suresh, Jennifer Jacobs, Charis Clevenger, Amanda Howard, Chelsea Brown, Brent Milne, Tom Fischaber, Tamara Sumner, James H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13395)  

**Abstract**: Human tutoring interventions play a crucial role in supporting student learning, improving academic performance, and promoting personal growth. This paper focuses on analyzing mathematics tutoring discourse using talk moves - a framework of dialogue acts grounded in Accountable Talk theory. However, scaling the collection, annotation, and analysis of extensive tutoring dialogues to develop machine learning models is a challenging and resource-intensive task. To address this, we present SAGA22, a compact dataset, and explore various modeling strategies, including dialogue context, speaker information, pretraining datasets, and further fine-tuning. By leveraging existing datasets and models designed for classroom teaching, our results demonstrate that supplementary pretraining on classroom data enhances model performance in tutoring settings, particularly when incorporating longer context and speaker information. Additionally, we conduct extensive ablation studies to underscore the challenges in talk move modeling. 

**Abstract (ZH)**: 人类辅导干预在支持学生学习、提高学术成绩和促进个人成长中发挥着重要作用。本文focus于利用对话动作——基于可问责对话理论的框架——分析数学辅导对话。然而，扩展收集、标注和分析大量辅导对话以开发机器学习模型是一项具有挑战性和资源密集的任务。为了解决这一问题，我们提出了SAGA22紧凑型数据集，并探索了包括对话上下文、演讲者信息、预训练数据集和进一步微调在内的各种建模策略。通过利用为课堂教学设计的现有数据集和模型，我们的结果表明，在辅导环境中，补充课堂数据的预训练可以提高模型性能，尤其是在结合更长的上下文和演讲者信息时。此外，我们进行了广泛的消融研究以强调谈话动作建模中的挑战。 

---
# Observing Dialogue in Therapy: Categorizing and Forecasting Behavioral Codes 

**Title (ZH)**: 在治疗中观察对话：行为编码的分类与预测 

**Authors**: Jie Cao, Michael Tanana, Zac E. Imel, Eric Poitras, David C. Atkins, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/1907.00326)  

**Abstract**: Automatically analyzing dialogue can help understand and guide behavior in domains such as counseling, where interactions are largely mediated by conversation. In this paper, we study modeling behavioral codes used to asses a psychotherapy treatment style called Motivational Interviewing (MI), which is effective for addressing substance abuse and related problems. Specifically, we address the problem of providing real-time guidance to therapists with a dialogue observer that (1) categorizes therapist and client MI behavioral codes and, (2) forecasts codes for upcoming utterances to help guide the conversation and potentially alert the therapist. For both tasks, we define neural network models that build upon recent successes in dialogue modeling. Our experiments demonstrate that our models can outperform several baselines for both tasks. We also report the results of a careful analysis that reveals the impact of the various network design tradeoffs for modeling therapy dialogue. 

**Abstract (ZH)**: 自动分析对话有助于理解并引导涉及咨询等领域的行为，其中对话主要由交流驱动。本文研究了用于评估一种有效的心理治疗风格（动机访谈）的行为代码建模方法，该风格适用于处理物质滥用及相关问题。具体来说，我们提出了一个对话观察者以实现实时指导，该观察者可以（1）分类治疗师和来访者使用的动机访谈行为代码，（2）预测未来言论的行为代码以帮助引导对话，并可能提醒治疗师。在两个任务中，我们定义了基于最近对话建模成功经验的神经网络模型。实验表明，我们的模型在两个任务上均优于若干基准。我们还报告了对各种网络设计权衡的仔细分析结果，揭示了其对治疗对话建模的影响。 

---
