# Do Chains-of-Thoughts of Large Language Models Suffer from Hallucinations, Cognitive Biases, or Phobias in Bayesian Reasoning? 

**Title (ZH)**: 大型语言模型的Chain-of-Thoughts会受到幻觉、认知偏见或贝叶斯推理恐惧症的影响吗？ 

**Authors**: Roberto Araya  

**Link**: [PDF](https://arxiv.org/pdf/2503.15268)  

**Abstract**: Learning to reason and carefully explain arguments is central to students' cognitive, mathematical, and computational thinking development. This is particularly challenging in problems under uncertainty and in Bayesian reasoning. With the new generation of large language models (LLMs) capable of reasoning using Chain-of-Thought (CoT), there is an excellent opportunity to learn with them as they explain their reasoning through a dialogue with their artificial internal voice. It is an engaging and excellent opportunity to learn Bayesian reasoning. Furthermore, given that different LLMs sometimes arrive at opposite solutions, CoT generates opportunities for deep learning by detailed comparisons of reasonings. However, unlike humans, we found that they do not autonomously explain using ecologically valid strategies like natural frequencies, whole objects, and embodied heuristics. This is unfortunate, as these strategies help humans avoid critical mistakes and have proven pedagogical value in Bayesian reasoning. In order to overcome these biases and aid understanding and learning, we included prompts that induce LLMs to use these strategies. We found that LLMs with CoT incorporate them but not consistently. They show persistent biases towards symbolic reasoning and avoidance or phobia of ecologically valid strategies. 

**Abstract (ZH)**: 学习推理和谨慎解释论证是学生认知、数学和计算思维发展中的核心。特别是在不确定性问题和贝叶斯推理中，这是一个尤为挑战性的任务。随着新一代大语言模型（LLMs）能够使用链式思考（CoT）进行推理，通过它们与人工内部语音的对话来解释其推理过程，学习贝叶斯推理成为一个引人入胜且优秀的契机。此外，由于不同的LLMs有时会得出完全相反的解决方案，CoT为通过仔细比较推理过程来进行深入学习提供了机会。然而，与人类不同，我们发现它们并未自主使用诸如自然频率、整体对象和体态启发法等生态有效策略进行解释。这令人遗憾，因为这些策略有助于人类避免关键错误，在贝叶斯推理中具有教学价值。为了克服这些偏见并促进理解和学习，我们加入了促使LLMs使用这些策略的提示。我们发现，具有CoT的LLMs可以融入这些策略，但不够一致。它们表现出持续倾向于符号推理，并避免或恐惧生态有效策略的偏见。 

---
# Aligning Crowd-sourced Human Feedback for Reinforcement Learning on Code Generation by Large Language Models 

**Title (ZH)**: 基于大规模语言模型的代码生成强化学习中众源人类反馈的对齐方法 

**Authors**: Man Fai Wong, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15129)  

**Abstract**: This paper studies how AI-assisted programming and large language models (LLM) improve software developers' ability via AI tools (LLM agents) like Github Copilot and Amazon CodeWhisperer, while integrating human feedback to enhance reinforcement learning (RLHF) with crowd-sourced computation to enhance text-to-code generation. Additionally, we demonstrate that our Bayesian optimization framework supports AI alignment in code generation by distributing the feedback collection burden, highlighting the value of collecting human feedback of good quality. Our empirical evaluations demonstrate the efficacy of this approach, showcasing how LLM agents can be effectively trained for improved text-to-code generation. Our Bayesian optimization framework can be designed for general domain-specific languages, promoting the alignment of large language model capabilities with human feedback in AI-assisted programming for code generation. 

**Abstract (ZH)**: 本文研究了AI辅助编程和大型语言模型（LLM）如何通过GitHub Copilot和Amazon CodeWhisperer等AI工具提升软件开发人员的能力，并结合人类反馈增强强化学习（RLHF），利用crowd-sourced计算提高文本到代码生成能力。此外，我们展示了我们的贝叶斯优化框架如何通过分散反馈收集负担来支持代码生成中的AI对齐，突显高质量人类反馈的价值。我们的实证评估证明了该方法的有效性，展示了如何有效训练LLM代理以提高文本到代码生成能力。我们的贝叶斯优化框架可以适用于通用领域特定语言，促进大型语言模型能力与AI辅助编程中的人类反馈在代码生成中的对齐。 

---
# Reasoning Effort and Problem Complexity: A Scaling Analysis in LLMs 

**Title (ZH)**: 推理努力与问题复杂性：大规模语言模型中的标度分析 

**Authors**: Benjamin Estermann, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2503.15113)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable text generation capabilities, and recent advances in training paradigms have led to breakthroughs in their reasoning performance. In this work, we investigate how the reasoning effort of such models scales with problem complexity. We use the infinitely scalable Tents puzzle, which has a known linear-time solution, to analyze this scaling behavior. Our results show that reasoning effort scales with problem size, but only up to a critical problem complexity. Beyond this threshold, the reasoning effort does not continue to increase, and may even decrease. This observation highlights a critical limitation in the logical coherence of current LLMs as problem complexity increases, and underscores the need for strategies to improve reasoning scalability. Furthermore, our results reveal significant performance differences between current state-of-the-art reasoning models when faced with increasingly complex logical puzzles. 

**Abstract (ZH)**: 大型语言模型的推理努力随着问题复杂性的增加按比例变化：基于无限可扩展Tents谜题的分析 

---
# TULIP: Towards Unified Language-Image Pretraining 

**Title (ZH)**: TULIP: 向统一的语言-图像预训练目标迈进 

**Authors**: Zineng Tang, Long Lian, Seun Eisape, XuDong Wang, Roei Herzig, Adam Yala, Alane Suhr, Trevor Darrell, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15485)  

**Abstract**: Despite the recent success of image-text contrastive models like CLIP and SigLIP, these models often struggle with vision-centric tasks that demand high-fidelity image understanding, such as counting, depth estimation, and fine-grained object recognition. These models, by performing language alignment, tend to prioritize high-level semantics over visual understanding, weakening their image understanding. On the other hand, vision-focused models are great at processing visual information but struggle to understand language, limiting their flexibility for language-driven tasks. In this work, we introduce TULIP, an open-source, drop-in replacement for existing CLIP-like models. Our method leverages generative data augmentation, enhanced image-image and text-text contrastive learning, and image/text reconstruction regularization to learn fine-grained visual features while preserving global semantic alignment. Our approach, scaling to over 1B parameters, outperforms existing state-of-the-art (SOTA) models across multiple benchmarks, establishing a new SOTA zero-shot performance on ImageNet-1K, delivering up to a $2\times$ enhancement over SigLIP on RxRx1 in linear probing for few-shot classification, and improving vision-language models, achieving over $3\times$ higher scores than SigLIP on MMVP. Our code/checkpoints are available at this https URL 

**Abstract (ZH)**: 尽管近日图像-文本对比模型如CLIP和SigLIP取得了成功，但这些模型在要求高保真图像理解的任务（如计数、深度估计和细粒度对象识别）中经常表现不佳。通过进行语言对齐，这些模型倾向于优先处理高层语义，而削弱了其图像理解能力。另一方面，专注于视觉的模型擅长处理视觉信息，但在理解和处理语言方面存在局限性，限制了它们在语言驱动任务中的灵活性。在本研究中，我们提出了TULIP，这是一个开源的即插即用替代现有CLIP类似模型的方法。我们的方法利用生成的数据增强、增强的图像-图像和文本-文本对比学习以及图像/文本重建正则化来学习细粒度的视觉特征，同时保持全局语义对齐。我们的方法在超过10亿参数的规模下，在多个基准测试中超过了现有最先进的（SOTA）模型，并在ImageNet-1K上建立了新的SOTA零样本性能，在RxRx1上实现了高达2倍的线性探针少样本分类性能提升，以及在MMVP上实现了SigLIP超过3倍的评分改进。我们的代码/检查点可在以下链接获取。 

---
# What Makes a Reward Model a Good Teacher? An Optimization Perspective 

**Title (ZH)**: 奖励模型成为一个好老师的原因：从优化的角度探讨 

**Authors**: Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D. Lee, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2503.15477)  

**Abstract**: The success of Reinforcement Learning from Human Feedback (RLHF) critically depends on the quality of the reward model. While this quality is primarily evaluated through accuracy, it remains unclear whether accuracy fully captures what makes a reward model an effective teacher. We address this question from an optimization perspective. First, we prove that regardless of how accurate a reward model is, if it induces low reward variance, then the RLHF objective suffers from a flat landscape. Consequently, even a perfectly accurate reward model can lead to extremely slow optimization, underperforming less accurate models that induce higher reward variance. We additionally show that a reward model that works well for one language model can induce low reward variance, and thus a flat objective landscape, for another. These results establish a fundamental limitation of evaluating reward models solely based on accuracy or independently of the language model they guide. Experiments using models of up to 8B parameters corroborate our theory, demonstrating the interplay between reward variance, accuracy, and reward maximization rate. Overall, our findings highlight that beyond accuracy, a reward model needs to induce sufficient variance for efficient optimization. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）的成功关键取决于奖励模型的质量。从优化的角度回答这一问题，我们证明，无论奖励模型的准确性如何，如果它引起低奖励方差，那么RLHF目标将遭受平坦的景观问题。因此，即使一个完全准确的奖励模型也可能导致极其缓慢的优化，而不如一些准确度较低但引起更高奖励方差的模型表现得好。此外，我们还证明一个适用于一个语言模型的奖励模型可能会为另一个语言模型引起低奖励方差，从而导致平坦的目标景观。这些结果确立了仅根据准确性和独立于所指导的语言模型来评估奖励模型的基本限制。使用多达8B参数的模型进行的实验支持我们的理论，展示了奖励方差、准确性和奖励最大化率之间的相互作用。总体而言，我们的研究结果强调，除了准确性外，奖励模型还需要引起足够的方差以实现高效的优化。 

---
# From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment 

**Title (ZH)**: 从1,000,000用户到每位用户：面向用户的个性化偏好扩展 

**Authors**: Jia-Nan Li, Jian Guan, Songhao Wu, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15463)  

**Abstract**: Large language models (LLMs) have traditionally been aligned through one-size-fits-all approaches that assume uniform human preferences, fundamentally overlooking the diversity in user values and needs. This paper introduces a comprehensive framework for scalable personalized alignment of LLMs. We establish a systematic preference space characterizing psychological and behavioral dimensions, alongside diverse persona representations for robust preference inference in real-world scenarios. Building upon this foundation, we introduce \textsc{AlignX}, a large-scale dataset of over 1.3 million personalized preference examples, and develop two complementary alignment approaches: \textit{in-context alignment} directly conditioning on persona representations and \textit{preference-bridged alignment} modeling intermediate preference distributions. Extensive experiments demonstrate substantial improvements over existing methods, with an average 17.06\% accuracy gain across four benchmarks while exhibiting a strong adaptation capability to novel preferences, robustness to limited user data, and precise preference controllability. These results validate our framework's effectiveness, advancing toward truly user-adaptive AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）的传统对齐方法采用一刀切的方式，假设人类偏好一致，从根本上忽视了用户价值观和需求的多样性。本文提出了一种全面的框架以实现可扩展的个性化LLM对齐。我们建立了一个系统化的偏好空间，刻画了心理和行为维度，并结合多样化的人物角色表示，以实现稳健的偏好推断。在此基础上，我们引入了\textsc{AlignX}，这是一个包含超过130万个性化偏好示例的大型数据集，并开发了两种互补的对齐方法：基于人物角色表示的上下文对齐和基于偏好桥梁的对齐方法。广泛的实验结果表明，与现有方法相比取得了显著改进，平均在四个基准测试中准确性提升17.06%，同时具有强大的新偏好适应能力和对有限用户数据的鲁棒性，以及精确的偏好可控性。这些结果验证了我们框架的有效性，推动了真正用户自适应AI系统的进步。 

---
# Probing the topology of the space of tokens with structured prompts 

**Title (ZH)**: 探究令牌空间的拓扑结构通过结构化提示 

**Authors**: Michael Robinson, Sourya Dey, Taisa Kushner  

**Link**: [PDF](https://arxiv.org/pdf/2503.15421)  

**Abstract**: This article presents a general and flexible method for prompting a large language model (LLM) to reveal its (hidden) token input embedding up to homeomorphism. Moreover, this article provides strong theoretical justification -- a mathematical proof for generic LLMs -- for why this method should be expected to work. With this method in hand, we demonstrate its effectiveness by recovering the token subspace of Llemma-7B. The results of this paper apply not only to LLMs but also to general nonlinear autoregressive processes. 

**Abstract (ZH)**: 本文提出了一种通用且灵活的方法，用于提示大型语言模型（LLM）通过同胚变换揭示其（隐藏的）标记输入嵌入。此外，本文提供了强大的理论证明——对通用LLM的数学证明——说明为何这种方法预期能够奏效。借助此方法，我们通过对Llemma-7B的标记子空间进行恢复，展示了其有效性。本文的结果不仅适用于LLM，还适用于一般非线性自回归过程。 

---
# Real-world validation of a multimodal LLM-powered pipeline for High-Accuracy Clinical Trial Patient Matching leveraging EHR data 

**Title (ZH)**: 基于EHR数据的多模态LLM赋能高精度临床试验患者匹配pipeline的现实世界验证 

**Authors**: Anatole Callies, Quentin Bodinier, Philippe Ravaud, Kourosh Davarpanah  

**Link**: [PDF](https://arxiv.org/pdf/2503.15374)  

**Abstract**: Background: Patient recruitment in clinical trials is hindered by complex eligibility criteria and labor-intensive chart reviews. Prior research using text-only models have struggled to address this problem in a reliable and scalable way due to (1) limited reasoning capabilities, (2) information loss from converting visual records to text, and (3) lack of a generic EHR integration to extract patient data.
Methods: We introduce a broadly applicable, integration-free, LLM-powered pipeline that automates patient-trial matching using unprocessed documents extracted from EHRs. Our approach leverages (1) the new reasoning-LLM paradigm, enabling the assessment of even the most complex criteria, (2) visual capabilities of latest LLMs to interpret medical records without lossy image-to-text conversions, and (3) multimodal embeddings for efficient medical record search. The pipeline was validated on the n2c2 2018 cohort selection dataset (288 diabetic patients) and a real-world dataset composed of 485 patients from 30 different sites matched against 36 diverse trials.
Results: On the n2c2 dataset, our method achieved a new state-of-the-art criterion-level accuracy of 93\%. In real-world trials, the pipeline yielded an accuracy of 87\%, undermined by the difficulty to replicate human decision-making when medical records lack sufficient information. Nevertheless, users were able to review overall eligibility in under 9 minutes per patient on average, representing an 80\% improvement over traditional manual chart reviews.
Conclusion: This pipeline demonstrates robust performance in clinical trial patient matching without requiring custom integration with site systems or trial-specific tailoring, thereby enabling scalable deployment across sites seeking to leverage AI for patient matching. 

**Abstract (ZH)**: 背景：临床试验中患者的招募受到了复杂入选标准和劳动密集型病历审查的阻碍。由于（1）有限的推理能力，（2）将视觉记录转换为文本时的信息损失，以及（3）缺乏通用的电子健康记录（EHR）集成来提取患者数据，先前仅使用文本模型的研究难以可靠且可扩展地解决这一问题。

方法：我们引入了一种广泛适用的、无需集成的、由大规模语言模型（LLM）驱动的pipeline，该pipeline利用未处理的从EHR提取的文档自动完成患者-试验匹配。我们的方法利用了（1）新的推理LLM范式，能够评估最复杂的标准；（2）最新LLM的视觉能力，能够在不进行损失性的图像到文本转换的情况下解释医疗记录；以及（3）多模态嵌入，用于高效地搜索医疗记录。该pipeline在n2c2 2018队列选择数据集（288名糖尿病患者）和一个由来自30个不同地点的485名患者组成的实际数据集（与36项不同试验匹配）上进行了验证。

结果：在n2c2数据集上，该方法实现了新的最佳标准级别准确率93%。在实际临床试验中，pipeline的准确率为87%，由于医疗记录缺乏充分信息而使人工决策复制变得困难。然而，用户仍然能够平均在每名患者不到9分钟内审查总体入组资格，这比传统的手动病历审查提高了80%。

结论：该pipeline在无需对站点系统进行定制集成或针对特定试验进行调整的情况下展示了在临床试验患者匹配中的稳健性能，从而使得使用AI进行患者匹配的部署能够在寻求利用AI的各个站点中实现可扩展性。 

---
# MAMM-Refine: A Recipe for Improving Faithfulness in Generation with Multi-Agent Collaboration 

**Title (ZH)**: MAMM-Refine: 多智能体合作改进生成忠实度的方法 

**Authors**: David Wan, Justin Chih-Yao Chen, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.15272)  

**Abstract**: Multi-agent collaboration among models has shown promise in reasoning tasks but is underexplored in long-form generation tasks like summarization and question-answering. We extend multi-agent multi-model reasoning to generation, specifically to improving faithfulness through refinement, i.e., revising model-generated outputs to remove factual inconsistencies. We investigate how iterative collaboration among multiple instances and types of large language models (LLMs) enhances subtasks in the refinement process, such as error detection, critiquing unfaithful sentences, and making corrections based on critiques. We design intrinsic evaluations for each subtask, with our findings indicating that both multi-agent (multiple instances) and multi-model (diverse LLM types) approaches benefit error detection and critiquing. Additionally, reframing critiquing and refinement as reranking rather than generation tasks improves multi-agent performance. We consolidate these insights into a final "recipe" called Multi-Agent Multi-Model Refinement (MAMM-Refine), where multi-agent and multi-model collaboration significantly boosts performance on three summarization datasets as well as on long-form question answering, demonstrating the effectiveness and generalizability of our recipe. 

**Abstract (ZH)**: 多agent多模型协作在生成任务中的推理研究：提升忠实性通过精炼 

---
# Automated Non-Functional Requirements Generation in Software Engineering with Large Language Models: A Comparative Study 

**Title (ZH)**: 使用大型语言模型在软件工程中自动生成非功能需求：一项比较研究 

**Authors**: Jomar Thomas Almonte, Santhosh Anitha Boominathan, Nathalia Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2503.15248)  

**Abstract**: Neglecting non-functional requirements (NFRs) early in software development can lead to critical challenges. Despite their importance, NFRs are often overlooked or difficult to identify, impacting software quality. To support requirements engineers in eliciting NFRs, we developed a framework that leverages Large Language Models (LLMs) to derive quality-driven NFRs from functional requirements (FRs). Using a custom prompting technique within a Deno-based pipeline, the system identifies relevant quality attributes for each functional requirement and generates corresponding NFRs, aiding systematic integration. A crucial aspect is evaluating the quality and suitability of these generated requirements. Can LLMs produce high-quality NFR suggestions? Using 34 functional requirements - selected as a representative subset of 3,964 FRs-the LLMs inferred applicable attributes based on the ISO/IEC 25010:2023 standard, generating 1,593 NFRs. A horizontal evaluation covered three dimensions: NFR validity, applicability of quality attributes, and classification precision. Ten industry software quality evaluators, averaging 13 years of experience, assessed a subset for relevance and quality. The evaluation showed strong alignment between LLM-generated NFRs and expert assessments, with median validity and applicability scores of 5.0 (means: 4.63 and 4.59, respectively) on a 1-5 scale. In the classification task, 80.4% of LLM-assigned attributes matched expert choices, with 8.3% near misses and 11.3% mismatches. A comparative analysis of eight LLMs highlighted variations in performance, with gemini-1.5-pro exhibiting the highest attribute accuracy, while llama-3.3-70B achieved higher validity and applicability scores. These findings provide insights into the feasibility of using LLMs for automated NFR generation and lay the foundation for further exploration of AI-assisted requirements engineering. 

**Abstract (ZH)**: 忽视软件开发早期的功能需求非功能性要求（NFRs）可能导致关键挑战。尽管NFRs很重要，但它们往往被忽视或难以识别，影响软件质量。为了支持需求工程师提取NFRs，我们开发了一个框架，利用大型语言模型（LLMs）从功能需求（FRs）中推导出质量驱动的NFRs。通过一个基于Deno的流水线中的定制提示技术，系统识别每个功能需求的相关质量属性并生成相应的NFRs，促进系统的集成。一个关键方面是评估这些生成需求的质量和适用性。大型语言模型能否生成高质量的NFR建议？使用34个功能需求——作为3,964个FRs的一个代表性子集——LLMs根据ISO/IEC 25010:2023标准推断适用的属性，生成1,593个NFRs。横向评估涵盖了三个维度：NFR有效性、质量属性的适用性以及分类精度。十名拥有平均13年经验的行业软件质量评估员评估了其中一部分的相关性和质量。评估结果显示LLM生成的NFR与专家评估之间有很强的契合度，中位有效性和适用性分数分别为5.0（平均分别为4.63和4.59）。在分类任务中，80.4%的LLM分配的属性与专家选择匹配，8.3%为接近匹配，11.3%为不匹配。对于八种大型语言模型的比较分析揭示了性能上的差异，gemini-1.5-pro在属性准确性上表现最佳，而llama-3.3-70B在有效性和适用性评分上更高。这些发现为使用大型语言模型进行自动化NFR生成提供了见解，并为AI辅助需求工程的进一步探索奠定了基础。 

---
# BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity? 

**Title (ZH)**: BigO(基准)——大型语言模型能否生成受控时间与空间复杂度的代码？ 

**Authors**: Pierre Chambon, Baptiste Roziere, Benoit Sagot, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2503.15242)  

**Abstract**: We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time. 

**Abstract (ZH)**: BigO(Bench): 一种用于评估生成语言模型在理解与生成具有指定时空复杂度代码方面能力的新基准 

---
# Exploring Large Language Models for Word Games:Who is the Spy? 

**Title (ZH)**: 探索大型语言模型在词游中的应用：谁是间谍？ 

**Authors**: Chentian Wei, Jiewei Chen, Jinzhu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15235)  

**Abstract**: Word games hold significant research value for natural language processing (NLP), game theory, and related fields due to their rule-based and situational nature. This study explores how large language models (LLMs) can be effectively involved in word games and proposes a training-free framework. "Shei Shi Wo Di" or "Who is the Spy" in English, is a classic word game. Using this game as an example, we introduce a Chain-of-Thought (CoT)-based scheduling framework to enable LLMs to achieve excellent performance in tasks such as inferring role words and disguising their identities. We evaluate the framework's performance based on game success rates and the accuracy of the LLM agents' analytical results. Experimental results affirm the framework's effectiveness, demonstrating notable improvements in LLM performance across multiple datasets. This work highlights the potential of LLMs in mastering situational reasoning and social interactions within structured game environments. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于规则和情境的词游游戏对于自然语言处理（NLP）、博弈论及相关领域的研究具有重要研究价值。本研究探讨大型语言模型（LLMs）在词游游戏中的有效应用，并提出了一种无需训练的框架。“谁是卧底”是一种经典的词游游戏。通过该游戏为例，我们介绍了一种基于Chain-of-Thought（CoT）的调度框架，以使LLMs在推断角色词和伪装身份等任务中表现出色。我们根据游戏成功率和LLMs代理分析结果的准确性评估该框架的性能。实验结果证实了该框架的有效性，展示了在多个数据集上LLMs性能的显著提升。本工作突显了LLMs在掌握结构化游戏环境中情境推理和社会互动方面的潜力。相关代码已在以下网址公开：此httpsURL。 

---
# Foundation models may exhibit staged progression in novel CBRN threat disclosure 

**Title (ZH)**: 基础模型可能在新的CBRN威胁披露中表现出阶段性的进展 

**Authors**: Kevin M Esvelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.15182)  

**Abstract**: The extent to which foundation models can disclose novel chemical, biological, radiation, and nuclear (CBRN) threats to expert users is unclear due to a lack of test cases. I leveraged the unique opportunity presented by an upcoming publication describing a novel catastrophic biothreat - "Technical Report on Mirror Bacteria: Feasibility and Risks" - to conduct a small controlled study before it became public. Graduate-trained biologists tasked with predicting the consequences of releasing mirror E. coli showed no significant differences in rubric-graded accuracy using Claude Sonnet 3.5 new (n=10) or web search only (n=2); both groups scored comparably to a web baseline (28 and 43 versus 36). However, Sonnet reasoned correctly when prompted by a report author, but a smaller model, Haiku 3.5, failed even with author guidance (80 versus 5). These results suggest distinct stages of model capability: Haiku is unable to reason about mirror life even with threat-aware expert guidance (Stage 1), while Sonnet correctly reasons only with threat-aware prompting (Stage 2). Continued advances may allow future models to disclose novel CBRN threats to naive experts (Stage 3) or unskilled users (Stage 4). While mirror life represents only one case study, monitoring new models' ability to reason about privately known threats may allow protective measures to be implemented before widespread disclosure. 

**Abstract (ZH)**: 基于基础模型对专家用户披露新型化学、生物、辐射和核（CBRN）威胁能力的研究：一项基于即将发布的新型灾难性生物威胁技术报告的小规模受控研究 

---
# Comparing Llama3 and DeepSeekR1 on Biomedical Text Classification Tasks 

**Title (ZH)**: 比较Llama3和DeepSeekR1在生物医学文本分类任务中的性能 

**Authors**: Yuting Guo, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2503.15169)  

**Abstract**: This study compares the performance of two open-source large language models (LLMs)-Llama3-70B and DeepSeekR1-distill-Llama3-70B-on six biomedical text classification tasks. Four tasks involve data from social media, while two tasks focus on clinical notes from electronic health records, and all experiments were performed in zero-shot settings. Performance metrics, including precision, recall, and F1 scores, were measured for each task, along with their 95% confidence intervals. Results demonstrated that DeepSeekR1-distill-Llama3-70B generally performs better in terms of precision on most tasks, with mixed results on recall. While the zero-shot LLMs demonstrated high F1 scores for some tasks, they grossly underperformed on others, for data from both sources. The findings suggest that model selection should be guided by the specific requirements of the health-related text classification tasks, particularly when considering the precision-recall trade-offs, and that, in the presence of annotated data, supervised classification approaches may be more reliable than zero-shot LLMs. 

**Abstract (ZH)**: 本研究比较了两个开源大规模语言模型（LLM）——Llama3-70B和DeepSeekR1-distill-Llama3-70B在六项生物医学文本分类任务中的性能。四项任务涉及社交媒体数据，而两项任务则专注于电子健康记录中的临床笔记，所有实验均在零样本设置下进行。测量了每个任务的精确度、召回率和F1分数，以及它们的95%置信区间。结果显示，DeepSeekR1-distill-Llama3-70B在大多数任务中通常在精确度方面表现更好，召回率的表现则参差不齐。虽然零样本LLM在某些任务上表现出高F1分数，但在其他任务上，无论是哪种数据源，它们的表现都非常糟糕。研究结果表明，在选择模型时应根据健康相关文本分类任务的具体要求进行，尤其是在考虑精确度与召回率之间的权衡时，而且在有标注数据的情况下，监督分类方法可能比零样本LLM更加可靠。 

---
# Increasing the Robustness of the Fine-tuned Multilingual Machine-Generated Text Detectors 

**Title (ZH)**: 增强细调多语言机器生成文本检测器的鲁棒性 

**Authors**: Dominik Macko, Robert Moro, Ivan Srba  

**Link**: [PDF](https://arxiv.org/pdf/2503.15128)  

**Abstract**: Since the proliferation of LLMs, there have been concerns about their misuse for harmful content creation and spreading. Recent studies justify such fears, providing evidence of LLM vulnerabilities and high potential of their misuse. Humans are no longer able to distinguish between high-quality machine-generated and authentic human-written texts. Therefore, it is crucial to develop automated means to accurately detect machine-generated content. It would enable to identify such content in online information space, thus providing an additional information about its credibility. This work addresses the problem by proposing a robust fine-tuning process of LLMs for the detection task, making the detectors more robust against obfuscation and more generalizable to out-of-distribution data. 

**Abstract (ZH)**: 自大规模语言模型的兴起以来，人们对其用于有害内容创作和传播的滥用表示担忧。近期研究证实了这种担忧，提供了关于语言模型漏洞及其滥用潜力的证据。人类现在难以区分高质量的机器生成文本和真实的人类撰写的文本。因此，开发准确检测机器生成内容的自动化方法至关重要。这将有助于在在线信息空间中识别此类内容，从而提供其可信度的额外信息。本工作通过提出一种稳健的大型语言模型微调过程来解决这个问题，使检测器更 robust 地抵抗混淆，并更具泛化性以应对分布外数据。 

---
# Towards Understanding the Safety Boundaries of DeepSeek Models: Evaluation and Findings 

**Title (ZH)**: 探索深Seek模型的安全边界：评估与发现 

**Authors**: Zonghao Ying, Guangyi Zheng, Yongxin Huang, Deyue Zhang, Wenxin Zhang, Quanchen Zou, Aishan Liu, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15092)  

**Abstract**: This study presents the first comprehensive safety evaluation of the DeepSeek models, focusing on evaluating the safety risks associated with their generated content. Our evaluation encompasses DeepSeek's latest generation of large language models, multimodal large language models, and text-to-image models, systematically examining their performance regarding unsafe content generation. Notably, we developed a bilingual (Chinese-English) safety evaluation dataset tailored to Chinese sociocultural contexts, enabling a more thorough evaluation of the safety capabilities of Chinese-developed models. Experimental results indicate that despite their strong general capabilities, DeepSeek models exhibit significant safety vulnerabilities across multiple risk dimensions, including algorithmic discrimination and sexual content. These findings provide crucial insights for understanding and improving the safety of large foundation models. Our code is available at this https URL. 

**Abstract (ZH)**: 本研究首次全面评估了DeepSeek模型的安全性，重点关注其生成内容相关的安全风险。我们的评估涵盖了DeepSeek最新一代的大语言模型、多模态大语言模型和文本到图像模型，系统地检查了它们在产生不当内容方面的性能。值得注意的是，我们开发了一个适用于中国社会文化背景的双语（中文-英文）安全评估数据集，使对中文开发模型的安全能力进行更全面的评估成为可能。实验结果表明，尽管DeepSeek模型具有强大的通用能力，但在多个安全风险维度上仍表现出显著的安全漏洞，包括算法歧视和性内容。这些发现为理解并改进大型基础模型的安全性提供了重要见解。我们的代码可在以下链接获取。 

---
# MASS: Mathematical Data Selection via Skill Graphs for Pretraining Large Language Models 

**Title (ZH)**: MASS：通过技能图进行数学数据选择以预训练大规模语言模型 

**Authors**: Jiazheng Li, Lu Yu, Qing Cui, Zhiqiang Zhang, Jun Zhou, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14917)  

**Abstract**: High-quality data plays a critical role in the pretraining and fine-tuning of large language models (LLMs), even determining their performance ceiling to some degree. Consequently, numerous data selection methods have been proposed to identify subsets of data that can effectively and efficiently enhance model performance. However, most of these methods focus on general data selection and tend to overlook the specific nuances of domain-related data. In this paper, we introduce MASS, a \textbf{MA}thematical data \textbf{S}election framework using the \textbf{S}kill graph for pretraining LLMs in the mathematical reasoning domain. By taking into account the unique characteristics of mathematics and reasoning, we construct a skill graph that captures the mathematical skills and their interrelations from a reference dataset. This skill graph guides us in assigning quality scores to the target dataset, enabling us to select the top-ranked subset which is further used to pretrain LLMs. Experimental results demonstrate the efficiency and effectiveness of MASS across different model sizes (1B and 7B) and pretraining datasets (web data and synthetic data). Specifically, in terms of efficiency, models trained on subsets selected by MASS can achieve similar performance to models trained on the original datasets, with a significant reduction in the number of trained tokens - ranging from 50\% to 70\% fewer tokens. In terms of effectiveness, when trained on the same amount of tokens, models trained on the data selected by MASS outperform those trained on the original datasets by 3.3\% to 5.9\%. These results underscore the potential of MASS to improve both the efficiency and effectiveness of pretraining LLMs. 

**Abstract (ZH)**: 数学数据选择框架MASS：基于技能图的大型语言模型数学推理领域预训练数据分析 

---
# Deep Contrastive Unlearning for Language Models 

**Title (ZH)**: 深度对比去学习语言模型 

**Authors**: Estrid He, Tabinda Sarwar, Ibrahim Khalil, Xun Yi, Ke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14900)  

**Abstract**: The past a few years have witnessed the great success of large language models, demonstrating powerful capabilities in comprehending textual data and generating human-like languages. Large language models achieve success by being trained on vast amounts of textual data, including online sources with copyrighted content and user-generated knowledge. However, this comes at a cost: the potential risk of exposing users' privacy and violating copyright protections. Thus, to safeguard individuals' "right to be forgotten", there has been increasing interests in machine unlearning -- the process of removing information carried by particular training samples from a model while not deteriorating its predictive quality. This is a challenging task due to the black-box nature of language models. Most existing studies focus on mitigating the impact of those forgot samples upon a model's outputs, and do not explicitly consider the geometric distributions of samples in the latent space of a model. To address this issue, we propose a machine unlearning framework, named Deep Contrastive Unlearning for fine-Tuning (DeepCUT) language models. Our proposed model achieves machine unlearning by directly optimizing the latent space of a model. Comprehensive experiments on real-world datasets demonstrate the effectiveness and efficiency of DeepCUT with consistent and significant improvement over baseline methods. 

**Abstract (ZH)**: 过去几年大型语言模型取得了巨大成功，展示了其在理解和生成类人类语言方面的强大能力。大型语言模型通过训练大量文本数据实现成功，包括包含版权内容的在线来源和用户生成的知识。然而，这伴随着潜在的风险：泄露用户的隐私和违反版权保护。因此，为了保护个人的“被遗忘权”，机器遗忘——即从模型中移除特定训练样本信息而不降低其预测质量的过程——引起了越来越多的关注。这一任务由于语言模型的黑匣子性质而极具挑战性。现有大多数研究集中于减轻被遗忘样本对模型输出的影响，而没有明确考虑模型潜在空间中样本的几何分布。为了解决这一问题，我们提出了一种机器遗忘框架，名为用于微调的语言模型的深度对比遗忘（DeepCUT）。我们提出的方法通过直接优化模型的潜在空间实现机器遗忘。在真实世界数据集上的全面实验表明，DeepCUT在基线方法上具有显著的有效性和效率，提供了一致的改进。 

---
# Mitigating Object Hallucinations in MLLMs via Multi-Frequency Perturbations 

**Title (ZH)**: 通过多频率扰动缓解MLLMs中的对象幻觉 

**Authors**: Shuo Li, Jiajun Sun, Guodong Zheng, Xiaoran Fan, Yujiong Shen, Yi Lu, Zhiheng Xi, Yuming Yang, Wenming Tan, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14895)  

**Abstract**: Recently, multimodal large language models (MLLMs) have demonstrated remarkable performance in visual-language tasks. However, the authenticity of the responses generated by MLLMs is often compromised by object hallucinations. We identify that a key cause of these hallucinations is the model's over-susceptibility to specific image frequency features in detecting objects. In this paper, we introduce Multi-Frequency Perturbations (MFP), a simple, cost-effective, and pluggable method that leverages both low-frequency and high-frequency features of images to perturb visual feature representations and explicitly suppress redundant frequency-domain features during inference, thereby mitigating hallucinations. Experimental results demonstrate that our method significantly mitigates object hallucinations across various model architectures. Furthermore, as a training-time method, MFP can be combined with inference-time methods to achieve state-of-the-art performance on the CHAIR benchmark. 

**Abstract (ZH)**: 最近，多模态大型语言模型（MLLMs）在视觉语言任务中展现了卓越的表现。然而，MLLMs生成的响应真实性常常受到物体错觉的损害。我们发现这些错觉的一个关键原因是模型对检测物体时过于敏感于特定图像频率特征。在本文中，我们提出了一种简单、成本效益高且易嵌入的方法——多频谱扰动（MFP），该方法利用图像的低频和高频特征来扰动视觉特征表示，并在推理时明确抑制冗余的频域特征，从而减轻错觉。实验结果表明，我们的方法显著减轻了各种模型架构中的物体错觉。此外，作为一种训练时方法，MFP可以与推理时方法结合使用，在CHAIR基准测试上达到当前最佳性能。 

---
# MetaLadder: Ascending Mathematical Solution Quality via Analogical-Problem Reasoning Transfer 

**Title (ZH)**: MetaLadder: 通过类问题推理转移提升数学解决方案质量 

**Authors**: Honglin Lin, Zhuoshi Pan, Yu Li, Qizhi Pei, Xin Gao, Mengzhang Cai, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14891)  

**Abstract**: Large Language Models (LLMs) have demonstrated promising capabilities in solving mathematical reasoning tasks, leveraging Chain-of-Thought (CoT) data as a vital component in guiding answer generation. Current paradigms typically generate CoT and answers directly for a given problem, diverging from human problem-solving strategies to some extent. Humans often solve problems by recalling analogous cases and leveraging their solutions to reason about the current task. Inspired by this cognitive process, we propose \textbf{MetaLadder}, a novel framework that explicitly prompts LLMs to recall and reflect on meta-problems, those structurally or semantically analogous problems, alongside their CoT solutions before addressing the target problem. Additionally, we introduce a problem-restating mechanism to enhance the model's comprehension of the target problem by regenerating the original question, which further improves reasoning accuracy. Therefore, the model can achieve reasoning transfer from analogical problems, mimicking human-like "learning from examples" and generalization abilities. Extensive experiments on mathematical benchmarks demonstrate that our MetaLadder significantly boosts LLMs' problem-solving accuracy, largely outperforming standard CoT-based methods (\textbf{10.3\%} accuracy gain) and other methods. Our code and data has been released at this https URL. 

**Abstract (ZH)**: 大型语言模型通过显式提示回忆和反映元问题来提升数学推理能力：MetaLadder框架及其实验分析 

---
# Envisioning an AI-Enhanced Mental Health Ecosystem 

**Title (ZH)**: 设想一种增强心理健康的人工智能生态系统 

**Authors**: Kellie Yu Hui Sim, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2503.14883)  

**Abstract**: The rapid advancement of Large Language Models (LLMs), reasoning models, and agentic AI approaches coincides with a growing global mental health crisis, where increasing demand has not translated into adequate access to professional support, particularly for underserved populations. This presents a unique opportunity for AI to complement human-led interventions, offering scalable and context-aware support while preserving human connection in this sensitive domain. We explore various AI applications in peer support, self-help interventions, proactive monitoring, and data-driven insights, using a human-centred approach that ensures AI supports rather than replaces human interaction. However, AI deployment in mental health fields presents challenges such as ethical concerns, transparency, privacy risks, and risks of over-reliance. We propose a hybrid ecosystem where where AI assists but does not replace human providers, emphasising responsible deployment and evaluation. We also present some of our early work and findings in several of these AI applications. Finally, we outline future research directions for refining AI-enhanced interventions while adhering to ethical and culturally sensitive guidelines. 

**Abstract (ZH)**: 大型语言模型、推理模型和自主AI方法的快速发展 coincides with an increasing global mental health crisis, where growing demand has not translated into adequate access to professional support, particularly for underserved populations. This presents a unique opportunity for AI to complement human-led interventions, offering scalable and context-aware support while preserving human connection in this sensitive domain. We explore various AI applications in peer support, self-help interventions, proactive monitoring, and data-driven insights, using a human-centred approach that ensures AI supports rather than replaces human interaction. However, AI deployment in mental health fields presents challenges such as ethical concerns, transparency, privacy risks, and risks of over-reliance. We propose a hybrid ecosystem where AI assists but does not replace human providers, emphasizing responsible deployment and evaluation. We also present some of our early work and findings in several of these AI applications. Finally, we outline future research directions for refining AI-enhanced interventions while adhering to ethical and culturally sensitive guidelines. 

---
# MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models 

**Title (ZH)**: MMDT：解码多模态基础模型的可靠性和安全性 

**Authors**: Chejian Xu, Jiawei Zhang, Zhaorun Chen, Chulin Xie, Mintong Kang, Yujin Potter, Zhun Wang, Zhuowen Yuan, Alexander Xiong, Zidi Xiong, Chenhui Zhang, Lingzhi Yuan, Yi Zeng, Peiyang Xu, Chengquan Guo, Andy Zhou, Jeffrey Ziwei Tan, Xuandong Zhao, Francesco Pinto, Zhen Xiang, Yu Gai, Zinan Lin, Dan Hendrycks, Bo Li, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.14827)  

**Abstract**: Multimodal foundation models (MMFMs) play a crucial role in various applications, including autonomous driving, healthcare, and virtual assistants. However, several studies have revealed vulnerabilities in these models, such as generating unsafe content by text-to-image models. Existing benchmarks on multimodal models either predominantly assess the helpfulness of these models, or only focus on limited perspectives such as fairness and privacy. In this paper, we present the first unified platform, MMDT (Multimodal DecodingTrust), designed to provide a comprehensive safety and trustworthiness evaluation for MMFMs. Our platform assesses models from multiple perspectives, including safety, hallucination, fairness/bias, privacy, adversarial robustness, and out-of-distribution (OOD) generalization. We have designed various evaluation scenarios and red teaming algorithms under different tasks for each perspective to generate challenging data, forming a high-quality benchmark. We evaluate a range of multimodal models using MMDT, and our findings reveal a series of vulnerabilities and areas for improvement across these perspectives. This work introduces the first comprehensive and unique safety and trustworthiness evaluation platform for MMFMs, paving the way for developing safer and more reliable MMFMs and systems. Our platform and benchmark are available at this https URL. 

**Abstract (ZH)**: 多模态基础模型（MMFMs）在自动驾驶、 healthcare 和虚拟助理等多个应用中发挥着关键作用。然而，多项研究揭示了这些模型的漏洞，例如文本到图像模型生成不安全的内容。现有的多模态模型基准主要评估这些模型的有用性，或者仅关注公平性、隐私等有限视角。在本文中，我们提出了第一个统一平台 MMDT（多模态解码信任），旨在为 MMFMs 提供全方位的安全性和可信度评估。该平台从多个视角评估模型，包括安全性、幻想、公平性/偏向性、隐私、对抗鲁棒性和离群值外推能力。我们在每个视角下设计了多种评估场景和红队算法，生成具有挑战性的数据，形成高质量的基准。我们使用 MMDT 评估了多种多模态模型，并发现了一系列跨视角的漏洞和改进领域。本研究引入了第一个全面且独特的 MMFMs 安全性和可信度评估平台，为开发更安全和更可靠的 MMFMs 和系统铺平了道路。我们的平台和基准可在以下网址获得：this https URL。 

---
# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving 

**Title (ZH)**: RAGO：检索增强生成服务的系统性能优化 

**Authors**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14649)  

**Abstract**: Retrieval-augmented generation (RAG), which combines large language models (LLMs) with retrievals from external knowledge databases, is emerging as a popular approach for reliable LLM serving. However, efficient RAG serving remains an open challenge due to the rapid emergence of many RAG variants and the substantial differences in workload characteristics across them. In this paper, we make three fundamental contributions to advancing RAG serving. First, we introduce RAGSchema, a structured abstraction that captures the wide range of RAG algorithms, serving as a foundation for performance optimization. Second, we analyze several representative RAG workloads with distinct RAGSchema, revealing significant performance variability across these workloads. Third, to address this variability and meet diverse performance requirements, we propose RAGO (Retrieval-Augmented Generation Optimizer), a system optimization framework for efficient RAG serving. Our evaluation shows that RAGO achieves up to a 2x increase in QPS per chip and a 55% reduction in time-to-first-token latency compared to RAG systems built on LLM-system extensions. 

**Abstract (ZH)**: 基于检索增强生成（RAG）的检索辅助生成（RAGSchema）抽象及其性能优化框架（RAGO） 

---
# Assessing Large Language Models for Automated Feedback Generation in Learning Programming Problem Solving 

**Title (ZH)**: 评估大规模语言模型在编程问题求解自动化反馈生成中的应用 

**Authors**: Priscylla Silva, Evandro Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.14630)  

**Abstract**: Providing effective feedback is important for student learning in programming problem-solving. In this sense, Large Language Models (LLMs) have emerged as potential tools to automate feedback generation. However, their reliability and ability to identify reasoning errors in student code remain not well understood. This study evaluates the performance of four LLMs (GPT-4o, GPT-4o mini, GPT-4-Turbo, and Gemini-1.5-pro) on a benchmark dataset of 45 student solutions. We assessed the models' capacity to provide accurate and insightful feedback, particularly in identifying reasoning mistakes. Our analysis reveals that 63\% of feedback hints were accurate and complete, while 37\% contained mistakes, including incorrect line identification, flawed explanations, or hallucinated issues. These findings highlight the potential and limitations of LLMs in programming education and underscore the need for improvements to enhance reliability and minimize risks in educational applications. 

**Abstract (ZH)**: 提供有效的反馈对于编程问题解决中的学生学习至关重要。在这种背景下，大型语言模型（LLMs）被视作自动反馈生成的潜在工具。然而，它们的可靠性和识别学生代码中的推理错误能力尚不明确。本研究评估了四种LLMs（GPT-4o、GPT-4o mini、GPT-4-Turbo和Gemini-1.5-pro）在包含45个学生解决方案的基准数据集上的性能。我们评估了模型提供准确且有洞察力的反馈的能力，特别是在识别推理错误方面的能力。我们的分析显示，63%的反馈提示是准确且完整的，而37%则包含错误，包括错误的行标识、不合理的解释或虚假的问题。这些发现凸显了LLMs在编程教育中的潜力和局限性，并强调了提高可靠性和减少教育应用中风险的必要性。 

---
# Robust Weight Imprinting: Insights from Neural Collapse and Proxy-Based Aggregation 

**Title (ZH)**: 稳健的权重印记：从神经崩溃和代理聚合中获得的见解 

**Authors**: Justus Westerhoff, Golzar Atefi, Mario Koddenbrock, Alexei Figueroa, Alexander Löser, Erik Rodner, Felix A. Gers  

**Link**: [PDF](https://arxiv.org/pdf/2503.14572)  

**Abstract**: The capacity of a foundation model allows for adaptation to new downstream tasks. Weight imprinting is a universal and efficient method to fulfill this purpose. It has been reinvented several times, but it has not been systematically studied. In this paper, we propose a framework for imprinting, identifying three main components: generation, normalization, and aggregation. This allows us to conduct an in-depth analysis of imprinting and a comparison of the existing work. We reveal the benefits of representing novel data with multiple proxies in the generation step and show the importance of proper normalization. We determine those proxies through clustering and propose a novel variant of imprinting that outperforms previous work. We motivate this by the neural collapse phenomenon -- an important connection that we can draw for the first time. Our results show an increase of up to 4% in challenging scenarios with complex data distributions for new classes. 

**Abstract (ZH)**: 基础模型的容量使其能够适应新的下游任务。权重印记是一种通用且高效的实现这一目标的方法。尽管它已被重新发明多次，但尚未进行系统性的研究。本文提出了一种印记框架，识别出生成、规范化和聚合三个主要组成部分，以开展对印记的深度分析和现有工作的比较。我们揭示了在生成步骤中使用多个代理表示新颖数据的好处，并强调了适当规范化的重要性。我们通过聚类确定这些代理，并提出了一种新的印记变体，其性能优于之前的工作。我们通过神经崩溃现象来阐述这一点，这是首次可以建立的重要联系。我们的结果表明，在复杂数据分布的新类场景中，性能最高可提升4%。 

---
# Policy Frameworks for Transparent Chain-of-Thought Reasoning in Large Language Models 

**Title (ZH)**: 大型语言模型中透明链式推理的政策框架 

**Authors**: Yihang Chen, Haikang Deng, Kaiqiao Han, Qingyue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.14521)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by decomposing complex problems into step-by-step solutions, improving performance on reasoning tasks. However, current CoT disclosure policies vary widely across different models in frontend visibility, API access, and pricing strategies, lacking a unified policy framework. This paper analyzes the dual-edged implications of full CoT disclosure: while it empowers small-model distillation, fosters trust, and enables error diagnosis, it also risks violating intellectual property, enabling misuse, and incurring operational costs. We propose a tiered-access policy framework that balances transparency, accountability, and security by tailoring CoT availability to academic, business, and general users through ethical licensing, structured reasoning outputs, and cross-tier safeguards. By harmonizing accessibility with ethical and operational considerations, this framework aims to advance responsible AI deployment while mitigating risks of misuse or misinterpretation. 

**Abstract (ZH)**: 全链推理披露的双刃剑影响及分级访问政策框架：促进负责任的AI部署的同时减轻滥用或误用风险 

---
