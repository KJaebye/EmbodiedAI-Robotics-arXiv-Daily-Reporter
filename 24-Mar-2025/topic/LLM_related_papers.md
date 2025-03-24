# Leveraging Language Models for Out-of-Distribution Recovery in Reinforcement Learning 

**Title (ZH)**: 利用语言模型进行强化学习中的分布外恢复 

**Authors**: Chan Kim, Seung-Woo Seo, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.17125)  

**Abstract**: Deep Reinforcement Learning (DRL) has demonstrated strong performance in robotic control but remains susceptible to out-of-distribution (OOD) states, often resulting in unreliable actions and task failure. While previous methods have focused on minimizing or preventing OOD occurrences, they largely neglect recovery once an agent encounters such states. Although the latest research has attempted to address this by guiding agents back to in-distribution states, their reliance on uncertainty estimation hinders scalability in complex environments. To overcome this limitation, we introduce Language Models for Out-of-Distribution Recovery (LaMOuR), which enables recovery learning without relying on uncertainty estimation. LaMOuR generates dense reward codes that guide the agent back to a state where it can successfully perform its original task, leveraging the capabilities of LVLMs in image description, logical reasoning, and code generation. Experimental results show that LaMOuR substantially enhances recovery efficiency across diverse locomotion tasks and even generalizes effectively to complex environments, including humanoid locomotion and mobile manipulation, where existing methods struggle. The code and supplementary materials are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于语言模型的离分布态恢复（LaMOuR）：无需不确定性估计的离分布态恢复方法 

---
# A Vehicle-Infrastructure Multi-layer Cooperative Decision-making Framework 

**Title (ZH)**: 基于车辆-基础设施多层协同决策框架 

**Authors**: Yiming Cui, Shiyu Fang, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.16552)  

**Abstract**: Autonomous driving has entered the testing phase, but due to the limited decision-making capabilities of individual vehicle algorithms, safety and efficiency issues have become more apparent in complex scenarios. With the advancement of connected communication technologies, autonomous vehicles equipped with connectivity can leverage vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) communications, offering a potential solution to the decision-making challenges from individual vehicle's perspective. We propose a multi-level vehicle-infrastructure cooperative decision-making framework for complex conflict scenarios at unsignalized intersections. First, based on vehicle states, we define a method for quantifying vehicle impacts and their propagation relationships, using accumulated impact to group vehicles through motif-based graph clustering. Next, within and between vehicle groups, a pass order negotiation process based on Large Language Models (LLM) is employed to determine the vehicle passage order, resulting in planned vehicle actions. Simulation results from ablation experiments show that our approach reduces negotiation complexity and ensures safer, more efficient vehicle passage at intersections, aligning with natural decision-making logic. 

**Abstract (ZH)**: 基于车辆-基础设施协同的无信号交叉口复杂冲突场景决策框架 

---
# Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models 

**Title (ZH)**: 向代理型推荐系统迈进：多模态大语言模型时代 

**Authors**: Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang, Ryan A. Rossi, Branislav Kveton, Dongruo Zhou, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16734)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to the emergence of agentic AI systems that extend beyond the capabilities of standalone models. By empowering LLMs to perceive external environments, integrate multimodal information, and interact with various tools, these agentic systems exhibit greater autonomy and adaptability across complex tasks. This evolution brings new opportunities to recommender systems (RS): LLM-based Agentic RS (LLM-ARS) can offer more interactive, context-aware, and proactive recommendations, potentially reshaping the user experience and broadening the application scope of RS. Despite promising early results, fundamental challenges remain, including how to effectively incorporate external knowledge, balance autonomy with controllability, and evaluate performance in dynamic, multimodal settings. In this perspective paper, we first present a systematic analysis of LLM-ARS: (1) clarifying core concepts and architectures; (2) highlighting how agentic capabilities -- such as planning, memory, and multimodal reasoning -- can enhance recommendation quality; and (3) outlining key research questions in areas such as safety, efficiency, and lifelong personalization. We also discuss open problems and future directions, arguing that LLM-ARS will drive the next wave of RS innovation. Ultimately, we foresee a paradigm shift toward intelligent, autonomous, and collaborative recommendation experiences that more closely align with users' evolving needs and complex decision-making processes. 

**Abstract (ZH)**: 近期大型语言模型的突破推动了具有自主能力的AI系统的出现，这些系统超越了单一模型的功能。通过赋予大型语言模型感知外部环境、整合多模态信息以及与各种工具交互的能力，这些自主系统在复杂任务中表现出更大的自主性和适应性。这种演变为推荐系统（RS）带来了新的机会：基于大型语言模型的自主推荐系统（LLM-ARS）可以提供更加互动、情境意识强且主动的推荐，有可能重塑用户体验并扩大推荐系统的应用范围。尽管早期结果颇具前景，但仍存在一些根本性的挑战，包括如何有效融入外部知识、如何平衡自主性和可控性以及如何在动态的多模态环境中评估性能。在本文中，我们首先对LLM-ARS进行了系统的分析：（1）阐明核心概念和架构；（2）强调自主能力如规划、记忆和多模态推理如何提高推荐质量；（3）概述关于安全性、效率和终生个性化等领域的关键研究问题。我们还讨论了开放性问题和未来方向，认为LLM-ARS将推动推荐系统创新的下一波浪潮。最终，我们预见到一种智能的、自主的和合作的推荐体验范式的转变，这种转变更能够与用户不断变化的需求和复杂的决策过程相契合。 

---
# Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning 

**Title (ZH)**: 通过临床经验学习提升大型语言模型代理的交互诊断能力 

**Authors**: Zhoujian Sun, Ziyi Liu, Cheng Luo, Jiebin Chu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16463)  

**Abstract**: Recent advances in large language models (LLMs) have shown promising results in medical diagnosis, with some studies indicating superior performance compared to human physicians in specific scenarios. However, the diagnostic capabilities of LLMs are often overestimated, as their performance significantly deteriorates in interactive diagnostic settings that require active information gathering. This study investigates the underlying mechanisms behind the performance degradation phenomenon and proposes a solution. We identified that the primary deficiency of LLMs lies in the initial diagnosis phase, particularly in information-gathering efficiency and initial diagnosis formation, rather than in the subsequent differential diagnosis phase. To address this limitation, we developed a plug-and-play method enhanced (PPME) LLM agent, leveraging over 3.5 million electronic medical records from Chinese and American healthcare facilities. Our approach integrates specialized models for initial disease diagnosis and inquiry into the history of the present illness, trained through supervised and reinforcement learning techniques. The experimental results indicate that the PPME LLM achieved over 30% improvement compared to baselines. The final diagnostic accuracy of the PPME LLM in interactive diagnostic scenarios approached levels comparable to those achieved using complete clinical data. These findings suggest a promising potential for developing autonomous diagnostic systems, although further validation studies are needed. 

**Abstract (ZH)**: 近期大规模语言模型在医疗诊断方面的进展显示出有希望的结果，一些研究在特定场景中表明其性能优于人类医师。然而，大规模语言模型的诊断能力往往被高估，在需要主动信息收集的交互式诊断环境中，其性能显著下降。本研究探讨了性能下降现象背后的机制并提出了解决方案。我们发现，大规模语言模型的主要不足在于初步诊断阶段，特别是信息收集效率和初步诊断形成上，而不是在后续的鉴别诊断阶段。为解决这一限制，我们开发了一种插件增强的大规模语言模型代理（PPME LLM），利用来自中国和美国医疗服务设施的超过350万份电子医疗记录。我们的方法通过监督学习和强化学习技术将专门用于初步疾病诊断和现病史查询的模型结合起来。实验结果表明，PPME LLM相比于基线模型实现了超过30%的改进。在交互式诊断场景中，PPME LLM的最终诊断准确性接近使用完整临床数据所达到的水平。这些发现表明了自主诊断系统开发的巨大潜力，但还需要进一步的验证研究。 

---
# Efficient Intent-Based Filtering for Multi-Party Conversations Using Knowledge Distillation from LLMs 

**Title (ZH)**: 使用来自大规模语言模型的知识蒸馏进行高效基于意图的多方对话过滤 

**Authors**: Reem Gody, Mohamed Abdelghaffar, Mohammed Jabreel, Ahmed Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2503.17336)  

**Abstract**: Large language models (LLMs) have showcased remarkable capabilities in conversational AI, enabling open-domain responses in chat-bots, as well as advanced processing of conversations like summarization, intent classification, and insights generation. However, these models are resource-intensive, demanding substantial memory and computational power. To address this, we propose a cost-effective solution that filters conversational snippets of interest for LLM processing, tailored to the target downstream application, rather than processing every snippet. In this work, we introduce an innovative approach that leverages knowledge distillation from LLMs to develop an intent-based filter for multi-party conversations, optimized for compute power constrained environments. Our method combines different strategies to create a diverse multi-party conversational dataset, that is annotated with the target intents and is then used to fine-tune the MobileBERT model for multi-label intent classification. This model achieves a balance between efficiency and performance, effectively filtering conversation snippets based on their intents. By passing only the relevant snippets to the LLM for further processing, our approach significantly reduces overall operational costs depending on the intents and the data distribution as demonstrated in our experiments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话AI领域展现出了 remarkable 的能力，能够生成开放领域对话，并对对话进行总结、意图分类和见解生成。然而，这些模型资源密集，需要大量内存和计算能力。为此，我们提出了一种经济高效的解决方案，通过对目标下游应用感兴趣的对话片段进行过滤以便LLM处理，而不是处理每个片段。在本文中，我们介绍了一种创新的方法，利用LLMs的知识蒸馏开发基于意图的过滤器，以适应计算能力受限的环境。我们的方法结合不同的策略创建了一个多元对话数据集，并对其进行标注以目标意图，然后用于微调MobileBERT模型以实现多标签意图分类。该模型在效率和性能之间达到了平衡，能够根据对话片段的意图对其进行有效过滤。通过只将相关的片段传递给LLM进行进一步处理，我们的方法在实验中展示了依赖于意图和数据分布的情况下显著降低了总体运营成本。 

---
# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging 

**Title (ZH)**: SafeMERGE: 在精选分层模型合并中保留细调大型语言模型的安全对齐 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2503.17239)  

**Abstract**: Fine-tuning large language models (LLMs) on downstream tasks can inadvertently erode their safety alignment, even for benign fine-tuning datasets. We address this challenge by proposing SafeMERGE, a post-fine-tuning framework that preserves safety while maintaining task utility. It achieves this by selectively merging fine-tuned and safety-aligned model layers only when those deviate from safe behavior, measured by a cosine similarity criterion. We evaluate SafeMERGE against other fine-tuning- and post-fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct models on GSM8K and PubMedQA tasks while exploring different merging strategies. We find that SafeMERGE consistently reduces harmful outputs compared to other baselines without significantly sacrificing performance, sometimes even enhancing it. The results suggest that our selective, subspace-guided, and per-layer merging method provides an effective safeguard against the inadvertent loss of safety in fine-tuned LLMs while outperforming simpler post-fine-tuning-stage defenses. 

**Abstract (ZH)**: Fine-tuning 大型语言模型 (LLMs) 在下游任务上的调整可能会无意中削弱其安全性对齐，即使是对于 benign 细调数据集也是如此。我们提出了一种名为 SafeMERGE 的后细调框架，该框架在保持安全性的前提下维持任务效用。通过仅在那些行为偏离安全标准的 fine-tuned 和安全性对齐的模型层之间选择性地进行合并，SafeMERGE 实现了这一目标，合并的方式基于余弦相似性的标准。我们在 Llama-2-7B-Chat 和 Qwen-2-7B-Instruct 模型上，以 GSM8K 和 PubMedQA 任务为评价标准，探讨了不同的合并策略。研究结果表明，SafeMERGE 在减少有害输出方面始终优于其他基准方法，而不显著牺牲性能，有时甚至可以提升性能。结果表明，我们的选择性、子空间引导的逐层合并方法提供了一种有效的保护措施，以防止在细调的大型语言模型中意外失去安全性，并且优于简单的后细调阶段防御措施。 

---
# FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs 

**Title (ZH)**: FactSelfCheck: LLMs的事实级黑盒幻觉检测 

**Authors**: Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17229)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only an 8% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常生成虚构内容，这对事实至关重要的应用构成了重大挑战。虽然现有的虚构内容检测方法通常在句子或段落级别进行操作，但我们提出了FactSelfCheck，这是一种新颖的黑盒抽样基于方法，能够实现细粒度的事实级检测。我们的方法将文本表示为由三元组形式的事实构成的知识图谱。通过对多个LLM响应中的事实一致性进行分析，我们可以在不依赖外部资源或训练数据的情况下计算细粒度的虚构得分。我们的评估表明，FactSelfCheck在性能上与领先的抽样基于方法相当，同时提供更详细的洞察。尤为值得注意的是，我们基于事实的方法在虚构内容纠正方面显著提升，与基线相比，事实内容提高了35%，而基于句子级别的SelfCheckGPT仅提高了8%。我们检测的粒度化特性能够更精确地识别和纠正虚构内容。 

---
# Automating Adjudication of Cardiovascular Events Using Large Language Models 

**Title (ZH)**: 使用大型语言模型自动 adjudicate 心血管事件 

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17222)  

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies. 

**Abstract (ZH)**: 使用大型语言模型自动化的临床试验心血管事件裁决框架 

---
# LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries 

**Title (ZH)**: LLMs偏好Python：关于LLMs对编程语言和库的偏见研究 

**Authors**: Lukas Twist, Jie M. Zhang, Mark Harman, Don Syme, Joost Noppen, Detlef Nauck  

**Link**: [PDF](https://arxiv.org/pdf/2503.17181)  

**Abstract**: Programming language and library choices are crucial to software reliability and security. Poor or inconsistent choices can lead to increased technical debt, security vulnerabilities, and even catastrophic failures in safety-critical systems. As Large Language Models (LLMs) play an increasing role in code generation, it is essential to understand how they make these decisions. However, little is known about their preferences when selecting programming languages and libraries for different coding tasks. To fill this gap, this study provides the first in-depth investigation into LLM preferences for programming languages and libraries used when generating code. We assess the preferences of eight diverse LLMs by prompting them to complete various coding tasks, including widely-studied benchmarks and the more practical task of generating the initial structural code for new projects (a crucial step that often determines a project's language or library choices).
Our findings reveal that LLMs heavily favour Python when solving language-agnostic problems, using it in 90%-97% of cases for benchmark tasks. Even when generating initial project code where Python is not a suitable language, it remains the most-used language in 58% of instances. Moreover, LLMs contradict their own language recommendations in 83% of project initialisation tasks, raising concerns about their reliability in guiding language selection. Similar biases toward well-established libraries further create serious discoverability challenges for newer open-source projects. These results highlight the need to improve LLMs' adaptability to diverse programming contexts and to develop mechanisms for mitigating programming language and library bias. 

**Abstract (ZH)**: 编程语言和库的选择对软件可靠性和安全性至关重要。不恰当或不一致的选择可能增加技术债务、安全漏洞，甚至导致安全关键系统出现灾难性故障。随着大型语言模型（LLMs）在代码生成中扮演越来越重要的角色，了解它们如何做出这些选择变得尤为重要。然而，关于它们在不同编程任务中选择编程语言和库的偏好知之甚少。为此，本研究首次深入探讨了LLMs在生成代码时选择编程语言和库的偏好。我们通过促使八种不同的LLMs完成各种编程任务，包括广泛研究的基准测试和生成新项目初始结构代码的更为实际的任务（这一步骤通常决定了项目的语言或库选择），来评估它们的偏好。

我们的研究发现，当解决跨语言问题时，LLMs强烈偏好使用Python，超过90%-97%的情况下会选择Python来完成基准测试任务。即使在生成初始项目代码的情况下，尽管Python不是最适合的语言，它仍然是58%情况下最常用的语言。此外，在58%的项目初始化任务中，LLMs与它们自己的语言推荐相悖，这引发了对其在指导语言选择方面可靠性的担忧。类似地，对现有库的偏向进一步给新的开源项目带来了严重的可发现性挑战。这些结果表明，需要改进LLMs在不同编程环境中的适应性，并开发机制以减轻编程语言和库的偏向。 

---
# PVChat: Personalized Video Chat with One-Shot Learning 

**Title (ZH)**: PVChat：基于单次学习的个性化视频聊天 

**Authors**: Yufei Shi, Weilong Yan, Gang Xu, Yumeng Li, Yuchen Li, Zhenxi Li, Fei Richard Yu, Ming Li, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17069)  

**Abstract**: Video large language models (ViLLMs) excel in general video understanding, e.g., recognizing activities like talking and eating, but struggle with identity-aware comprehension, such as "Wilson is receiving chemotherapy" or "Tom is discussing with Sarah", limiting their applicability in smart healthcare and smart home environments. To address this limitation, we propose a one-shot learning framework PVChat, the first personalized ViLLM that enables subject-aware question answering (QA) from a single video for each subject. Our approach optimizes a Mixture-of-Heads (MoH) enhanced ViLLM on a synthetically augmented video-QA dataset, leveraging a progressive image-to-video learning strategy. Specifically, we introduce an automated augmentation pipeline that synthesizes identity-preserving positive samples and retrieves hard negatives from existing video corpora, generating a diverse training dataset with four QA types: existence, appearance, action, and location inquiries. To enhance subject-specific learning, we propose a ReLU Routing MoH attention mechanism, alongside two novel objectives: (1) Smooth Proximity Regularization for progressive learning through exponential distance scaling and (2) Head Activation Enhancement for balanced attention routing. Finally, we adopt a two-stage training strategy, transitioning from image pre-training to video fine-tuning, enabling a gradual learning process from static attributes to dynamic representations. We evaluate PVChat on diverse datasets covering medical scenarios, TV series, anime, and real-world footage, demonstrating its superiority in personalized feature understanding after learning from a single video, compared to state-of-the-art ViLLMs. 

**Abstract (ZH)**: 基于视频的个性化大型语言模型（PVChat）：一种单视频学习框架，实现主体意识下的问答能力 

---
# Token Dynamics: Towards Efficient and Dynamic Video Token Representation for Video Large Language Models 

**Title (ZH)**: Token 动态分析：面向高效且动态的视频令牌表示以用于视频大规模语言模型 

**Authors**: Haichao Zhang, Zhuowei Li, Dimitris Metaxas, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16980)  

**Abstract**: Token-based video representation has emerged as a promising approach for enabling large language models to interpret video content. However, existing token reduction techniques, such as token pruning and token merging, often disrupt essential spatial-temporal positional embeddings, failing to adequately balance computational efficiency with fewer tokens. Consequently, these methods result in relatively lengthy token sequences, limiting their applicability in scenarios requiring extreme token compression, such as video large language models. In this paper, we introduce the novel task of extreme short token reduction, aiming to represent extensive video sequences with a minimal number of tokens. To address this challenge, we propose Token Dynamics, a new video representation framework that dynamically reduces token count while preserving spatial-temporal coherence. Specifically, we disentangle video representations by separating visual embeddings from grid-level motion information, structuring them into: 1. a concise token base, created by clustering tokens that describe object-level content; 2. a token dynamics map, capturing detailed spatial-temporal motion patterns across grids. Furthermore, we introduce a cross-dynamics attention mechanism that integrates motion features into the token base without increasing token length, thereby maintaining compactness and spatial-temporal integrity. The experiments demonstrate a reduction of token count to merely 0.07% of the original tokens, with only a minor performance drop of 1.13%. Additionally, we propose two novel subtasks within extreme token reduction (fixed-length and adaptive-length compression), both effectively representing long token sequences for video-language tasks. Our method offers significantly lower theoretical complexity, fewer tokens, and enhanced throughput, thus providing an efficient solution for video LLMs. 

**Abstract (ZH)**: 基于令牌的视频表示作为使大型语言模型能够解释视频内容的一种有前途的方法已经 emergence。然而，现有的令牌缩减技术，例如令牌裁剪和令牌合并，往往会破坏重要的空时位置嵌入，无法充分平衡计算效率和较少的令牌数量。因此，这些方法会导致相对较长的令牌序列，限制了其在需要极端令牌压缩的场景（如视频大型语言模型）中的应用。在本文中，我们引入了一项全新的极端短令牌缩减任务，旨在使用最少的令牌表示大量的视频序列。为了解决这一挑战，我们提出了一种新的视频表示框架——令牌动态，它可以在保持空时连贯性的同时动态减少令牌数量。具体而言，我们通过分离视觉嵌入和网格级别的运动信息来解耦视频表示，将它们结构化为：1. 一个简洁的令牌基础，通过聚类描述对象级内容的令牌创建；2. 一个令牌动态映射，捕捉网格之间详细的空时运动模式。此外，我们还引入了一种跨动态注意力机制，可以在不增加令牌长度的情况下将运动特征整合到令牌基础中，从而保持紧凑性和空时完整性。实验结果表明，令牌数量减少了原始令牌的0.07%，性能仅下降了1.13%。此外，我们还在极端令牌缩减中提出了两个新的子任务（固定长度和自适应长度压缩），两者都能有效地表示长令牌序列以供视频-语言任务使用。我们的方法提供了显著更低的理论复杂度、更少的令牌和增强的吞吐量，从而为视频LLMs提供了高效的解决方案。 

---
# Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks 

**Title (ZH)**: 评估大型语言模型输出的一致性和可重复性：跨 diverse 金融和会计任务的证据 

**Authors**: Julian Junyan Wang, Victor Xiaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16974)  

**Abstract**: This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple Generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks. 

**Abstract (ZH)**: 本研究提供了对金融与会计研究中大型语言模型（LLM）输出一致性和可再现性的首次全面评估。通过在五个常见任务（分类、情感分析、总结、文本生成和预测）上进行广泛实验（共50次独立运行），评估了给定相同输入时LLM的输出一致性。使用三个OpenAI模型（GPT-3.5-turbo、GPT-4o-mini和GPT-4o），我们从多样化的金融源文本和数据中生成了超过340万条输出，涵盖MD&A、FOMC声明、金融新闻文章、收益会议 transcript 和财务报表。研究发现，输出在任务上表现出显著但任务依赖的一致性，二分类和情感分析几乎完全可再现，而复杂任务则表现出更大的变化性。进阶模型在一致性与可再现性上并不始终表现得更好，且出现特定任务模式。LLM在一致性上显著优于专家人工注释者，并在专家分歧显著的情况下仍保持高水平的一致性。进一步发现，简单聚合策略（跨3-5次运行）显著提高了一致性。模拟分析表明，尽管LLM输出存在可测量的不一致性，下游统计推断依然表现出色。这些发现缓解了所谓的“G-黑客”现象，即从多个生成AI运行中选择性报告有利结果的问题，证明了在金融与会计任务中这种风险较低。 

---
# TEMPO: Temporal Preference Optimization of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment 

**Title (ZH)**: TEMPO：通过难度调度和预SFT对齐优化视频LLMs的时间偏好优化 

**Authors**: Shicheng Li, Lei Li, Kun Ouyang, Shuhuai Ren, Yuanxin Liu, Yuanxing Zhang, Fuzheng Zhang, Lingpeng Kong, Qi Liu, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.16929)  

**Abstract**: Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPO (TEMporal Preference Optimization), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and ``Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPO as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs. 

**Abstract (ZH)**: 视频大型语言模型（Video LLMs）通过利用两阶段范式取得了显著成功：在大规模视频-文本数据上进行预训练以实现视觉-语言对齐，随后进行监督微调（SFT）以获得特定任务的能力。然而，现有方法在处理时间推理方面存在困难，因为数据中的时间对应关系较弱，并且在训练过程中依赖于下一个令牌预测范式。为了解决这些局限性，我们提出了一种系统框架TEMPO（TEMporal Preference Optimization），通过直接偏好优化（DPO）增强视频大型语言模型的时间推理能力。为了促进这一点，我们引入了一种自动偏好数据生成管道，该管道系统地构建偏好对，通过选择富含时间信息的视频、设计视频特定的扰动策略，并最终评估模型对干净和扰动视频输入的响应。我们的时间对齐包括两个关键创新：循序渐进的学习方法，逐步提高扰动难度以提高模型的稳健性和适应性；以及“预SFT对齐”，在指令微调之前应用偏好优化以优先考虑细粒度的时间理解。广泛的经验表明，通过相对较小的自动生成DPO数据集，我们的方法在多个基准上持续提高了视频大型语言模型的性能。我们进一步分析了DPO数据在不同架构中的可迁移性以及难度调度在优化中的作用。我们的研究结果突显了TEMPO作为SFT方法的可扩展和高效补充的重要性，为开发可靠的视频大型语言模型铺平了道路。 

---
# RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation 

**Title (ZH)**: RustEvo^2: 一种基于API演化的LLM驱动的Rust代码生成演变基准测试 

**Authors**: Linxi Liang, Jing Gong, Mingwei Liu, Chong Wang, Guangsheng Ou, Yanlin Wang, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16922)  

**Abstract**: Large Language Models (LLMs) have become pivotal tools for automating code generation in software development. However, these models face significant challenges in producing version-aware code for rapidly evolving languages like Rust, where frequent Application Programming Interfaces (API) changes across versions lead to compatibility issues and correctness errors. Existing benchmarks lack systematic evaluation of how models navigate API transitions, relying on labor-intensive manual curation and offering limited version-specific insights. To address this gap, we present RustEvo, a novel framework for constructing dynamic benchmarks that evaluate the ability of LLMs to adapt to evolving Rust APIs. RustEvo automates dataset creation by synthesizing 588 API changes (380 from Rust standard libraries, 208 from 15 third-party crates) into programming tasks mirroring real-world challenges. These tasks cover four API evolution categories: Stabilizations, Signature Changes, Behavioral Changes, and Deprecations, reflecting their actual distribution in the Rust ecosystem.
Experiments on state-of-the-art (SOTA) LLMs reveal significant performance variations: models achieve a 65.8% average success rate on stabilized APIs but only 38.0% on behavioral changes, highlighting difficulties in detecting semantic shifts without signature alterations. Knowledge cutoff dates strongly influence performance, with models scoring 56.1% on before-cutoff APIs versus 32.5% on after-cutoff tasks. Retrieval-Augmented Generation (RAG) mitigates this gap, improving success rates by 13.5% on average for APIs released after model training. Our findings underscore the necessity of our evolution-aware benchmarks to advance the adaptability of LLMs in fast-paced software ecosystems. The framework and the benchmarks are publicly released at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为软件开发中自动化代码生成的关键工具。然而，这些模型在生成版本感知代码时面临巨大挑战，特别是在像Rust这样的快速演进语言中，频繁的应用程序编程接口（API）变化导致兼容性问题和正确性错误。现有基准测试缺乏系统性评估模型如何导航API过渡，依赖于耗时的手动整理，并提供有限的版本特定洞察。为弥补这一差距，我们提出了RustEvo，一种新的框架，用于构建动态基准测试以评估LLMs适应Rust API演进的能力。RustEvo通过合成588个API变化（包括380个来自Rust标准库和208个来自15个第三方crate），创建模拟现实挑战的编程任务。这些任务涵盖了四个API演进类别：稳定化、签名变化、行为变化和弃用，反映了Rust生态系统中这些类别的实际分布。 

---
# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs 

**Title (ZH)**: 稀疏逻辑采样：在大规模语言模型中加速知识蒸馏 

**Authors**: Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16870)  

**Abstract**: Knowledge distillation can be a cost-effective technique to distill knowledge in Large Language Models, if the teacher output logits can be pre-computed and cached. However, successfully applying this to pre-training remains largely unexplored. In this work, we prove that naive approaches for sparse knowledge distillation such as caching Top-K probabilities, while intuitive, provide biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration. We propose an importance-sampling-based method `Random Sampling Knowledge Distillation', which provides unbiased estimates, preserves the gradient in expectation, and requires storing significantly sparser logits. Our method enables faster training of student models with marginal overhead (<10%) compared to cross-entropy based training, while maintaining competitive performance compared to full distillation, across a range of model sizes from 300M to 3B. 

**Abstract (ZH)**: 基于重要性采样的随机采样知识蒸馏在大规模语言模型中提供无偏估计并保持梯度，在不同模型大小（从300M到3B）下实现较快的训练速度同时保持竞争力性能。 

---
# Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models 

**Title (ZH)**: 想象听见：听觉知识生成可以成为语言模型的有效辅助工具 

**Authors**: Suho Yoo, Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16853)  

**Abstract**: Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing and querying the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach. 

**Abstract (ZH)**: 语言模型在仅文本数据预训练的情况下往往难以完成需要听觉常识知识的任务。为解决这一问题，已有工作通过扩展语言模型以从外部音频数据库检索知识予以应对，但这种方法存在一些局限性，如数据库中缺乏相关音频以及构建和查询数据库所导致的高昂成本。为应对这些问题，我们提出了一种名为Imagine to Hear的新型方法，该方法利用生成模型动态生成听觉知识。我们的框架能够从给定提示中检测出多个与音频相关的文本片段，并生成相应的听觉知识。我们开发了几种机制来高效处理多样的听觉知识，包括基于CLAP的拒绝采样器和语言-音频融合模块。实验结果表明，我们的方法在不依赖外部数据库的情况下，在AuditoryBench上实现了最先进的性能，突显了基于生成的方法的有效性。 

---
# The Deployment of End-to-End Audio Language Models Should Take into Account the Principle of Least Privilege 

**Title (ZH)**: 端到端音频语言模型的部署应考虑最小权限原则 

**Authors**: Luxi He, Xiangyu Qi, Michel Liao, Inyoung Cheong, Prateek Mittal, Danqi Chen, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16833)  

**Abstract**: We are at a turning point for language models that accept audio input. The latest end-to-end audio language models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this position paper, we urge a closer examination of how these models are built and deployed. We argue that the principle of least privilege should guide decisions on whether to deploy cascaded or end-to-end models. Specifically, evaluations should assess (1) whether end-to-end modeling is necessary for a given application; and (2), the appropriate scope of information access. Finally, We highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs. 

**Abstract (ZH)**: 语言模型接受音频输入正处在一个转折点。最新的端到端音频语言模型（Audio LMs）直接处理语音而非依赖于单独的转录步骤。这一转变保留了诸如语调或多说话者存在的详细信息，这些信息在转录过程中会丢失。然而，这也引入了新的安全风险，包括可能误用说话者身份线索和其他敏感的嗓音属性，这可能会带来法律上的影响。在本文中，我们呼吁更仔细地审视这些模型的构建和部署方式。我们认为最小权限原则应指导是否部署级联或端到端模型的决策。具体而言，评估应考虑（1）给定应用是否需要端到端建模；（2）适当的信息访问范围。最后，我们指出现有音频LM基准中的相关缺口，并识别必须解决的关键开放研究问题，包括技术和政策相关问题，以确保端到端音频LMs的负责任部署。 

---
# Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models 

**Title (ZH)**: 工具链：在冻结语言模型的CoT推理中利用大量未见过的工具 

**Authors**: Mengsong Wu, Tong Zhu, Han Han, Xiang Zhang, Wenbiao Shao, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16779)  

**Abstract**: Tool learning can further broaden the usage scenarios of large language models (LLMs). However most of the existing methods either need to finetune that the model can only use tools seen in the training data, or add tool demonstrations into the prompt with lower efficiency. In this paper, we present a new Tool Learning method Chain-of-Tools. It makes full use of the powerful semantic representation capability of frozen LLMs to finish tool calling in CoT reasoning with a huge and flexible tool pool which may contain unseen tools. Especially, to validate the effectiveness of our approach in the massive unseen tool scenario, we construct a new dataset SimpleToolQuestions. We conduct experiments on two numerical reasoning benchmarks (GSM8K-XL and FuncQA) and two knowledge-based question answering benchmarks (KAMEL and SimpleToolQuestions). Experimental results show that our approach performs better than the baseline. We also identify dimensions of the model output that are critical in tool selection, enhancing the model interpretability. Our code and data are available at: this https URL . 

**Abstract (ZH)**: Tool学习可以进一步拓宽大规模语言模型的使用场景。然而，现有方法要么需要微调模型只能使用训练数据中 seen 的工具，要么在提示中添加工具演示以降低效率。在本文中，我们提出了一种新的Tool学习方法：Chain-of-Tools。该方法充分利用了冻结的大规模语言模型的强大语义表示能力，在包含未seen工具的庞大且灵活的工具池中进行CoT推理以完成工具调用。特别是，为了验证我们的方法在大量未seen工具场景下的有效性，我们构建了一个新的数据集SimpleToolQuestions。我们在两个数值推理基准（GSM8K-XL和FuncQA）和两个基于知识的问答基准（KAMEL和SimpleToolQuestions）上进行了实验。实验结果表明，我们的方法优于baseline。我们还识别了模型输出中对于工具选择至关重要的维度，提高了模型的可解释性。我们的代码和数据可在以下链接获取：this https URL。 

---
# Echoes of Power: Investigating Geopolitical Bias in US and China Large Language Models 

**Title (ZH)**: 权力回声：探究美国和中国大型语言模型中的地缘政治偏见 

**Authors**: Andre G. C. Pacheco, Athus Cavalini, Giovanni Comarela  

**Link**: [PDF](https://arxiv.org/pdf/2503.16679)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating human-like text, transforming human-machine interactions. However, their widespread adoption has raised concerns about their potential to influence public opinion and shape political narratives. In this work, we investigate the geopolitical biases in US and Chinese LLMs, focusing on how these models respond to questions related to geopolitics and international relations. We collected responses from ChatGPT and DeepSeek to a set of geopolitical questions and evaluated their outputs through both qualitative and quantitative analyses. Our findings show notable biases in both models, reflecting distinct ideological perspectives and cultural influences. However, despite these biases, for a set of questions, the models' responses are more aligned than expected, indicating that they can address sensitive topics without necessarily presenting directly opposing viewpoints. This study highlights the potential of LLMs to shape public discourse and underscores the importance of critically assessing AI-generated content, particularly in politically sensitive contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为生成类人类文本的强大工具，正在改变人机交互方式。然而，它们的广泛应用也引发了对其可能影响公众意见和塑造政治叙事潜在影响的担忧。在这项研究中，我们调查了中美LLMs在地缘政治偏见方面的情况，重点关注这些模型对与地缘政治和国际关系相关问题的响应。我们收集了ChatGPT和DeepSeek对一组地缘政治问题的回答，并通过定性和定量分析评估了它们的输出。研究结果表明，这两种模型都存在明显的偏见，反映出不同的意识形态视角和文化影响。然而，尽管存在这些偏见，对于一组问题，模型的回答比预期更加一致，表明它们可以在不直接呈现对立观点的情况下处理敏感话题。本研究强调了LLMs在塑造公众话语方面的潜力，并强调了在政治敏感背景下批判性评估AI生成内容的重要性。 

---
# Accelerating Transformer Inference and Training with 2:4 Activation Sparsity 

**Title (ZH)**: 用2:4激活稀疏性加速Transformer推理和训练 

**Authors**: Daniel Haziza, Timothy Chou, Dhruv Choudhary, Luca Wehrstedt, Francisco Massa, Jiecao Yu, Geonhwa Jeong, Supriya Rao, Patrick Labatut, Jesse Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16672)  

**Abstract**: In this paper, we demonstrate how to leverage 2:4 sparsity, a popular hardware-accelerated GPU sparsity pattern, to activations to accelerate large language model training and inference. Crucially we exploit the intrinsic sparsity found in Squared-ReLU activations to provide this acceleration with no accuracy loss. Our approach achieves up to 1.3x faster Feed Forward Network (FFNs) in both the forwards and backwards pass. This work highlights the potential for sparsity to play a key role in accelerating large language model training and inference. 

**Abstract (ZH)**: 本文展示了如何利用2:4稀疏性这一流行的硬件加速GPU稀疏模式来加速大型语言模型的训练和推理，同时通过对Squared-ReLU激活中固有的稀疏性的利用，实现了零准确率损失的加速。我们的方法在前向和反向传播中分别实现了最多1.3倍更快的前馈网络(FFNs)速度。本文突显了稀疏性在加速大型语言模型训练和推理中的潜在关键作用。 

---
# Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms 

**Title (ZH)**: 代码演化图：理解大型语言模型驱动的算法设计 

**Authors**: Niki van Stein, Anna V. Kononova, Lars Kotthoff, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2503.16668)  

**Abstract**: Large Language Models (LLMs) have demonstrated great promise in generating code, especially when used inside an evolutionary computation framework to iteratively optimize the generated algorithms. However, in some cases they fail to generate competitive algorithms or the code optimization stalls, and we are left with no recourse because of a lack of understanding of the generation process and generated codes. We present a novel approach to mitigate this problem by enabling users to analyze the generated codes inside the evolutionary process and how they evolve over repeated prompting of the LLM. We show results for three benchmark problem classes and demonstrate novel insights. In particular, LLMs tend to generate more complex code with repeated prompting, but additional complexity can hurt algorithmic performance in some cases. Different LLMs have different coding ``styles'' and generated code tends to be dissimilar to other LLMs. These two findings suggest that using different LLMs inside the code evolution frameworks might produce higher performing code than using only one LLM. 

**Abstract (ZH)**: 大型语言模型在进化计算框架中生成代码并迭代优化代码方面展现了巨大的潜力，但有时无法生成竞争性的算法或代码优化停滞不前，由于缺乏对生成过程和生成代码的理解，我们对此束手无策。我们提出了一种新的方法，使用户能够在进化过程中分析生成的代码及其在重复提示LLM时如何演变。我们展示了三种基准问题类别的结果，并展示了新的见解。具体而言，随着重复提示LLM，生成的代码变得更加复杂，但额外的复杂性在某些情况下可能损害算法性能。不同LLM具有不同的编码风格，生成的代码往往与其他LLM生成的代码不同。这两项发现表明，在代码进化框架中使用不同的LLM可能产生性能更高的代码，而不仅仅使用一个LLM。 

---
# Investigating Retrieval-Augmented Generation in Quranic Studies: A Study of 13 Open-Source Large Language Models 

**Title (ZH)**: 调查《古兰经》研究中的检索增强生成：13个开源大规模语言模型的研究 

**Authors**: Zahra Khalila, Arbi Haza Nasution, Winda Monika, Aytug Onan, Yohei Murakami, Yasir Bin Ismail Radi, Noor Mohammad Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.16581)  

**Abstract**: Accurate and contextually faithful responses are critical when applying large language models (LLMs) to sensitive and domain-specific tasks, such as answering queries related to quranic studies. General-purpose LLMs often struggle with hallucinations, where generated responses deviate from authoritative sources, raising concerns about their reliability in religious contexts. This challenge highlights the need for systems that can integrate domain-specific knowledge while maintaining response accuracy, relevance, and faithfulness. In this study, we investigate 13 open-source LLMs categorized into large (e.g., Llama3:70b, Gemma2:27b, QwQ:32b), medium (e.g., Gemma2:9b, Llama3:8b), and small (e.g., Llama3.2:3b, Phi3:3.8b). A Retrieval-Augmented Generation (RAG) is used to make up for the problems that come with using separate models. This research utilizes a descriptive dataset of Quranic surahs including the meanings, historical context, and qualities of the 114 surahs, allowing the model to gather relevant knowledge before responding. The models are evaluated using three key metrics set by human evaluators: context relevance, answer faithfulness, and answer relevance. The findings reveal that large models consistently outperform smaller models in capturing query semantics and producing accurate, contextually grounded responses. The Llama3.2:3b model, even though it is considered small, does very well on faithfulness (4.619) and relevance (4.857), showing the promise of smaller architectures that have been well optimized. This article examines the trade-offs between model size, computational efficiency, and response quality while using LLMs in domain-specific applications. 

**Abstract (ZH)**: 准确且上下文忠实的响应对于将大型语言模型应用于宗教研究等敏感和领域特定任务至关重要。通用大型语言模型常常会遇到幻觉问题，导致生成的响应偏离权威来源，这在宗教场景中引发了其可靠性的担忧。这一挑战凸显了系统整合领域特定知识并保持响应准确性、相关性和忠实性的重要性。本研究调查了13个开源大型语言模型（包括Llama3:70b、Gemma2:27b、QwQ:32b等大型模型，Gemma2:9b、Llama3:8b等中型模型，以及Llama3.2:3b、Phi3:3.8b等小型模型），并使用检索增强生成（RAG）来弥补单独模型使用时的问题。本研究利用描述性的古兰经章节数据集，包括114个章节的意义、历史背景和特性，使模型能在回应前获取相关知识。模型使用由人类评估者设定的三个关键指标进行评估：上下文相关性、答案忠实性和答案相关性。研究发现，大型模型在捕捉查询语义和生成准确、上下文相关响应方面始终优于小型模型。尽管Llama3.2:3b模型属于小型模型，但其在忠实性（4.619）和相关性（4.857）方面表现优异，显示出已优化的小型架构的潜力。本文探讨了在领域特定应用中使用大型语言模型时，模型规模、计算效率与响应质量之间的权衡。 

---
# Extract, Match, and Score: An Evaluation Paradigm for Long Question-context-answer Triplets in Financial Analysis 

**Title (ZH)**: 提取、匹配和评分：金融分析中长期问题-背景-答案 triplet的评估范式 

**Authors**: Bo Hu, Han Yuan, Vlad Pandelea, Wuqiong Luo, Yingzhu Zhao, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16575)  

**Abstract**: The rapid advancement of large language models (LLMs) has sparked widespread adoption across diverse applications, making robust evaluation frameworks crucial for assessing their performance. While conventional evaluation metrics remain applicable for shorter texts, their efficacy diminishes when evaluating the quality of long-form answers. This limitation is particularly critical in real-world scenarios involving extended questions, extensive context, and long-form answers, such as financial analysis or regulatory compliance. In this paper, we use a practical financial use case to illustrate applications that handle "long question-context-answer triplets". We construct a real-world financial dataset comprising long triplets and demonstrate the inadequacies of traditional metrics. To address this, we propose an effective Extract, Match, and Score (EMS) evaluation approach tailored to the complexities of long-form LLMs' outputs, providing practitioners with a reliable methodology for assessing LLMs' performance in complex real-world scenarios. 

**Abstract (ZH)**: 大规模语言模型的迅速发展促成了其在多种应用中的广泛应用，因此构建 robust 的评估框架对于评估其性能至关重要。虽然传统的评估指标适用于较短的文本，但在评估长文答复的质量时效果减弱。这一限制在涉及扩展问题、丰富背景和长文答复的实际场景中尤为重要，例如财务分析或合规性。本文通过一个实际的财务应用场景来展示处理“长问题-背景-答复三元组”的应用。我们构建了一个包含长三元组的现实世界财务数据集，并展示了传统指标的不足。为了解决这个问题，我们提出了一种适应长文生成模型输出复杂性的有效 Extract-Match-Score (EMS) 评估方法，为评估复杂实际场景中大语言模型的性能提供了可靠的方法。 

---
# Gene42: Long-Range Genomic Foundation Model With Dense Attention 

**Title (ZH)**: Gene42: 长范围基因组基础模型与密集注意力 

**Authors**: Kirill Vishniakov, Boulbaba Ben Amor, Engin Tekin, Nancy A. ElNaker, Karthik Viswanathan, Aleksandr Medvedev, Aahan Singh, Maryam Nadeem, Mohammad Amaan Sayeed, Praveenkumar Kanithi, Tiago Magalhaes, Natalia Vassilieva, Dwarikanath Mahapatra, Marco Pimentel, and Shadab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16565)  

**Abstract**: We introduce Gene42, a novel family of Genomic Foundation Models (GFMs) designed to manage context lengths of up to 192,000 base pairs (bp) at a single-nucleotide resolution. Gene42 models utilize a decoder-only (LLaMA-style) architecture with a dense self-attention mechanism. Initially trained on fixed-length sequences of 4,096 bp, our models underwent continuous pretraining to extend the context length to 192,000 bp. This iterative extension allowed for the comprehensive processing of large-scale genomic data and the capture of intricate patterns and dependencies within the human genome. Gene42 is the first dense attention model capable of handling such extensive long context lengths in genomics, challenging state-space models that often rely on convolutional operators among other mechanisms. Our pretrained models exhibit notably low perplexity values and high reconstruction accuracy, highlighting their strong ability to model genomic data. Extensive experiments on various genomic benchmarks have demonstrated state-of-the-art performance across multiple tasks, including biotype classification, regulatory region identification, chromatin profiling prediction, variant pathogenicity prediction, and species classification. The models are publicly available at this http URL. 

**Abstract (ZH)**: Gene42：一种新型的基因组基础模型家族，设计用于管理高达192,000碱基对的上下文长度 

---
# Chem42: a Family of chemical Language Models for Target-aware Ligand Generation 

**Title (ZH)**: Chem42：一种面向靶点的化学语言模型家族，用于配体生成 

**Authors**: Aahan Singh, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Mohammad Amaan Sayeed, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2503.16563)  

**Abstract**: Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at this http URL. 

**Abstract (ZH)**: Revolutionizing 药物发现需求的不仅仅是理解分子相互作用——还需要能够设计针对特定生物靶标的新颖配体的生成模型。Chem42：一种先进的生成化学语言模型，通过集成原子级相互作用与互补蛋白质语言模型Prot42的多模态输入，实现分子结构、相互作用及结合模式的复杂跨模态表示，从而能够创建结构合理、易于合成且具有增强靶标特异性的配体。Chem42在多种蛋白质靶标下的评估表明，它在化学有效性、目标感知设计以及预测结合亲和力方面超过了现有方法。通过缩小可行药物候选物的搜索空间，Chem42可以加速药物发现流程，提供一种强大的生成人工智能工具，用于精准医疗。我们的Chem42模型在分子性质预测、条件分子生成及目标感知配体设计方面设立了新的基准。模型已公开，访问地址见相应链接。 

---
# Gender and content bias in Large Language Models: a case study on Google Gemini 2.0 Flash Experimental 

**Title (ZH)**: 大型语言模型中的性别和内容偏见：对Google Gemini 2.0 Flash Experimental的案例研究 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2503.16534)  

**Abstract**: This study evaluates the biases in Gemini 2.0 Flash Experimental, a state-of-the-art large language model (LLM) developed by Google, focusing on content moderation and gender disparities. By comparing its performance to ChatGPT-4o, examined in a previous work of the author, the analysis highlights some differences in ethical moderation practices. Gemini 2.0 demonstrates reduced gender bias, notably with female-specific prompts achieving a substantial rise in acceptance rates compared to results obtained by ChatGPT-4o. It adopts a more permissive stance toward sexual content and maintains relatively high acceptance rates for violent prompts, including gender-specific cases. Despite these changes, whether they constitute an improvement is debatable. While gender bias has been reduced, this reduction comes at the cost of permitting more violent content toward both males and females, potentially normalizing violence rather than mitigating harm. Male-specific prompts still generally receive higher acceptance rates than female-specific ones. These findings underscore the complexities of aligning AI systems with ethical standards, highlighting progress in reducing certain biases while raising concerns about the broader implications of the model's permissiveness. Ongoing refinements are essential to achieve moderation practices that ensure transparency, fairness, and inclusivity without amplifying harmful content. 

**Abstract (ZH)**: 本研究评估了由谷歌开发的最先进的大型语言模型（LLM）Gemini 2.0 Flash Experimental中的偏见问题，重点关注内容审核和性别差距。通过将其性能与作者在之前工作中研究的ChatGPT-4o进行比较，分析突显了一些伦理审核实践的差异。Gemini 2.0展示了减少性别偏见的情况，特别是在针对女性的具体提示中，接受率大幅提升，高于ChatGPT-4o的结果。它在性内容方面采取了更为宽松的立场，并且对于暴力提示（包括性别特定情况）保持了较高的接受率。尽管存在这些变化，但是否构成了改进仍然是值得争议的问题。虽然性别偏见有所减少，但这种减少是以容忍更多针对男女的暴力内容为代价的，这可能会正常化暴力行为，而不是减轻伤害。男性特定提示的接受率仍然普遍高于女性特定提示。这些发现突显了将AI系统与伦理标准对齐的复杂性，尽管减少了某些偏见，但仍引起了关于模型宽松程度更广泛影响的担忧。持续优化对于实现确保透明度、公平性和包容性的审核实践至关重要，同时不放大有害内容。 

---
# From Patient Consultations to Graphs: Leveraging LLMs for Patient Journey Knowledge Graph Construction 

**Title (ZH)**: 从患者咨询到图构建：利用大语言模型构建患者旅程知识图谱 

**Authors**: Hassan S. Al Khatib, Sudip Mittal, Shahram Rahimi, Nina Marhamati, Sean Bozorgzad  

**Link**: [PDF](https://arxiv.org/pdf/2503.16533)  

**Abstract**: The transition towards patient-centric healthcare necessitates a comprehensive understanding of patient journeys, which encompass all healthcare experiences and interactions across the care spectrum. Existing healthcare data systems are often fragmented and lack a holistic representation of patient trajectories, creating challenges for coordinated care and personalized interventions. Patient Journey Knowledge Graphs (PJKGs) represent a novel approach to addressing the challenge of fragmented healthcare data by integrating diverse patient information into a unified, structured representation. This paper presents a methodology for constructing PJKGs using Large Language Models (LLMs) to process and structure both formal clinical documentation and unstructured patient-provider conversations. These graphs encapsulate temporal and causal relationships among clinical encounters, diagnoses, treatments, and outcomes, enabling advanced temporal reasoning and personalized care insights. The research evaluates four different LLMs, such as Claude 3.5, Mistral, Llama 3.1, and Chatgpt4o, in their ability to generate accurate and computationally efficient knowledge graphs. Results demonstrate that while all models achieved perfect structural compliance, they exhibited variations in medical entity processing and computational efficiency. The paper concludes by identifying key challenges and future research directions. This work contributes to advancing patient-centric healthcare through the development of comprehensive, actionable knowledge graphs that support improved care coordination and outcome prediction. 

**Abstract (ZH)**: 基于患者的医疗知识图谱构建方法学：利用大规模语言模型整合结构化与非结构化患者数据以支持个性化医疗决策 

---
# Enhancing LLM Generation with Knowledge Hypergraph for Evidence-Based Medicine 

**Title (ZH)**: 增强LLM生成能力以支持基于证据的医学知识图谱 

**Authors**: Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhengwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16530)  

**Abstract**: Evidence-based medicine (EBM) plays a crucial role in the application of large language models (LLMs) in healthcare, as it provides reliable support for medical decision-making processes. Although it benefits from current retrieval-augmented generation~(RAG) technologies, it still faces two significant challenges: the collection of dispersed evidence and the efficient organization of this evidence to support the complex queries necessary for EBM. To tackle these issues, we propose using LLMs to gather scattered evidence from multiple sources and present a knowledge hypergraph-based evidence management model to integrate these evidence while capturing intricate relationships. Furthermore, to better support complex queries, we have developed an Importance-Driven Evidence Prioritization (IDEP) algorithm that utilizes the LLM to generate multiple evidence features, each with an associated importance score, which are then used to rank the evidence and produce the final retrieval results. Experimental results from six datasets demonstrate that our approach outperforms existing RAG techniques in application domains of interest to EBM, such as medical quizzing, hallucination detection, and decision support. Testsets and the constructed knowledge graph can be accessed at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于证据的医学（EBM）在医疗领域应用大型语言模型（LLMs）中扮演着重要角色，因为它为医学决策过程提供了可靠的支撑。尽管EBM受益于当前的检索增强生成（RAG）技术，但它仍然面临两个重大挑战：分散证据的收集和组织，以支持EBM所需的复杂查询。为解决这些问题，我们提出使用LLM从多个源收集分散的证据，并通过知识超图为基础的证据管理模型整合这些证据，同时捕获复杂的相互关系。此外，为了更好地支持复杂查询，我们开发了一种基于重要性驱动的证据优先级排序（IDEP）算法，该算法利用LLM生成多个具有相关重要性评分的证据特征，然后根据这些特征对证据进行排序并产生最终检索结果。来自六个数据集的实验结果表明，我们的方法在适用于EBM的应用领域（如医学测验、幻觉检测和决策支持）中优于现有RAG技术。测试集和构建的知识图谱可访问 <https://this https URL>。 

---
# HDLCoRe: A Training-Free Framework for Mitigating Hallucinations in LLM-Generated HDL 

**Title (ZH)**: HDLCoRe: 无需训练的框架，用于减轻LLM生成的HDL中的幻觉 

**Authors**: Heng Ping, Shixuan Li, Peiyu Zhang, Anzhe Cheng, Shukai Duan, Nikos Kanakaris, Xiongye Xiao, Wei Yang, Shahin Nazarian, Andrei Irimia, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16528)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, when applied to hardware description languages (HDL), these models exhibit significant limitations due to data scarcity, resulting in hallucinations and incorrect code generation. To address these challenges, we propose HDLCoRe, a training-free framework that enhances LLMs' HDL generation capabilities through prompt engineering techniques and retrieval-augmented generation (RAG). Our approach consists of two main components: (1) an HDL-aware Chain-of-Thought (CoT) prompting technique with self-verification that classifies tasks by complexity and type, incorporates domain-specific knowledge, and guides LLMs through step-by-step self-simulation for error correction; and (2) a two-stage heterogeneous RAG system that addresses formatting inconsistencies through key component extraction and efficiently retrieves relevant HDL examples through sequential filtering and re-ranking. HDLCoRe eliminates the need for model fine-tuning while substantially improving LLMs' HDL generation capabilities. Experimental results demonstrate that our framework achieves superior performance on the RTLLM2.0 benchmark, significantly reducing hallucinations and improving both syntactic and functional correctness. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在代码生成任务中展现了显著的能力。然而，当应用于硬件描述语言（HDL）时，这些模型由于数据稀缺性表现出显著的局限性，导致生成幻觉和错误代码。为了应对这些挑战，我们提出了一种无需微调的HDLCoRe框架，通过提示工程技术和服务检索增强生成（RAG）技术来增强LLMs的HDL生成能力。我们的方法包括两个主要组成部分：（1）一种awareHDLCoRe有序思考（CoT）提示技术，带有自我验证功能，通过分类任务的复杂性和类型、整合领域特定知识，并通过逐步自我模拟引导LLMs进行错误校正；以及（2）一种两阶段异构RAG系统，通过关键组件提取解决格式不一致问题，并通过顺序过滤和重新排名高效检索相关HDL示例。HDLCoRe消除了模型微调的需要，同时大幅提高LLMs的HDL生成能力。实验结果表明，我们的框架在RTLLM2.0基准测试中实现了优异的性能，显著减少了幻觉，并提高了语法和功能正确性。 

---
# LLM Generated Persona is a Promise with a Catch 

**Title (ZH)**: LLM生成的人格是一种有 promise 的承诺，但也伴随着问题。 

**Authors**: Ang Li, Haozhe Chen, Hongseok Namkoong, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16527)  

**Abstract**: The use of large language models (LLMs) to simulate human behavior has gained significant attention, particularly through personas that approximate individual characteristics. Persona-based simulations hold promise for transforming disciplines that rely on population-level feedback, including social science, economic analysis, marketing research, and business operations. Traditional methods to collect realistic persona data face significant challenges. They are prohibitively expensive and logistically challenging due to privacy constraints, and often fail to capture multi-dimensional attributes, particularly subjective qualities. Consequently, synthetic persona generation with LLMs offers a scalable, cost-effective alternative. However, current approaches rely on ad hoc and heuristic generation techniques that do not guarantee methodological rigor or simulation precision, resulting in systematic biases in downstream tasks. Through extensive large-scale experiments including presidential election forecasts and general opinion surveys of the U.S. population, we reveal that these biases can lead to significant deviations from real-world outcomes. Our findings underscore the need to develop a rigorous science of persona generation and outline the methodological innovations, organizational and institutional support, and empirical foundations required to enhance the reliability and scalability of LLM-driven persona simulations. To support further research and development in this area, we have open-sourced approximately one million generated personas, available for public access and analysis at this https URL. 

**Abstract (ZH)**: 大型语言模型在模拟人类行为中的应用，特别是在基于个性特征的模拟方面已引起广泛关注。这种基于个性的模拟为依赖于群体反馈的学科带来了变革潜力，包括社会科学、经济分析、市场营销研究和企业运营。传统方法收集真实个性数据面临重大挑战。由于隐私限制，这些方法费用高昂且在物流上具有挑战性，且往往无法捕捉多维度特征，特别是主观品质。因此，使用大型语言模型生成合成个性提供了一种可扩展且成本效益高的替代方案。然而，目前的方法依赖于难以保证方法论严谨性和模拟精确性的随意生成技术，导致下游任务中出现系统性偏差。通过包括美国总统选举预测和一般公众意见调查在内的大规模实验，我们揭示了这些偏差可能导致与现实世界结果的重大偏差。我们的研究强调了需要发展严谨的个性生成科学，并概述了提高大型语言模型驱动的个性模拟可靠性和可扩展性的方法论创新、组织和制度支持以及实证基础。为支持该领域的进一步研究和发展，我们已开源约一百万个生成的个性特征，并可通过以下链接进行公共访问和分析：[此 https URL]。 

---
# KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference 

**Title (ZH)**: KVShare：具有语义意识的键值缓存共享以实现高效的大型语言模型推理 

**Authors**: Huan Yang, Renji Zhang, Deyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16525)  

**Abstract**: This paper presents KVShare, a multi-user Key-Value (KV) Cache sharing technology based on semantic similarity, designed to enhance the inference efficiency of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Addressing the limitations of existing prefix caching (strict text prefix matching) and semantic caching (loss of response diversity), KVShare achieves fine-grained KV cache reuse through semantic alignment algorithms and differential editing operations. Experiments on real-world user conversation datasets demonstrate that KVShare improves KV cache hit rates by over 60%, while maintaining output quality comparable to full computation (no significant degradation in BLEU and Rouge-L metrics). This approach effectively reduces GPU resource consumption and is applicable to scenarios with repetitive queries, such as healthcare and education. 

**Abstract (ZH)**: 基于语义相似性的多用户键值缓存技术KVShare：提高大型语言模型和多模态大型语言模型的推理效率 

---
# Using LLMs for Automated Privacy Policy Analysis: Prompt Engineering, Fine-Tuning and Explainability 

**Title (ZH)**: 使用大语言模型进行自动化隐私政策分析：提示工程、微调与解释性 

**Authors**: Yuxin Chen, Peng Tang, Weidong Qiu, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16516)  

**Abstract**: Privacy policies are widely used by digital services and often required for legal purposes. Many machine learning based classifiers have been developed to automate detection of different concepts in a given privacy policy, which can help facilitate other automated tasks such as producing a more reader-friendly summary and detecting legal compliance issues. Despite the successful applications of large language models (LLMs) to many NLP tasks in various domains, there is very little work studying the use of LLMs for automated privacy policy analysis, therefore, if and how LLMs can help automate privacy policy analysis remains under-explored. To fill this research gap, we conducted a comprehensive evaluation of LLM-based privacy policy concept classifiers, employing both prompt engineering and LoRA (low-rank adaptation) fine-tuning, on four state-of-the-art (SOTA) privacy policy corpora and taxonomies. Our experimental results demonstrated that combining prompt engineering and fine-tuning can make LLM-based classifiers outperform other SOTA methods, \emph{significantly} and \emph{consistently} across privacy policy corpora/taxonomies and concepts. Furthermore, we evaluated the explainability of the LLM-based classifiers using three metrics: completeness, logicality, and comprehensibility. For all three metrics, a score exceeding 91.1\% was observed in our evaluation, indicating that LLMs are not only useful to improve the classification performance, but also to enhance the explainability of detection results. 

**Abstract (ZH)**: 基于大型语言模型的隐私政策概念分类器的综合评估：从提示工程到LoRA微调 

---
# Highlighting Case Studies in LLM Literature Review of Interdisciplinary System Science 

**Title (ZH)**: LLM领域跨学科系统科学文献综述中的案例研究亮点 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2503.16515)  

**Abstract**: Large Language Models (LLMs) were used to assist four Commonwealth Scientific and Industrial Research Organisation (CSIRO) researchers to perform systematic literature reviews (SLR). We evaluate the performance of LLMs for SLR tasks in these case studies. In each, we explore the impact of changing parameters on the accuracy of LLM responses. The LLM was tasked with extracting evidence from chosen academic papers to answer specific research questions. We evaluate the models' performance in faithfully reproducing quotes from the literature and subject experts were asked to assess the model performance in answering the research questions. We developed a semantic text highlighting tool to facilitate expert review of LLM responses.
We found that state of the art LLMs were able to reproduce quotes from texts with greater than 95% accuracy and answer research questions with an accuracy of approximately 83%. We use two methods to determine the correctness of LLM responses; expert review and the cosine similarity of transformer embeddings of LLM and expert answers. The correlation between these methods ranged from 0.48 to 0.77, providing evidence that the latter is a valid metric for measuring semantic similarity. 

**Abstract (ZH)**: 大型语言模型（LLMs）被用于辅助四名 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 研究人员进行系统文献综述（SLR）。我们在这些案例研究中评估了LLMs在SLR任务中的性能。在每个案例中，我们探索了改变参数对LLM响应准确性的影响。LLM的任务是从选定的学术论文中提取证据以回答特定的研究问题。我们评估了模型在忠实再现文献引文方面的表现，并邀请领域专家对模型回答研究问题的性能进行评估。我们开发了一种语义文本高亮工具，以促进领域专家对LLM响应的审查。我们发现，最先进的LLMs能够在超过95%的准确率下忠实再现文本引文，并在大约83%的准确率下回答研究问题。我们使用两种方法来确定LLM响应的正确性；专家审查和转换器嵌入表示的余弦相似度。这些方法的相关性范围从0.48到0.77，提供了后者是衡量语义相似性的有效度量标准的证据。 

---
# VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric 

**Title (ZH)**: VeriMind: 自主代理模型在新型评价指标下的自动化Verilog生成 

**Authors**: Bardia Nadimi, Ghali Omar Boutaib, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16514)  

**Abstract**: Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis. 

**Abstract (ZH)**: 基于大型语言模型的VeriMind框架：一种用于Verilog代码生成的代理型LLM框架 

---
# Token-Level Uncertainty-Aware Objective for Language Model Post-Training 

**Title (ZH)**: 基于令牌级别的不确定性意识目标函数的模型后训练 

**Authors**: Tingkai Liu, Ari S. Benjamin, Anthony M. Zador  

**Link**: [PDF](https://arxiv.org/pdf/2503.16511)  

**Abstract**: In the current work, we connect token-level uncertainty in causal language modeling to two types of training objectives: 1) masked maximum likelihood (MLE), 2) self-distillation. We show that masked MLE is effective in reducing epistemic uncertainty, and serve as an effective token-level automatic curriculum learning technique. However, masked MLE is prone to overfitting and requires self-distillation regularization to improve or maintain performance on out-of-distribution tasks. We demonstrate significant performance gain via the proposed training objective - combined masked MLE and self-distillation - across multiple architectures (Gemma, LLaMA, Phi) and datasets (Alpaca, ShareGPT, GSM8K), mitigating overfitting while maintaining adaptability during post-training. Our findings suggest that uncertainty-aware training provides an effective mechanism for enhancing language model training. 

**Abstract (ZH)**: 在当前工作中，我们将因果语言模型中的 token 级别不确定性与两种训练目标连接起来：1) 掩码最大似然估计 (MLE)，2) 自我蒸馏。我们表明，掩码 MLE 在减少epistemic不确定性方面有效，并作为有效的 token 级别自动 Curriculum Learning 技术。然而，掩码 MLE 容易过拟合，并需要通过自我蒸馏正则化来提高或保持在分布外任务中的性能。通过提出的训练目标——结合掩码 MLE 和自我蒸馏——我们在多个架构 (Gemma, LLaMA, Phi) 和数据集 (Alpaca, ShareGPT, GSM8K) 上实现了显著的性能提升，同时在后训练期间减轻过拟合并保持适应性。我们的研究结果表明，不确定性感知的训练为增强语言模型训练提供了一种有效机制。 

---
# Conversational AI as a Coding Assistant: Understanding Programmers' Interactions with and Expectations from Large Language Models for Coding 

**Title (ZH)**: Conversational AI作为编程辅助：理解程序员与大规模语言模型在编程中的互动及其期望 

**Authors**: Mehmet Akhoroz, Caglar Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16508)  

**Abstract**: Conversational AI interfaces powered by large language models (LLMs) are increasingly used as coding assistants. However, questions remain about how programmers interact with LLM-based conversational agents, the challenges they encounter, and the factors influencing adoption. This study investigates programmers' usage patterns, perceptions, and interaction strategies when engaging with LLM-driven coding assistants. Through a survey, participants reported both the benefits, such as efficiency and clarity of explanations, and the limitations, including inaccuracies, lack of contextual awareness, and concerns about over-reliance. Notably, some programmers actively avoid LLMs due to a preference for independent learning, distrust in AI-generated code, and ethical considerations. Based on our findings, we propose design guidelines for improving conversational coding assistants, emphasizing context retention, transparency, multimodal support, and adaptability to user preferences. These insights contribute to the broader understanding of how LLM-based conversational agents can be effectively integrated into software development workflows while addressing adoption barriers and enhancing usability. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的对话式AI接口作为编码助手的使用：程序员的交互模式、挑战及影响因素研究 

---
# Llms, Virtual Users, and Bias: Predicting Any Survey Question Without Human Data 

**Title (ZH)**: 大语言模型、虚拟用户和偏见：无需人类数据预测任何调研问题 

**Authors**: Enzo Sinacola, Arnault Pachot, Thierry Petit  

**Link**: [PDF](https://arxiv.org/pdf/2503.16498)  

**Abstract**: Large Language Models (LLMs) offer a promising alternative to traditional survey methods, potentially enhancing efficiency and reducing costs. In this study, we use LLMs to create virtual populations that answer survey questions, enabling us to predict outcomes comparable to human responses. We evaluate several LLMs-including GPT-4o, GPT-3.5, Claude 3.5-Sonnet, and versions of the Llama and Mistral models-comparing their performance to that of a traditional Random Forests algorithm using demographic data from the World Values Survey (WVS). LLMs demonstrate competitive performance overall, with the significant advantage of requiring no additional training data. However, they exhibit biases when predicting responses for certain religious and population groups, underperforming in these areas. On the other hand, Random Forests demonstrate stronger performance than LLMs when trained with sufficient data. We observe that removing censorship mechanisms from LLMs significantly improves predictive accuracy, particularly for underrepresented demographic segments where censored models struggle. These findings highlight the importance of addressing biases and reconsidering censorship approaches in LLMs to enhance their reliability and fairness in public opinion research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为传统调查方法提供了有前景的替代方案，有可能提高效率并降低成本。本研究使用LLMs创建虚拟人口以回答调查问题，从而能够预测与人类回应相近的结果。我们评估了几种LLMs，包括GPT-4o、GPT-3.5、Claude 3.5-Sonnet以及Llama和Mistral模型的不同版本，并将它们的性能与使用世界价值观调查（WVS）的人口统计数据训练的传统随机森林算法进行比较。总体而言，LLMs展示出了竞争力，其显著优势在于不需要额外的训练数据。然而，它们在预测某些宗教和人口群体的回答时表现出偏差，这些区域的性能欠佳。另一方面，当使用足够数据训练时，随机森林表现出比LLMs更好的性能。我们发现，移除LLMs中的审核机制显著提高了预测准确性，尤其是对于审核模型难以处理的未充分代表的人口细分群体。这些发现强调了在公共意见研究中解决偏见并重新考虑审核方法的重要性，以提高LLMs的可靠性和公平性。 

---
# AI-Powered Episodic Future Thinking 

**Title (ZH)**: AI驱动的 episodic 未来思考 

**Authors**: Sareh Ahmadi, Michelle Rockwell, Megan Stuart, Allison Tegge, Xuan Wang, Jeffrey Stein, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2503.16484)  

**Abstract**: Episodic Future Thinking (EFT) is an intervention that involves vividly imagining personal future events and experiences in detail. It has shown promise as an intervention to reduce delay discounting - the tendency to devalue delayed rewards in favor of immediate gratification - and to promote behavior change in a range of maladaptive health behaviors. We present EFTeacher, an AI chatbot powered by the GPT-4-Turbo large language model, designed to generate EFT cues for users with lifestyle-related conditions. To evaluate the chatbot, we conducted a user study that included usability assessments and user evaluations based on content characteristics questionnaires, followed by semi-structured interviews. The study provides qualitative insights into participants' experiences and interactions with the chatbot and its usability. Our findings highlight the potential application of AI chatbots based on Large Language Models (LLMs) in EFT interventions, and offer design guidelines for future behavior-oriented applications. 

**Abstract (ZH)**: episodic未来思考(EFT)是一种干预措施，涉及生动地详细想象个人未来事件和体验。它显示出减少延迟折扣（倾向于以即时满足为优先而低估延迟奖励）以及促进各种不良健康行为改变的潜力。我们介绍了EFTeacher，一个基于GPT-4-Turbo大规模语言模型的AI聊天机器人，旨在为与生活方式相关条件的用户提供EFT提示。为评估聊天机器人，我们进行了一项用户研究，包括使用可用性评估和基于内容特性问卷的用户评价，随后进行了半结构化访谈。该研究提供了关于参与者与聊天机器人互动经验及可用性的定性见解。我们的研究结果强调了基于大规模语言模型（LLMs）的AI聊天机器人在EFT干预中的潜在应用，并为未来目标行为应用的设计提供了指导方针。 

---
# Human Preferences for Constructive Interactions in Language Model Alignment 

**Title (ZH)**: 人类在语言模型对齐中对建设性互动的偏好 

**Authors**: Yara Kyrychenko, Jon Roozenbeek, Brandon Davidson, Sander van der Linden, Ramit Debnath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16480)  

**Abstract**: As large language models (LLMs) enter the mainstream, aligning them to foster constructive dialogue rather than exacerbate societal divisions is critical. Using an individualized and multicultural alignment dataset of over 7,500 conversations of individuals from 74 countries engaging with 21 LLMs, we examined how linguistic attributes linked to constructive interactions are reflected in human preference data used for training AI. We found that users consistently preferred well-reasoned and nuanced responses while rejecting those high in personal storytelling. However, users who believed that AI should reflect their values tended to place less preference on reasoning in LLM responses and more on curiosity. Encouragingly, we observed that users could set the tone for how constructive their conversation would be, as LLMs mirrored linguistic attributes, including toxicity, in user queries. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）进入主流，引导它们促进建设性对话而不是加剧社会分歧至关重要。通过使用包含来自74个国家的7,500多场对话的数据集，这些对话涉及21种LLM，并具有个性化和多元文化的特点，我们研究了与建设性互动相关的语言属性如何在用于训练AI的人类偏好数据中体现。我们发现，用户倾向于偏好有充足理由和细腻的回应，而非过度个人化的的故事讲述。然而，那些认为AI应该反映其价值观的用户倾向于在LLM回应中较少看重理性分析，而更看重好奇心。令人鼓舞的是，我们观察到用户能够通过其查询的语言属性，包括有害内容，来设定对话的建设性基调。 

---
# LeRAAT: LLM-Enabled Real-Time Aviation Advisory Tool 

**Title (ZH)**: LeRAAT: LLM驱动的实时航空 advisement 工具 

**Authors**: Marc R. Schlichting, Vale Rasmussen, Heba Alazzeh, Houjun Liu, Kiana Jafari, Amelia F. Hardy, Dylan M. Asmar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16477)  

**Abstract**: In aviation emergencies, high-stakes decisions must be made in an instant. Pilots rely on quick access to precise, context-specific information -- an area where emerging tools like large language models (LLMs) show promise in providing critical support. This paper introduces LeRAAT, a framework that integrates LLMs with the X-Plane flight simulator to deliver real-time, context-aware pilot assistance. The system uses live flight data, weather conditions, and aircraft documentation to generate recommendations aligned with aviation best practices and tailored to the particular situation. It employs a Retrieval-Augmented Generation (RAG) pipeline that extracts and synthesizes information from aircraft type-specific manuals, including performance specifications and emergency procedures, as well as aviation regulatory materials, such as FAA directives and standard operating procedures. We showcase the framework in both a virtual reality and traditional on-screen simulation, supporting a wide range of research applications such as pilot training, human factors research, and operational decision support. 

**Abstract (ZH)**: 在航空紧急事件中，高风险决策必须瞬时作出。飞行员依赖于快速获取精确的、情境相关的信息——这是新兴工具如大规模语言模型（LLMs）展现出关键支持潜力的领域。本文介绍了一种名为LeRAAT的框架，该框架将LLMs与X-Plane飞行模拟器集成，以提供实时的情境感知飞行员辅助。该系统利用实时飞行数据、天气状况和航空器文档来生成符合航空最佳实践且针对特定情境的建议。它采用检索增强生成（RAG）管道，从特定类型的航空器手册中提取和合成信息，包括性能规范和应急程序，以及航空监管材料，如联邦航空管理局（FAA）的指令和标准操作程序。我们通过虚拟现实和传统屏幕模拟展示了该框架，支持飞行员培训、人因研究和操作决策支持等多种研究应用。 

---
# ACE, Action and Control via Explanations: A Proposal for LLMs to Provide Human-Centered Explainability for Multimodal AI Assistants 

**Title (ZH)**: ACE：动作与控制借助解释：为多模态AI助手机给人中心化可解释性的提案 

**Authors**: Elizabeth Anne Watkins, Emanuel Moss, Ramesh Manuvinakurike, Meng Shi, Richard Beckwith, Giuseppe Raffa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16466)  

**Abstract**: In this short paper we address issues related to building multimodal AI systems for human performance support in manufacturing domains. We make two contributions: we first identify challenges of participatory design and training of such systems, and secondly, to address such challenges, we propose the ACE paradigm: "Action and Control via Explanations". Specifically, we suggest that LLMs can be used to produce explanations in the form of human interpretable "semantic frames", which in turn enable end users to provide data the AI system needs to align its multimodal models and representations, including computer vision, automatic speech recognition, and document inputs. ACE, by using LLMs to "explain" using semantic frames, will help the human and the AI system to collaborate, together building a more accurate model of humans activities and behaviors, and ultimately more accurate predictive outputs for better task support, and better outcomes for human users performing manual tasks. 

**Abstract (ZH)**: 在本短文中，我们探讨了构建适用于制造领域的人机协作智能系统面临的问题，并提出了两个贡献：首先，我们识别出了参与设计和培训这类系统所面临的挑战；其次，为了应对这些挑战，我们提出了ACE范式：“通过解释进行行动与控制”。具体而言，我们建议可以利用大规模语言模型（LLM）生成人类可解释的“语义框架”形式的解释，从而使得最终用户能够提供AI系统所需的用于其跨模态模型和表示的数据，包括计算机视觉、自动语音识别和文档输入。通过利用LLM用语义框架“解释”的方式，ACE将有助于人与智能系统合作，共同构建更准确的人类活动和行为模型，并最终提供更准确的任务预测输出，以更好地支持进行手工任务的人类用户。 

---
# OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents 

**Title (ZH)**: OS-Kairos: 适应性交互的MLLM驱动GUI代理模型 

**Authors**: Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16465)  

**Abstract**: Autonomous graphical user interface (GUI) agents powered by multimodal large language models have shown great promise. However, a critical yet underexplored issue persists: over-execution, where the agent executes tasks in a fully autonomous way, without adequate assessment of its action confidence to compromise an adaptive human-agent collaboration. This poses substantial risks in complex scenarios, such as those involving ambiguous user instructions, unexpected interruptions, and environmental hijacks. To address the issue, we introduce OS-Kairos, an adaptive GUI agent capable of predicting confidence levels at each interaction step and efficiently deciding whether to act autonomously or seek human intervention. OS-Kairos is developed through two key mechanisms: (i) collaborative probing that annotates confidence scores at each interaction step; (ii) confidence-driven interaction that leverages these confidence scores to elicit the ability of adaptive interaction. Experimental results show that OS-Kairos substantially outperforms existing models on our curated dataset featuring complex scenarios, as well as on established benchmarks such as AITZ and Meta-GUI, with 24.59\%$\sim$87.29\% improvements in task success rate. OS-Kairos facilitates an adaptive human-agent collaboration, prioritizing effectiveness, generality, scalability, and efficiency for real-world GUI interaction. The dataset and codes are available at this https URL. 

**Abstract (ZH)**: 由多模态大语言模型驱动的自主图形用户界面（GUI）代理展现了巨大的潜力。然而，一个关键而未被充分探索的问题依然存在：过度执行，代理在完全自主执行任务时，缺乏对其行为信心的充分评估，从而破坏了适应性的人机协作。在复杂场景中，如模糊用户指令、意外中断和环境干预的情况下，这带来了显著的风险。为了解决这一问题，我们提出了OS-Kairos，这是一种能够预测每次交互步骤的信心水平并在必要时寻求人类干预以决定是否自主行动的适应性GUI代理。OS-Kairos通过两种关键机制开发：（i）协作探针，为每个交互步骤标注信心分数；（ii）信心驱动交互，利用这些信心分数以实现适应性交互的能力。实验结果显示，OS-Kairos在我们的精心设计的数据集以及AITZ和Meta-GUI等标准基准上表现出色，任务成功率提高了24.59%至87.29%。OS-Kairos促进了适应性的人机协作，优先考虑实际应用场景中GUI交互的有效性、通用性、可扩展性和效率。数据集和代码可在以下链接获取：this https URL 

---
# Beyond Final Answers: Evaluating Large Language Models for Math Tutoring 

**Title (ZH)**: 超越最终答案：评估大型语言模型在数学辅导中的应用 

**Authors**: Adit Gupta, Jennifer Reddig, Tommaso Calo, Daniel Weitekamp, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16460)  

**Abstract**: Researchers have made notable progress in applying Large Language Models (LLMs) to solve math problems, as demonstrated through efforts like GSM8k, ProofNet, AlphaGeometry, and MathOdyssey. This progress has sparked interest in their potential use for tutoring students in mathematics. However, the reliability of LLMs in tutoring contexts -- where correctness and instructional quality are crucial -- remains underexplored. Moreover, LLM problem-solving capabilities may not necessarily translate into effective tutoring support for students. In this work, we present two novel approaches to evaluate the correctness and quality of LLMs in math tutoring contexts. The first approach uses an intelligent tutoring system for college algebra as a testbed to assess LLM problem-solving capabilities. We generate benchmark problems using the tutor, prompt a diverse set of LLMs to solve them, and compare the solutions to those generated by the tutor. The second approach evaluates LLM as tutors rather than problem solvers. We employ human evaluators, who act as students seeking tutoring support from each LLM. We then assess the quality and correctness of the support provided by the LLMs via a qualitative coding process. We applied these methods to evaluate several ChatGPT models, including 3.5 Turbo, 4, 4o, o1-mini, and o1-preview. Our findings show that when used as problem solvers, LLMs generate correct final answers for 85.5% of the college algebra problems tested. When employed interactively as tutors, 90% of LLM dialogues show high-quality instructional support; however, many contain errors -- only 56.6% are entirely correct. We conclude that, despite their potential, LLMs are not yet suitable as intelligent tutors for math without human oversight or additional mechanisms to ensure correctness and quality. 

**Abstract (ZH)**: 研究人员在利用大型语言模型（LLMs）解决数学问题方面取得了显著进展，如GSM8k、ProofNet、AlphaGeometry和MathOdyssey等努力所展示的那样。这一进展引发了它们在数学教学中的潜在应用兴趣。然而，LLMs在教学情境下的可靠性——正确性和教学质量至关重要——仍需进一步探索。此外，LLMs的解题能力不一定能转化为有效的学生教学支持。在本项工作中，我们提出了两种新的方法来评估LLMs在数学教学情境下的正确性和质量。第一种方法使用一个大学代数智能教学系统作为实验平台，评估LLMs的解题能力。我们利用该系统生成基准问题，促使一系列不同的LLMs求解这些问题，并将它们的答案与教师生成的答案进行比较。第二种方法将LLMs评估为辅导者而非解题者。我们采用了人类评价者，他们作为寻求LLMs教学支持的学生。然后，我们通过定性的编码过程来评估LLMs提供的支持的质量和正确性。我们应用这些方法评估了多个ChatGPT模型，包括3.5 Turbo、4、4o、o1-mini和o1-preview。研究结果显示，当作为解题者使用时，LLMs在测试的大学代数问题中生成正确最终答案的比例为85.5%。当作为交互式辅导者使用时，90%的LLM对话显示出高质量的教学支持，但其中许多包含错误，仅56.6%是完全正确的。因此，尽管潜力巨大，但在没有人类监督或额外机制确保正确性和质量的情况下，LLMs尚未适合用作数学智能辅导者。 

---
# Integrating Personality into Digital Humans: A Review of LLM-Driven Approaches for Virtual Reality 

**Title (ZH)**: 将人格融入数字人类：基于LLM的虚拟现实方法综述 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16457)  

**Abstract**: The integration of large language models (LLMs) into virtual reality (VR) environments has opened new pathways for creating more immersive and interactive digital humans. By leveraging the generative capabilities of LLMs alongside multimodal outputs such as facial expressions and gestures, virtual agents can simulate human-like personalities and emotions, fostering richer and more engaging user experiences. This paper provides a comprehensive review of methods for enabling digital humans to adopt nuanced personality traits, exploring approaches such as zero-shot, few-shot, and fine-tuning. Additionally, it highlights the challenges of integrating LLM-driven personality traits into VR, including computational demands, latency issues, and the lack of standardized evaluation frameworks for multimodal interactions. By addressing these gaps, this work lays a foundation for advancing applications in education, therapy, and gaming, while fostering interdisciplinary collaboration to redefine human-computer interaction in VR. 

**Abstract (ZH)**: 大型语言模型（LLMs）与虚拟现实（VR）环境的集成开启了创建更具沉浸感和互动性的数字人类的新途径。通过利用LLMs的生成能力以及面部表情和手势等多模态输出，虚拟代理可以模拟类似人类的性格和情绪，促进更丰富和更具吸引力的用户体验。本文提供了一种全面的方法回顾，探讨了零样本、少量样本和微调等方法，使数字人类能够采用细腻的性格特征。同时，本文还突出了将LLMs驱动的性格特征集成到VR中所面临的挑战，包括计算需求、延迟问题以及缺乏针对多模态交互的标准评估框架。通过解决这些差距，本文为基础视听觉交互应用在教育、治疗和游戏领域的发展奠定了基础，并促进了跨学科合作，以重新定义VR中的交互方式。 

---
# Position: Beyond Assistance -- Reimagining LLMs as Ethical and Adaptive Co-Creators in Mental Health Care 

**Title (ZH)**: 位置：超越辅助——重新构想在心理健康护理中作为伦理化和适应性共创者的语言模型 

**Authors**: Abeer Badawi, Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, Shaina Raza, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16456)  

**Abstract**: This position paper argues for a fundamental shift in how Large Language Models (LLMs) are integrated into the mental health care domain. We advocate for their role as co-creators rather than mere assistive tools. While LLMs have the potential to enhance accessibility, personalization, and crisis intervention, their adoption remains limited due to concerns about bias, evaluation, over-reliance, dehumanization, and regulatory uncertainties. To address these challenges, we propose two structured pathways: SAFE-i (Supportive, Adaptive, Fair, and Ethical Implementation) Guidelines for ethical and responsible deployment, and HAAS-e (Human-AI Alignment and Safety Evaluation) Framework for multidimensional, human-centered assessment. SAFE-i provides a blueprint for data governance, adaptive model engineering, and real-world integration, ensuring LLMs align with clinical and ethical standards. HAAS-e introduces evaluation metrics that go beyond technical accuracy to measure trustworthiness, empathy, cultural sensitivity, and actionability. We call for the adoption of these structured approaches to establish a responsible and scalable model for LLM-driven mental health support, ensuring that AI complements-rather than replaces-human expertise. 

**Abstract (ZH)**: 这一立场论文主张在精神健康护理领域对大型语言模型（LLMs）的整合方式进行根本性转变。我们提倡将其视为协作共创者而非仅仅是辅助工具。尽管LLMs有可能提高可及性、个性化和危机干预，但由于偏见、评估、过度依赖、去人性化和监管不确定性等方面的担忧，其采用仍受到限制。为解决这些挑战，我们提出两条结构化的路径：SAFE-i（支持性、适应性、公平性和伦理实施）准则，用于负责任和伦理的部署，以及HAAS-e（人类-人工智能契合与安全性评估）框架，用于多维度的人本评估。SAFE-i为数据治理、适应性模型工程和实际整合提供了蓝图，确保LLMs符合临床和伦理标准。HAAS-e引入了超越技术准确性的评估标准，以衡量可信度、同理心、文化敏感性和可操作性。我们呼吁采用这些结构化方法，建立LLM驱动精神健康支持的负责任和可扩展模式，确保人工智能补充而非取代人类专长。 

---
# DreamLLM-3D: Affective Dream Reliving using Large Language Model and 3D Generative AI 

**Title (ZH)**: DreamLLM-3D: 基于大型语言模型和3D生成AI的情绪梦境重温 

**Authors**: Pinyao Liu, Keon Ju Lee, Alexander Steinmaurer, Claudia Picard-Deland, Michelle Carr, Alexandra Kitson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16439)  

**Abstract**: We present DreamLLM-3D, a composite multimodal AI system behind an immersive art installation for dream re-experiencing. It enables automated dream content analysis for immersive dream-reliving, by integrating a Large Language Model (LLM) with text-to-3D Generative AI. The LLM processes voiced dream reports to identify key dream entities (characters and objects), social interaction, and dream sentiment. The extracted entities are visualized as dynamic 3D point clouds, with emotional data influencing the color and soundscapes of the virtual dream environment. Additionally, we propose an experiential AI-Dreamworker Hybrid paradigm. Our system and paradigm could potentially facilitate a more emotionally engaging dream-reliving experience, enhancing personal insights and creativity. 

**Abstract (ZH)**: 我们提出DreamLLM-3D，一个结合多模态AI系统的沉浸式艺术装置，用于梦境再体验。该系统通过将大型语言模型（LLM）与文本到3D生成AI集成，实现自动化的梦境内容分析，以支持沉浸式梦境再体验。此外，我们提出了 experiential AI-Dreamworker Hybrid 帕累托改进。我们的系统和帕累托改进有望促进更富有情感共鸣的梦境再体验，增强个人洞察力和创造力。 

---
# Haunted House: A text-based game for comparing the flexibility of mental models in humans and LLMs 

**Title (ZH)**: 鬼屋：基于文本的游戏，用于比较人类和大语言模型的心理模型灵活性 

**Authors**: Brett Puppart, Paul-Henry Paltmann, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2503.16437)  

**Abstract**: This study introduces "Haunted House" a novel text-based game designed to compare the performance of humans and large language models (LLMs) in model-based reasoning. Players must escape from a house containing nine rooms in a 3x3 grid layout while avoiding the ghost. They are guided by verbal clues that they get each time they move. In Study 1, the results from 98 human participants revealed a success rate of 31.6%, significantly outperforming seven state-of-the-art LLMs tested. Out of 140 attempts across seven LLMs, only one attempt resulted in a pass by Claude 3 Opus. Preliminary results suggested that GPT o3-mini-high performance might be higher, but not at the human level. Further analysis of 29 human participants' moves in Study 2 indicated that LLMs frequently struggled with random and illogical moves, while humans exhibited such errors less frequently. Our findings suggest that current LLMs encounter difficulties in tasks that demand active model-based reasoning, offering inspiration for future benchmarks. 

**Abstract (ZH)**: 这项研究介绍了“鬼屋”这款新型文本游戏，旨在通过模型基于的推理任务比较人类和大型语言模型（LLMs）的表现。玩家必须在一个3x3网格布局的房子里通过九个房间，同时避免遇鬼。他们通过每次移动获得言语线索。研究1的98名人类参与者的结果显示成功率为31.6%，显著优于测试的七种最先进的LLMs。在七种LLM的140次尝试中，仅Claude 3 Opus有一次通过。初步结果显示，GPT o3-mini-high性能可能更高，但尚未达到人类水平。研究2中29名人类参与者的行为分析表明，LLMs在处理随机和不合逻辑的移动时经常遇到困难，而 humans则较少出现此类错误。我们的研究发现表明，当前的LLMs在需要主动模型推理的任务中存在困难，为未来的基准测试提供了启示。 

---
