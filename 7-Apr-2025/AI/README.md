# Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning 

**Title (ZH)**: 更大的语言模型意味着更好的推理能力？一种推理能力预训练缩放定律 

**Authors**: Xinyi Wang, Shawn Tan, Mingyu Jin, William Yang Wang, Rameswar Panda, Yikang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03635)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种要求复杂推理的任务中展现了 remarkable 能力。然而，缩放对其推理能力的影响依然不够清晰。在本文中，我们介绍了一个合成的多跳推理环境，旨在紧密复制真实世界大规模知识图谱的结构和分布。我们的推理任务涉及补全图中的缺失边，这需要高级的多跳推理并模拟真实世界的推理场景。为了评估这一点，我们从不完整的图的三元组中从零开始预训练语言模型（LMs），并评估其推断缺失边的能力。有趣的是，我们观察到过度参数化可能由于过度记忆而损害推理性能。我们研究了影响这一凹形损失曲线的各种因素，包括图结构、模型大小和训练步数。为了预测特定知识图谱的最佳模型大小，我们发现了一种经验缩放，它线性地将知识图谱搜索熵映射到最优模型大小。本文为大型语言模型（LLMs）中缩放与推理之间的关系提供了新的见解，揭示了可能优化其推理任务性能的方式。 

---
# Towards deployment-centric multimodal AI beyond vision and language 

**Title (ZH)**: 面向部署的多模态AI超越视觉和语言 

**Authors**: Xianyuan Liu, Jiayang Zhang, Shuo Zhou, Thijs L. van der Plas, Avish Vijayaraghavan, Anastasiia Grishina, Mengdie Zhuang, Daniel Schofield, Christopher Tomlinson, Yuhan Wang, Ruizhe Li, Louisa van Zeeland, Sina Tabakhi, Cyndie Demeocq, Xiang Li, Arunav Das, Orlando Timmerman, Thomas Baldwin-McDonald, Jinge Wu, Peizhen Bai, Zahraa Al Sahili, Omnia Alwazzan, Thao N. Do, Mohammod N.I. Suvon, Angeline Wang, Lucia Cipolina-Kun, Luigi A. Moretti, Lucas Farndale, Nitisha Jain, Natalia Efremova, Yan Ge, Marta Varela, Hak-Keung Lam, Oya Celiktutan, Ben R. Evans, Alejandro Coca-Castro, Honghan Wu, Zahraa S. Abdallah, Chen Chen, Valentin Danchev, Nataliya Tkachenko, Lei Lu, Tingting Zhu, Gregory G. Slabaugh, Roger K. Moore, William K. Cheung, Peter H. Charlton, Haiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03603)  

**Abstract**: Multimodal artificial intelligence (AI) integrates diverse types of data via machine learning to improve understanding, prediction, and decision-making across disciplines such as healthcare, science, and engineering. However, most multimodal AI advances focus on models for vision and language data, while their deployability remains a key challenge. We advocate a deployment-centric workflow that incorporates deployment constraints early to reduce the likelihood of undeployable solutions, complementing data-centric and model-centric approaches. We also emphasise deeper integration across multiple levels of multimodality and multidisciplinary collaboration to significantly broaden the research scope beyond vision and language. To facilitate this approach, we identify common multimodal-AI-specific challenges shared across disciplines and examine three real-world use cases: pandemic response, self-driving car design, and climate change adaptation, drawing expertise from healthcare, social science, engineering, science, sustainability, and finance. By fostering multidisciplinary dialogue and open research practices, our community can accelerate deployment-centric development for broad societal impact. 

**Abstract (ZH)**: 多模态人工智能通过机器学习整合多种类型的数据，以提高跨医学、科学和工程等领域的问题理解、预测和决策能力。然而，大多数多模态人工智能的进步集中在视觉和语言数据模型上，部署仍是一个关键挑战。我们提倡一种以部署为中心的工作流程，早期整合部署约束以减少不可部署解决方案的可能性，同时补充数据为中心和模型为中心的方法。我们还强调在多个层面实现更深层次的多模态融合，并促进跨学科合作，以显著拓宽研究范围，超越视觉和语言领域。为了实现这一目标，我们确定了跨学科共享的多模态人工智能特定挑战，并分析了三个实际应用案例：疫情应对、自动驾驶汽车设计和气候变化适应，吸取了医学、社会科学、工程学、科学、可持续发展和金融领域的专业知识。通过促进跨学科对话和开放研究实践，我们的社群可以加速以部署为中心的发展，以实现广泛的社会影响。 

---
# Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web 

**Title (ZH)**: Talk2X —— 一个促进基于LLM的聊天机器人在网络部署的开源工具包 

**Authors**: Lars Krupp, Daniel Geißler, Peter Hevesi, Marco Hirsch, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2504.03343)  

**Abstract**: Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web. 

**Abstract (ZH)**: LLM驱动的聊天机器人集成到网站中，提供了导航和信息检索的替代手段，改变了用户访问网络信息的方式。然而，主要的封闭源解决方案限制了其在网络主机中的普及，并且在实现细节和能源效率方面缺乏透明度。在此工作中，我们提出了我们公开提供的代理Talk2X，它利用调整后的检索增强生成方法(RAG)结合自动生成的向量数据库，以促进能源效率。Talk2X的架构可以通用化到任意网站，为开发人员提供易于集成的工具。通过混合方法评估Talk2X的易用性，要求用户从开放科学仓库中获取特定资产。与标准用户-网站交互相比，Talk2X显著提高了任务完成时间、正确性和用户体验，支持用户快速找到特定信息。我们的发现为网络信息访问方式的持续范式转变贡献了技术进步。 

---
# Monte Carlo Graph Coloring 

**Title (ZH)**: 蒙特卡洛图着色 

**Authors**: Tristan Cazenave, Benjamin Negrevergne, Florian Sikora  

**Link**: [PDF](https://arxiv.org/pdf/2504.03277)  

**Abstract**: Graph Coloring is probably one of the most studied and famous problem in graph algorithms. Exact methods fail to solve instances with more than few hundred vertices, therefore, a large number of heuristics have been proposed. Nested Monte Carlo Search (NMCS) and Nested Rollout Policy Adaptation (NRPA) are Monte Carlo search algorithms for single player games. Surprisingly, few work has been dedicated to evaluating Monte Carlo search algorithms to combinatorial graph problems. In this paper we expose how to efficiently apply Monte Carlo search to Graph Coloring and compare this approach to existing ones. 

**Abstract (ZH)**: 图着色可能是图算法中研究最多和最著名的问题之一。精确方法无法解决具有几百个以上顶点的实例，因此提出了大量的启发式方法。嵌套蒙特卡洛搜索（NMCS）和嵌套展开策略适应（NRPA）是单人游戏的蒙特卡洛搜索算法。令人惊讶的是，很少有工作专门评估蒙特卡洛搜索算法在组合图问题上的效果。在这项工作中，我们展示了如何有效地将蒙特卡洛搜索应用于图着色，并将这种方法与现有方法进行比较。 

---
# Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators 

**Title (ZH)**: 所见即所信：基于基础模型的不确定性估计的信念空间规划 

**Authors**: Linfeng Zhao, Willie McClinton, Aidan Curtis, Nishanth Kumar, Tom Silver, Leslie Pack Kaelbling, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03245)  

**Abstract**: Generalizable robotic mobile manipulation in open-world environments poses significant challenges due to long horizons, complex goals, and partial observability. A promising approach to address these challenges involves planning with a library of parameterized skills, where a task planner sequences these skills to achieve goals specified in structured languages, such as logical expressions over symbolic facts. While vision-language models (VLMs) can be used to ground these expressions, they often assume full observability, leading to suboptimal behavior when the agent lacks sufficient information to evaluate facts with certainty. This paper introduces a novel framework that leverages VLMs as a perception module to estimate uncertainty and facilitate symbolic grounding. Our approach constructs a symbolic belief representation and uses a belief-space planner to generate uncertainty-aware plans that incorporate strategic information gathering. This enables the agent to effectively reason about partial observability and property uncertainty. We demonstrate our system on a range of challenging real-world tasks that require reasoning in partially observable environments. Simulated evaluations show that our approach outperforms both vanilla VLM-based end-to-end planning or VLM-based state estimation baselines by planning for and executing strategic information gathering. This work highlights the potential of VLMs to construct belief-space symbolic scene representations, enabling downstream tasks such as uncertainty-aware planning. 

**Abstract (ZH)**: 开放世界环境中的通用机器人移动操作面临着长期展望、复杂目标以及部分可观测性的显著挑战。一种有前景的方法是使用参数化技能库进行规划，其中任务规划器将这些技能序列化以实现用结构化语言（如符号事实的逻辑表达式）指定的目标。虽然视觉语言模型可以用于实现这些表达式的语义化，但它们通常假设完全可观测性，当代理缺乏足够的信息来确定性地评估事实时，会导致次优行为。本文介绍了一种新颖的框架，利用视觉语言模型作为感知模块来估计不确定性并促进符号化语义化。我们的方法构建了符号性信念表示，并使用信念空间规划器生成考虑策略性信息收集的不确定性意识计划。这使代理能够有效地推理部分可观测性和属性不确定性。我们在一系列需要在部分可观测环境中进行推理的具有挑战性的实际任务上展示了我们的系统。模拟评估表明，与基于视觉语言模型的端到端规划或基于视觉语言模型的状态估计基线相比，我们的方法通过计划和执行策略性的信息收集来表现出色。这项工作强调了视觉语言模型在构建信念空间符号化场景表示方面的潜力，从而支持后续任务如不确定性意识规划。 

---
# DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments 

**Title (ZH)**: DeepResearcher: 通过强化学习在现实环境中扩大深度研究 

**Authors**: Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03160)  

**Abstract**: Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the ability to formulate plans, cross-validate information from multiple sources, engage in self-reflection to redirect research, and maintain honesty when unable to find definitive answers. Our results highlight that end-to-end training in real-world web environments is not merely an implementation detail but a fundamental requirement for developing robust research capabilities aligned with real-world applications. We release DeepResearcher at this https URL. 

**Abstract (ZH)**: 具有网络搜索能力的大语言模型在深入研究任务中展现了 impressive 的潜力。然而，当前的方法主要依赖于手工设计的提示（基于提示工程的方法）或在受控检索增强生成（RAG）环境中使用强化学习（RAG-基于的方法），这些方法在捕捉现实世界互动的复杂性方面显得力不从心。本文介绍了 DeepResearcher，这是首个通过在现实世界环境中扩展强化学习训练基于大语言模型的深度研究代理的全面框架。与假设所有必要信息都存在于固定语料库中的 RAG-基于的方法不同，我们的方法训练代理能够应对开放网络的嘈杂、无结构和动态性。我们实现了一个专有的多代理架构，其中浏览代理从各种网页结构中提取相关信息，并克服了显著的技术挑战。在开放领域研究任务上的广泛实验表明，DeepResearcher 在基于提示工程的基线方法上取得了高达 28.9 分点的显著改进，在基于 RAG 的强化学习代理上取得了高达 7.2 分点的改进。我们的情 qualitative 分析揭示了端到端强化学习训练中出现的认知行为，包括制定计划、从多个来源验证信息、进行自我反思以重新定向研究、以及在无法找到确切答案时保持诚实的能力。我们的结果强调，在现实世界网络环境中进行端到端训练不仅是实现细节，而是开发与实际应用相契合的强大研究能力的基本要求。我们在此 https:// 指向的地址发布了 DeepResearcher。 

---
# LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph 

**Title (ZH)**: LightPROF：面向知识图谱的大语言模型轻量级推理框架 

**Authors**: Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui Wang, Tat-Seng Chua, Xiao Zhang, Zhen Cai  

**Link**: [PDF](https://arxiv.org/pdf/2504.03137)  

**Abstract**: Large Language Models (LLMs) have impressive capabilities in text understanding and zero-shot reasoning. However, delays in knowledge updates may cause them to reason incorrectly or produce harmful results. Knowledge Graphs (KGs) provide rich and reliable contextual information for the reasoning process of LLMs by structurally organizing and connecting a wide range of entities and relations. Existing KG-based LLM reasoning methods only inject KGs' knowledge into prompts in a textual form, ignoring its structural information. Moreover, they mostly rely on close-source models or open-source models with large parameters, which poses challenges to high resource consumption. To address this, we propose a novel Lightweight and efficient Prompt learning-ReasOning Framework for KGQA (LightPROF), which leverages the full potential of LLMs to tackle complex reasoning tasks in a parameter-efficient manner. Specifically, LightPROF follows a "Retrieve-Embed-Reason process", first accurately, and stably retrieving the corresponding reasoning graph from the KG through retrieval module. Next, through a Transformer-based Knowledge Adapter, it finely extracts and integrates factual and structural information from the KG, then maps this information to the LLM's token embedding space, creating an LLM-friendly prompt to be used by the LLM for the final reasoning. Additionally, LightPROF only requires training Knowledge Adapter and can be compatible with any open-source LLM. Extensive experiments on two public KGQA benchmarks demonstrate that LightPROF achieves superior performance with small-scale LLMs. Furthermore, LightPROF shows significant advantages in terms of input token count and reasoning time. 

**Abstract (ZH)**: 基于知识图谱的大语言模型轻量化高效提示学习推理框架（LightPROF） 

---
# Language Models Guidance with Multi-Aspect-Cueing: A Case Study for Competitor Analysis 

**Title (ZH)**: 多方面线索引导的语言模型应用：竞争对手分析案例研究 

**Authors**: Amir Hadifar, Christopher Ochs, Arjan Van Ewijk  

**Link**: [PDF](https://arxiv.org/pdf/2504.02984)  

**Abstract**: Competitor analysis is essential in modern business due to the influence of industry rivals on strategic planning. It involves assessing multiple aspects and balancing trade-offs to make informed decisions. Recent Large Language Models (LLMs) have demonstrated impressive capabilities to reason about such trade-offs but grapple with inherent limitations such as a lack of knowledge about contemporary or future realities and an incomplete understanding of a market's competitive landscape. In this paper, we address this gap by incorporating business aspects into LLMs to enhance their understanding of a competitive market. Through quantitative and qualitative experiments, we illustrate how integrating such aspects consistently improves model performance, thereby enhancing analytical efficacy in competitor analysis. 

**Abstract (ZH)**: 竞争对手分析对于现代企业战略规划至关重要，因为它受行业竞争对手的影响。这涉及评估多个方面并平衡权衡以做出明智的决策。近年来的大规模语言模型展示了在处理这种权衡方面的强大能力，但面临着诸如缺乏对当前或未来现实的了解以及对市场竞争格局的不完全理解等固有局限性。在本文中，我们通过将商业方面纳入大规模语言模型来弥补这一差距，以增强其对竞争市场的理解。通过定量和定性实验，我们展示了整合此类方面如何一致地提高模型性能，从而增强竞争对手分析中的分析功效。 

---
# Bonsai: Interpretable Tree-Adaptive Grounded Reasoning 

**Title (ZH)**: Bonsai: 可解释的树适应性 grounded 推理 

**Authors**: Kate Sanders, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03640)  

**Abstract**: To develop general-purpose collaborative agents, humans need reliable AI systems that can (1) adapt to new domains and (2) transparently reason with uncertainty to allow for verification and correction. Black-box models demonstrate powerful data processing abilities but do not satisfy these criteria due to their opaqueness, domain specificity, and lack of uncertainty awareness. We introduce Bonsai, a compositional and probabilistic reasoning system that generates adaptable inference trees by retrieving relevant grounding evidence and using it to compute likelihoods of sub-claims derived from broader natural language inferences. Bonsai's reasoning power is tunable at test-time via evidence scaling and it demonstrates reliable handling of varied domains including transcripts, photographs, videos, audio, and databases. Question-answering and human alignment experiments demonstrate that Bonsai matches the performance of domain-specific black-box methods while generating interpretable, grounded, and uncertainty-aware reasoning traces. 

**Abstract (ZH)**: 开发通用协作代理，需要可靠的AI系统，这些系统能够（1）适应新领域，（2）透明地处理不确定性以允许验证和纠正。虽然黑盒模型展示了强大的数据处理能力，但由于其不透明性、领域特定性和缺乏不确定性意识，无法满足这些标准。我们引入了Bonsai，这是一种组合性和概率性推理系统，通过检索相关基础证据并使用这些证据计算源自更广泛的自然语言推理的子命题的似然性，生成适应性推理树。Bonsai的推理能力可以在测试时通过证据缩放进行调节，并且在包括转录、照片、视频、音频和数据库在内的各种领域中展现出可靠的处理能力。问答和人类对齐实验表明，Bonsai在性能上达到了领域特定黑盒方法的水平，同时生成可解释、基于证据和不确定性意识的推理轨迹。 

---
# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models 

**Title (ZH)**: Nemotron-H: 一类准确高效的混合Mamba-Transformer模型 

**Authors**: NVIDIA, Aaron Blakeman, Aarti Basant, Abhinav Khattar, Adithya Renduchintala, Akhiad Bercovich, Aleksander Ficek, Alexis Bjorlin, Ali Taghibakhshi, Amala Sanjay Deshmukh, Ameya Sunil Mahabaleshwarkar, Andrew Tao, Anna Shors, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Bobby Chen, Boris Ginsburg, Boxin Wang, Brandon Norick, Brian Butterfield, Bryan Catanzaro, Carlo del Mundo, Chengyu Dong, Christine Harvey, Christopher Parisien, Dan Su, Daniel Korzekwa, Danny Yin, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Denys Fridman, Dima Rekesh, Ding Ma, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Dusan Stosic, Eileen Long, Elad Segal, Ellie Evans, Eric Chung, Erick Galinkin, Evelina Bakhturina, Ewa Dobrowolska, Fei Jia, Fuxiao Liu, Gargi Prasad, Gerald Shen, Guilin Liu, Guo Chen, Haifeng Qian, Helen Ngo, Hongbin Liu, Hui Li, Igor Gitman, Ilia Karmanov, Ivan Moshkov, Izik Golan, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jarno Seppanen, Jason Lu, Jason Sewall, Jiaqi Zeng, Jiaxuan You, Jimmy Zhang, Jing Zhang, Jining Huang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jon Barker, Jonathan Cohen, Joseph Jennings, Jupinder Parmar, Karan Sapra, Kari Briski, Kateryna Chumachenko, Katherine Luna, Keshav Santhanam, Kezhi Kong, Kirthi Sivamani, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Lawrence McAfee, Leon Derczynski, Lindsey Pavao, Luis Vega, Lukas Voegtle, Maciej Bala, Maer Rodrigues de Melo, Makesh Narsimhan Sreedhar, Marcin Chochowski, Markus Kliegl  

**Link**: [PDF](https://arxiv.org/pdf/2504.03624)  

**Abstract**: As inference-time scaling becomes critical for enhanced reasoning capabilities, it is increasingly becoming important to build models that are efficient to infer. We introduce Nemotron-H, a family of 8B and 56B/47B hybrid Mamba-Transformer models designed to reduce inference cost for a given accuracy level. To achieve this goal, we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token. We show that Nemotron-H models offer either better or on-par accuracy compared to other similarly-sized state-of-the-art open-sourced Transformer models (e.g., Qwen-2.5-7B/72B and Llama-3.1-8B/70B), while being up to 3$\times$ faster at inference. To further increase inference speed and reduce the memory required at inference time, we created Nemotron-H-47B-Base from the 56B model using a new compression via pruning and distillation technique called MiniPuzzle. Nemotron-H-47B-Base achieves similar accuracy to the 56B model, but is 20% faster to infer. In addition, we introduce an FP8-based training recipe and show that it can achieve on par results with BF16-based training. This recipe is used to train the 56B model. All Nemotron-H models will be released, with support in Hugging Face, NeMo, and Megatron-LM. 

**Abstract (ZH)**: 基于推理时可扩展性的增强推理能力：Nemotron-H系列模型的设计与实现 

---
# Align to Structure: Aligning Large Language Models with Structural Information 

**Title (ZH)**: 结构对齐：大型语言模型与结构信息的对齐 

**Authors**: Zae Myung Kim, Anand Ramachandran, Farideh Tavazoee, Joo-Kyung Kim, Oleg Rokhlenko, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03622)  

**Abstract**: Generating long, coherent text remains a challenge for large language models (LLMs), as they lack hierarchical planning and structured organization in discourse generation. We introduce Structural Alignment, a novel method that aligns LLMs with human-like discourse structures to enhance long-form text generation. By integrating linguistically grounded discourse frameworks into reinforcement learning, our approach guides models to produce coherent and well-organized outputs. We employ a dense reward scheme within a Proximal Policy Optimization framework, assigning fine-grained, token-level rewards based on the discourse distinctiveness relative to human writing. Two complementary reward models are evaluated: the first improves readability by scoring surface-level textual features to provide explicit structuring, while the second reinforces deeper coherence and rhetorical sophistication by analyzing global discourse patterns through hierarchical discourse motifs, outperforming both standard and RLHF-enhanced models in tasks such as essay generation and long-document summarization. All training data and code will be publicly shared at this https URL. 

**Abstract (ZH)**: Generating 长篇连贯文本仍然是大规模语言模型（LLMs）的挑战，因为它们在话语生成中缺乏层次规划和结构化组织。我们引入了结构对齐方法，该方法将LLMs与类似人类的话语结构对齐，以提高长文本生成能力。通过将基于语言学的话语框架整合到强化学习中，我们的方法引导模型生成连贯且组织良好的输出。我们采用密集奖励方案，在近端策略优化框架中，基于话语独特性相对于人类写作进行细粒度、token级别奖励的分配。两种互补的奖励模型进行了评估：第一个通过评分表层文本特征来提高可读性，提供明确的结构化指导，而第二个通过分析通过分层话语模式来强化更深层次的连贯性和修辞 sophistication，超越了标准模型和基于RLHF的增强模型，在诸如论文生成和长文档摘要等任务中表现更佳。所有训练数据和代码将在以下网址公开：this https URL。 

---
# Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task 

**Title (ZH)**: 多语言检索增强生成用于知识密集型任务 

**Authors**: Leonardo Ranaldi, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03616)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone of contemporary NLP, enhancing large language models (LLMs) by allowing them to access richer factual contexts through in-context retrieval. While effective in monolingual settings, especially in English, its use in multilingual tasks remains unexplored. This paper investigates the effectiveness of RAG across multiple languages by proposing novel approaches for multilingual open-domain question-answering. We evaluate the performance of various multilingual RAG strategies, including question-translation (tRAG), which translates questions into English before retrieval, and Multilingual RAG (MultiRAG), where retrieval occurs directly across multiple languages. Our findings reveal that tRAG, while useful, suffers from limited coverage. In contrast, MultiRAG improves efficiency by enabling multilingual retrieval but introduces inconsistencies due to cross-lingual variations in the retrieved content. To address these issues, we propose Crosslingual RAG (CrossRAG), a method that translates retrieved documents into a common language (e.g., English) before generating the response. Our experiments show that CrossRAG significantly enhances performance on knowledge-intensive tasks, benefiting both high-resource and low-resource languages. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为当代NLP的基石，通过上下文检索使大规模语言模型（LLMs）能够访问更丰富的事实性上下文。尽管在单语环境，尤其是英语环境中非常有效，但它在多语言任务中的应用尚未被探索。本文通过提出新的多语言开放域问答方法，考察了RAG在多种语言环境下的有效性。我们评估了各种多语言RAG策略的表现，包括问题翻译（tRAG），即将问题翻译成英语后再进行检索，以及多语言RAG（MultiRAG），其中检索可以直接跨越多种语言进行。我们的研究发现，tRAG虽然有用，但覆盖范围有限。相比之下，MultiRAG通过允许多语言检索提高了效率，但由于检索内容在跨语言中的差异性，引入了不一致性。为了解决这些问题，我们提出了跨语言RAG（CrossRAG）方法，在生成响应前将检索到的文档翻译成一种通用语言（如英语）。我们的实验结果显示，CrossRAG在知识密集型任务上显著提升了性能，受益于高资源和低资源语言。 

---
# Autonomous and Self-Adapting System for Synthetic Media Detection and Attribution 

**Title (ZH)**: 自主适应性合成媒体检测与归属系统 

**Authors**: Aref Azizpour, Tai D. Nguyen, Matthew C. Stamm  

**Link**: [PDF](https://arxiv.org/pdf/2504.03615)  

**Abstract**: Rapid advances in generative AI have enabled the creation of highly realistic synthetic images, which, while beneficial in many domains, also pose serious risks in terms of disinformation, fraud, and other malicious applications. Current synthetic image identification systems are typically static, relying on feature representations learned from known generators; as new generative models emerge, these systems suffer from severe performance degradation. In this paper, we introduce the concept of an autonomous self-adaptive synthetic media identification system -- one that not only detects synthetic images and attributes them to known sources but also autonomously identifies and incorporates novel generators without human intervention. Our approach leverages an open-set identification strategy with an evolvable embedding space that distinguishes between known and unknown sources. By employing an unsupervised clustering method to aggregate unknown samples into high-confidence clusters and continuously refining its decision boundaries, our system maintains robust detection and attribution performance even as the generative landscape evolves. Extensive experiments demonstrate that our method significantly outperforms existing approaches, marking a crucial step toward universal, adaptable forensic systems in the era of rapidly advancing generative models. 

**Abstract (ZH)**: 快速发展的生成式AI使高度逼真的合成图像得以创建，尽管在许多领域有益，但也带来了信息发布误导、欺诈及其他恶意应用的重大风险。当前的合成图像识别系统通常是静态的，依赖于从已知生成器学习到的特征表示；随着新生成模型的出现，这些系统会遭受严重的性能退化。本文介绍了一种自主自适应合成媒体识别系统——不仅能检测合成图像并将其归属到已知来源，还能自主识别和整合新的生成器而不需人工干预。我们的方法利用开放集识别策略和可进化的嵌入空间，以区分已知和未知来源。通过使用无监督聚类方法将未知样本聚类为高置信度群组，并不断优化其决策边界，我们的系统能够在生成模型不断发展的背景下保持稳健的检测和归属性能。详尽的实验表明，我们的方法显著优于现有方法，标志着向适应快速发展的生成模型时代通用可适应的取证系统的关键一步。 

---
# APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay 

**Title (ZH)**: APIGen-MT: 由模拟代理-人类互动驱动的多轮数据生成管线 

**Authors**: Akshara Prabhakar, Zuxin Liu, Weiran Yao, Jianguo Zhang, Ming Zhu, Shiyu Wang, Zhiwei Liu, Tulika Awalgaonkar, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03601)  

**Abstract**: Training effective AI agents for multi-turn interactions requires high-quality data that captures realistic human-agent dynamics, yet such data is scarce and expensive to collect manually. We introduce APIGen-MT, a two-phase framework that generates verifiable and diverse multi-turn agent data. In the first phase, our agentic pipeline produces detailed task blueprints with ground-truth actions, leveraging a committee of LLM reviewers and iterative feedback loops. These blueprints are then transformed into complete interaction trajectories through simulated human-agent interplay. We train a family of models -- the xLAM-2-fc-r series with sizes ranging from 1B to 70B parameters. Our models outperform frontier models such as GPT-4o and Claude 3.5 on $\tau$-bench and BFCL benchmarks, with the smaller models surpassing their larger counterparts, particularly in multi-turn settings, while maintaining superior consistency across multiple trials. Comprehensive experiments demonstrate that our verified blueprint-to-details approach yields high-quality training data, enabling the development of more reliable, efficient, and capable agents. We open-source both the synthetic data collected and the trained xLAM-2-fc-r models to advance research in AI agents. Models are available on HuggingFace at this https URL and project website is this https URL 

**Abstract (ZH)**: 训练高效的多轮交互AI代理需要高质量的数据来捕捉现实的人机动态，但这类数据稀缺且手动收集成本高昂。我们介绍了一种两阶段框架APIGen-MT，用于生成可验证和多样的多轮交互代理数据。在第一阶段，我们的代理管道利用LLM评审员委员会和迭代反馈循环生成详细的任务蓝图，包含真实动作。随后，这些蓝图被转换为完整的交互轨迹，通过模拟的人机交互过程。我们训练了一系列模型——包括从1亿到70亿参数的xLAM-2-fc-r系列。我们的模型在$\tau$-bench和BFCL基准上优于前沿模型如GPT-4o和Claude 3.5，小型模型在多轮交互设置中尤其优于大型模型，同时在多次试验中保持了更高的一致性。全面的实验表明，我们的验证蓝图到详细信息的方法生成了高质量的训练数据，使得能够开发出更可靠、更高效和更有能力的代理。我们开源了收集的合成数据和训练的xLAM-2-fc-r模型，以促进代理人的AI研究。模型可在HuggingFace上获取，链接为这个=https://huggingface.co/，项目网站为这个=https://。 

---
# MedSAM2: Segment Anything in 3D Medical Images and Videos 

**Title (ZH)**: MedSAM2: 三维医学图像和视频中的实例分割 

**Authors**: Jun Ma, Zongxin Yang, Sumin Kim, Bihui Chen, Mohammed Baharoon, Adibvafa Fallahpour, Reza Asakereh, Hongwei Lyu, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03600)  

**Abstract**: Medical image and video segmentation is a critical task for precision medicine, which has witnessed considerable progress in developing task or modality-specific and generalist models for 2D images. However, there have been limited studies on building general-purpose models for 3D images and videos with comprehensive user studies. Here, we present MedSAM2, a promptable segmentation foundation model for 3D image and video segmentation. The model is developed by fine-tuning the Segment Anything Model 2 on a large medical dataset with over 455,000 3D image-mask pairs and 76,000 frames, outperforming previous models across a wide range of organs, lesions, and imaging modalities. Furthermore, we implement a human-in-the-loop pipeline to facilitate the creation of large-scale datasets resulting in, to the best of our knowledge, the most extensive user study to date, involving the annotation of 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames, demonstrating that MedSAM2 can reduce manual costs by more than 85%. MedSAM2 is also integrated into widely used platforms with user-friendly interfaces for local and cloud deployment, making it a practical tool for supporting efficient, scalable, and high-quality segmentation in both research and healthcare environments. 

**Abstract (ZH)**: 可提示的3D医学图像与视频分割基础模型MedSAM2 

---
# EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline 

**Title (ZH)**: EnrichIndex：使用LLMs离线丰富检索索引 

**Authors**: Peter Baile Chen, Tomer Wolfson, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.03598)  

**Abstract**: Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs. 

**Abstract (ZH)**: 现有的信息检索系统在目标文档的语言与用户查询语言高度匹配时表现出色。然而，现实中的检索系统往往需要隐式推断文档的相关性。例如，检索技术文本或表格时，其相关性可能通过特定的专业术语或结构隐含表达，而非明确体现在内容中。大型语言模型（LLMs）通过利用其推理能力，具有识别此类隐含相关性的巨大潜力。然而，当前的LLM增强检索受到高延迟和计算成本的限制，因为LLM通常在线上为每个查询重新计算查询-文档的相关性。为解决这一问题，我们引入了EnrichIndex，这是一种检索方法，通过在摄取数据时对所有文档进行一次扫描，使用LLM离线构建语义增强的检索索引。此外，语义增强的索引可以补充现有的在线检索方法，提升LLM重排序的性能。我们在五个涉及段落和表格的检索任务上评估了EnrichIndex，发现它优于强大的在线LLM基线系统，平均提升召回率@10分别为11.7分和NDCG@10分别为10.6分。在线调用LLM时，它处理的令牌数量减少了293.3倍，大大降低了在线延迟和成本。总体而言，EnrichIndex是一种有效的方法，通过利用LLMs的强大推理能力，在线下构建更好的检索索引。 

---
# Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin for Real-World Robot Policy Evaluation 

**Title (ZH)**: 实即虚：通过动态数字孪生桥接仿真到现实的差距以评估真实世界机器人政策 

**Authors**: Jad Abou-Chakra, Lingfeng Sun, Krishan Rana, Brandon May, Karl Schmeckpeper, Maria Vittoria Minniti, Laura Herlant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03597)  

**Abstract**: Recent advancements in behavior cloning have enabled robots to perform complex manipulation tasks. However, accurately assessing training performance remains challenging, particularly for real-world applications, as behavior cloning losses often correlate poorly with actual task success. Consequently, researchers resort to success rate metrics derived from costly and time-consuming real-world evaluations, making the identification of optimal policies and detection of overfitting or underfitting impractical. To address these issues, we propose real-is-sim, a novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment. By continuously aligning the simulated world with the physical world, demonstrations can be collected in the real world with states extracted from the simulator. The simulator enables flexible state representations by rendering image inputs from any viewpoint or extracting low-level state information from objects embodied within the scene. During training, policies can be directly evaluated within the simulator in an offline and highly parallelizable manner. Finally, during deployment, policies are run within the simulator where the real robot directly tracks the simulated robot's joints, effectively decoupling policy execution from real hardware and mitigating traditional domain-transfer challenges. We validate real-is-sim on the PushT manipulation task, demonstrating strong correlation between success rates obtained in the simulator and real-world evaluations. Videos of our system can be found at this https URL. 

**Abstract (ZH)**: 最近行为克隆的进步使机器人能够执行复杂的操作任务。然而，准确评估训练性能仍然颇具挑战性，特别是在实际应用中，因为行为克隆损失往往与实际任务成功率的相关性较差。因此，研究人员转向通过昂贵且耗时的实地评估获取的成功率指标，这使得识别最优策略和检测过拟合或欠拟合变得 impractical。为解决这些问题，我们提出了一种新颖的行为克隆框架——real-is-sim，该框架在整个策略开发管道（数据收集、训练和部署）中均采用了动态数字孪生（基于 Embodied Gaussians）。通过持续使模拟世界与物理世界保持一致，可以在现实世界中收集演示，同时从模拟器中提取状态。模拟器通过从任何视角生成图像输入或提取嵌入场景中的物体的底层状态信息，提供了灵活的状态表示方式。在训练过程中，策略可以直接在模拟器中离线且高度并行地进行评估。最后，在部署阶段，策略在模拟器中运行，真实机器人直接跟踪模拟机器人关节的实际动作，从而将策略执行与真实硬件解耦，并减轻传统的领域迁移挑战。我们在 PushT 操作任务上验证了 real-is-sim，展示了模拟器和实地评估中成功率之间的强相关性。有关系统视频请访问 this https URL。 

---
# SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement 

**Title (ZH)**: SynWorld: 有agency的动作知识精炼的虚拟场景合成 

**Authors**: Runnan Fang, Xiaobin Wang, Yuan Liang, Shuofei Qiao, Jialong Wu, Zekun Xi, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03561)  

**Abstract**: In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at this https URL. 

**Abstract (ZH)**: 基于代理与环境的交互，在代理扩展其能力的过程中，它们通过规划和执行行动来实现。然而，当逻辑语言模型（LLM）驱动的代理部署在新颖环境中或需要导航非标准行动空间时，它们面临着重大挑战。为了使代理能够自主探索环境、优化工作流并增强对行动的理解，我们提出了SynWorld框架，该框架允许代理合成多步行动调用的可能场景，并使用蒙特卡洛树搜索（MCTS）探索来有效精炼其当前环境中的行动知识。我们的实验表明，SynWorld是一种有效且通用的方法，用于在新环境中学习行动知识。代码可在以下网址获得：this https URL。 

---
# Agentic Knowledgeable Self-awareness 

**Title (ZH)**: 代理知识型自我意识 

**Authors**: Shuofei Qiao, Zhisong Qiu, Baochang Ren, Xiaobin Wang, Xiangyuan Ru, Ningyu Zhang, Xiang Chen, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03553)  

**Abstract**: Large Language Models (LLMs) have achieved considerable performance across various agentic planning tasks. However, traditional agent planning approaches adopt a "flood irrigation" methodology that indiscriminately injects gold trajectories, external feedback, and domain knowledge into agent models. This practice overlooks the fundamental human cognitive principle of situational self-awareness during decision-making-the ability to dynamically assess situational demands and strategically employ resources during decision-making. We propose agentic knowledgeable self-awareness to address this gap, a novel paradigm enabling LLM-based agents to autonomously regulate knowledge utilization. Specifically, we propose KnowSelf, a data-centric approach that applies agents with knowledgeable self-awareness like humans. Concretely, we devise a heuristic situation judgement criterion to mark special tokens on the agent's self-explored trajectories for collecting training data. Through a two-stage training process, the agent model can switch between different situations by generating specific special tokens, achieving optimal planning effects with minimal costs. Our experiments demonstrate that KnowSelf can outperform various strong baselines on different tasks and models with minimal use of external knowledge. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类代理规划任务中取得了显著性能。然而，传统的代理规划方法采用了一种“泛灌式”方法，随意注入黄金轨迹、外部反馈和领域知识到代理模型中。这一做法忽视了决策过程中情景自我意识的基本人类认知原则——在决策时动态评估情境需求并战略性地运用资源的能力。我们提出代理知识型自我意识来填补这一空白，这是一种新型范式，使基于LLM的代理能够自主调节知识的应用。具体地，我们提出了KnowSelf，一种以数据为中心的方法，使代理具有类似人类的知识型自我意识。具体而言，我们设计了一种启发式的情景判断标准，在代理探索的轨迹上标记特殊标记以收集训练数据。通过两阶段训练过程，代理模型可以根据生成特定特殊标记在不同情境间切换，实现最小成本下的最优规划效果。我们的实验结果显示，KnowSelf在不同任务和模型上能够优于多种强大基线，且对外部知识的使用最少。代码可在以下链接获取。 

---
# MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation 

**Title (ZH)**: MultiMed-ST：大规模多对多医疗语音多语言翻译 

**Authors**: Khai Le-Duc, Tuyen Tran, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03546)  

**Abstract**: Multilingual speech translation (ST) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: this https URL. 

**Abstract (ZH)**: 多语言医疗语音翻译在医疗领域的应用通过消除语言障碍实现高效沟通，缓解专业人员短缺，并在疫情等期间促进诊断和治疗的改进。在本文中，我们首次系统地研究了医疗领域的语音翻译，通过发布包含六种语言（越南语、英语、德语、法语、繁体中文、简体中文）所有翻译方向的大型语音翻译数据集MultiMed-ST及相应的模型，对该数据集进行了全面分析，它是最大的医疗机器翻译数据集，也是所有领域中最大的多对多多语言语音翻译数据集。此外，我们还进行了迄今为止最广泛的语音翻译研究，包括经验基线、双语-多语比较研究、端到端与级联比较研究、特定任务与多任务序列到序列比较研究、代码转换分析以及定量-定性错误分析。所有代码、数据和模型均可在线获取：this https URL。 

---
# Dense Neural Network Based Arrhythmia Classification on Low-cost and Low-compute Micro-controller 

**Title (ZH)**: 基于密集神经网络的低-cost低计算量微控制器心律失常分类 

**Authors**: Md Abu Obaida Zishan, H M Shihab, Sabik Sadman Islam, Maliha Alam Riya, Gazi Mashrur Rahman, Jannatun Noor  

**Link**: [PDF](https://arxiv.org/pdf/2504.03531)  

**Abstract**: The electrocardiogram (ECG) monitoring device is an expensive albeit essential device for the treatment and diagnosis of cardiovascular diseases (CVD). The cost of this device typically ranges from $2000 to $10000. Several studies have implemented ECG monitoring systems in micro-controller units (MCU) to reduce industrial development costs by up to 20 times. However, to match industry-grade systems and display heartbeats effectively, it is essential to develop an efficient algorithm for detecting arrhythmia (irregular heartbeat). Hence in this study, a dense neural network is developed to detect arrhythmia on the Arduino Nano. The Nano consists of the ATMega328 microcontroller with a 16MHz clock, 2KB of SRAM, and 32KB of program memory. Additionally, the AD8232 SparkFun Single-Lead Heart Rate Monitor is used as the ECG sensor. The implemented neural network model consists of two layers (excluding the input) with 10 and four neurons respectively with sigmoid activation function. However, four approaches are explored to choose the appropriate activation functions. The model has a size of 1.267 KB, achieves an F1 score (macro-average) of 78.3\% for classifying four types of arrhythmia, an accuracy rate of 96.38%, and requires 0.001314 MOps of floating-point operations (FLOPs). 

**Abstract (ZH)**: 基于Arduino Nano的心电图心动过速检测方法研究 

---
# Quantifying Robustness: A Benchmarking Framework for Deep Learning Forecasting in Cyber-Physical Systems 

**Title (ZH)**: 定量评估鲁棒性：在网络物理系统中深度学习预测的基准框架 

**Authors**: Alexander Windmann, Henrik Steude, Daniel Boschmann, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.03494)  

**Abstract**: Cyber-Physical Systems (CPS) in domains such as manufacturing and energy distribution generate complex time series data crucial for Prognostics and Health Management (PHM). While Deep Learning (DL) methods have demonstrated strong forecasting capabilities, their adoption in industrial CPS remains limited due insufficient robustness. Existing robustness evaluations primarily focus on formal verification or adversarial perturbations, inadequately representing the complexities encountered in real-world CPS scenarios. To address this, we introduce a practical robustness definition grounded in distributional robustness, explicitly tailored to industrial CPS, and propose a systematic framework for robustness evaluation. Our framework simulates realistic disturbances, such as sensor drift, noise and irregular sampling, enabling thorough robustness analyses of forecasting models on real-world CPS datasets. The robustness definition provides a standardized score to quantify and compare model performance across diverse datasets, assisting in informed model selection and architecture design. Through extensive empirical studies evaluating prominent DL architectures (including recurrent, convolutional, attention-based, modular, and structured state-space models) we demonstrate the applicability and effectiveness of our approach. We publicly release our robustness benchmark to encourage further research and reproducibility. 

**Abstract (ZH)**: 工业 CPS 领域（如制造业和能源分配）生成的复杂时间序列数据对于预测与健康管理 (PHM) 至关重要。尽管深度学习 (DL) 方法在预测能力上表现出色，但其在工业 CPS 中的应用仍因鲁棒性不足而受到限制。现有的鲁棒性评估主要集中在形式验证或对抗性扰动上，未能充分反映现实世界 CPS 场景中的复杂性。为解决这一问题，我们引入了一个基于分布鲁棒性的实用鲁棒性定义，该定义特别针对工业 CPS 设计，并提出了一套系统的鲁棒性评估框架。该框架模拟了实际干扰，如传感器漂移、噪声和不规则采样，从而对实际工业 CPS 数据集上的预测模型进行全面的鲁棒性分析。鲁棒性定义提供了一个标准化评分，用于衡量和比较模型在不同数据集上的性能，有助于模型选择和架构设计的知情决策。通过广泛的实际研究评估了主流 DL 架构（包括循环、卷积、注意力、模块化和结构化状态空间模型），我们证明了我们方法的适用性和有效性。我们公开发布了鲁棒性基准，以鼓励进一步的研究和可再现性。 

---
# BUFF: Bayesian Uncertainty Guided Diffusion Probabilistic Model for Single Image Super-Resolution 

**Title (ZH)**: BUFF: 基于贝叶斯不确定性引导扩散的概率模型单张图像超分辨率 

**Authors**: Zihao He, Shengchuan Zhang, Runze Hu, Yunhang Shen, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03490)  

**Abstract**: Super-resolution (SR) techniques are critical for enhancing image quality, particularly in scenarios where high-resolution imagery is essential yet limited by hardware constraints. Existing diffusion models for SR have relied predominantly on Gaussian models for noise generation, which often fall short when dealing with the complex and variable texture inherent in natural scenes. To address these deficiencies, we introduce the Bayesian Uncertainty Guided Diffusion Probabilistic Model (BUFF). BUFF distinguishes itself by incorporating a Bayesian network to generate high-resolution uncertainty masks. These masks guide the diffusion process, allowing for the adjustment of noise intensity in a manner that is both context-aware and adaptive. This novel approach not only enhances the fidelity of super-resolved images to their original high-resolution counterparts but also significantly mitigates artifacts and blurring in areas characterized by complex textures and fine details. The model demonstrates exceptional robustness against complex noise patterns and showcases superior adaptability in handling textures and edges within images. Empirical evidence, supported by visual results, illustrates the model's robustness, especially in challenging scenarios, and its effectiveness in addressing common SR issues such as blurring. Experimental evaluations conducted on the DIV2K dataset reveal that BUFF achieves a notable improvement, with a +0.61 increase compared to baseline in SSIM on BSD100, surpassing traditional diffusion approaches by an average additional +0.20dB PSNR gain. These findings underscore the potential of Bayesian methods in enhancing diffusion processes for SR, paving the way for future advancements in the field. 

**Abstract (ZH)**: Bayesian不确定性引导扩散概率模型（ BUFF）：用于增强图像质量的复杂纹理场景超分辨率 

---
# Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej 

**Title (ZH)**: 印度结构化法律文件生成：一种基于模型的包装器方法与VidhikDastaavej 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03486)  

**Abstract**: Automating legal document drafting can significantly enhance efficiency, reduce manual effort, and streamline legal workflows. While prior research has explored tasks such as judgment prediction and case summarization, the structured generation of private legal documents in the Indian legal domain remains largely unaddressed. To bridge this gap, we introduce VidhikDastaavej, a novel, anonymized dataset of private legal documents, and develop NyayaShilp, a fine-tuned legal document generation model specifically adapted to Indian legal texts. We propose a Model-Agnostic Wrapper (MAW), a two-step framework that first generates structured section titles and then iteratively produces content while leveraging retrieval-based mechanisms to ensure coherence and factual accuracy. We benchmark multiple open-source LLMs, including instruction-tuned and domain-adapted versions, alongside proprietary models for comparison. Our findings indicate that while direct fine-tuning on small datasets does not always yield improvements, our structured wrapper significantly enhances coherence, factual adherence, and overall document quality while mitigating hallucinations. To ensure real-world applicability, we developed a Human-in-the-Loop (HITL) Document Generation System, an interactive user interface that enables users to specify document types, refine section details, and generate structured legal drafts. This tool allows legal professionals and researchers to generate, validate, and refine AI-generated legal documents efficiently. Extensive evaluations, including expert assessments, confirm that our framework achieves high reliability in structured legal drafting. This research establishes a scalable and adaptable foundation for AI-assisted legal drafting in India, offering an effective approach to structured legal document generation. 

**Abstract (ZH)**: 自动化法律文书起草可以显著提升效率、减少手动工作量并简化法律工作流程。尽管前期研究已经探索了判决预测和案例总结等任务，印度法律领域的结构化生成私密法律文书仍然 largely未被关注。为弥补这一不足，我们引入了 VidhikDastaavej，一个新型的匿名私密法律文书数据集，并开发了 NyayaShilp，一种专门适应印度法律文本的微调法律文书生成模型。我们提出了一种模型无关包装器（MAW），这是一种两步框架，首先生成结构化的部分标题，然后利用检索机制迭代生产内容，以确保连贯性和事实准确性。我们使用多种开源LLM进行基准测试，包括指令微调和领域适应版本，以及对照专有模型。研究结果表明，虽然直接在小数据集上进行微调并不总是能够带来改进，但我们的结构化包装器在提高连贯性、事实准确性和总体文档质量方面表现出色，并减少了幻觉现象。为了确保实际应用，我们开发了一个人机交互文档生成系统（HITL），这是一种交互式用户界面，使用户能够指定文档类型、细化部分详情并生成结构化的法律草案。该工具使法律专业人士和研究人员能够有效地生成、验证和细化AI生成的法律文书。广泛的评估，包括专家评估，证实了我们框架在结构化法律起草中的高可靠性。本研究为印度法律辅助起草奠定了可扩展和适应性强的基础，提供了一种有效的结构化法律文书生成方法。 

---
# Physics-informed 4D X-ray image reconstruction from ultra-sparse spatiotemporal data 

**Title (ZH)**: 基于物理约束的4D X射线图像超稀疏空时数据重建 

**Authors**: Zisheng Yao, Yuhe Zhang, Zhe Hu, Robert Klöfkorn, Tobias Ritschel, Pablo Villanueva-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.03469)  

**Abstract**: The unprecedented X-ray flux density provided by modern X-ray sources offers new spatiotemporal possibilities for X-ray imaging of fast dynamic processes. Approaches to exploit such possibilities often result in either i) a limited number of projections or spatial information due to limited scanning speed, as in time-resolved tomography, or ii) a limited number of time points, as in stroboscopic imaging, making the reconstruction problem ill-posed and unlikely to be solved by classical reconstruction approaches. 4D reconstruction from such data requires sample priors, which can be included via deep learning (DL). State-of-the-art 4D reconstruction methods for X-ray imaging combine the power of AI and the physics of X-ray propagation to tackle the challenge of sparse views. However, most approaches do not constrain the physics of the studied process, i.e., a full physical model. Here we present 4D physics-informed optimized neural implicit X-ray imaging (4D-PIONIX), a novel physics-informed 4D X-ray image reconstruction method combining the full physical model and a state-of-the-art DL-based reconstruction method for 4D X-ray imaging from sparse views. We demonstrate and evaluate the potential of our approach by retrieving 4D information from ultra-sparse spatiotemporal acquisitions of simulated binary droplet collisions, a relevant fluid dynamic process. We envision that this work will open new spatiotemporal possibilities for various 4D X-ray imaging modalities, such as time-resolved X-ray tomography and more novel sparse acquisition approaches like X-ray multi-projection imaging, which will pave the way for investigations of various rapid 4D dynamics, such as fluid dynamics and composite testing. 

**Abstract (ZH)**: 现代X射线源提供的前所未有的X射线通量密度为快速动态过程的X射线成像提供了新的时空可能性。利用此类可能性的方法往往会导致时间分辨层析成像中投影或空间信息有限，或者在荧光成像中时间点有限，从而使重建问题变得病态，难以通过经典重建方法解决。从稀疏数据中进行4D重建需要样本先验，可以通过深度学习（DL）纳入其中。最新的4D重建方法结合了人工智能和X射线传播的物理原理，以应对稀疏视图挑战。然而，大多数方法并未约束所研究过程的物理定律，即整个物理模型。我们提出了4D物理约束优化神经隐式X射线成像（4D-PIONIX），这是一种结合完整物理模型和基于DL的4D X射线成像稀疏视图先进重建方法的新颖物理约束4D X射线图像重建方法。通过从模拟二元液滴碰撞的超稀疏时空采集中检索4D信息，我们展示了并评估了我们方法的潜力。我们设想，这项工作将为时间分辨X射线层析成像以及X射线多投影成像等更新型稀疏采集方法开辟新的时空可能性，从而为介观动力学和复合材料测试等各种快速4D动力学的研究铺平道路。 

---
# SpectR: Dynamically Composing LM Experts with Spectral Routing 

**Title (ZH)**: SpectR: 动态组合语言模型专家的谱路由方法 

**Authors**: William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03454)  

**Abstract**: Training large, general-purpose language models poses significant challenges. The growing availability of specialized expert models, fine-tuned from pretrained models for specific tasks or domains, offers a promising alternative. Leveraging the potential of these existing expert models in real-world applications requires effective methods to select or merge the models best suited for a given task. This paper introduces SPECTR, an approach for dynamically composing expert models at each time step during inference. Notably, our method requires no additional training and enables flexible, token- and layer-wise model combinations. Our experimental results demonstrate that SPECTR improves routing accuracy over alternative training-free methods, increasing task performance across expert domains. 

**Abstract (ZH)**: 大语言模型的训练面临着重大挑战。随着专业化专家模型的可用性不断提升，这些模型是从预训练模型微调而来，专门用于特定任务或领域，为真实世界应用提供了有 promise 的替代方案。利用这些现有专家模型的潜力需要有效的方法来选择或组合最适合给定任务的模型。本文介绍了 SPECTR，一种在推理的每个时间步骤动态组合专家模型的方法。值得注意的是，我们的方法无需额外训练，并允许灵活的、按token和层级的模型组合。我们的实验结果表明，与替代的无需训练的方法相比，SPECTR 在不同专家领域提高了任务性能。 

---
# The AI Cosmologist I: An Agentic System for Automated Data Analysis 

**Title (ZH)**: AI天体哲学家I：自主系统及其在自动化数据分析中的应用 

**Authors**: Adam Moss  

**Link**: [PDF](https://arxiv.org/pdf/2504.03424)  

**Abstract**: We present the AI Cosmologist, an agentic system designed to automate cosmological/astronomical data analysis and machine learning research workflows. This implements a complete pipeline from idea generation to experimental evaluation and research dissemination, mimicking the scientific process typically performed by human researchers. The system employs specialized agents for planning, coding, execution, analysis, and synthesis that work together to develop novel approaches. Unlike traditional auto machine-learning systems, the AI Cosmologist generates diverse implementation strategies, writes complete code, handles execution errors, analyzes results, and synthesizes new approaches based on experimental outcomes. We demonstrate the AI Cosmologist capabilities across several machine learning tasks, showing how it can successfully explore solution spaces, iterate based on experimental results, and combine successful elements from different approaches. Our results indicate that agentic systems can automate portions of the research process, potentially accelerating scientific discovery. The code and experimental data used in this paper are available on GitHub at this https URL. Example papers included in the appendix demonstrate the system's capability to autonomously produce complete scientific publications, starting from only the dataset and task description 

**Abstract (ZH)**: 我们介绍了一智能宇宙学家，这是一种自主系统，旨在自动化宇宙学/天文学数据解析及机器学习研究流程。该系统实现从想法生成到实验评估和研究成果发布的完整工作流程，模拟人类研究人员通常执行的科学过程。该系统运用专门的代理进行规划、编程、执行、分析和综合，共同开发新的方法。与传统的自动机器学习系统不同，智能宇宙学家能够生成多样化的实施策略，编写完整的代码，处理执行错误，分析结果，并根据实验结果综合新的方法。我们在多个机器学习任务中展示了智能宇宙学家的能力，显示出它如何成功探索解空间、基于实验结果迭代以及结合不同方法的成功元素。我们的结果表明，自主系统可以自动化研究过程的部分环节，可能加速科学发现。本文使用的代码和实验数据可在 GitHub 上找到：this https URL。附录中的示例论文展示了该系统从仅数据集和任务描述开始，自主产生完整科学出版物的能力。 

---
# Autonomous state-space segmentation for Deep-RL sparse reward scenarios 

**Title (ZH)**: 自主状态空间分割用于深度强化学习稀疏奖励场景 

**Authors**: Gianluca Maselli, Vieri Giuliano Santucci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03420)  

**Abstract**: Dealing with environments with sparse rewards has always been crucial for systems developed to operate in autonomous open-ended learning settings. Intrinsic Motivations could be an effective way to help Deep Reinforcement Learning algorithms learn in such scenarios. In fact, intrinsic reward signals, such as novelty or curiosity, are generally adopted to improve exploration when extrinsic rewards are delayed or absent. Building on previous works, we tackle the problem of learning policies in the presence of sparse rewards by proposing a two-level architecture that alternates an ''intrinsically driven'' phase of exploration and autonomous sub-goal generation, to a phase of sparse reward, goal-directed policy learning. The idea is to build several small networks, each one specialized on a particular sub-path, and use them as starting points for future exploration without the need to further explore from scratch previously learnt paths. Two versions of the system have been trained and tested in the Gym SuperMarioBros environment without considering any additional extrinsic reward. The results show the validity of our approach and the importance of autonomously segment the environment to generate an efficient path towards the final goal. 

**Abstract (ZH)**: 处理稀疏奖励环境一直是为自主开放式学习环境开发的系统的关键问题。内在动机可能是帮助深度强化学习算法在这种场景下学习的有效方法。实际上，诸如新颖性或好奇心之类的内在奖励信号通常被采用，以改进探索，尤其是当外部奖励延迟或缺失时。在此基础上，我们通过提出交替进行“内在驱动”的探索和自主子目标生成阶段与稀疏奖励、目标导向策略学习阶段的两层架构来解决在稀疏奖励环境下的策略学习问题。该架构旨在构建多个专门针对特定子路径的小型网络，并将它们用作未来探索的起点，而无需从头开始进一步探索先前学习的路径。在Gym SuperMarioBros环境中，该系统在不考虑任何额外外部奖励的情况下进行了训练和测试。实验结果证明了我们方法的有效性，并强调了自主分割环境以生成高效路径的重要性。 

---
# Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning 

**Title (ZH)**: 面向推理导向的强化学习在线难度过滤 

**Authors**: Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, Donghyun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2504.03380)  

**Abstract**: Reasoning-Oriented Reinforcement Learning (RORL) enhances the reasoning ability of Large Language Models (LLMs). However, due to the sparsity of rewards in RORL, effective training is highly dependent on the selection of problems of appropriate difficulty. Although curriculum learning attempts to address this by adjusting difficulty, it often relies on static schedules, and even recent online filtering methods lack theoretical grounding and a systematic understanding of their effectiveness. In this work, we theoretically and empirically show that curating the batch with the problems that the training model achieves intermediate accuracy on the fly can maximize the effectiveness of RORL training, namely balanced online difficulty filtering. We first derive that the lower bound of the KL divergence between the initial and the optimal policy can be expressed with the variance of the sampled accuracy. Building on those insights, we show that balanced filtering can maximize the lower bound, leading to better performance. Experimental results across five challenging math reasoning benchmarks show that balanced online filtering yields an additional 10% in AIME and 4% improvements in average over plain GRPO. Moreover, further analysis shows the gains in sample efficiency and training time efficiency, exceeding the maximum reward of plain GRPO within 60% training time and the volume of the training set. 

**Abstract (ZH)**: 基于推理的强化学习（RORL）增强了大型语言模型（LLMs）的推理能力。然而，由于RORL中奖励稀疏，有效的训练高度依赖于适当难度问题的选择。尽管课程学习试图通过调整难度来应对这一问题，但通常依赖于静态计划，而且即使是最近的在线筛选方法也缺乏理论依据和对其有效性的系统理解。在本工作中，我们从理论上和实验上证明，动态筛选训练模型在运行中达到中间准确率的问题批次可以最大化RORL训练的效果，即平衡在线难度筛选。我们首先推导出初始策略和最优策略之间的KL散度下界可以用采样准确率的方差来表示。基于这些见解，我们展示了平衡筛选可以使下界最大化，从而提高性能。跨五个具有挑战性的数学推理基准的实验结果显示，平衡在线筛选在AIME上额外提升了10%，在平均值上提升了4%，超过了普通GRPO的性能。进一步分析表明，平衡在线筛选提高了样本效率和训练时间效率，在60%的训练时间内超过了普通GRPO的最大奖励，且在训练集规模上也有所超越。 

---
# Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency 

**Title (ZH)**: 边缘AI中可持续的大型语言模型推理：评估量化大型语言模型的能效、输出准确性和推理延迟 

**Authors**: Erik Johannes Husom, Arda Goknil, Merve Astekin, Lwin Khin Shar, Andre Kåsen, Sagar Sen, Benedikt Andreas Mithassel, Ahmet Soylu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03360)  

**Abstract**: Deploying Large Language Models (LLMs) on edge devices presents significant challenges due to computational constraints, memory limitations, inference speed, and energy consumption. Model quantization has emerged as a key technique to enable efficient LLM inference by reducing model size and computational overhead. In this study, we conduct a comprehensive analysis of 28 quantized LLMs from the Ollama library, which applies by default Post-Training Quantization (PTQ) and weight-only quantization techniques, deployed on an edge device (Raspberry Pi 4 with 4GB RAM). We evaluate energy efficiency, inference performance, and output accuracy across multiple quantization levels and task types. Models are benchmarked on five standardized datasets (CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, and HumanEval), and we employ a high-resolution, hardware-based energy measurement tool to capture real-world power consumption. Our findings reveal the trade-offs between energy efficiency, inference speed, and accuracy in different quantization settings, highlighting configurations that optimize LLM deployment for resource-constrained environments. By integrating hardware-level energy profiling with LLM benchmarking, this study provides actionable insights for sustainable AI, bridging a critical gap in existing research on energy-aware LLM deployment. 

**Abstract (ZH)**: 将大型语言模型（LLMs）部署在边缘设备上由于计算限制、内存限制、推理速度和能耗等方面提出了显著挑战。模型量化已成为一种关键技术，通过减小模型大小和计算开销，从而实现高效的LLM推理。在本研究中，我们对Ollama库中的28个量化LLM进行了全面分析，这些模型默认使用后训练量化（PTQ）和权重唯量化技术，并在具有4GB RAM的Raspberry Pi 4边缘设备上部署。我们评估了不同量化级别和任务类型下的能效、推理性能和输出准确性。模型在五个标准化数据集（CommonsenseQA、BIG-Bench Hard、TruthfulQA、GSM8K和HumanEval）上进行了基准测试，并使用高分辨率的硬件基能测量工具捕获实际能耗。我们的研究结果揭示了在不同量化设置下能效、推理速度和准确性的权衡，并突显了针对资源受限环境优化LLM部署的配置。通过结合硬件级能效分析与LLM基准测试，本研究为可持续AI提供了可操作的见解，填补了现有研究中能量感知LLM部署的关键空白。 

---
# Decentralized Collective World Model for Emergent Communication and Coordination 

**Title (ZH)**: 去中心化集体世界模型用于 emergent 通信与协调 

**Authors**: Kentaro Nomura, Tatsuya Aoki, Tadahiro Taniguchi, Takato Horii  

**Link**: [PDF](https://arxiv.org/pdf/2504.03353)  

**Abstract**: We propose a fully decentralized multi-agent world model that enables both symbol emergence for communication and coordinated behavior through temporal extension of collective predictive coding. Unlike previous research that focuses on either communication or coordination separately, our approach achieves both simultaneously. Our method integrates world models with communication channels, enabling agents to predict environmental dynamics, estimate states from partial observations, and share critical information through bidirectional message exchange with contrastive learning for message alignment. Using a two-agent trajectory drawing task, we demonstrate that our communication-based approach outperforms non-communicative models when agents have divergent perceptual capabilities, achieving the second-best coordination after centralized models. Importantly, our distributed approach with constraints preventing direct access to other agents' internal states facilitates the emergence of more meaningful symbol systems that accurately reflect environmental states. These findings demonstrate the effectiveness of decentralized communication for supporting coordination while developing shared representations of the environment. 

**Abstract (ZH)**: 我们提出了一种完全去中心化的多智能体世界模型，该模型通过时间扩展的集体预测编码实现通信中的符号涌现和协调行为。与以往专注于通信或协调其中一项的研究不同，我们的方法能够在同时实现两者。我们的方法将世界模型与通信渠道集成，使智能体能够预测环境动力学、从部分观测中估计状态，并通过对比学习进行双向消息交换以优化信息共享。通过一个双智能体轨迹绘制任务，我们展示了在智能体具有不同的感知能力时，基于通信的方法优于非通信模型，并在中心化模型之后实现了次优的协调。重要的是，我们的分布式方法通过限制直接访问其他智能体的内部状态，促进了更具有意义的符号系统的涌现，这些符号系统准确反映了环境状态。这些发现证明了去中心化通信在支持协调并发展环境共享表征方面的有效性。 

---
# EOOD: Entropy-based Out-of-distribution Detection 

**Title (ZH)**: EOOD: 基于熵的分布外检测 

**Authors**: Guide Yang, Chao Hou, Weilong Peng, Xiang Fang, Yongwei Nie, Peican Zhu, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03342)  

**Abstract**: Deep neural networks (DNNs) often exhibit overconfidence when encountering out-of-distribution (OOD) samples, posing significant challenges for deployment. Since DNNs are trained on in-distribution (ID) datasets, the information flow of ID samples through DNNs inevitably differs from that of OOD samples. In this paper, we propose an Entropy-based Out-Of-distribution Detection (EOOD) framework. EOOD first identifies specific block where the information flow differences between ID and OOD samples are more pronounced, using both ID and pseudo-OOD samples. It then calculates the conditional entropy on the selected block as the OOD confidence score. Comprehensive experiments conducted across various ID and OOD settings demonstrate the effectiveness of EOOD in OOD detection and its superiority over state-of-the-art methods. 

**Abstract (ZH)**: 基于熵的异分布检测框架（Entropy-based Out-Of-distribution Detection, EOOD） 

---
# Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification 

**Title (ZH)**: 留意提示：改进声音分类的音频生成提示策略 

**Authors**: Francesca Ronchini, Ho-Hsiang Wu, Wei-Cheng Lin, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03329)  

**Abstract**: This paper investigates the design of effective prompt strategies for generating realistic datasets using Text-To-Audio (TTA) models. We also analyze different techniques for efficiently combining these datasets to enhance their utility in sound classification tasks. By evaluating two sound classification datasets with two TTA models, we apply a range of prompt strategies. Our findings reveal that task-specific prompt strategies significantly outperform basic prompt approaches in data generation. Furthermore, merging datasets generated using different TTA models proves to enhance classification performance more effectively than merely increasing the training dataset size. Overall, our results underscore the advantages of these methods as effective data augmentation techniques using synthetic data. 

**Abstract (ZH)**: 本文研究了使用Text-To-Audio (TTA)模型生成现实数据集的有效提示策略设计。我们还分析了不同技术以高效地结合这些数据集，以增强其在声音分类任务中的实用性。通过使用两种TTA模型评估两种声音分类数据集，我们应用了一系列提示策略。我们的研究结果表明，专门针对任务的提示策略在数据生成方面显著优于基本提示方法。此外，使用不同TTA模型生成的数据集的合并被证明比单纯增加训练数据集规模更有效地提高分类性能。总体而言，我们的结果强调了这些方法作为使用合成数据的有效数据扩增技术的优势。 

---
# Policy Optimization Algorithms in a Unified Framework 

**Title (ZH)**: 统一框架下的策略优化算法 

**Authors**: Shuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03328)  

**Abstract**: Policy optimization algorithms are crucial in many fields but challenging to grasp and implement, often due to complex calculations related to Markov decision processes and varying use of discount and average reward setups. This paper presents a unified framework that applies generalized ergodicity theory and perturbation analysis to clarify and enhance the application of these algorithms. Generalized ergodicity theory sheds light on the steady-state behavior of stochastic processes, aiding understanding of both discounted and average rewards. Perturbation analysis provides in-depth insights into the fundamental principles of policy optimization algorithms. We use this framework to identify common implementation errors and demonstrate the correct approaches. Through a case study on Linear Quadratic Regulator problems, we illustrate how slight variations in algorithm design affect implementation outcomes. We aim to make policy optimization algorithms more accessible and reduce their misuse in practice. 

**Abstract (ZH)**: 通用化 ergodic 理论和扰动分析在强化学习算法统一框架中的应用：提升算法理解和应用 

---
# Noise Augmented Fine Tuning for Mitigating Hallucinations in Large Language Models 

**Title (ZH)**: 噪声增强微调以减轻大型语言模型的幻觉 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03302)  

**Abstract**: Large language models (LLMs) often produce inaccurate or misleading content-hallucinations. To address this challenge, we introduce Noise-Augmented Fine-Tuning (NoiseFiT), a novel framework that leverages adaptive noise injection based on the signal-to-noise ratio (SNR) to enhance model robustness. In particular, NoiseFiT selectively perturbs layers identified as either high-SNR (more robust) or low-SNR (potentially under-regularized) using a dynamically scaled Gaussian noise. We further propose a hybrid loss that combines standard cross-entropy, soft cross-entropy, and consistency regularization to ensure stable and accurate outputs under noisy training conditions. Our theoretical analysis shows that adaptive noise injection is both unbiased and variance-preserving, providing strong guarantees for convergence in expectation. Empirical results on multiple test and benchmark datasets demonstrate that NoiseFiT significantly reduces hallucination rates, often improving or matching baseline performance in key tasks. These findings highlight the promise of noise-driven strategies for achieving robust, trustworthy language modeling without incurring prohibitive computational overhead. Given the comprehensive and detailed nature of our experiments, we have publicly released the fine-tuning logs, benchmark evaluation artifacts, and source code online at W&B, Hugging Face, and GitHub, respectively, to foster further research, accessibility and reproducibility. 

**Abstract (ZH)**: Large语言模型（LLMs）often产生不准确或误导性的内容幻觉。为解决这一挑战，我们引入了噪声增强微调（NoiseFiT）框架，该框架基于信噪比（SNR）适应性地注入噪声以增强模型稳健性。特别是，NoiseFiT针对识别为高SNR（更稳健）或低SNR（可能欠正则化）的层，使用动态缩放的高斯噪声进行选择性扰动。我们还提出了一种混合损失函数，结合了标准交叉熵、软交叉熵和一致性正则化，以确保在嘈杂训练条件下稳定准确的输出。我们的理论分析表明，适应性噪声注入既无偏又保持方差，为期望收敛提供了强大的保证。在多个测试和基准数据集上的实验证明，NoiseFiT显著降低了幻觉率，通常在关键任务上优于或匹配基线性能。这些发现突显了噪声驱动策略在不增加计算开销的情况下实现稳健、可信的语言建模的潜力。鉴于实验的全面性和详细性，我们已将微调日志、基准评估资源和源代码分别.publicly发布在W&B、Hugging Face和GitHub上，以促进进一步的研究、访问性和可再现性。 

---
# Stance-Driven Multimodal Controlled Statement Generation: New Dataset and Task 

**Title (ZH)**: 基于立场的多模态受控语句生成：新数据集与任务 

**Authors**: Bingqian Wang, Quan Fang, Jiachen Sun, Xiaoxiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03295)  

**Abstract**: Formulating statements that support diverse or controversial stances on specific topics is vital for platforms that enable user expression, reshape political discourse, and drive social critique and information dissemination. With the rise of Large Language Models (LLMs), controllable text generation towards specific stances has become a promising research area with applications in shaping public opinion and commercial marketing. However, current datasets often focus solely on pure texts, lacking multimodal content and effective context, particularly in the context of stance detection. In this paper, we formally define and study the new problem of stance-driven controllable content generation for tweets with text and images, where given a multimodal post (text and image/video), a model generates a stance-controlled response. To this end, we create the Multimodal Stance Generation Dataset (StanceGen2024), the first resource explicitly designed for multimodal stance-controllable text generation in political discourse. It includes posts and user comments from the 2024 U.S. presidential election, featuring text, images, videos, and stance annotations to explore how multimodal political content shapes stance expression. Furthermore, we propose a Stance-Driven Multimodal Generation (SDMG) framework that integrates weighted fusion of multimodal features and stance guidance to improve semantic consistency and stance control. We release the dataset and code (this https URL) for public use and further research. 

**Abstract (ZH)**: 基于多模态内容生成的立场驱动可控表达：以2024年美国大选推文为例 

---
# Towards Effective EU E-Participation: The Development of AskThePublic 

**Title (ZH)**: 向有效的欧盟电子参与迈进：AskThePublic的发展 

**Authors**: Kilian Sprenkamp, Nils Messerschmidt, Amir Sartipi, Igor Tchappi, Xiaohui Wu, Liudmila Zavolokina, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03287)  

**Abstract**: E-participation platforms can be an important asset for governments in increasing trust and fostering democratic societies. By engaging non-governmental and private institutions, domain experts, and even the general public, policymakers can make informed and inclusive decisions. Drawing on the Media Richness Theory and applying the Design Science Research method, we explore how a chatbot can be designed to improve the effectiveness of the policy-making process of existing citizen involvement platforms. Leveraging the Have Your Say platform, which solicits feedback on European Commission initiatives and regulations, a Large Language Model based chatbot, called AskThePublic is created, providing policymakers, journalists, researchers, and interested citizens with a convenient channel to explore and engage with public input. By conducting 11 semistructured interviews, the results show that the participants value the interactive and structured responses as well as enhanced language capabilities, thus increasing their likelihood of engaging with AskThePublic over the existing platform. An outlook for future iterations is provided and discussed with regard to the perspectives of the different stakeholders. 

**Abstract (ZH)**: 电子参与平台可以成为政府增加信任和促进民主社会的重要资产。通过与非政府和私营机构、领域专家，甚至普通公众互动，决策者可以作出更为明智和包容的决策。基于媒体丰富性理论和设计科学方法，我们探讨了如何设计聊天机器人以提高现有公民参与平台政策制定过程的有效性。利用征询公众意见平台（Have Your Say），该平台就欧洲委员会倡议和法规征求公众反馈，我们创建了一个基于大规模语言模型的聊天机器人AskThePublic，为决策者、记者、研究人员和感兴趣的公民提供了一个便捷的渠道来探索和参与公众意见。通过进行11次半结构化访谈，结果显示参与者认为交互式和结构化的回应以及增强的语言能力提高了他们使用AskThePublic平台的意愿。对未来迭代的展望与不同利益相关者的视角进行了讨论。 

---
# JanusDDG: A Thermodynamics-Compliant Model for Sequence-Based Protein Stability via Two-Fronts Multi-Head Attention 

**Title (ZH)**: JanusDDG: 一种基于序列的蛋白质稳定性thermodynamics合规模型通过两前沿多头注意力机制 

**Authors**: Guido Barducci, Ivan Rossi, Francesco Codicè, Cesare Rollo, Valeria Repetto, Corrado Pancotti, Virginia Iannibelli, Tiziana Sanavia, Piero Fariselli  

**Link**: [PDF](https://arxiv.org/pdf/2504.03278)  

**Abstract**: Understanding how residue variations affect protein stability is crucial for designing functional proteins and deciphering the molecular mechanisms underlying disease-related mutations. Recent advances in protein language models (PLMs) have revolutionized computational protein analysis, enabling, among other things, more accurate predictions of mutational effects. In this work, we introduce JanusDDG, a deep learning framework that leverages PLM-derived embeddings and a bidirectional cross-attention transformer architecture to predict $\Delta \Delta G$ of single and multiple-residue mutations while simultaneously being constrained to respect fundamental thermodynamic properties, such as antisymmetry and transitivity. Unlike conventional self-attention, JanusDDG computes queries (Q) and values (V) as the difference between wild-type and mutant embeddings, while keys (K) alternate between the two. This cross-interleaved attention mechanism enables the model to capture mutation-induced perturbations while preserving essential contextual information. Experimental results show that JanusDDG achieves state-of-the-art performance in predicting $\Delta \Delta G$ from sequence alone, matching or exceeding the accuracy of structure-based methods for both single and multiple mutations. 

**Abstract (ZH)**: 理解残基变异如何影响蛋白质稳定性对于设计功能性蛋白质和解析与疾病相关的突变分子机制至关重要。蛋白质语言模型（PLMs）的 recent 进展革命性地革新了蛋白质计算分析，使得更准确地预测突变效应成为可能。本文介绍了 JanusDDG，一个利用 PLM 提取的嵌入和双向交叉注意力变换器架构的深度学习框架，用于同时预测单个和多个残基突变的 $\Delta \Delta G$ 值，并遵守热力学基本性质如反对称性和传递性。与传统的自我注意力不同，JanusDDG 计算查询（Q）和值（V）为野生型和突变嵌入之差，而密钥（K）交替使用两者。这种交叉交错的注意力机制使模型能够捕捉突变引起的扰动同时保留重要上下文信息。实验结果表明，JanusDDG 在仅从序列预测 $\Delta \Delta G$ 方面达到了最先进的性能，对单突变和多突变的准确性与结构基于方法相当或超越。 

---
# Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review of Generative Social Simulations 

**Title (ZH)**: 大型语言模型能否解决基于代理的建模问题？对生成性社会模拟的批判性回顾 

**Authors**: Maik Larooij, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.03274)  

**Abstract**: Recent advancements in AI have reinvigorated Agent-Based Models (ABMs), as the integration of Large Language Models (LLMs) has led to the emergence of ``generative ABMs'' as a novel approach to simulating social systems. While ABMs offer means to bridge micro-level interactions with macro-level patterns, they have long faced criticisms from social scientists, pointing to e.g., lack of realism, computational complexity, and challenges of calibrating and validating against empirical data. This paper reviews the generative ABM literature to assess how this new approach adequately addresses these long-standing criticisms. Our findings show that studies show limited awareness of historical debates. Validation remains poorly addressed, with many studies relying solely on subjective assessments of model `believability', and even the most rigorous validation failing to adequately evidence operational validity. We argue that there are reasons to believe that LLMs will exacerbate rather than resolve the long-standing challenges of ABMs. The black-box nature of LLMs moreover limit their usefulness for disentangling complex emergent causal mechanisms. While generative ABMs are still in a stage of early experimentation, these findings question of whether and how the field can transition to the type of rigorous modeling needed to contribute to social scientific theory. 

**Abstract (ZH)**: Recent advancements in AI have reinvigorated Agent-Based Models (ABMs), as the integration of Large Language Models (LLMs) has led to the emergence of ``generative ABMs'' as a novel approach to simulating social systems. 

---
# Verification of Autonomous Neural Car Control with KeYmaera X 

**Title (ZH)**: 基于KeYmaera X对自主神经控制汽车系统的验证 

**Authors**: Enguerrand Prebet, Samuel Teuber, André Platzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03272)  

**Abstract**: This article presents a formal model and formal safety proofs for the ABZ'25 case study in differential dynamic logic (dL). The case study considers an autonomous car driving on a highway avoiding collisions with neighbouring cars. Using KeYmaera X's dL implementation, we prove absence of collision on an infinite time horizon which ensures that safety is preserved independently of trip length. The safety guarantees hold for time-varying reaction time and brake force. Our dL model considers the single lane scenario with cars ahead or behind. We demonstrate that dL with its tools is a rigorous foundation for runtime monitoring, shielding, and neural network verification. Doing so sheds light on inconsistencies between the provided specification and simulation environment highway-env of the ABZ'25 study. We attempt to fix these inconsistencies and uncover numerous counterexamples which also indicate issues in the provided reinforcement learning environment. 

**Abstract (ZH)**: 本文在微分动态逻辑（dL）框架下，提出了一种形式模型以及对ABZ'25案例研究中的形式安全证明。该案例研究考虑了一辆自动驾驶汽车在高速公路上行驶，避免与相邻车辆发生碰撞。利用KeYmaera X的dL实现，我们在无限时间 horizon 上证明了无碰撞条件，确保安全性独立于旅行距离得以保持。这些安全保证适用于时间变化的反应时间和制动强度。我们的dL模型考虑了一车道场景，其中包含前方或后方的车辆。本文证明了dL及其工具可以作为运行时监控、屏蔽和神经网络验证的严谨基础，并揭示了ABZ'25研究中提供规范和仿真环境highway-env之间的一致性问题。本文尝试修正这些问题，并发现了大量反例，这些反例还指出了提供强化学习环境中的问题。 

---
# An Extended Symbolic-Arithmetic Model for Teaching Double-Black Removal with Rotation in Red-Black Trees 

**Title (ZH)**: 扩展的符号算术模型：在红黑树中教学旋转删除双黑节点的方法 

**Authors**: Kennedy E. Ehimwenma, Hongyu Zhou, Junfeng Wang, Ze Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03259)  

**Abstract**: Double-black (DB) nodes have no place in red-black (RB) trees. So when DB nodes are formed, they are immediately removed. The removal of DB nodes that cause rotation and recoloring of other connected nodes poses greater challenges in the teaching and learning of RB trees. To ease this difficulty, this paper extends our previous work on the symbolic arithmetic algebraic (SA) method for removing DB nodes. The SA operations that are given as, Red + Black = Black; Black - Black = Red; Black + Black = DB; and DB - Black = Black removes DB nodes and rebalances black heights in RB trees. By extension, this paper projects three SA mathematical equations, namely, general symbolic arithmetic rule; partial symbolic arithmetic rule1; and partial symbolic arithmetic rule2. The removal of a DB node ultimately affects black heights in RB trees. To balance black heights using the SA equations, all the RB tree cases, namely, LR, RL, LL, and RR, were considered in this work; and the position of the nodes connected directly or indirectly to the DB node was also tested. In this study, to balance a RB tree, the issues considered w.r.t. the different cases of the RB tree were i) whether a DB node has an inner, outer, or both inner and outer black nephews; or ii) whether a DB node has an inner, outer or both inner and outer red nephews. The nephews r and x in this work are the children of the sibling s to a DB, and further up the tree, the parent p of a DB is their grandparent g. Thus, r and x have indirect relationships to a DB at the point of formation of the DB node. The novelty of the SA equations is in their effectiveness in the removal of DB that involves rotation of nodes as well as the recoloring of nodes along any simple path so as to balance black heights in a tree. 

**Abstract (ZH)**: 双黑（DB）节点在红黑（RB）树中没有位置。因此，当形成DB节点时，会立即删除。删除DB节点导致的旋转和重新着色连接节点的任务增加了在教学和学习RB树时的难度。为简化这一难度，本文扩展了我们之前关于符号算术代数（SA）方法去除DB节点的工作。提供的SA运算包括：红 + 黑 = 黑；黑 - 黑 = 红；黑 + 黑 = 双黑；双黑 - 黑 = 黑，这些运算用于去除DB节点并重新平衡RB树中的黑高。进一步地，本文提出了三个SA数学方程，即：一般符号算术规则；部分符号算术规则1；和部分符号算术规则2。删除一个DB节点最终会影响RB树中的黑高。为了使用SA方程平衡黑高，本研究考虑了所有RB树案例，包括LR、RL、LL和RR，并且还测试了直接或间接连接到DB节点的节点的位置。在这项研究中，为了平衡RB树，考虑了不同RB树案例的问题，包括i) DB节点是否有内部、外部或两者的黑色侄节点；或ii) DB节点是否有内部、外部或两者的红色侄节点。在这个研究中，r和x是DB的兄弟节点s的子节点，而在更高级别的树中，DB的父节点p是它们的祖父节点g。因此，r和x在形成DB节点时与DB有间接关系。SA方程的创新之处在于它们在涉及节点旋转和沿任何简单路径重新着色以平衡树中的黑高方面的有效性。 

---
# Rotation Invariance in Floor Plan Digitization using Zernike Moments 

**Title (ZH)**: 使用Zernike矩的地板平面图数字化的旋转不变性研究 

**Authors**: Marius Graumann, Jan Marius Stürmer, Tobias Koch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03241)  

**Abstract**: Nowadays, a lot of old floor plans exist in printed form or are stored as scanned raster images. Slight rotations or shifts may occur during scanning. Bringing floor plans of this form into a machine readable form to enable further use, still poses a problem. Therefore, we propose an end-to-end pipeline that pre-processes the image and leverages a novel approach to create a region adjacency graph (RAG) from the pre-processed image and predict its nodes. By incorporating normalization steps into the RAG feature extraction, we significantly improved the rotation invariance of the RAG feature calculation. Moreover, applying our method leads to an improved F1 score and IoU on rotated data. Furthermore, we proposed a wall splitting algorithm for partitioning walls into segments associated with the corresponding rooms. 

**Abstract (ZH)**: 如今，大量旧楼计划以印刷形式存在或以扫描的位图图像形式存储。扫描过程中可能会发生轻微的旋转或偏移。将这种形式的楼计划转换成机器可读的格式以供进一步使用，仍存在一个问题。因此，我们提出了一种端到端的流程，该流程预处理图像并利用新颖的方法从预处理的图像中创建区域相邻图（RAG）并预测其节点。通过将归一化步骤纳入RAG特征提取，我们显著提高了RAG特征计算的旋转不变性。此外，应用我们的方法可提高旋转数据上的F1分数和IoU。同时，我们还提出了一种墙壁分割算法，用于将墙壁分割成与相应房间相关的段。 

---
# Malware Detection in Docker Containers: An Image is Worth a Thousand Logs 

**Title (ZH)**: Docker容器中的恶意软件检测：一幅图胜过千行日志 

**Authors**: Akis Nousias, Efklidis Katsaros, Evangelos Syrmos, Panagiotis Radoglou-Grammatikis, Thomas Lagkas, Vasileios Argyriou, Ioannis Moscholios, Evangelos Markakis, Sotirios Goudos, Panagiotis Sarigiannidis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03238)  

**Abstract**: Malware detection is increasingly challenged by evolving techniques like obfuscation and polymorphism, limiting the effectiveness of traditional methods. Meanwhile, the widespread adoption of software containers has introduced new security challenges, including the growing threat of malicious software injection, where a container, once compromised, can serve as entry point for further cyberattacks. In this work, we address these security issues by introducing a method to identify compromised containers through machine learning analysis of their file systems. We cast the entire software containers into large RGB images via their tarball representations, and propose to use established Convolutional Neural Network architectures on a streaming, patch-based manner. To support our experiments, we release the COSOCO dataset--the first of its kind--containing 3364 large-scale RGB images of benign and compromised software containers at this https URL. Our method detects more malware and achieves higher F1 and Recall scores than all individual and ensembles of VirusTotal engines, demonstrating its effectiveness and setting a new standard for identifying malware-compromised software containers. 

**Abstract (ZH)**: 恶意软件检测 increasingly受到混淆技术和多形性等演变技术的挑战，限制了传统方法的有效性。同时，软件容器的广泛应用引入了新的安全挑战，包括恶意软件注入的日益严重威胁，一旦容器被攻破，就可能成为进一步网络攻击的入口。在此项工作中，我们通过机器学习分析容器文件系统来识别受感染的容器，以应对这些安全问题。我们将整个软件容器通过其tarball表示转化为大规模的RGB图像，并提出了一种基于流式、补丁级的卷积神经网络架构。为支持我们的实验，我们发布了COSOCO数据集——这是首个包含3364个良性与受感染的软件容器大规模RGB图像的数据集，可在以下链接访问：https://cosoco.alicloudapi.com。我们的方法检测到更多的恶意软件，并且在F1和召回率上优于所有单个和组合的VirusTotal引擎，证明了其有效性并为识别受恶意软件感染的软件容器设定了新的标准。 

---
# Crash Time Matters: HybridMamba for Fine-Grained Temporal Localization in Traffic Surveillance Footage 

**Title (ZH)**: 碰撞时间至关重要：HybridMamba在交通监控视频中实现细粒度时间定位 

**Authors**: Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03235)  

**Abstract**: Traffic crash detection in long-form surveillance videos is critical for emergency response and infrastructure planning but remains difficult due to the brief and rare nature of crash events. We introduce HybridMamba, a novel architecture that combines visual transformers with state-space temporal modeling to achieve accurate crash time localization. Our method uses multi-level token compression and hierarchical temporal processing to remain computationally efficient without sacrificing temporal resolution. Evaluated on a large-scale dataset from the Iowa Department of Transportation, HybridMamba achieves a mean absolute error of 1.50 seconds, with 65.2 percent of predictions within one second of the ground truth. It outperforms recent video-language models such as TimeChat and VideoLLaMA2 by up to 2.8 seconds, while using significantly fewer parameters. Our results demonstrate strong generalization across videos ranging from 2 to 40 minutes in diverse conditions. HybridMamba offers a robust and efficient solution for fine-grained temporal localization in traffic surveillance. The code will be released upon publication. 

**Abstract (ZH)**: 基于长时监控视频的交通事故检测对于应急响应和基础设施规划至关重要，但由于事故事件短暂且稀少，这一任务仍具有挑战性。我们提出了一种名为HybridMamba的新型架构，该架构结合了视觉变换器与状态空间时间建模，以实现精确的事故时间定位。该方法通过多级令牌压缩和层次时间处理，在保持时空分辨率的同时保持了高效的计算性能。在爱荷华州交通运输部的大规模数据集上评估，HybridMamba实现了1.50秒的平均绝对误差，其中65.2%的预测与真实值相差不超过1秒。与近期的视频语言模型TimeChat和VideoLLaMA2相比，HybridMamba在性能上高出多达2.8秒，同时参数量显著减少。我们的结果表明，HybridMamba在不同条件下的2到40分钟视频中展现出强大的泛化能力。HybridMamba为交通监控中的细粒度时间定位提供了稳健且高效的解决方案。代码将在出版后发布。 

---
# Think When You Need: Self-Adaptive Chain-of-Thought Learning 

**Title (ZH)**: 需要时思考：自适应链式思维学习 

**Authors**: Junjie Yang, Ke Lin, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03234)  

**Abstract**: Chain of Thought (CoT) reasoning enhances language models' performance but often leads to inefficient "overthinking" on simple problems. We identify that existing approaches directly penalizing reasoning length fail to account for varying problem complexity. Our approach constructs rewards through length and quality comparisons, guided by theoretical assumptions that jointly enhance solution correctness with conciseness. Moreover, we further demonstrate our method to fuzzy tasks where ground truth is unavailable. Experiments across multiple reasoning benchmarks demonstrate that our method maintains accuracy while generating significantly more concise explanations, effectively teaching models to "think when needed." 

**Abstract (ZH)**: Chain of Thought (CoT)推理提高了语言模型的性能，但往往会针对简单问题进行无效的“过度思考”。我们发现，现有直接惩罚推理长度的方法未能考虑问题复杂度的差异。我们通过长度和质量比较构建奖励，基于理论假设以同时提升解决方案的正确性和简洁性。此外，我们还进一步展示了本方法在地面真相不可用的模糊任务中的应用效果。跨多个推理基准的实验表明，本方法在保持准确性的同时生成了明显更加简洁的解释，有效地教会模型“按需思考”。 

---
# Persuasive Calibration 

**Title (ZH)**: 说服性校准 

**Authors**: Yiding Feng, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03211)  

**Abstract**: We introduce and study the persuasive calibration problem, where a principal aims to provide trustworthy predictions about underlying events to a downstream agent to make desired decisions. We adopt the standard calibration framework that regulates predictions to be unbiased conditional on their own value, and thus, they can reliably be interpreted at the face value by the agent. Allowing a small calibration error budget, we aim to answer the following question: what is and how to compute the optimal predictor under this calibration error budget, especially when there exists incentive misalignment between the principal and the agent? We focus on standard Lt-norm Expected Calibration Error (ECE) metric.
We develop a general framework by viewing predictors as post-processed versions of perfectly calibrated predictors. Using this framework, we first characterize the structure of the optimal predictor. Specifically, when the principal's utility is event-independent and for L1-norm ECE, we show: (1) the optimal predictor is over-(resp. under-) confident for high (resp. low) true expected outcomes, while remaining perfectly calibrated in the middle; (2) the miscalibrated predictions exhibit a collinearity structure with the principal's utility function. On the algorithmic side, we provide a FPTAS for computing approximately optimal predictor for general principal utility and general Lt-norm ECE. Moreover, for the L1- and L-Infinity-norm ECE, we provide polynomial-time algorithms that compute the exact optimal predictor. 

**Abstract (ZH)**: 我们引入并研究了说服性校准问题，其中主要方旨在为下游代理提供关于潜在事件的可信预测，以便代理作出 desired 的决策。我们采用标准的校准框架，该框架要求预测在有条件偏差的情况下保持无偏，从而使代理可以可靠地按面值解读这些预测。允许校准误差的轻微预算，我们旨在回答以下问题：在存在主要方与代理之间激励不一致的情况下，在这种校准误差预算下最优预测器是什么，以及如何计算这一最优预测器？我们关注标准 Lt-范数期望校准误差（ECE）度量。我们开发了一个通用框架，将预测器视为完美校准预测器的后处理版本。利用这一框架，我们首先描述了最优预测器的结构。特别是在主要方的效用与事件无关且对于 L1-范数 ECE 时，我们证明：（1）最优预测器对于高（低）真实预期结果而言是过度（不足）自信的，而在中间保持完美校准；（2）非校准预测表现出与主要方效用函数的共线性结构。在算法方面，我们为一般主要方效用和一般 Lt-范数 ECE 提供了一个近似多项式时间算法来计算近最优预测器。此外，对于 L1-和 L-无穷范数 ECE，我们提供了计算精确最优预测器的多项式时间算法。 

---
# Augmenting Human Cognition With Generative AI: Lessons From AI-Assisted Decision-Making 

**Title (ZH)**: 利用生成式AI增强人类认知：来自AI辅助决策的教训 

**Authors**: Zelun Tony Zhang, Leon Reicherts  

**Link**: [PDF](https://arxiv.org/pdf/2504.03207)  

**Abstract**: How can we use generative AI to design tools that augment rather than replace human cognition? In this position paper, we review our own research on AI-assisted decision-making for lessons to learn. We observe that in both AI-assisted decision-making and generative AI, a popular approach is to suggest AI-generated end-to-end solutions to users, which users can then accept, reject, or edit. Alternatively, AI tools could offer more incremental support to help users solve tasks themselves, which we call process-oriented support. We describe findings on the challenges of end-to-end solutions, and how process-oriented support can address them. We also discuss the applicability of these findings to generative AI based on a recent study in which we compared both approaches to assist users in a complex decision-making task with LLMs. 

**Abstract (ZH)**: 如何使用生成型AI设计增强而非替代人类认知的工具？在这篇立场论文中，我们回顾了自己在AI辅助决策方面的研究，以从中汲取教训。我们观察到，在AI辅助决策和生成型AI中，一个流行的方法是向用户建议完整的AI生成解决方案，用户可以选择接受、拒绝或编辑。或者，AI工具可以提供更多逐步的支持，帮助用户自己解决问题，我们称之为过程导向支持。我们描述了端到端解决方案面临的挑战，以及过程导向支持如何解决这些问题。我们还基于一项最近的研究讨论了这些发现适用于生成型AI的适用性，该研究比较了两种方法在使用大型语言模型辅助复杂决策任务时的效果。 

---
# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward 

**Title (ZH)**: 增强个性化多轮对话的 Curiosity 奖励机制 

**Authors**: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.03206)  

**Abstract**: Effective conversational agents must be able to personalize their behavior to suit a user's preferences, personality, and attributes, whether they are assisting with writing tasks or operating in domains like education or healthcare. Current training methods like Reinforcement Learning from Human Feedback (RLHF) prioritize helpfulness and safety but fall short in fostering truly empathetic, adaptive, and personalized interactions. Traditional approaches to personalization often rely on extensive user history, limiting their effectiveness for new or context-limited users. To overcome these limitations, we propose to incorporate an intrinsic motivation to improve the conversational agents's model of the user as an additional reward alongside multi-turn RLHF. This reward mechanism encourages the agent to actively elicit user traits by optimizing conversations to increase the accuracy of its user model. Consequently, the policy agent can deliver more personalized interactions through obtaining more information about the user. We applied our method both education and fitness settings, where LLMs teach concepts or recommend personalized strategies based on users' hidden learning style or lifestyle attributes. Using LLM-simulated users, our approach outperformed a multi-turn RLHF baseline in revealing information about the users' preferences, and adapting to them. 

**Abstract (ZH)**: 有效的对话代理必须能够个性化其行为以适应用户的需求、个性和属性，无论是协助写作任务还是在教育或医疗等领域中操作。当前的训练方法如基于人类反馈的强化学习（RLHF）注重帮助性和安全性，但在培养真正具有同理心、适应性和个性化的互动方面存在不足。传统个性化方法往往依赖于用户的历史数据，这限制了其在新用户或上下文受限用户中的有效性。为了克服这些限制，我们建议在多回合RLHF的同时引入一种内在动机，以改进对话代理对用户的模型作为额外的奖励。这种奖励机制鼓励代理通过优化对话以增加其用户模型的准确性来主动获取用户的特征。因此，策略代理可以通过获取更多关于用户的详细信息来提供更加个性化的互动。我们在教育和健身环境中应用了这种方法，其中LLM根据用户的隐藏学习风格或生活方式属性教授概念或推荐个性化策略。使用LLM模拟用户，我们的方法在揭示用户偏好方面优于多回合RLHF基准模型，并能够更好地适应这些偏好。 

---
# Endo3R: Unified Online Reconstruction from Dynamic Monocular Endoscopic Video 

**Title (ZH)**: Endo3R：统一的动态单目内窥视频在线重建 

**Authors**: Jiaxin Guo, Wenzhen Dong, Tianyu Huang, Hao Ding, Ziyi Wang, Haomin Kuang, Qi Dou, Yun-Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03198)  

**Abstract**: Reconstructing 3D scenes from monocular surgical videos can enhance surgeon's perception and therefore plays a vital role in various computer-assisted surgery tasks. However, achieving scale-consistent reconstruction remains an open challenge due to inherent issues in endoscopic videos, such as dynamic deformations and textureless surfaces. Despite recent advances, current methods either rely on calibration or instrument priors to estimate scale, or employ SfM-like multi-stage pipelines, leading to error accumulation and requiring offline optimization. In this paper, we present Endo3R, a unified 3D foundation model for online scale-consistent reconstruction from monocular surgical video, without any priors or extra optimization. Our model unifies the tasks by predicting globally aligned pointmaps, scale-consistent video depths, and camera parameters without any offline optimization. The core contribution of our method is expanding the capability of the recent pairwise reconstruction model to long-term incremental dynamic reconstruction by an uncertainty-aware dual memory mechanism. The mechanism maintains history tokens of both short-term dynamics and long-term spatial consistency. Notably, to tackle the highly dynamic nature of surgical scenes, we measure the uncertainty of tokens via Sampson distance and filter out tokens with high uncertainty. Regarding the scarcity of endoscopic datasets with ground-truth depth and camera poses, we further devise a self-supervised mechanism with a novel dynamics-aware flow loss. Abundant experiments on SCARED and Hamlyn datasets demonstrate our superior performance in zero-shot surgical video depth prediction and camera pose estimation with online efficiency. Project page: this https URL. 

**Abstract (ZH)**: 从单目手术视频重建一致标度的3D场景可以增强外科医生的感知，并在各种计算机辅助手术任务中发挥重要作用。然而，由于内窥镜视频固有的动态变形和无纹理表面等问题，实现一致标度的重建仍然是一个开放的挑战。尽管取得了近期的进步，当前的方法要么依赖标定或器械先验来估计标度，要么采用类似SfM的多阶段流水线，导致错误累积，并需要离线优化。在本文中，我们提出Endo3R，一种用于在线一致标度重建的统一3D基础模型，无需任何先验或额外优化。我们的模型通过预测全局对齐的点图、一致标度的视频深度和相机参数来统一任务，而不进行任何离线优化。我们的方法的核心贡献是通过一种不确定性意识的双记忆机制，将最近的两两重建模型扩展到长期增量动态重建。该机制维护了短期动态和长期空间一致性的历史令牌。值得注意的是，为了解决手术场景的高度动态性质，我们通过Sampson距离来测量令牌的不确定性，并过滤掉具有高不确定性的令牌。鉴于内窥镜数据集稀缺且缺乏地面真实深度和相机姿态，我们进一步提出了一种自我监督机制，并设计了一种新的动态意识流损失。在SCARED和Hamlyn数据集上的大量实验表明，我们的方法在零样本手术视频深度预测和相机姿态估计方面具有在线效率和优越的表现。项目页面：this https URL。 

---
# Learning Natural Language Constraints for Safe Reinforcement Learning of Language Agents 

**Title (ZH)**: 学习自然语言约束以实现语言代理的安全强化学习 

**Authors**: Jaymari Chua, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03185)  

**Abstract**: Generalizable alignment is a core challenge for deploying Large Language Models (LLMs) safely in real-world NLP applications. Current alignment methods, including Reinforcement Learning from Human Feedback (RLHF), often fail to guarantee constraint satisfaction outside their training distribution due to their reliance on implicit, post-hoc preferences. Inspired by a paradigm shift to first curate data before tuning, we introduce a new framework for safe language alignment that learns natural language constraints from positive and negative demonstrations as a primary step. From inferring both a task-specific reward function and latent constraint functions, our approach fosters adaptation to novel safety requirements and robust generalization under domain shifts and adversarial inputs. We formalize the framework within a Constrained Markov Decision Process (CMDP) and validate it via a text-based navigation environment, demonstrating safe adaptation to changing danger zones. Our experiments show fewer violations upon domain shift when following a safe navigation path, and we achieve zero violations by applying learned constraints to a distilled BERT model as a fine-tuning technique. This work offers a promising path toward building safety-critical and more generalizable LLMs for practical NLP settings. 

**Abstract (ZH)**: 通用对齐是安全部署大型语言模型（LLMs）于实际NLP应用中的核心挑战。现有的对齐方法，包括人类反馈强化学习（RLHF），往往由于依赖于隐式的、事后偏好的原因，在其训练分布之外无法保证约束满足。借鉴在调整之前先整理数据的范式转变，我们提出了一种新的安全语言对齐框架，其主要步骤是从正反示例中学习自然语言约束。通过推断任务特定的奖励函数和潜在约束函数，该方法促进了对新型安全要求的适应，并在领域转换和对抗性输入下实现了稳健的泛化。我们在约束马尔可夫决策过程（CMDP）中形式化了该框架，并通过文本导航环境进行验证，展示了在危险区域变化时安全适应的能力。实验结果表明，在遵循安全导航路径时，领域转换后的违规次数较少，并通过将学习到的约束应用于蒸馏BERT模型进行微调实现了零违规。这项工作为构建关键安全性和更具泛化性的LLMs提供了有前景的道路。 

---
# Real-Time Roadway Obstacle Detection for Electric Scooters Using Deep Learning and Multi-Sensor Fusion 

**Title (ZH)**: 基于深度学习和多传感器融合的电动滑板车实时道路障碍检测 

**Authors**: Zeyang Zheng, Arman Hosseini, Dong Chen, Omid Shoghli, Arsalan Heydarian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03171)  

**Abstract**: The increasing adoption of electric scooters (e-scooters) in urban areas has coincided with a rise in traffic accidents and injuries, largely due to their small wheels, lack of suspension, and sensitivity to uneven surfaces. While deep learning-based object detection has been widely used to improve automobile safety, its application for e-scooter obstacle detection remains unexplored. This study introduces a novel ground obstacle detection system for e-scooters, integrating an RGB camera, and a depth camera to enhance real-time road hazard detection. Additionally, the Inertial Measurement Unit (IMU) measures linear vertical acceleration to identify surface vibrations, guiding the selection of six obstacle categories: tree branches, manhole covers, potholes, pine cones, non-directional cracks, and truncated domes. All sensors, including the RGB camera, depth camera, and IMU, are integrated within the Intel RealSense Camera D435i. A deep learning model powered by YOLO detects road hazards and utilizes depth data to estimate obstacle proximity. Evaluated on the seven hours of naturalistic riding dataset, the system achieves a high mean average precision (mAP) of 0.827 and demonstrates excellent real-time performance. This approach provides an effective solution to enhance e-scooter safety through advanced computer vision and data fusion. The dataset is accessible at this https URL, and the project code is hosted on this https URL. 

**Abstract (ZH)**: 电助力踏板车(e-scooter)在城市区域的日益普及 coincided with 交通事故和伤害的增加，主要原因在于其小型车轮、缺乏减震装置以及对不平路面的敏感性。尽管基于深度学习的目标检测技术已被广泛应用于汽车安全改进，但其在电助力踏板车障碍检测方面的应用尚未得到探索。本研究引入了一种新型电助力踏板车地面障碍检测系统，结合RGB相机和深度相机以增强实时道路障碍检测。此外，惯性测量单元(IMU)测量线性垂直加速度以识别路面振动，并据此将障碍物分为六类：树枝、井盖、坑洞、松果、无方向裂缝和截顶圆顶。所有传感器，包括RGB相机、深度相机和IMU，均集成在Intel RealSense Camera D435i中。由YOLO驱动的深度学习模型检测道路障碍，并利用深度数据估算障碍物距离。该系统在七小时的自然骑行数据集上进行了评估，平均精度(mAP)达到0.827，并展示了出色的实时性能。该方法通过先进的计算机视觉和数据融合为提高电助力踏板车安全提供了一种有效解决方案。数据集可访问于此 https URL，项目代码托管于此 https URL。 

---
# NuScenes-SpatialQA: A Spatial Understanding and Reasoning Benchmark for Vision-Language Models in Autonomous Driving 

**Title (ZH)**: NuScenes-空间QA：自动驾驶中视觉语言模型的空间理解与推理基准 

**Authors**: Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03164)  

**Abstract**: Recent advancements in Vision-Language Models (VLMs) have demonstrated strong potential for autonomous driving tasks. However, their spatial understanding and reasoning-key capabilities for autonomous driving-still exhibit significant limitations. Notably, none of the existing benchmarks systematically evaluate VLMs' spatial reasoning capabilities in driving scenarios. To fill this gap, we propose NuScenes-SpatialQA, the first large-scale ground-truth-based Question-Answer (QA) benchmark specifically designed to evaluate the spatial understanding and reasoning capabilities of VLMs in autonomous driving. Built upon the NuScenes dataset, the benchmark is constructed through an automated 3D scene graph generation pipeline and a QA generation pipeline. The benchmark systematically evaluates VLMs' performance in both spatial understanding and reasoning across multiple dimensions. Using this benchmark, we conduct extensive experiments on diverse VLMs, including both general and spatial-enhanced models, providing the first comprehensive evaluation of their spatial capabilities in autonomous driving. Surprisingly, the experimental results show that the spatial-enhanced VLM outperforms in qualitative QA but does not demonstrate competitiveness in quantitative QA. In general, VLMs still face considerable challenges in spatial understanding and reasoning. 

**Abstract (ZH)**: Recent advancements in Vision-Language Models (VLMs) have demonstrated strong potential for autonomous driving tasks. However, their spatial understanding and reasoning-key capabilities for autonomous driving-still exhibit significant limitations. Notably, none of the existing benchmarks systematically evaluate VLMs' spatial reasoning capabilities in driving scenarios. To fill this gap, we propose NuScenes-SpatialQA, the first large-scale ground-truth-based Question-Answer (QA) benchmark specifically designed to evaluate the spatial understanding and reasoning capabilities of VLMs in autonomous driving. 

---
# A Human Digital Twin Architecture for Knowledge-based Interactions and Context-Aware Conversations 

**Title (ZH)**: 基于知识交互与情境 Awareness 对话的人机数字孪生架构 

**Authors**: Abdul Mannan Mohammed, Azhar Ali Mohammad, Jason A. Ortiz, Carsten Neumann, Grace Bochenek, Dirk Reiners, Carolina Cruz-Neira  

**Link**: [PDF](https://arxiv.org/pdf/2504.03147)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and Machine Learning (ML) are creating new opportunities for Human-Autonomy Teaming (HAT) in tasks, missions, and continuous coordinated activities. A major challenge is enabling humans to maintain awareness and control over autonomous assets, while also building trust and supporting shared contextual understanding. To address this, we present a real-time Human Digital Twin (HDT) architecture that integrates Large Language Models (LLMs) for knowledge reporting, answering, and recommendation, embodied in a visual interface.
The system applies a metacognitive approach to enable personalized, context-aware responses aligned with the human teammate's expectations. The HDT acts as a visually and behaviorally realistic team member, integrated throughout the mission lifecycle, from training to deployment to after-action review. Our architecture includes speech recognition, context processing, AI-driven dialogue, emotion modeling, lip-syncing, and multimodal feedback. We describe the system design, performance metrics, and future development directions for more adaptive and realistic HAT systems. 

**Abstract (ZH)**: 近期人工智能（AI）和机器学习（ML）的发展为人类与自主系统团队协作（HAT）在任务、使命及持续协调活动中带来了新机遇。主要挑战在于使人类能够保持对自主资产的意识和控制，同时建立信任并支持共享的上下文理解。为此，我们提出了一种实时人类数字孪生（HDT）架构，该架构集成了大型语言模型（LLMs）进行知识报告、回答和推荐，并体现在可视化界面中。该系统采用元认知方法，以实现与人类队友期望相一致的个性化、上下文感知响应。HDT作为在使命生命周期中（从训练到部署再到事后审查）具有视觉和行为真实性的团队成员发挥作用。我们的架构包括语音识别、上下文处理、AI驱动对话、情绪建模、唇同步和多模态反馈。我们阐述了系统设计、性能指标以及开发更适应和真实的HAT系统的未来方向。 

---
# Hierarchical Modeling for Medical Visual Question Answering with Cross-Attention Fusion 

**Title (ZH)**: 医疗跨注意力融合的分层建模视觉问答 

**Authors**: Junkai Zhang, Bin Li, Shoujun Zhou, Yue Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.03135)  

**Abstract**: Medical Visual Question Answering (Med-VQA) answers clinical questions using medical images, aiding diagnosis. Designing the MedVQA system holds profound importance in assisting clinical diagnosis and enhancing diagnostic accuracy. Building upon this foundation, Hierarchical Medical VQA extends Medical VQA by organizing medical questions into a hierarchical structure and making level-specific predictions to handle fine-grained distinctions. Recently, many studies have proposed hierarchical MedVQA tasks and established datasets, However, several issues still remain: (1) imperfect hierarchical modeling leads to poor differentiation between question levels causing semantic fragmentation across hierarchies. (2) Excessive reliance on implicit learning in Transformer-based cross-modal self-attention fusion methods, which obscures crucial local semantic correlations in medical scenarios. To address these issues, this study proposes a HiCA-VQA method, including two modules: Hierarchical Prompting for fine-grained medical questions and Hierarchical Answer Decoders. The hierarchical prompting module pre-aligns hierarchical text prompts with image features to guide the model in focusing on specific image regions according to question types, while the hierarchical decoder performs separate predictions for questions at different levels to improve accuracy across granularities. The framework also incorporates a cross-attention fusion module where images serve as queries and text as key-value pairs. Experiments on the Rad-Restruct benchmark demonstrate that the HiCA-VQA framework better outperforms existing state-of-the-art methods in answering hierarchical fine-grained questions. This study provides an effective pathway for hierarchical visual question answering systems, advancing medical image understanding. 

**Abstract (ZH)**: 基于层次结构的医疗视觉问答（HiCA-VQA）：提高细粒度医疗图像理解 

---
# GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction 

**Title (ZH)**: GraphSeg：通过图边增加和收缩实现的分段3D表示 

**Authors**: Haozhan Tang, Tianyi Zhang, Oliver Kroemer, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03129)  

**Abstract**: Robots operating in unstructured environments often require accurate and consistent object-level representations. This typically requires segmenting individual objects from the robot's surroundings. While recent large models such as Segment Anything (SAM) offer strong performance in 2D image segmentation. These advances do not translate directly to performance in the physical 3D world, where they often over-segment objects and fail to produce consistent mask correspondences across views. In this paper, we present GraphSeg, a framework for generating consistent 3D object segmentations from a sparse set of 2D images of the environment without any depth information. GraphSeg adds edges to graphs and constructs dual correspondence graphs: one from 2D pixel-level similarities and one from inferred 3D structure. We formulate segmentation as a problem of edge addition, then subsequent graph contraction, which merges multiple 2D masks into unified object-level segmentations. We can then leverage \emph{3D foundation models} to produce segmented 3D representations. GraphSeg achieves robust segmentation with significantly fewer images and greater accuracy than prior methods. We demonstrate state-of-the-art performance on tabletop scenes and show that GraphSeg enables improved performance on downstream robotic manipulation tasks. Code available at this https URL. 

**Abstract (ZH)**: robots在非结构化环境中的操作通常需要准确且一致的对象级表示。这通常需要将个体物体从机器人的周围环境中分割出来。虽然近年来的大模型如Segment Anything (SAM)在2D图像分割方面表现出强大的性能，但这些进步并未直接转化为在物理3D世界中的性能，其中它们往往会过度分割物体，并且无法在不同视角下产生一致的掩码对应关系。在本文中，我们提出GraphSeg框架，用于从环境的稀疏2D图像集中生成一致的3D对象分割，无需任何深度信息。GraphSeg会向图中添加边并构建双重对应图：一个是基于2D像素级相似性的，另一个是基于推断出的3D结构的。我们将分割问题形式化为边添加问题，随后进行图收缩，从而将多个2D掩码合并为统一的对象级分割。我们还可以利用\emph{3D基础模型}来生成分割的3D表示。GraphSeg通过使用显著较少的图像实现更鲁棒且更精确的分割，优于先前的方法。我们在桌面上场景中展示了最先进的性能，并证明GraphSeg能够使下游机器人 manipulation任务的表现得以提升。代码已发布于这个httpsURL。 

---
# Graph Network Modeling Techniques for Visualizing Human Mobility Patterns 

**Title (ZH)**: 图网络建模技术在可视化人类移动模式中的应用 

**Authors**: Sinjini Mitra, Anuj Srivastava, Avipsa Roy, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2504.03119)  

**Abstract**: Human mobility analysis at urban-scale requires models to represent the complex nature of human movements, which in turn are affected by accessibility to nearby points of interest, underlying socioeconomic factors of a place, and local transport choices for people living in a geographic region. In this work, we represent human mobility and the associated flow of movements as a grapyh. Graph-based approaches for mobility analysis are still in their early stages of adoption and are actively being researched. The challenges of graph-based mobility analysis are multifaceted - the lack of sufficiently high-quality data to represent flows at high spatial and teporal resolution whereas, limited computational resources to translate large voluments of mobility data into a network structure, and scaling issues inherent in graph models etc. The current study develops a methodology by embedding graphs into a continuous space, which alleviates issues related to fast graph matching, graph time-series modeling, and visualization of mobility dynamics. Through experiments, we demonstrate how mobility data collected from taxicab trajectories could be transformed into network structures and patterns of mobility flow changes, and can be used for downstream tasks reporting approx 40% decrease in error on average in matched graphs vs unmatched ones. 

**Abstract (ZH)**: 城市规模下的人类移动性分析需要能够代表人类移动复杂性的模型，这些模型受到附近兴趣点的可达性、地方的经济社会因素以及地理区域内居民的本地交通选择的影响。在本研究中，我们将人类移动性和相关移动流表示为图形。基于图形的方法在移动性分析中的应用尚处于早期阶段，且正受到广泛关注和研究。图形导向的移动性分析面临的挑战是多方面的——缺乏足够高质量的数据来表示高空间和时间分辨率的流动，计算资源限制了大规模移动性数据转化为网络结构的能力，以及图形模型固有的缩放问题等。当前研究开发了一种将图形嵌入连续空间的方法，以缓解快速图形匹配、图形时间序列建模以及移动性动态可视化等方面的问题。通过实验，我们展示了如何将来自出租车轨迹的移动数据转化为网络结构和移动流变化的模式，并且可以用于下游任务，结果显示在匹配的图形中与未匹配的图形相比，匹配错误率平均降低了约40%。 

---
# NuWa: Deriving Lightweight Task-Specific Vision Transformers for Edge Devices 

**Title (ZH)**: NuWa: 为边缘设备提取轻量级任务特定视觉变压器 

**Authors**: Ziteng Wei, Qiang He, Bing Li, Feifei Chen, Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03118)  

**Abstract**: Vision Transformers (ViTs) excel in computer vision tasks but lack flexibility for edge devices' diverse needs. A vital issue is that ViTs pre-trained to cover a broad range of tasks are \textit{over-qualified} for edge devices that usually demand only part of a ViT's knowledge for specific tasks. Their task-specific accuracy on these edge devices is suboptimal. We discovered that small ViTs that focus on device-specific tasks can improve model accuracy and in the meantime, accelerate model inference. This paper presents NuWa, an approach that derives small ViTs from the base ViT for edge devices with specific task requirements. NuWa can transfer task-specific knowledge extracted from the base ViT into small ViTs that fully leverage constrained resources on edge devices to maximize model accuracy with inference latency assurance. Experiments with three base ViTs on three public datasets demonstrate that compared with state-of-the-art solutions, NuWa improves model accuracy by up to $\text{11.83}\%$ and accelerates model inference by 1.29$\times$ - 2.79$\times$. Code for reproduction is available at this https URL. 

**Abstract (ZH)**: Vision Transformers (ViTs)在计算机视觉任务中表现出色，但在应对边缘设备多样化需求时缺乏灵活性。一个关键问题是，预先训练以覆盖广泛任务的ViTs对通常只需特定任务部分知识的边缘设备来说过于胜任，这导致它们在这些边缘设备上的任务特定准确性欠佳。我们发现，专注于特定任务的较小ViTs可以提高模型精度，并同时加速模型推理。本文提出NuWa方法，旨在为具有特定任务需求的边缘设备生成小型ViTs。NuWa能够将从基础ViT提取的任务特定知识转移到小型ViTs中，充分利用边缘设备受限的资源，确保在保证推理延迟的前提下最大化模型精度。在三个基础ViT和三个公开数据集上的实验结果显示，与当前最佳解决方案相比，NuWa可提高模型精度高达11.83%，并将模型推理速度提升1.29至2.79倍。相关代码可在以下链接中复制再现：this https URL。 

---
# Multi-Granularity Vision Fastformer with Fusion Mechanism for Skin Lesion Segmentation 

**Title (ZH)**: 多粒度视觉Fastformer融合机制皮肤病变分割 

**Authors**: Xuanyu Liu, Huiyun Yao, Jinggui Gao, Zhongyi Guo, Xue Zhang, Yulin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03108)  

**Abstract**: Background:Convolutional Neural Networks(CNN) and Vision Transformers(ViT) are the main techniques used in Medical image segmentation. However, CNN is limited to local contextual information, and ViT's quadratic complexity results in significant computational costs. At the same time, equipping the model to distinguish lesion boundaries with varying degrees of severity is also a challenge encountered in skin lesion segmentation. Purpose:This research aims to optimize the balance between computational costs and long-range dependency modelling and achieve excellent generalization across lesions with different degrees of severity. Methods:we propose a lightweight U-shape network that utilizes Vision Fastformer with Fusion Mechanism (VFFM-UNet). We inherit the advantages of Fastformer's additive attention mechanism, combining element-wise product and matrix product for comprehensive feature extraction and channel reduction to save computational costs. In order to accurately identify the lesion boundaries with varying degrees of severity, we designed Fusion Mechanism including Multi-Granularity Fusion and Channel Fusion, which can process the feature maps in the granularity and channel levels to obtain different contextual information. Results:Comprehensive experiments on the ISIC2017, ISIC2018 and PH2 datasets demonstrate that VFFM-UNet outperforms existing state-of-the-art models regarding parameter numbers, computational complexity and segmentation performance. In short, compared to MISSFormer, our model achieves superior segmentation performance while reducing parameter and computation costs by 101x and 15x, respectively. Conclusions:Both quantitative and qualitative analyses show that VFFM-UNet sets a new benchmark by reaching an ideal balance between parameter numbers, computational complexity, and segmentation performance compared to existing state-of-the-art models. 

**Abstract (ZH)**: 背景：卷积神经网络(CNN)和视觉变换器(ViT)是医学图像分割的主要技术。然而，CNN局限于局部上下文信息，而ViT的二次复杂度导致了显著的计算成本。同时，识别不同严重程度的皮肤病变边界也是一个挑战。目的：本研究旨在优化计算成本与长距离依赖建模之间的平衡，并实现对不同严重程度病变的优秀泛化能力。方法：我们提出了一种轻量级U型网络，利用结合了融合机制的视觉快速变换器(VFFM-UNet)。我们继承了Fastformer的加性注意机制的优点，结合元素级乘积和矩阵乘积进行全面的特征提取，并通过通道减少来节省计算成本。为了准确识别不同严重程度的病变边界，我们设计了包括多粒度融合和通道融合的融合机制，可以在粒度和通道级别处理特征图以获得不同的上下文信息。结果：在ISIC2017、ISIC2018和PH2数据集上的综合实验表明，VFFM-UNet在参数数量、计算复杂度和分割性能方面优于现有最先进的模型。简而言之，与MISSFormer相比，我们的模型在参数和计算成本分别减少了101倍和15倍的同时，实现了更优秀的分割性能。结论：定量和定性分析表明，VFFM-UNet在参数数量、计算复杂度和分割性能方面达到了理想的平衡，设立了新的基准，优于现有的最先进的模型。 

---
# Post-processing for Fair Regression via Explainable SVD 

**Title (ZH)**: 通过可解释的SVD进行公平回归后处理 

**Authors**: Zhiqun Zuo, Ding Zhu, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2504.03093)  

**Abstract**: This paper presents a post-processing algorithm for training fair neural network regression models that satisfy statistical parity, utilizing an explainable singular value decomposition (SVD) of the weight matrix. We propose a linear transformation of the weight matrix, whereby the singular values derived from the SVD of the transformed matrix directly correspond to the differences in the first and second moments of the output distributions across two groups. Consequently, we can convert the fairness constraints into constraints on the singular values. We analytically solve the problem of finding the optimal weights under these constraints. Experimental validation on various datasets demonstrates that our method achieves a similar or superior fairness-accuracy trade-off compared to the baselines without using the sensitive attribute at the inference time. 

**Abstract (ZH)**: 基于可解释奇异值分解的公平神经网络回归模型后处理算法：利用权重矩阵的线性变换实现统计对等约束 

---
# Machine Learning-Based Detection and Analysis of Suspicious Activities in Bitcoin Wallet Transactions in the USA 

**Title (ZH)**: 基于机器学习的美国比特币钱包交易中可疑活动的检测与分析 

**Authors**: Md Zahidul Islam, Md Shahidul Islam, Biswajit Chandra das, Syed Ali Reza, Proshanta Kumar Bhowmik, Kanchon Kumar Bishnu, Md Shafiqur Rahman, Redoyan Chowdhury, Laxmi Pant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03092)  

**Abstract**: The dramatic adoption of Bitcoin and other cryptocurrencies in the USA has revolutionized the financial landscape and provided unprecedented investment and transaction efficiency opportunities. The prime objective of this research project is to develop machine learning algorithms capable of effectively identifying and tracking suspicious activity in Bitcoin wallet transactions. With high-tech analysis, the study aims to create a model with a feature for identifying trends and outliers that can expose illicit activity. The current study specifically focuses on Bitcoin transaction information in America, with a strong emphasis placed on the importance of knowing about the immediate environment in and through which such transactions pass through. The dataset is composed of in-depth Bitcoin wallet transactional information, including important factors such as transaction values, timestamps, network flows, and addresses for wallets. All entries in the dataset expose information about financial transactions between wallets, including received and sent transactions, and such information is significant for analysis and trends that can represent suspicious activity. This study deployed three accredited algorithms, most notably, Logistic Regression, Random Forest, and Support Vector Machines. In retrospect, Random Forest emerged as the best model with the highest F1 Score, showcasing its ability to handle non-linear relationships in the data. Insights revealed significant patterns in wallet activity, such as the correlation between unredeemed transactions and final balances. The application of machine algorithms in tracking cryptocurrencies is a tool for creating transparent and secure U.S. markets. 

**Abstract (ZH)**: 美国比特币及其他加密货币的急剧采用已革新了金融格局，提供了前所未有的投资和交易效率机会。本研究项目的主要目标是开发能够有效识别和跟踪比特币钱包交易中可疑活动的机器学习算法。通过高科技分析，研究旨在创建一个具有识别趋势和异常值的功能的模型，以揭露非法活动。本研究特别关注美国的比特币交易信息，强调了解此类交易途径和环境的重要性。数据集包含深入的比特币钱包交易信息，包括交易值、时间戳、网络流和钱包地址等重要因素。数据集中所有条目都提供了关于钱包之间金融交易的信息，包括接收和发送的交易，这些信息对于分析和识别可疑活动趋势至关重要。本研究部署了三种认证算法，特别是逻辑回归、随机森林和支持向量机。回顾来看，随机森林模型以最高的F1分数脱颖而出，展示了其处理数据中非线性关系的能力。研究揭示了钱包活动中的显著模式，如未兑现交易与最终余额之间的关联。在追踪加密货币方面应用机器算法是创建透明和安全美国市场的工具。 

---
# From Questions to Insights: Exploring XAI Challenges Reported on Stack Overflow Questions 

**Title (ZH)**: 从问题到洞察：探索Stack Overflow问题中报告的XAI挑战 

**Authors**: Saumendu Roy, Saikat Mondal, Banani Roy, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.03085)  

**Abstract**: The lack of interpretability is a major barrier that limits the practical usage of AI models. Several eXplainable AI (XAI) techniques (e.g., SHAP, LIME) have been employed to interpret these models' performance. However, users often face challenges when leveraging these techniques in real-world scenarios and thus submit questions in technical Q&A forums like Stack Overflow (SO) to resolve these challenges. We conducted an exploratory study to expose these challenges, their severity, and features that can make XAI techniques more accessible and easier to use. Our contributions to this study are fourfold. First, we manually analyzed 663 SO questions that discussed challenges related to XAI techniques. Our careful investigation produced a catalog of seven challenges (e.g., disagreement issues). We then analyzed their prevalence and found that model integration and disagreement issues emerged as the most prevalent challenges. Second, we attempt to estimate the severity of each XAI challenge by determining the correlation between challenge types and answer metadata (e.g., the presence of accepted answers). Our analysis suggests that model integration issues is the most severe challenge. Third, we attempt to perceive the severity of these challenges based on practitioners' ability to use XAI techniques effectively in their work. Practitioners' responses suggest that disagreement issues most severely affect the use of XAI techniques. Fourth, we seek agreement from practitioners on improvements or features that could make XAI techniques more accessible and user-friendly. The majority of them suggest consistency in explanations and simplified integration. Our study findings might (a) help to enhance the accessibility and usability of XAI and (b) act as the initial benchmark that can inspire future research. 

**Abstract (ZH)**: 缺乏可解释性是限制AI模型实际应用的重大障碍。多种可解释AI(XAI)技术（例如SHAP、LIME）被用于解释这些模型的性能。然而，用户在实际应用场景中使用这些技术时常常面临挑战，并在技术问答论坛（如Stack Overflow）上提出问题以解决这些挑战。我们开展了探索性研究以揭示这些挑战、它们的严重程度以及能够使XAI技术更易于使用和获取的特征。本研究的贡献包括四个方面。首先，我们手动分析了663篇讨论与XAI技术相关的挑战的Stack Overflow问题，仔细调查并生成了七类挑战的清单（例如，分歧问题）。然后，我们分析了这些挑战的普遍性，发现模型整合问题和分歧问题是最常见的挑战。第二，我们尝试通过确定挑战类型与答案元数据（如接受答案的存在）的相关性来估计每类XAI挑战的严重程度。我们的分析表明，模型整合问题是最严重的挑战。第三，我们基于实践者在其工作中有效使用XAI技术的能力来感知这些挑战的严重性。实践者的回应表明，分歧问题对XAI技术的使用影响最大。第四，我们寻求实践者对可能使XAI技术更易于使用和用户友好的改进或特性的认同。大多数实践者建议保持解释的一致性和简化集成。本研究的发现可能有助于提高XAI的可访问性和易用性，并成为激励未来研究的初步基准。 

---
# Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning 

**Title (ZH)**: 针对适应性对手的联邦学习中基于身份的识别集成 

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip  

**Link**: [PDF](https://arxiv.org/pdf/2504.03077)  

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats. 

**Abstract (ZH)**: 基于身份的识别在联邦学习中的应用：应对重新连接恶意客户端的安全措施 

---
# AD-GPT: Large Language Models in Alzheimer's Disease 

**Title (ZH)**: AD-GPT: 阿尔茨海默病中的大规模语言模型 

**Authors**: Ziyu Liu, Lintao Tang, Zeliang Sun, Zhengliang Liu, Yanjun Lyu, Wei Ruan, Yangshuang Xu, Liang Shan, Jiyoon Shin, Xiaohe Chen, Dajiang Zhu, Tianming Liu, Rongjie Liu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03071)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for medical information retrieval, yet their accuracy and depth remain limited in specialized domains such as Alzheimer's disease (AD), a growing global health challenge. To address this gap, we introduce AD-GPT, a domain-specific generative pre-trained transformer designed to enhance the retrieval and analysis of AD-related genetic and neurobiological information. AD-GPT integrates diverse biomedical data sources, including potential AD-associated genes, molecular genetic information, and key gene variants linked to brain regions. We develop a stacked LLM architecture combining Llama3 and BERT, optimized for four critical tasks in AD research: (1) genetic information retrieval, (2) gene-brain region relationship assessment, (3) gene-AD relationship analysis, and (4) brain region-AD relationship mapping. Comparative evaluations against state-of-the-art LLMs demonstrate AD-GPT's superior precision and reliability across these tasks, underscoring its potential as a robust and specialized AI tool for advancing AD research and biomarker discovery. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医学信息检索中展现出强大的工具潜力，但在阿尔茨海默病（AD）等专科领域，其准确性和深度仍然有限。为弥补这一不足，我们引入了AD-GPT，这是一种专门设计的生成预训练变换器，旨在增强与AD相关的遗传和神经生物学信息的检索和分析。AD-GPT 结合多种生物医学数据源，包括潜在的 AD 相关基因、分子遗传信息以及与大脑区域相关的关键基因变异。我们开发了一种层叠大型语言模型架构，结合了 Llama3 和 BERT，优化了AD研究中四个关键任务：（1）遗传信息检索，（2）基因-大脑区域关系评估，（3）基因-AD 关系分析，以及（4）大脑区域-AD 关系映射。与最先进的大型语言模型的对比评估表明，AD-GPT 在这些任务中的精度和可靠性显著优于其他模型，其潜在价值在于作为一种强大且专门的人工智能工具，推动AD研究和生物标志物发现的进步。 

---
# Properties of Fixed Points of Generalised Extra Gradient Methods Applied to Min-Max Problems 

**Title (ZH)**: 广义额外梯度方法应用于最小-最大问题的不动点性质 

**Authors**: Amir Ali Farzin, Yuen-Man Pun, Philipp Braun, Iman Shames  

**Link**: [PDF](https://arxiv.org/pdf/2504.03069)  

**Abstract**: This paper studies properties of fixed points of generalised Extra-gradient (GEG) algorithms applied to min-max problems. We discuss connections between saddle points of the objective function of the min-max problem and GEG fixed points. We show that, under appropriate step-size selections, the set of saddle points (Nash equilibria) is a subset of stable fixed points of GEG. Convergence properties of the GEG algorithm are obtained through a stability analysis of a discrete-time dynamical system. The results and benefits when compared to existing methods are illustrated through numerical examples. 

**Abstract (ZH)**: 本文研究了广义Extra-gradient（GEG）算法在解决极小极大问题中的不动点性质。我们讨论了极小极大问题目标函数鞍点与其GEG不动点之间的联系。我们证明，在适当步长选择下，鞍点集合（纳什均衡集）是GEG稳定不动点的子集。通过离散时间动态系统的稳定性分析获得了GEG算法的收敛性质。通过数值例子展示了与现有方法相比的结果和优势。 

---
# Design of AI-Powered Tool for Self-Regulation Support in Programming Education 

**Title (ZH)**: 基于AI的编程教育自我调节支持工具设计 

**Authors**: Huiyong Li, Boxuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03068)  

**Abstract**: Large Language Model (LLM) tools have demonstrated their potential to deliver high-quality assistance by providing instant, personalized feedback that is crucial for effective programming education. However, many of these tools operate independently from institutional Learning Management Systems, which creates a significant disconnect. This isolation limits the ability to leverage learning materials and exercise context for generating tailored, context-aware feedback. Furthermore, previous research on self-regulated learning and LLM support mainly focused on knowledge acquisition, not the development of important self-regulation skills. To address these challenges, we developed CodeRunner Agent, an LLM-based programming assistant that integrates the CodeRunner, a student-submitted code executing and automated grading plugin in Moodle. CodeRunner Agent empowers educators to customize AI-generated feedback by incorporating detailed context from lecture materials, programming questions, student answers, and execution results. Additionally, it enhances students' self-regulated learning by providing strategy-based AI responses. This integrated, context-aware, and skill-focused approach offers promising avenues for data-driven improvements in programming education. 

**Abstract (ZH)**: 大型语言模型工具通过提供即时个性化反馈展示了其在有效编程教育中的高质辅助潜力，但这些工具通常独立于机构的学习管理系统运行，这造成了显著的分离。这种隔离限制了利用学习材料和练习背景生成个性化、情境感知反馈的能力。此外，以往关于自我调节学习和大型语言模型支持的研究主要集中在知识获取上，而忽略了培养重要自我调节技能。为应对这些挑战，我们开发了CodeRunner Agent，这是一种基于大型语言模型的编程助手，它整合了Moodle中的CodeRunner插件，该插件用于学生提交代码的执行和自动评分。CodeRunner Agent使教育者能够通过融入详细的讲义材料、编程问题、学生答案和执行结果来定制人工智能生成的反馈。此外，它通过提供基于策略的人工智能响应来增强学生的自我调节学习能力。这种整合、情境感知和技能导向的方法为编程教育的数据驱动改进提供了有希望的途径。 

---
# Context-Aware Self-Adaptation for Domain Generalization 

**Title (ZH)**: 基于上下文的自适应方法研究：领域泛化 

**Authors**: Hao Yan, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03064)  

**Abstract**: Domain generalization aims at developing suitable learning algorithms in source training domains such that the model learned can generalize well on a different unseen testing domain. We present a novel two-stage approach called Context-Aware Self-Adaptation (CASA) for domain generalization. CASA simulates an approximate meta-generalization scenario and incorporates a self-adaptation module to adjust pre-trained meta source models to the meta-target domains while maintaining their predictive capability on the meta-source domains. The core concept of self-adaptation involves leveraging contextual information, such as the mean of mini-batch features, as domain knowledge to automatically adapt a model trained in the first stage to new contexts in the second stage. Lastly, we utilize an ensemble of multiple meta-source models to perform inference on the testing domain. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on standard benchmarks. 

**Abstract (ZH)**: 领域泛化旨在开发在源训练领域中有效的学习算法，使得模型能够在不同的未见测试领域中良好泛化。我们提出了一种名为Context-Aware Self-Adaptation (CASA)的新型两阶段方法，用于领域泛化。CASA模拟了近似元泛化场景，并集成了一个自我调整模块，在保留元源域预测能力的同时，将预训练的元源域模型调整到元目标域。自我调整的核心概念涉及利用上下文信息（如mini-batch特征的均值）作为领域知识，自动调整第一阶段训练的模型以适应第二阶段的新上下文。最后，我们使用多个元源域模型的集成在测试域中进行推理。实验结果表明，我们提出的方法在标准基准测试中取得了最先进的性能。 

---
# Cooperative Inference for Real-Time 3D Human Pose Estimation in Multi-Device Edge Networks 

**Title (ZH)**: 多设备边缘网络中的实时3D人体姿态协作推理 

**Authors**: Hyun-Ho Choi, Kangsoo Kim, Ki-Ho Lee, Kisong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.03052)  

**Abstract**: Accurate and real-time three-dimensional (3D) pose estimation is challenging in resource-constrained and dynamic environments owing to its high computational complexity. To address this issue, this study proposes a novel cooperative inference method for real-time 3D human pose estimation in mobile edge computing (MEC) networks. In the proposed method, multiple end devices equipped with lightweight inference models employ dual confidence thresholds to filter ambiguous images. Only the filtered images are offloaded to an edge server with a more powerful inference model for re-evaluation, thereby improving the estimation accuracy under computational and communication constraints. We numerically analyze the performance of the proposed inference method in terms of the inference accuracy and end-to-end delay and formulate a joint optimization problem to derive the optimal confidence thresholds and transmission time for each device, with the objective of minimizing the mean per-joint position error (MPJPE) while satisfying the required end-to-end delay constraint. To solve this problem, we demonstrate that minimizing the MPJPE is equivalent to maximizing the sum of the inference accuracies for all devices, decompose the problem into manageable subproblems, and present a low-complexity optimization algorithm to obtain a near-optimal solution. The experimental results show that a trade-off exists between the MPJPE and end-to-end delay depending on the confidence thresholds. Furthermore, the results confirm that the proposed cooperative inference method achieves a significant reduction in the MPJPE through the optimal selection of confidence thresholds and transmission times, while consistently satisfying the end-to-end delay requirement in various MEC environments. 

**Abstract (ZH)**: 资源受限和动态环境下基于移动边缘计算的实时三维人体姿态估计联合推断方法 

---
# Task as Context Prompting for Accurate Medical Symptom Coding Using Large Language Models 

**Title (ZH)**: 基于任务为导向的上下文提示以实现大型语言模型在医疗症状编码中的准确编码 

**Authors**: Chengyang He, Wenlong Zhang, Violet Xinying Chen, Yue Ning, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03051)  

**Abstract**: Accurate medical symptom coding from unstructured clinical text, such as vaccine safety reports, is a critical task with applications in pharmacovigilance and safety monitoring. Symptom coding, as tailored in this study, involves identifying and linking nuanced symptom mentions to standardized vocabularies like MedDRA, differentiating it from broader medical coding tasks. Traditional approaches to this task, which treat symptom extraction and linking as independent workflows, often fail to handle the variability and complexity of clinical narratives, especially for rare cases. Recent advancements in Large Language Models (LLMs) offer new opportunities but face challenges in achieving consistent performance. To address these issues, we propose Task as Context (TACO) Prompting, a novel framework that unifies extraction and linking tasks by embedding task-specific context into LLM prompts. Our study also introduces SYMPCODER, a human-annotated dataset derived from Vaccine Adverse Event Reporting System (VAERS) reports, and a two-stage evaluation framework to comprehensively assess both symptom linking and mention fidelity. Our comprehensive evaluation of multiple LLMs, including Llama2-chat, Jackalope-7b, GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o, demonstrates TACO's effectiveness in improving flexibility and accuracy for tailored tasks like symptom coding, paving the way for more specific coding tasks and advancing clinical text processing methodologies. 

**Abstract (ZH)**: 从未结构化临床文本中准确编码医学症状：一项针对疫苗安全报告的应用研究 

---
# Safety Modulation: Enhancing Safety in Reinforcement Learning through Cost-Modulated Rewards 

**Title (ZH)**: 安全调节：通过成本调节奖励增强强化学习的安全性 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03040)  

**Abstract**: Safe Reinforcement Learning (Safe RL) aims to train an RL agent to maximize its performance in real-world environments while adhering to safety constraints, as exceeding safety violation limits can result in severe consequences. In this paper, we propose a novel safe RL approach called Safety Modulated Policy Optimization (SMPO), which enables safe policy function learning within the standard policy optimization framework through safety modulated rewards. In particular, we consider safety violation costs as feedback from the RL environments that are parallel to the standard awards, and introduce a Q-cost function as safety critic to estimate expected future cumulative costs. Then we propose to modulate the rewards using a cost-aware weighting function, which is carefully designed to ensure the safety limits based on the estimation of the safety critic, while maximizing the expected rewards. The policy function and the safety critic are simultaneously learned through gradient descent during online interactions with the environment. We conduct experiments using multiple RL environments and the experimental results demonstrate that our method outperforms several classic and state-of-the-art comparison methods in terms of overall safe RL performance. 

**Abstract (ZH)**: 安全强化学习（Safe RL）旨在训练一个RL代理在符合安全约束的情况下最大化其实用效果，因为超出安全违规限制可能导致严重后果。在本文中，我们提出了一种新颖的安全RL方法，称为安全性调节策略优化（SMPO），该方法可以通过安全性调节奖励在标准策略优化框架中实现安全策略函数的学习。特别地，我们将安全违规成本作为与标准奖励并行的RL环境反馈，并引入Q成本函数作为安全性评估器来估计预期的未来累积成本。然后，我们提出使用成本感知加权函数调节奖励，该函数旨在基于安全性评估器的估计确保安全限制，同时最大化预期奖励。策略函数和安全性评估器在与环境进行在线交互过程中同时通过梯度下降进行学习。我们在多个RL环境中进行了实验，并且实验结果表明，我们的方法在整体安全RL性能上优于几种经典和最先进的比较方法。 

---
# Deep Reinforcement Learning via Object-Centric Attention 

**Title (ZH)**: 基于对象中心注意力的深度强化学习 

**Authors**: Jannis Blüml, Cedric Derstroff, Bjarne Gregori, Elisabeth Dillies, Quentin Delfosse, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2504.03024)  

**Abstract**: Deep reinforcement learning agents, trained on raw pixel inputs, often fail to generalize beyond their training environments, relying on spurious correlations and irrelevant background details. To address this issue, object-centric agents have recently emerged. However, they require different representations tailored to the task specifications. Contrary to deep agents, no single object-centric architecture can be applied to any environment. Inspired by principles of cognitive science and Occam's Razor, we introduce Object-Centric Attention via Masking (OCCAM), which selectively preserves task-relevant entities while filtering out irrelevant visual information. Specifically, OCCAM takes advantage of the object-centric inductive bias. Empirical evaluations on Atari benchmarks demonstrate that OCCAM significantly improves robustness to novel perturbations and reduces sample complexity while showing similar or improved performance compared to conventional pixel-based RL. These results suggest that structured abstraction can enhance generalization without requiring explicit symbolic representations or domain-specific object extraction pipelines. 

**Abstract (ZH)**: 基于对象中心注意力掩模的强化学习代理 

---
# The Dual-Route Model of Induction 

**Title (ZH)**: 双重路径归纳模型 

**Authors**: Sheridan Feucht, Eric Todd, Byron Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2504.03022)  

**Abstract**: Prior work on in-context copying has shown the existence of induction heads, which attend to and promote individual tokens during copying. In this work we introduce a new type of induction head: concept-level induction heads, which copy entire lexical units instead of individual tokens. Concept induction heads learn to attend to the ends of multi-token words throughout training, working in parallel with token-level induction heads to copy meaningful text. We show that these heads are responsible for semantic tasks like word-level translation, whereas token induction heads are vital for tasks that can only be done verbatim, like copying nonsense tokens. These two "routes" operate independently: in fact, we show that ablation of token induction heads causes models to paraphrase where they would otherwise copy verbatim. In light of these findings, we argue that although token induction heads are vital for specific tasks, concept induction heads may be more broadly relevant for in-context learning. 

**Abstract (ZH)**: 基于上下文复制的先前工作已经表明存在引导头，它们在复制过程中关注并促进个别令牌。在这项工作中，我们引入了一种新的引导头类型：概念级引导头，它们复制整个词汇单元，而不仅仅是个别令牌。概念引导头在训练过程中学会关注多令牌词的末尾，并与令牌级引导头并行工作以复制有意义的文字。我们展示了这些引导头负责诸如词级翻译等语义任务，而令牌级引导头对于只能逐字完成的任务，如复制无意义的令牌，则至关重要。这两种“路径”是独立运作的：实际上，我们证明了去除令牌级引导头会导致模型在原本应逐字复制的地方进行改写。鉴于这些发现，我们认为虽然令牌级引导头对特定任务至关重要，但概念级引导头可能更广泛地适用于基于上下文的学习。 

---
# Localized Definitions and Distributed Reasoning: A Proof-of-Concept Mechanistic Interpretability Study via Activation Patching 

**Title (ZH)**: 局部定义与分布式推理：基于激活修补的机理可解释性概念验证研究 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2504.02976)  

**Abstract**: This study investigates the localization of knowledge representation in fine-tuned GPT-2 models using Causal Layer Attribution via Activation Patching (CLAP), a method that identifies critical neural layers responsible for correct answer generation. The model was fine-tuned on 9,958 PubMed abstracts (epilepsy: 20,595 mentions, EEG: 11,674 mentions, seizure: 13,921 mentions) using two configurations with validation loss monitoring for early stopping. CLAP involved (1) caching clean (correct answer) and corrupted (incorrect answer) activations, (2) computing logit difference to quantify model preference, and (3) patching corrupted activations with clean ones to assess recovery. Results revealed three findings: First, patching the first feedforward layer recovered 56% of correct preference, demonstrating that associative knowledge is distributed across multiple layers. Second, patching the final output layer completely restored accuracy (100% recovery), indicating that definitional knowledge is localised. The stronger clean logit difference for definitional questions further supports this localized representation. Third, minimal recovery from convolutional layer patching (13.6%) suggests low-level features contribute marginally to high-level reasoning. Statistical analysis confirmed significant layer-specific effects (p<0.01). These findings demonstrate that factual knowledge is more localized and associative knowledge depends on distributed representations. We also showed that editing efficacy depends on task type. Our findings not only reconcile conflicting observations about localization in model editing but also emphasize on using task-adaptive techniques for reliable, interpretable updates. 

**Abstract (ZH)**: 本研究使用因果层 attribution 通过激活补丁方法（CLAP）探究了微调后 GPT-2 模型中知识表示的局部化，该方法识别出负责正确答案生成的關鍵神经层。模型在包含 9,958 个PubMed摘要（癫痫：20,595 次提及，脑电图：11,674 次提及，发作：13,921 次提及）的数据集上进行了微调，并通过监控验证损失实现了早停。CLAP 包括以下步骤：(1) 缓存干净（正确答案）和受损（错误答案）的激活；(2) 计算 logits 差异以量化模型偏好；(3) 用干净激活替换受损激活以评估恢复情况。结果揭示了三项发现：首先，修复第一个前馈层恢复了 56% 的偏好，表明关联性知识分布在多个层中。其次，修复最终输出层完全恢复了准确性（100% 恢复），表明定义性知识是局部化的。定义性问题的 stronger clean logits 差异进一步支持了这种局部表示。第三，卷积层修复的 minimal 恢复（13.6%）表明低级特征对高级推理的贡献微乎其微。统计分析证实了特定层的显著效应（p<0.01）。这些发现表明，事实性知识更局部化，而关联性知识依赖于分布式表示。此外，我们还表明了编辑效果依赖于任务类型。我们的发现不仅弥合了关于模型编辑中局部化的矛盾观察，还强调了使用任务适配技术进行可靠且可解释的更新的重要性。 

---
# Improved Compact Genetic Algorithms with Efficient Caching 

**Title (ZH)**: 改进的高效缓存紧凑遗传算法 

**Authors**: Prasanta Dutta, Anirban Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2504.02972)  

**Abstract**: Compact Genetic Algorithms (cGAs) are condensed variants of classical Genetic Algorithms (GAs) that use a probability vector representation of the population instead of the complete population. cGAs have been shown to significantly reduce the number of function evaluations required while producing outcomes similar to those of classical GAs. However, cGAs have a tendency to repeatedly generate the same chromosomes as they approach convergence, resulting in unnecessary evaluations of identical chromosomes. This article introduces the concept of caching in cGAs as a means of avoiding redundant evaluations of the same chromosomes. Our proposed approach operates equivalently to cGAs, but enhances the algorithm's time efficiency by reducing the number of function evaluations. We also present a data structure for efficient cache maintenance to ensure low overhead. The proposed caching approach has an asymptotically constant time complexity on average. The proposed method further generalizes the caching mechanism with higher selection pressure for elitism-based cGAs. We conduct a rigorous analysis based on experiments on benchmark optimization problems using two well-known cache replacement strategies. The results demonstrate that caching significantly reduces the number of function evaluations required while maintaining the same level of performance accuracy. 

**Abstract (ZH)**: 基于缓存的紧凑遗传算法：提高时间效率并保持性能准确性 

---
# Global-Order GFlowNets 

**Title (ZH)**: 全球秩序GFlowNets 

**Authors**: Lluís Pastor-Pérez, Javier Alonso-Garcia, Lukas Mauch  

**Link**: [PDF](https://arxiv.org/pdf/2504.02968)  

**Abstract**: Order-Preserving (OP) GFlowNets have demonstrated remarkable success in tackling complex multi-objective (MOO) black-box optimization problems using stochastic optimization techniques. Specifically, they can be trained online to efficiently sample diverse candidates near the Pareto front. A key advantage of OP GFlowNets is their ability to impose a local order on training samples based on Pareto dominance, eliminating the need for scalarization - a common requirement in other approaches like Preference-Conditional GFlowNets. However, we identify an important limitation of OP GFlowNets: imposing a local order on training samples can lead to conflicting optimization objectives. To address this issue, we introduce Global-Order GFlowNets, which transform the local order into a global one, thereby resolving these conflicts. Our experimental evaluations on various benchmarks demonstrate the efficacy and promise of our proposed method. 

**Abstract (ZH)**: Order-Preserving (OP) GFlowNets在使用随机优化技术解决复杂的多目标（MOO）黑盒优化问题中取得了显著成功。具体而言，它们可以在线训练以高效地采样 Pareto 前沿附近的多样化候选解决方案。OP GFlowNets的一个关键优势是能够基于帕累托支配对训练样本施加局部顺序，从而消除其他方法（如偏好条件GFlowNets）中常见的标量化需求。然而，我们识别了OP GFlowNets的一个重要局限性：对训练样本施加局部顺序可能导致优化目标冲突。为了解决这一问题，我们引入了全局顺序GFlowNets，将局部顺序转换为全局顺序，从而解决这些冲突。我们在各种基准上的实验评估展示了我们提出方法的有效性和潜力。 

---
# CoLa -- Learning to Interactively Collaborate with Large LMs 

**Title (ZH)**: CoLa -- 学习与大型语言模型互动协作 

**Authors**: Abhishek Sharma, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.02965)  

**Abstract**: LLMs' remarkable ability to tackle a wide range of language tasks opened new opportunities for collaborative human-AI problem solving. LLMs can amplify human capabilities by applying their intuitions and reasoning strategies at scale. We explore whether human guides can be simulated, by generalizing from human demonstrations of guiding an AI system to solve complex language problems. We introduce CoLa, a novel self-guided learning paradigm for training automated $\textit{guides}$ and evaluate it on two QA datasets, a puzzle-solving task, and a constrained text generation task. Our empirical results show that CoLa consistently outperforms competitive approaches across all domains. Moreover, a small-sized trained guide outperforms a strong model like GPT-4 when acting as a guide. We compare the strategies employed by humans and automated guides by conducting a human study on a QA dataset. We show that automated guides outperform humans by adapting their strategies to reasoners' capabilities and conduct qualitative analyses highlighting distinct differences in guiding strategies. 

**Abstract (ZH)**: 大规模语言模型的卓越能力为人类-AI协作解决问题开辟了新机遇。大规模应用人类直觉和推理策略，可以增强人类能力。我们探索通过从人类指导AI系统解决复杂语言问题的示范中进行泛化，模拟人类指导者的能力。我们引入了CoLa，一种新颖的自我指导学习范式，用于训练自动化的“指导者”，并分别在两个问答数据集、一个解谜任务和一个受约束的文本生成任务上进行评估。我们的实验证明，CoLa在所有领域均优于竞争性方法。此外，小型训练指导者在作为指导者时比强大的模型如GPT-4表现更好。我们通过一项关于问答数据集的人类研究，比较了人类和自动化指导者所采用的策略。我们展示了自动化指导者通过适应理解决策者的能力而优于人类，并进行了定性分析，突显了指导策略的差异。 

---
# Digital Forensics in the Age of Large Language Models 

**Title (ZH)**: 大型语言模型时代的数字取证 

**Authors**: Zhipeng Yin, Zichong Wang, Weifeng Xu, Jun Zhuang, Pallab Mozumder, Antoinette Smith, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02963)  

**Abstract**: Digital forensics plays a pivotal role in modern investigative processes, utilizing specialized methods to systematically collect, analyze, and interpret digital evidence for judicial proceedings. However, traditional digital forensic techniques are primarily based on manual labor-intensive processes, which become increasingly insufficient with the rapid growth and complexity of digital data. To this end, Large Language Models (LLMs) have emerged as powerful tools capable of automating and enhancing various digital forensic tasks, significantly transforming the field. Despite the strides made, general practitioners and forensic experts often lack a comprehensive understanding of the capabilities, principles, and limitations of LLM, which limits the full potential of LLM in forensic applications. To fill this gap, this paper aims to provide an accessible and systematic overview of how LLM has revolutionized the digital forensics approach. Specifically, it takes a look at the basic concepts of digital forensics, as well as the evolution of LLM, and emphasizes the superior capabilities of LLM. To connect theory and practice, relevant examples and real-world scenarios are discussed. We also critically analyze the current limitations of applying LLMs to digital forensics, including issues related to illusion, interpretability, bias, and ethical considerations. In addition, this paper outlines the prospects for future research, highlighting the need for effective use of LLMs for transparency, accountability, and robust standardization in the forensic process. 

**Abstract (ZH)**: 大型语言模型在数字取证中的革命性作用：理论与实践、挑战与前景 

---
# Level Up Peer Review in Education: Investigating genAI-driven Gamification system and its influence on Peer Feedback Effectiveness 

**Title (ZH)**: 提升教育中的同行评审水平：探究由生成式AI驱动的游戏化系统及其对同行反馈有效性的影响 

**Authors**: Rafal Wlodarski, Leonardo da Silva Sousa, Allison Connell Pensky  

**Link**: [PDF](https://arxiv.org/pdf/2504.02962)  

**Abstract**: In software engineering (SE), the ability to review code and critique designs is essential for professional practice. However, these skills are rarely emphasized in formal education, and peer feedback quality and engagement can vary significantly among students. This paper introduces Socratique, a gamified peer-assessment platform integrated with Generative AI (GenAI) assistance, designed to develop students' peer-review skills in a functional programming course. By incorporating game elements, Socratique aims to motivate students to provide more feedback, while the GenAI assistant offers real-time support in crafting high quality, constructive comments. To evaluate the impact of this approach, we conducted a randomized controlled experiment with master's students comparing a treatment group with a gamified, GenAI-driven setup against a control group with minimal gamification. Results show that students in the treatment group provided significantly more voluntary feedback, with higher scores on clarity, relevance, and specificity - all key aspects of effective code and design reviews. This study provides evidence for the effectiveness of combining gamification and AI to improve peer review processes, with implications for fostering review-related competencies in software engineering curricula. 

**Abstract (ZH)**: 软件工程中，审查代码和批评设计的能力对于专业实践至关重要。然而，这些技能在正规教育中 rarely 被重视，同学之间的反馈质量和参与度差异显著。本文介绍了 Socratique，一个融入生成式人工智能 (GenAI) 助手的游戏化同伴评估平台，旨在通过功能编程课程提高学生的同行评审技能。通过融入游戏元素，Socratique 旨在激励学生提供更多反馈，而 GenAI 助手则提供即时支持，帮助编写高质量、建设性的评论。为了评估该方法的影响，我们对硕士生进行了随机对照实验，将接受游戏化和 GenAI 驱动设置的干预组与仅进行少量游戏化的对照组进行了比较。结果表明，干预组的学生提供了更多的自愿反馈，且在清晰度、相关性和具体性等方面得分更高——这些都是有效代码和设计审查的关键方面。本研究为结合游戏化和人工智能以改善同行评审过程的有效性提供了证据，并对软件工程课程中培养审查相关技能具有重要影响。 

---
# VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning 

**Title (ZH)**: VARGPT-v1.1: 通过迭代指令调优和强化学习改进视觉自回归统一模型 

**Authors**: Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.02949)  

**Abstract**: In this work, we present VARGPT-v1.1, an advanced unified visual autoregressive model that builds upon our previous framework VARGPT. The model preserves the dual paradigm of next-token prediction for visual understanding and next-scale generation for image synthesis. Specifically, VARGPT-v1.1 integrates: (1) a novel training strategy combining iterative visual instruction tuning with reinforcement learning through Direct Preference Optimization (DPO), (2) an expanded training corpus containing 8.3M visual-generative instruction pairs, (3) an upgraded language model backbone using Qwen2, (4) enhanced image generation resolution, and (5) emergent image editing capabilities without architectural modifications. These advancements enable VARGPT-v1.1 to achieve state-of-the-art performance in multimodal understanding and text-to-image instruction-following tasks, demonstrating significant improvements in both comprehension and generation metrics. Notably, through visual instruction tuning, the model acquires image editing functionality while maintaining architectural consistency with its predecessor, revealing the potential for unified visual understanding, generation, and editing. Our findings suggest that well-designed unified visual autoregressive models can effectively adopt flexible training strategies from large language models (LLMs), exhibiting promising scalability. The codebase and model weights are publicly available at this https URL. 

**Abstract (ZH)**: 在本文中，我们呈现了VARGPT-v1.1，这是一种先进的统一视觉自回归模型，基于我们之前的框架VARGPT。该模型保留了视觉理解中的下一个令牌预测和图像合成中的下一个尺度生成的双重范式。具体而言，VARGPT-v1.1 集成了：(1) 一种新颖的训练策略，结合迭代的视觉指令调优与直接偏好优化(DPO)强化学习，(2) 包含8.3M视觉生成指令对的扩展训练语料库，(3) 升级的基于Qwen2的语言模型主干，(4) 增强的图像生成分辨率，以及(5) 无需架构修改的 Emergent 图像编辑能力。这些进步使VARGPT-v1.1 在多模态理解和文本到图像指令跟随任务中达到了最先进的性能，显示出在理解和生成方面的重要改进。通过视觉指令调优，模型获得了图像编辑功能，同时保持了与前任的一致架构，揭示了统一视觉理解、生成和编辑的潜力。我们的研究结果表明，设计良好的统一视觉自回归模型可以有效采用来自大型语言模型(LLMs)的灵活训练策略，显示出强大的可扩展性。相关代码库和模型权重可在以下链接公开获取。 

---
# Graph Attention for Heterogeneous Graphs with Positional Encoding 

**Title (ZH)**: 基于位置编码的异构图注意力模型 

**Authors**: Nikhil Shivakumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.02938)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as the de facto standard for modeling graph data, with attention mechanisms and transformers significantly enhancing their performance on graph-based tasks. Despite these advancements, the performance of GNNs on heterogeneous graphs often remains complex, with networks generally underperforming compared to their homogeneous counterparts. This work benchmarks various GNN architectures to identify the most effective methods for heterogeneous graphs, with a particular focus on node classification and link prediction. Our findings reveal that graph attention networks excel in these tasks. As a main contribution, we explore enhancements to these attention networks by integrating positional encodings for node embeddings. This involves utilizing the full Laplacian spectrum to accurately capture both the relative and absolute positions of each node within the graph, further enhancing performance on downstream tasks such as node classification and link prediction. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为建模图数据的事实标准，注意力机制和变压器极大提升了其在图基任务上的性能。尽管取得了这些进展，GNNs在异构图上的性能仍然复杂，通常表现不如其同质版本。本研究对多种GNN架构进行了基准测试，以识别最适合异构图的有效方法，特别关注节点分类和链接预测。我们的研究表明，图注意力网络在这些任务上表现出色。作为主要贡献，我们通过集成节点嵌入的位置编码来探索这些注意力网络的增强方法，使用完整的拉普拉斯谱准确捕获图中每个节点的相对和绝对位置，进一步提升了下游任务如节点分类和链接预测的性能。 

---
# Robustly identifying concepts introduced during chat fine-tuning using crosscoders 

**Title (ZH)**: robustly识别聊天微调过程中引入的概念 Using Crosscoders 

**Authors**: Julian Minder, Clement Dumas, Caden Juang, Bilal Chugtai, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2504.02922)  

**Abstract**: Model diffing is the study of how fine-tuning changes a model's representations and internal algorithms. Many behaviours of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent's presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of genuinely chat-specific latents that are both interpretable and causally effective, representing concepts such as $\textit{false information}$ and $\textit{personal question}$, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat tuning modifies language model behavior. 

**Abstract (ZH)**: 跨模型差异分析是研究微调如何改变模型的表示和内部算法的研究。许多感兴趣的表征行为在微调过程中被引入，跨模型差异分析提供了一种有前景的方法来解释这些行为。跨模型差异分析方法Crosscoders最近通过在基模型和微调模型中学习一个可解释概念的共享字典，这些概念以潜在方向表示，使我们能够追踪概念在微调过程中如何变化或新出现。值得注意的是，先前的工作观察到基模型中没有方向的概念，并推测这些模型特异性潜在变量是在微调过程中引入的概念。然而，我们发现源自Crosscoders L1训练损失的两个问题可能导致错误地将这些概念归因于微调后的模型，而实际上这两个模型中都存在这些概念。我们开发了潜在缩放来通过更准确地测量每个潜在变量在模型中的存在情况来标记这些问题。在将Gemma 2 2B基模型和聊天模型进行比较的实验中，我们观察到标准的Crosscoders严重受到这些问题的影响。基于这些洞见，我们训练了一个使用BatchTopK损失的Crosscoders，并展示了它显著缓解了这些问题，发现了更多真正属于聊天模型和高度可解释的概念。我们建议研究人员采用类似的技术。使用BatchTopK Crosscoders，我们成功地识别出一组真正属于聊天模型的潜在变量，这些潜在变量不仅是可解释的，而且具有因果效果，代表了诸如“虚假信息”和“私人问题”等概念，以及显示不同拒绝触发器偏好变化的多个拒绝相关的潜在变量。总体而言，我们的工作推进了基于Crosscoders的方法在模型差异分析中的最佳实践，并证明了这种方法可以提供关于聊天微调如何修改语言模型行为的具体见解。 

---
# Bias in Large Language Models Across Clinical Applications: A Systematic Review 

**Title (ZH)**: 大型语言模型在临床应用中的偏见：一项系统评价 

**Authors**: Thanathip Suenghataiphorn, Narisara Tribuddharat, Pojsakorn Danpanichkul, Narathorn Kulthamrongsri  

**Link**: [PDF](https://arxiv.org/pdf/2504.02917)  

**Abstract**: Background: Large language models (LLMs) are rapidly being integrated into healthcare, promising to enhance various clinical tasks. However, concerns exist regarding their potential for bias, which could compromise patient care and exacerbate health inequities. This systematic review investigates the prevalence, sources, manifestations, and clinical implications of bias in LLMs. Methods: We conducted a systematic search of PubMed, OVID, and EMBASE from database inception through 2025, for studies evaluating bias in LLMs applied to clinical tasks. We extracted data on LLM type, bias source, bias manifestation, affected attributes, clinical task, evaluation methods, and outcomes. Risk of bias was assessed using a modified ROBINS-I tool. Results: Thirty-eight studies met inclusion criteria, revealing pervasive bias across various LLMs and clinical applications. Both data-related bias (from biased training data) and model-related bias (from model training) were significant contributors. Biases manifested as: allocative harm (e.g., differential treatment recommendations); representational harm (e.g., stereotypical associations, biased image generation); and performance disparities (e.g., variable output quality). These biases affected multiple attributes, most frequently race/ethnicity and gender, but also age, disability, and language. Conclusions: Bias in clinical LLMs is a pervasive and systemic issue, with a potential to lead to misdiagnosis and inappropriate treatment, particularly for marginalized patient populations. Rigorous evaluation of the model is crucial. Furthermore, the development and implementation of effective mitigation strategies, coupled with continuous monitoring in real-world clinical settings, are essential to ensure the safe, equitable, and trustworthy deployment of LLMs in healthcare. 

**Abstract (ZH)**: 背景：大规模语言模型（LLMs）正迅速被整合到医疗保健领域，有望增强各种临床任务。然而，其潜在的偏见引起了关注，这可能会损害患者护理并加剧健康不平等。本系统综述探讨了LLMs在临床任务中偏见的普遍性、来源、表现形式及其临床影响。 

---
# Haphazard Inputs as Images in Online Learning 

**Title (ZH)**: 随机输入作为在线学习中的图像 

**Authors**: Rohit Agarwal, Aryan Dessai, Arif Ahmed Sekh, Krishna Agarwal, Alexander Horsch, Dilip K. Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2504.02912)  

**Abstract**: The field of varying feature space in online learning settings, also known as haphazard inputs, is very prominent nowadays due to its applicability in various fields. However, the current solutions to haphazard inputs are model-dependent and cannot benefit from the existing advanced deep-learning methods, which necessitate inputs of fixed dimensions. Therefore, we propose to transform the varying feature space in an online learning setting to a fixed-dimension image representation on the fly. This simple yet novel approach is model-agnostic, allowing any vision-based models to be applicable for haphazard inputs, as demonstrated using ResNet and ViT. The image representation handles the inconsistent input data seamlessly, making our proposed approach scalable and robust. We show the efficacy of our method on four publicly available datasets. The code is available at this https URL. 

**Abstract (ZH)**: 在线学习环境中变量特征空间的处理：从乱序输入到固定维度图像表示的转换 

---
# Noiser: Bounded Input Perturbations for Attributing Large Language Models 

**Title (ZH)**: Noiser: 有限输入扰动的大语言模型归因方法 

**Authors**: Mohammad Reza Ghasemi Madani, Aryo Pradipta Gema, Gabriele Sarti, Yu Zhao, Pasquale Minervini, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2504.02911)  

**Abstract**: Feature attribution (FA) methods are common post-hoc approaches that explain how Large Language Models (LLMs) make predictions. Accordingly, generating faithful attributions that reflect the actual inner behavior of the model is crucial. In this paper, we introduce Noiser, a perturbation-based FA method that imposes bounded noise on each input embedding and measures the robustness of the model against partially noised input to obtain the input attributions. Additionally, we propose an answerability metric that employs an instructed judge model to assess the extent to which highly scored tokens suffice to recover the predicted output. Through a comprehensive evaluation across six LLMs and three tasks, we demonstrate that Noiser consistently outperforms existing gradient-based, attention-based, and perturbation-based FA methods in terms of both faithfulness and answerability, making it a robust and effective approach for explaining language model predictions. 

**Abstract (ZH)**: 基于扰动的特征归因方法Noiser通过在每个输入嵌入上施加有界噪声并测量模型对部分噪声输入的鲁棒性来生成输入归因。此外，我们提出了一种可回答性度量，利用一个指令指导的判断模型评估高分词汇恢复预测输出的程度。通过在六种大语言模型和三种任务上的全面评估，我们证明Noiser在忠实性和可回答性方面均优于现有的梯度基、注意力基和扰动基特征归因方法，使它成为解释语言模型预测结果的稳健且有效的方法。 

---
# Systematic Literature Review: Explainable AI Definitions and Challenges in Education 

**Title (ZH)**: 系统文献综述：可解释人工智能在教育中的定义与挑战 

**Authors**: Zaid M. Altukhi, Sojen Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02910)  

**Abstract**: Explainable AI (XAI) seeks to transform black-box algorithmic processes into transparent ones, enhancing trust in AI applications across various sectors such as education. This review aims to examine the various definitions of XAI within the literature and explore the challenges of XAI in education. Our goal is to shed light on how XAI can contribute to enhancing the educational field. This systematic review, utilising the PRISMA method for rigorous and transparent research, identified 19 relevant studies. Our findings reveal 15 definitions and 62 challenges. These challenges are categorised using thematic analysis into seven groups: explainability, ethical, technical, human-computer interaction (HCI), trustworthiness, policy and guideline, and others, thereby deepening our understanding of the implications of XAI in education. Our analysis highlights the absence of standardised definitions for XAI, leading to confusion, especially because definitions concerning ethics, trustworthiness, technicalities, and explainability tend to overlap and vary. 

**Abstract (ZH)**: 可解释的人工智能（XAI）旨在将黑盒算法过程转变为透明过程，增强跨教育等各领域的AI应用信任。本文旨在于文献中回顾XAI的各种定义，并探索教育领域中XAI的挑战。我们的目标是阐明XAI如何为教育领域做出贡献。本系统综述采用PRISMA方法进行严格和透明的研究，共识别出19项相关研究。研究发现15种定义和62项挑战。这些挑战通过主题分析被归类为七大类：可解释性、伦理、技术、人机交互（HCI）、可信度、政策与指南，以及其他，从而加深了我们对教育领域中XAI影响的理解。我们的分析指出，缺乏标准化的XAI定义导致了 confusion，特别是因为与伦理、可信度、技术和可解释性相关的定义往往重叠且变化不一。 

---
# Enhancing Chart-to-Code Generation in Multimodal Large Language Models via Iterative Dual Preference Learning 

**Title (ZH)**: 通过迭代双重偏好学习增强多模态大语言模型的图表到代码生成 

**Authors**: Zhihan Zhang, Yixin Cao, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02906)  

**Abstract**: Chart-to-code generation, the process of converting chart images into executable plotting scripts, provides a lossless representation of chart information, requiring models to accurately capture and summarize all visual and structural elements. However, this remains a significant challenge for multimodal large language models (MLLMs), which are not inherently well-aligned with code generation tasks. To bridge this gap, we introduce Chart2Code, a novel iterative dual preference learning framework designed to enhance MLLMs' chart-to-code generation capabilities through structured code variant generation and fine-grained dual reward signals. We validate Chart2Code across three MLLMs and find that iterative preference learning consistently improves out-of-distribution chart-to-code generation quality. Throughout this process, our dual scoring method, which evaluates both the textual code structure and its visual representation, leads to greater performance improvements, even with a reduced preference dataset size. Further analysis explores the key components of our framework and highlights the interplay between chart-to-code generation and broader chart reasoning, paving the way for future advancements in chart comprehension. 

**Abstract (ZH)**: Chart-to-code 生成：一种通过迭代双重偏好学习框架增强多模态大规模语言模型图表生成能力的方法 

---
# How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence 

**Title (ZH)**: 训练后重塑大语言模型：知识、诚实、谢绝与自信的机制视角 

**Authors**: Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Himabindu Lakkaraju, Yizhou Sun, Shichang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02904)  

**Abstract**: Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by linear vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. 

**Abstract (ZH)**: Post-训练对于大型语言模型（LLMs）的成功至关重要，它将预训练的基础模型转换为更有用且更对齐的后训练模型。虽然已有大量研究探讨了后训练算法并基于输出评估了后训练模型，但关于后训练如何从内部重塑LLMs的研究仍显不足。在本文中，我们从四个角度机械地比较基础模型和后训练模型，以更好地理解后训练的效果。我们的研究发现，跨模型家族和数据集表明：（1）后训练不改变事实知识的存储位置，它适应基础模型的知识表示并发展新的知识表示；（2）真实性和拒绝都可以通过隐藏表示空间中的线性向量来表示。基模型和后训练模型的真实方向非常相似，且能够有效转移；（3）拒绝方向在基模型和后训练模型之间不同，并显示出有限的前向可转移性；（4）基模型和后训练模型之间的自信度差异不能归因于熵神经元。我们的研究 insights 了后训练期间保留和改变的基本机制，促进了下游任务如模型引导，并有可能为未来的研究提供释义能力和LLM后训练的益处。 

---
# Beyond Accuracy: The Role of Calibration in Self-Improving Large Language Models 

**Title (ZH)**: 超越准确性：校准在自改进大型语言模型中的作用 

**Authors**: Liangjie Huang, Dawei Li, Huan Liu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.02902)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable self-improvement capabilities, whereby models iteratively revise their outputs through self-generated feedback. While this reflective mechanism has shown promise in enhancing task performance, recent studies suggest that it may also introduce undesirable biases-most notably, self-bias, or the tendency of LLMs to favor their own prior outputs. In this work, we extend this line of inquiry by investigating the impact on confidence estimation. We evaluate three representative self-improvement paradigms-basic prompting, Chain-of-Thought (CoT) prompting, and tuning-based methods and find that iterative self-improvement can lead to systematic overconfidence, as evidenced by a steadily increasing Expected Calibration Error (ECE) and lower accuracy with high confidence. We then further explore the integration of confidence calibration techniques with self-improvement. Specifically, we compare three strategies: (1) applying calibration after multiple rounds of self-improvement, (2) calibrating before self-improvement, and (3) applying calibration iteratively at each self-improvement step. Our results show that iterative calibration is most effective in reducing ECE, yielding improved calibration. Our work pioneers the study of self-improving LLMs from a calibration perspective, offering valuable insights into balancing model performance and reliability. 

**Abstract (ZH)**: 大型语言模型的自提高能力可以通过自动生成的反馈迭代修订其输出，显示出显著的自我改进能力。虽然这种反思机制展示了提升任务性能的潜力，但 recent 研究表明，它也可能引入不 desirable 的偏差——最典型的是自偏差，即大型语言模型倾向于偏好其自身的先验输出。在本文中，我们在此研究方向上进一步探讨了对其自提高对置信度估计的影响。我们评估了三种代表性的自提高范式——基本提示、思维链（CoT）提示和基于调优的方法，发现迭代自我提高可能导致系统性过自信，这体现在预期校准误差（ECE）的持续增加和高置信度下的更低准确性。然后，我们进一步探索了校准技术与自提高的整合。具体来说，我们比较了三种策略：（1）在多轮自我提高后应用校准；（2）在自我提高前应用校准；（3）在每次自我提高步骤中迭代应用校准。我们的结果表明，迭代校准在降低 ECE 方面最有效，从而提高了校准效果。本研究从校准角度探讨了自我提高的大型语言模型，为我们平衡模型性能和可靠性提供了宝贵的见解。 

---
# Hide and Seek in Noise Labels: Noise-Robust Collaborative Active Learning with LLM-Powered Assistance 

**Title (ZH)**: 在噪声标签中隐藏与寻找：基于LLM助力的噪声鲁棒协作主动学习 

**Authors**: Bo Yuan, Yulin Chen, Yin Zhang, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02901)  

**Abstract**: Learning from noisy labels (LNL) is a challenge that arises in many real-world scenarios where collected training data can contain incorrect or corrupted labels. Most existing solutions identify noisy labels and adopt active learning to query human experts on them for denoising. In the era of large language models (LLMs), although we can reduce the human effort to improve these methods, their performances are still subject to accurately separating the clean and noisy samples from noisy data. In this paper, we propose an innovative collaborative learning framework NoiseAL based on active learning to combine LLMs and small models (SMs) for learning from noisy labels. During collaborative training, we first adopt two SMs to form a co-prediction network and propose a dynamic-enhanced threshold strategy to divide the noisy data into different subsets, then select the clean and noisy samples from these subsets to feed the active annotator LLMs to rectify noisy samples. Finally, we employ different optimization objectives to conquer subsets with different degrees of label noises. Extensive experiments on synthetic and real-world noise datasets further demonstrate the superiority of our framework over state-of-the-art baselines. 

**Abstract (ZH)**: 基于主动学习的NoiseAL框架：结合大语言模型和小型模型从噪声标签中学习 

---
# Meat-Free Day Reduces Greenhouse Gas Emissions but Poses Challenges for Customer Retention and Adherence to Dietary Guidelines 

**Title (ZH)**: 无肉日减少温室气体排放但面临客户留存和饮食指南遵守的挑战 

**Authors**: Giuseppe Russo, Kristina Gligorić, Vincent Moreau, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2504.02899)  

**Abstract**: Reducing meat consumption is crucial for achieving global environmental and nutritional targets. Meat-Free Day (MFD) is a widely adopted strategy to address this challenge by encouraging plant-based diets through the removal of animal-based meals. We assessed the environmental, behavioral, and nutritional impacts of MFD by implementing 67 MFDs over 18 months (once a week on a randomly chosen day) across 12 cafeterias on a large university campus, analyzing over 400,000 food purchases. MFD reduced on-campus food-related greenhouse gas (GHG) emissions on treated days by 52.9% and contributed to improved fiber (+26.9%) and cholesterol (-4.5%) consumption without altering caloric intake. These nutritional benefits were, however, accompanied by a 27.6% decrease in protein intake and a 34.2% increase in sugar consumption. Moreover, the increase in plant-based meals did not carry over to subsequent days, as evidenced by a 3.5% rebound in animal-based meal consumption on days immediately following treated days. MFD also led to a 16.8% drop in on-campus meal sales on treated this http URL Carlo simulations suggest that if 8.7% of diners were to eat burgers off-campus on treated days, MFD's GHG savings would be fully negated. As our analysis identifies on-campus customer retention as the main challenge to MFD effectiveness, we recommend combining MFD with customer retention interventions to ensure environmental and nutritional benefits. 

**Abstract (ZH)**: 减少肉类消费对于实现全球环境和营养目标至关重要。"无肉日"（MFD）通过推广植物基饮食来应对这一挑战，研究表明，实施MFD可以降低校内与食物相关的温室气体排放，改善纤维和胆固醇的摄入，同时不影响热量摄入。然而，这也伴随着蛋白质摄入量的下降和糖分摄入量的增加。此外，植物基餐食的增加并未持续到后续日子，表明在MFD后立即消费的动物基餐食有所回升。模拟结果显示，如果8.7%的就餐者在MFD在外就餐，将抵消MFD的温室气体减排效果。鉴于分析指出在校顾客留存是MFD效果的主要挑战，建议结合顾客留存措施以确保环境和营养效益。 

---
# UAC: Uncertainty-Aware Calibration of Neural Networks for Gesture Detection 

**Title (ZH)**: UAC：面向手势检测的不确定性意识校准的神经网络 

**Authors**: Farida Al Haddad, Yuxin Wang, Malcolm Mielle  

**Link**: [PDF](https://arxiv.org/pdf/2504.02895)  

**Abstract**: Artificial intelligence has the potential to impact safety and efficiency in safety-critical domains such as construction, manufacturing, and healthcare. For example, using sensor data from wearable devices, such as inertial measurement units (IMUs), human gestures can be detected while maintaining privacy, thereby ensuring that safety protocols are followed. However, strict safety requirements in these domains have limited the adoption of AI, since accurate calibration of predicted probabilities and robustness against out-of-distribution (OOD) data is necessary.
This paper proposes UAC (Uncertainty-Aware Calibration), a novel two-step method to address these challenges in IMU-based gesture recognition. First, we present an uncertainty-aware gesture network architecture that predicts both gesture probabilities and their associated uncertainties from IMU data. This uncertainty is then used to calibrate the probabilities of each potential gesture. Second, an entropy-weighted expectation of predictions over multiple IMU data windows is used to improve accuracy while maintaining correct calibration.
Our method is evaluated using three publicly available IMU datasets for gesture detection and is compared to three state-of-the-art calibration methods for neural networks: temperature scaling, entropy maximization, and Laplace approximation. UAC outperforms existing methods, achieving improved accuracy and calibration in both OOD and in-distribution scenarios. Moreover, we find that, unlike our method, none of the state-of-the-art methods significantly improve the calibration of IMU-based gesture recognition models. In conclusion, our work highlights the advantages of uncertainty-aware calibration of neural networks, demonstrating improvements in both calibration and accuracy for gesture detection using IMU data. 

**Abstract (ZH)**: 人工智能有潜力在建筑、制造和医疗等关键安全领域影响安全性和效率。例如，通过穿戴设备（如惯性测量单元IMU）的传感器数据，可以在保护隐私的同时检测人类手势，从而确保遵守安全规范。然而，这些领域严格的安全要求限制了人工智能的应用，因为准确的预测概率校准和对分布外（OOD）数据的鲁棒性是必要的。

本文提出了一种新颖的两步方法UAC（不确定性感知校准）来解决基于IMU的手势识别中的这些挑战。首先，我们提出了一种不确定性感知的手势网络架构，该架构可以从IMU数据中预测手势的概率及其相关不确定性。然后，使用这些不确定性来校准每种潜在手势的概率。其次，通过在多个IMU数据窗口上的熵加权预测期望来提高准确性同时保持正确的校准。

我们的方法使用三个公开可用的IMU数据集进行了手势检测的评估，并与三种最先进的神经网络校准方法（温度缩放、熵最大化、拉普拉斯近似）进行了比较。UAC在OOD和同分布场景中均实现了更好的准确性和校准，此外，我们发现与我们的方法不同，最先进的方法没有在基于IMU的手势识别模型的校准方面显著提高。总之，我们的工作突显了不确定性感知校准神经网络的优势，展示了使用IMU数据进行手势检测时在校准和准确性方面的改进。 

---
# OnRL-RAG: Real-Time Personalized Mental Health Dialogue System 

**Title (ZH)**: OnRL-RAG：实时个性化心理健康对话系统 

**Authors**: Ahsan Bilal, Beiyu Lin, Mehdi Zaeifi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02894)  

**Abstract**: Large language models (LLMs) have been widely used for various tasks and applications. However, LLMs and fine-tuning are limited to the pre-trained data. For example, ChatGPT's world knowledge until 2021 can be outdated or inaccurate. To enhance the capabilities of LLMs, Retrieval-Augmented Generation (RAG), is proposed to augment LLMs with additional, new, latest details and information to LLMs. While RAG offers the correct information, it may not best present it, especially to different population groups with personalizations. Reinforcement Learning from Human Feedback (RLHF) adapts to user needs by aligning model responses with human preference through feedback loops. In real-life applications, such as mental health problems, a dynamic and feedback-based model would continuously adapt to new information and offer personalized assistance due to complex factors fluctuating in a daily environment. Thus, we propose an Online Reinforcement Learning-based Retrieval-Augmented Generation (OnRL-RAG) system to detect and personalize the responding systems to mental health problems, such as stress, anxiety, and depression. We use an open-source dataset collected from 2028 College Students with 28 survey questions for each student to demonstrate the performance of our proposed system with the existing systems. Our system achieves superior performance compared to standard RAG and simple LLM via GPT-4o, GPT-4o-mini, Gemini-1.5, and GPT-3.5. This work would open up the possibilities of real-life applications of LLMs for personalized services in the everyday environment. The results will also help researchers in the fields of sociology, psychology, and neuroscience to align their theories more closely with the actual human daily environment. 

**Abstract (ZH)**: 基于在线强化学习的检索增强生成（OnRL-RAG）系统：用于心理健康问题的个性化响应平台 

---
# Automated Survey Collection with LLM-based Conversational Agents 

**Title (ZH)**: 基于LLM的对话代理自动化调查收集 

**Authors**: Kurmanbek Kaiyrbekov, Nicholas J Dobbins, Sean D Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2504.02891)  

**Abstract**: Objective: Traditional phone-based surveys are among the most accessible and widely used methods to collect biomedical and healthcare data, however, they are often costly, labor intensive, and difficult to scale effectively. To overcome these limitations, we propose an end-to-end survey collection framework driven by conversational Large Language Models (LLMs).
Materials and Methods: Our framework consists of a researcher responsible for designing the survey and recruiting participants, a conversational phone agent powered by an LLM that calls participants and administers the survey, a second LLM (GPT-4o) that analyzes the conversation transcripts generated during the surveys, and a database for storing and organizing the results. To test our framework, we recruited 8 participants consisting of 5 native and 3 non-native english speakers and administered 40 surveys. We evaluated the correctness of LLM-generated conversation transcripts, accuracy of survey responses inferred by GPT-4o and overall participant experience.
Results: Survey responses were successfully extracted by GPT-4o from conversation transcripts with an average accuracy of 98% despite transcripts exhibiting an average per-line word error rate of 7.7%. While participants noted occasional errors made by the conversational LLM agent, they reported that the agent effectively conveyed the purpose of the survey, demonstrated good comprehension, and maintained an engaging interaction.
Conclusions: Our study highlights the potential of LLM agents in conducting and analyzing phone surveys for healthcare applications. By reducing the workload on human interviewers and offering a scalable solution, this approach paves the way for real-world, end-to-end AI-powered phone survey collection systems. 

**Abstract (ZH)**: 目标：传统的基于电话的调查是收集生物医学和卫生健康数据最为便捷和广泛使用的方法，然而，它们往往成本高昂、劳动密集且难以有效扩展。为克服这些局限，我们提出了一种由对话型大规模语言模型（LLMs）驱动的端到端调查数据收集框架。 

---
# Scaling Test-time Compute for Low-resource Languages: Multilingual Reasoning in LLMs 

**Title (ZH)**: 低资源语言测试时计算量缩放：LLMs中的多语言推理 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02890)  

**Abstract**: Recent advances in test-time compute scaling have enabled Large Language Models (LLMs) to tackle deep reasoning tasks by generating a chain-of-thought (CoT) that includes trial and error, backtracking, and intermediate reasoning steps before producing the final answer. However, these techniques have been applied predominantly to popular languages, such as English, leaving reasoning in low-resource languages underexplored and misaligned. In this work, we investigate the multilingual mechanism by which LLMs internally operate in a latent space biased toward their inherently dominant language. To leverage this phenomenon for low-resource languages, we train models to generate the CoT in English while outputting the final response in the target language, given input in the low-resource language. Our experiments demonstrate that this approach, named English-Pivoted CoT Training, outperforms other baselines, including training to generate both the CoT and the final response solely in the target language, with up to 28.33% improvement. Further analysis provides novel insights into the relationships between reasoning and multilinguality of LLMs, prompting for better approaches in developing multilingual large reasoning models 

**Abstract (ZH)**: Recent advances in test-time compute scaling have enabled Large Language Models (LLMs) to tackle deep reasoning tasks by generating a chain-of-thought (CoT) that includes trial and error, backtracking, and intermediate reasoning steps before producing the final answer. However, these techniques have been applied predominantly to popular languages, such as English, leaving reasoning in low-resource languages underexplored and misaligned. In this work, we investigate the multilingual mechanism by which LLMs internally operate in a latent space biased toward their inherently dominant language. To leverage this phenomenon for low-resource languages, we train models to generate the CoT in English while outputting the final response in the target language, given input in the low-resource language. Our experiments demonstrate that this approach, named English-Pivoted CoT Training, outperforms other baselines, including training to generate both the CoT and the final response solely in the target language, with up to 28.33% improvement. Further analysis provides novel insights into the relationships between reasoning and multilinguality of LLMs, prompting for better approaches in developing multilingual large reasoning models。 

---
# Embedding Method for Knowledge Graph with Densely Defined Ontology 

**Title (ZH)**: 基于密集定义本体的知识图嵌入方法 

**Authors**: Takanori Ugai  

**Link**: [PDF](https://arxiv.org/pdf/2504.02889)  

**Abstract**: Knowledge graph embedding (KGE) is a technique that enhances knowledge graphs by addressing incompleteness and improving knowledge retrieval. A limitation of the existing KGE models is their underutilization of ontologies, specifically the relationships between properties. This study proposes a KGE model, TransU, designed for knowledge graphs with well-defined ontologies that incorporate relationships between properties. The model treats properties as a subset of entities, enabling a unified representation. We present experimental results using a standard dataset and a practical dataset. 

**Abstract (ZH)**: 基于本体的知识图嵌入模型（TransU）：面向具定义关系属性的知识图谱 

---
# Global Rice Multi-Class Segmentation Dataset (RiceSEG): A Comprehensive and Diverse High-Resolution RGB-Annotated Images for the Development and Benchmarking of Rice Segmentation Algorithms 

**Title (ZH)**: 全球多类水稻分割数据集（RiceSEG）：高分辨率RGB标注图像的综合多样数据集，用于水稻分割算法的开发与 benchmarking 

**Authors**: Junchi Zhou, Haozhou Wang, Yoichiro Kato, Tejasri Nampally, P. Rajalakshmi, M. Balram, Keisuke Katsura, Hao Lu, Yue Mu, Wanneng Yang, Yangmingrui Gao, Feng Xiao, Hongtao Chen, Yuhao Chen, Wenjuan Li, Jingwen Wang, Fenghua Yu, Jian Zhou, Wensheng Wang, Xiaochun Hu, Yuanzhu Yang, Yanfeng Ding, Wei Guo, Shouyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02880)  

**Abstract**: Developing computer vision-based rice phenotyping techniques is crucial for precision field management and accelerating breeding, thereby continuously advancing rice production. Among phenotyping tasks, distinguishing image components is a key prerequisite for characterizing plant growth and development at the organ scale, enabling deeper insights into eco-physiological processes. However, due to the fine structure of rice organs and complex illumination within the canopy, this task remains highly challenging, underscoring the need for a high-quality training dataset. Such datasets are scarce, both due to a lack of large, representative collections of rice field images and the time-intensive nature of annotation. To address this gap, we established the first comprehensive multi-class rice semantic segmentation dataset, RiceSEG. We gathered nearly 50,000 high-resolution, ground-based images from five major rice-growing countries (China, Japan, India, the Philippines, and Tanzania), encompassing over 6,000 genotypes across all growth stages. From these original images, 3,078 representative samples were selected and annotated with six classes (background, green vegetation, senescent vegetation, panicle, weeds, and duckweed) to form the RiceSEG dataset. Notably, the sub-dataset from China spans all major genotypes and rice-growing environments from the northeast to the south. Both state-of-the-art convolutional neural networks and transformer-based semantic segmentation models were used as baselines. While these models perform reasonably well in segmenting background and green vegetation, they face difficulties during the reproductive stage, when canopy structures are more complex and multiple classes are involved. These findings highlight the importance of our dataset for developing specialized segmentation models for rice and other crops. 

**Abstract (ZH)**: 基于计算机视觉的水稻表型技术开发对于精确田间管理及加速育种具有重要意义，从而不断推进水稻生产。水稻表型任务中的图像组件区分是表征器官尺度植物生长和发育的关键前提，有助于深入理解生态生理过程。然而，由于水稻器官的精细结构和冠层内的复杂光照，这一任务依然极具挑战性，迫切需要高质量的训练数据集。这类数据集稀缺，主要是因为缺乏大型、代表性强的水稻田图像集合以及注释工作量大。为填补这一空白，我们建立了首个全面的多类水稻语义分割数据集RiceSEG。我们从五个主要水稻种植国家（中国、日本、印度、菲律宾和坦桑尼亚）收集了近50,000张高分辨率地面图像，涵盖了超过6,000个基因型的所有生长阶段。从这些原始图像中，选择了3,078个代表性样本，并根据六类（背景、绿色植被、衰老植被、穗轴、杂草和水葫芦）进行了标注，形成了RiceSEG数据集。特别地，中国子数据集涵盖了东北至南部的所有主要基因型和水稻种植环境。使用了最新的卷积神经网络和基于变压器的语义分割模型作为基准。尽管这些模型在分割背景和绿色植被方面表现良好，但在生殖期，由于冠层结构更加复杂且涉及多个类别，它们面临困难。这些发现强调了我们的数据集对于开发专门化的水稻及其他作物分割模型的重要性。 

---
# Scraping the Shadows: Deep Learning Breakthroughs in Dark Web Intelligence 

**Title (ZH)**: 刮开阴影：暗网intelligence深度学习突破 

**Authors**: Ingmar Bakermans, Daniel De Pascale, Gonçalo Marcelino, Giuseppe Cascavilla, Zeno Geradts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02872)  

**Abstract**: Darknet markets (DNMs) facilitate the trade of illegal goods on a global scale. Gathering data on DNMs is critical to ensuring law enforcement agencies can effectively combat crime. Manually extracting data from DNMs is an error-prone and time-consuming task. Aiming to automate this process we develop a framework for extracting data from DNMs and evaluate the application of three state-of-the-art Named Entity Recognition (NER) models, ELMo-BiLSTM \citep{ShahEtAl2022}, UniversalNER \citep{ZhouEtAl2024}, and GLiNER \citep{ZaratianaEtAl2023}, at the task of extracting complex entities from DNM product listing pages. We propose a new annotated dataset, which we use to train, fine-tune, and evaluate the models. Our findings show that state-of-the-art NER models perform well in information extraction from DNMs, achieving 91% Precision, 96% Recall, and an F1 score of 94%. In addition, fine-tuning enhances model performance, with UniversalNER achieving the best performance. 

**Abstract (ZH)**: Darknet市场(DNMs)促进了全球非法商品的交易。收集DNM数据是确保执法机构能有效打击犯罪的关键。手动从DNM中提取数据是一项易出错且耗时的任务。为了自动化这一过程，我们开发了一个从DNM中抽取数据的框架，并评估了三种最先进的命名实体识别(NER)模型，即ELMo-BiLSTM、UniversalNER和GLiNER，在从DNM产品列表页抽取复杂实体任务中的应用。我们提出一个新的标注数据集，用于训练、微调和评估这些模型。我们的研究发现，最先进的NER模型在从DNM中提取信息方面表现良好，精度为91%，召回率为96%，F1分为94%。此外，模型微调提升了性能，UniversalNER表现最佳。 

---
# Synthesized Annotation Guidelines are Knowledge-Lite Boosters for Clinical Information Extraction 

**Title (ZH)**: 合成注释指南是临床信息提取的知识轻型助推器 

**Authors**: Enshuo Hsu, Martin Ugbala, Krishna Kumar Kookal, Zouaidi Kawtar, Nicholas L. Rider, Muhammad F. Walji, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02871)  

**Abstract**: Generative information extraction using large language models, particularly through few-shot learning, has become a popular method. Recent studies indicate that providing a detailed, human-readable guideline-similar to the annotation guidelines traditionally used for training human annotators can significantly improve performance. However, constructing these guidelines is both labor- and knowledge-intensive. Additionally, the definitions are often tailored to meet specific needs, making them highly task-specific and often non-reusable. Handling these subtle differences requires considerable effort and attention to detail. In this study, we propose a self-improving method that harvests the knowledge summarization and text generation capacity of LLMs to synthesize annotation guidelines while requiring virtually no human input. Our zero-shot experiments on the clinical named entity recognition benchmarks, 2012 i2b2 EVENT, 2012 i2b2 TIMEX, 2014 i2b2, and 2018 n2c2 showed 25.86%, 4.36%, 0.20%, and 7.75% improvements in strict F1 scores from the no-guideline baseline. The LLM-synthesized guidelines showed equivalent or better performance compared to human-written guidelines by 1.15% to 4.14% in most tasks. In conclusion, this study proposes a novel LLM self-improving method that requires minimal knowledge and human input and is applicable to multiple biomedical domains. 

**Abstract (ZH)**: 使用大规模语言模型进行生成式信息提取，特别是通过少样本学习，已成为一种流行的方法。最近的研究表明，提供类似于传统训练人类注释者所使用的注释指南的详细、人可读的指南可以显著提高性能。然而，构建这些指南既耗费劳动也耗费知识。此外，这些定义通常根据特定需求量身定制，因此高度特定于任务，往往不可重用。处理这些细微差异需要大量的努力和细节注意。在这项研究中，我们提出了一种自我改进的方法，该方法利用大规模语言模型的知识总结和文本生成能力合成注释指南，几乎不需要人工输入。我们在临床命名实体识别基准测试，2012 i2b2 EVENT、2012 i2b2 TIMEX、2014 i2b2 和 2018 n2c2 上的零样本实验显示，相对于无指南 baseline，在严格 F1 分数上分别提高了 25.86%、4.36%、0.20% 和 7.75%。在大多数任务中，由大规模语言模型合成的指南的性能与人工编写的指南相媲美或更优，差距在 1.15% 至 4.14% 之间。综上所述，本研究提出了一种需要最少知识和人工输入、适用于多个生物医学领域的新型大规模语言模型自我改进方法。 

---
# AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening 

**Title (ZH)**: 基于LLM的AI招聘：一种上下文感知且可解释的多agent筛选简历框架 

**Authors**: Frank P.-W. Lo, Jianing Qiu, Zeyu Wang, Haibao Yu, Yeming Chen, Gao Zhang, Benny Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02870)  

**Abstract**: Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows. 

**Abstract (ZH)**: 基于LLM的多agent框架在求职简历筛选中的应用 

---
# Multi-Agent LLM Judge: automatic personalized LLM judge design for evaluating natural language generation applications 

**Title (ZH)**: 多智能体LLM裁判：个性化自动LLM裁判设计以评估自然语言生成应用 

**Authors**: Hongliu Cao, Ilias Driouich, Robin Singh, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02867)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across diverse domains, yet they still encounter challenges such as insufficient domain-specific knowledge, biases, and hallucinations. This underscores the need for robust evaluation methodologies to accurately assess LLM-based applications. Traditional evaluation methods, which rely on word overlap or text embeddings, are inadequate for capturing the nuanced semantic information necessary to evaluate dynamic, open-ended text generation. Recent research has explored leveraging LLMs to mimic human reasoning and decision-making processes for evaluation purposes known as LLM-as-a-judge framework. However, these existing frameworks have two significant limitations. First, they lack the flexibility to adapt to different text styles, including various answer and ground truth styles, thereby reducing their generalization performance. Second, the evaluation scores produced by these frameworks are often skewed and hard to interpret, showing a low correlation with human judgment. To address these challenges, we propose a novel dynamic multi-agent system that automatically designs personalized LLM judges for various natural language generation applications. This system iteratively refines evaluation prompts and balances the trade-off between the adaptive requirements of downstream tasks and the alignment with human perception. Our experimental results show that the proposed multi-agent LLM Judge framework not only enhances evaluation accuracy compared to existing methods but also produces evaluation scores that better align with human perception. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个领域展现了令人印象深刻的性能，但仍面临诸如领域特定知识不足、偏差和幻觉等问题。这强调了需要稳健的评估方法以准确评估基于LLM的应用。传统的评估方法依赖于词重叠或文本嵌入，无法捕捉动态、开放式文本生成所需的细微语义信息。近期的研究探讨了利用LLM模仿人类推理和决策过程来进行评估，这一框架被称为LLM作为评估者框架。然而，这些现有框架存在两个显著局限性。首先，它们缺乏适应不同文本风格的灵活性，包括各种答案和真值风格，从而降低了其泛化性能。其次，这些框架产生的评估分数往往失真且难以解释，与人类判断的相关性较低。为应对这些挑战，我们提出了一种新的动态多Agent系统，该系统能够自动为各种自然语言生成应用设计个性化的LLM评估者。该系统迭代优化评估提示，并权衡下游任务的适应需求与与人类感知的对齐。实验结果显示，所提出的多Agent LLM评估者框架不仅在评估准确性上优于现有方法，而且产生的评估分数与人类感知更好地对齐。 

---
# Computer Vision and Deep Learning for 4D Augmented Reality 

**Title (ZH)**: 计算机视觉与深度学习在4D增强现实中的应用 

**Authors**: Karthik Shivashankar  

**Link**: [PDF](https://arxiv.org/pdf/2504.02860)  

**Abstract**: The prospect of 4D video in Extended Reality (XR) platform is huge and exciting, it opens a whole new way of human computer interaction and the way we perceive the reality and consume multimedia. In this thesis, we have shown that feasibility of rendering 4D video in Microsoft mixed reality platform. This enables us to port any 3D performance capture from CVSSP into XR product like the HoloLens device with relative ease. However, if the 3D model is too complex and is made up of millions of vertices, the data bandwidth required to port the model is a severe limitation with the current hardware and communication system. Therefore, in this project we have also developed a compact representation of both shape and appearance of the 4d video sequence using deep learning models to effectively learn the compact representation of 4D video sequence and reconstruct it without affecting the shape and appearance of the video sequence. 

**Abstract (ZH)**: 4D视频在扩展现实(XR)平台上的前景巨大而令人兴奋：基于Microsoft混合现实平台的实现与挑战 

---
# Exploration of Multi-Element Collaborative Research and Application for Modern Power System Based on Generative Large Models 

**Title (ZH)**: 基于生成式大型模型的现代电力系统多元素协同研究与应用探索 

**Authors**: Lu Cheng, Qixiu Zhang, Beibei Xu, Zhiwei Huang, Cirun Zhang, Yanan Lyu, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02855)  

**Abstract**: The transition to intelligent, low-carbon power systems necessitates advanced optimization strategies for managing renewable energy integration, energy storage, and carbon emissions. Generative Large Models (GLMs) provide a data-driven approach to enhancing forecasting, scheduling, and market operations by processing multi-source data and capturing complex system dynamics. This paper explores the role of GLMs in optimizing load-side management, energy storage utilization, and electricity carbon, with a focus on Smart Wide-area Hybrid Energy Systems with Storage and Carbon (SGLSC). By leveraging spatiotemporal modeling and reinforcement learning, GLMs enable dynamic energy scheduling, improve grid stability, enhance carbon trading strategies, and strengthen resilience against extreme weather events. The proposed framework highlights the transformative potential of GLMs in achieving efficient, adaptive, and low-carbon power system operations. 

**Abstract (ZH)**: 智能低碳电力系统转型需要先进的优化策略来管理可再生能源集成、能量存储和碳排放。生成型大型模型（GLMs）通过处理多源数据并捕捉复杂系统动力学，为增强预测、调度和市场运营提供数据驱动的方法。本文探讨了GLMs在优化负荷侧管理、储能利用和电力碳排放方面的作用，重点关注含储能和碳的智能广域混合能源系统（SGLSC）。通过利用时空建模和强化学习，GLMs使能动态能源调度，提高电网稳定性，提升碳交易策略，并增强对极端天气事件的韧性。提出的框架突显了GLMs在实现高效、适应性和低碳电力系统运行方面的变革潜力。 

---
# Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks 

**Title (ZH)**: 使用扩散图网络学习复杂流体模拟的分布 

**Authors**: Mario Lino, Tobias Pfaff, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2504.02843)  

**Abstract**: Physical systems with complex unsteady dynamics, such as fluid flows, are often poorly represented by a single mean solution. For many practical applications, it is crucial to access the full distribution of possible states, from which relevant statistics (e.g., RMS and two-point correlations) can be derived. Here, we propose a graph-based latent diffusion (or alternatively, flow-matching) model that enables direct sampling of states from their equilibrium distribution, given a mesh discretization of the system and its physical parameters. This allows for the efficient computation of flow statistics without running long and expensive numerical simulations. The graph-based structure enables operations on unstructured meshes, which is critical for representing complex geometries with spatially localized high gradients, while latent-space diffusion modeling with a multi-scale GNN allows for efficient learning and inference of entire distributions of solutions. A key finding is that the proposed networks can accurately learn full distributions even when trained on incomplete data from relatively short simulations. We apply this method to a range of fluid dynamics tasks, such as predicting pressure distributions on 3D wing models in turbulent flow, demonstrating both accuracy and computational efficiency in challenging scenarios. The ability to directly sample accurate solutions, and capturing their diversity from short ground-truth simulations, is highly promising for complex scientific modeling tasks. 

**Abstract (ZH)**: 具有复杂非稳态动力学的物理系统，如流体流动，往往无法通过单一的平均解来良好表示。对于许多实际应用而言，访问所有可能状态的完整分布至关重要，从中可以推导出相关统计量（例如，均方根值和两点相关性）。在这里，我们提出了一种基于图的潜在扩散（或等价地，流匹配）模型，该模型可以在给定系统及其物理参数的网格离散化的情况下直接从稳态分布中抽样状态。这使得可以在不运行长时间和昂贵的数值模拟的情况下高效计算流场统计量。基于图的结构能够对无结构网格进行操作，这对于表示具有空间局部高梯度的复杂几何形状至关重要，而多尺度GNN的潜在空间扩散建模则允许高效地学习和推断整个解的分布。一个关键发现是，所提出的网络即使在基于相对较短模拟的不完整数据训练时也能准确地学习完整的分布。我们应用此方法解决了一系列流体动力学任务，例如在湍流流动中预测三维机翼模型的压力分布，证明了在挑战性场景中其准确性和计算效率。直接从简短的真实数据模拟中抽样准确的解并捕获其多样性，对于复杂的科学技术建模任务具有高度的前景。 

---
# A First-Principles Based Risk Assessment Framework and the IEEE P3396 Standard 

**Title (ZH)**: 基于第一性原理的风险评估框架和IEEE P3396标准 

**Authors**: Richard J. Tong, Marina Cortês, Jeanine A. DeFalco, Mark Underwood, Janusz Zalewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.00091)  

**Abstract**: Generative Artificial Intelligence (AI) is enabling unprecedented automation in content creation and decision support, but it also raises novel risks. This paper presents a first-principles risk assessment framework underlying the IEEE P3396 Recommended Practice for AI Risk, Safety, Trustworthiness, and Responsibility. We distinguish between process risks (risks arising from how AI systems are built or operated) and outcome risks (risks manifest in the AI system's outputs and their real-world effects), arguing that generative AI governance should prioritize outcome risks. Central to our approach is an information-centric ontology that classifies AI-generated outputs into four fundamental categories: (1) Perception-level information, (2) Knowledge-level information, (3) Decision/Action plan information, and (4) Control tokens (access or resource directives). This classification allows systematic identification of harms and more precise attribution of responsibility to stakeholders (developers, deployers, users, regulators) based on the nature of the information produced. We illustrate how each information type entails distinct outcome risks (e.g. deception, misinformation, unsafe recommendations, security breaches) and requires tailored risk metrics and mitigations. By grounding the framework in the essence of information, human agency, and cognition, we align risk evaluation with how AI outputs influence human understanding and action. The result is a principled approach to AI risk that supports clear accountability and targeted safeguards, in contrast to broad application-based risk categorizations. We include example tables mapping information types to risks and responsibilities. This work aims to inform the IEEE P3396 Recommended Practice and broader AI governance with a rigorous, first-principles foundation for assessing generative AI risks while enabling responsible innovation. 

**Abstract (ZH)**: Generative人工智能（AI）正在实现内容创造和决策支持前所未有的自动化，但也带来了新的风险。本文提出了一种基于IEEE P3396推荐实践的AI风险、安全、可靠性和责任的第一性原理风险评估框架。本文将风险分为过程风险（AI系统构建或运行方式引起的风险）和结果风险（AI系统输出及其现实影响中的风险），认为生成型AI治理应优先考虑结果风险。我们方法的核心是一种以信息为中心的概念分类，将AI生成的输出分为四大基本类别：（1）感知级信息，（2）知识级信息，（3）决策/行动方案信息，（4）控制标记（访问或资源指令）。这种分类有助于系统地识别危害，并根据所产生信息的性质更精确地将责任归咎于相关方（开发者、部署者、用户、监管者）。每种信息类型都伴随着不同的结果风险（例如，误导、虚假信息、不安全的建议、安全漏洞），需要特定的风险度量和缓解措施。通过将框架建立在信息的本质、人类代理和认知的基础上，我们将风险评估与AI输出如何影响人类理解与行动相结合。由此形成一种基于原则的AI风险管理方法，支持明确的责任分配和针对性的保障措施，与基于应用的广泛风险分类不同。本文还提供了示例表格，映射信息类型和风险、责任。本研究旨在为IEEE P3396推荐实践和更广泛的AI治理提供一个严谨的第一性原理基础，以评估生成型AI风险，同时促进负责任的创新。 

---
