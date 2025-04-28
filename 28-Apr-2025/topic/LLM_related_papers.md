# LLM Agent Swarm for Hypothesis-Driven Drug Discovery 

**Title (ZH)**: 基于假设的药物发现agents蜂群 swarm for 基于大规模语言模型的假设驱动药物发现 

**Authors**: Kevin Song, Andrew Trotter, Jake Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17967)  

**Abstract**: Drug discovery remains a formidable challenge: more than 90 percent of candidate molecules fail in clinical evaluation, and development costs often exceed one billion dollars per approved therapy. Disparate data streams, from genomics and transcriptomics to chemical libraries and clinical records, hinder coherent mechanistic insight and slow progress. Meanwhile, large language models excel at reasoning and tool integration but lack the modular specialization and iterative memory required for regulated, hypothesis-driven workflows. We introduce PharmaSwarm, a unified multi-agent framework that orchestrates specialized LLM "agents" to propose, validate, and refine hypotheses for novel drug targets and lead compounds. Each agent accesses dedicated functionality--automated genomic and expression analysis; a curated biomedical knowledge graph; pathway enrichment and network simulation; interpretable binding affinity prediction--while a central Evaluator LLM continuously ranks proposals by biological plausibility, novelty, in silico efficacy, and safety. A shared memory layer captures validated insights and fine-tunes underlying submodels over time, yielding a self-improving system. Deployable on low-code platforms or Kubernetes-based microservices, PharmaSwarm supports literature-driven discovery, omics-guided target identification, and market-informed repurposing. We also describe a rigorous four-tier validation pipeline spanning retrospective benchmarking, independent computational assays, experimental testing, and expert user studies to ensure transparency, reproducibility, and real-world impact. By acting as an AI copilot, PharmaSwarm can accelerate translational research and deliver high-confidence hypotheses more efficiently than traditional pipelines. 

**Abstract (ZH)**: 制药领域的药物发现依然是一项艰巨的挑战：超过90%的候选分子在临床评估中失败，且每种获批疗法的研发成本常常超过十亿美元。异质的数据流，包括基因组学、转录组学、化学库以及临床记录，阻碍了连贯的机制洞察并减缓了研究进展。与此同时，大型语言模型在推理和工具集成方面表现出色，但在符合监管要求、基于假设的迭代工作中缺乏模块化的特殊化和持续记忆功能。我们介绍了PharmaSwarm，这是一种统一的多代理框架，它可以协调专门的大型语言模型“代理”来提出、验证和完善针对新药靶点和先导化合物的假设。每个代理都接入特定的功能——自动基因组和表达分析；经过编目的生物医学知识图谱；途径富集和网络模拟；可解释的结合亲和力预测——而中心的评估LSTM持续根据生物可行性、新颖性、虚拟效果和安全性对提案进行排名。共享的内存层捕获验证过的洞察，并随时间微调底层子模型，从而产生一个自我改进的系统。PharmaSwarm 可部署在低代码平台或基于Kubernetes的微服务上，支持文献驱动的发现、组学导向的目标识别以及市场驱动的再利用。我们还描述了一个严格的四级验证管道，涵盖回顾性基准测试、独立计算实验、实验测试和专家用户研究，以确保透明性、可再现性和实际影响。通过充当AI副驾，PharmaSwarm 可以加速转化研究，并比传统管道更高效地提供高置信度假设。 

---
# Kimi-Audio Technical Report 

**Title (ZH)**: Kimi-Audio技术报告 

**Authors**: KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, Jun Chen, Yanru Chen, Yulun Du, Weiran He, Zhenxing Hu, Guokun Lai, Qingcheng Li, Yangyang Liu, Weidong Sun, Jianzhou Wang, Yuzhi Wang, Yuefeng Wu, Yuxin Wu, Dongchao Yang, Hao Yang, Ying Yang, Zhilin Yang, Aoxiong Yin, Ruibin Yuan, Yutong Zhang, Zaida Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18425)  

**Abstract**: We present Kimi-Audio, an open-source audio foundation model that excels in audio understanding, generation, and conversation. We detail the practices in building Kimi-Audio, including model architecture, data curation, training recipe, inference deployment, and evaluation. Specifically, we leverage a 12.5Hz audio tokenizer, design a novel LLM-based architecture with continuous features as input and discrete tokens as output, and develop a chunk-wise streaming detokenizer based on flow matching. We curate a pre-training dataset that consists of more than 13 million hours of audio data covering a wide range of modalities including speech, sound, and music, and build a pipeline to construct high-quality and diverse post-training data. Initialized from a pre-trained LLM, Kimi-Audio is continual pre-trained on both audio and text data with several carefully designed tasks, and then fine-tuned to support a diverse of audio-related tasks. Extensive evaluation shows that Kimi-Audio achieves state-of-the-art performance on a range of audio benchmarks including speech recognition, audio understanding, audio question answering, and speech conversation. We release the codes, model checkpoints, as well as the evaluation toolkits in this https URL. 

**Abstract (ZH)**: 我们介绍了Kimi-Audio，一个在音频理解、生成和对话方面表现出色的开源音频基础模型。我们详细介绍了Kimi-Audio的构建实践，包括模型架构、数据整理、训练方法、推理部署和评估。具体来说，我们利用了一个12.5Hz的音频分词器，设计了一种基于LLM的新架构，以连续特征作为输入，离散令牌作为输出，并开发了一种基于流动匹配的分块流式反分词器。我们整理了一个包含超过1300万小时音频数据的预训练数据集，涵盖包括语音、声音和音乐在内的多种模态，并构建了一条管线来构建高质量和多样化的后训练数据。从预训练的LLM初始化后，Kimi-Audio在音频和文本数据上进行了连续预训练，并通过精心设计的任务进行微调，以支持各种音频相关的任务。广泛评估表明，Kimi-Audio在包括语音识别、音频理解、音频问答和语音对话在内的多种音频基准上取得了最先进的性能。我们在此处发布了代码、模型检查点以及评估工具包。 

---
# LLMpatronous: Harnessing the Power of LLMs For Vulnerability Detection 

**Title (ZH)**: LLMpatronous: 利用大语言模型进行漏洞检测 

**Authors**: Rajesh Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2504.18423)  

**Abstract**: Despite the transformative impact of Artificial Intelligence (AI) across various sectors, cyber security continues to rely on traditional static and dynamic analysis tools, hampered by high false positive rates and superficial code comprehension. While generative AI offers promising automation capabilities for software development, leveraging Large Language Models (LLMs) for vulnerability detection presents unique challenges. This paper explores the potential and limitations of LLMs in identifying vulnerabilities, acknowledging inherent weaknesses such as hallucinations, limited context length, and knowledge cut-offs. Previous attempts employing machine learning models for vulnerability detection have proven ineffective due to limited real-world applicability, feature engineering challenges, lack of contextual understanding, and the complexities of training models to keep pace with the evolving threat landscape. Therefore, we propose a robust AI-driven approach focused on mitigating these limitations and ensuring the quality and reliability of LLM based vulnerability detection. Through innovative methodologies combining Retrieval-Augmented Generation (RAG) and Mixtureof-Agents (MoA), this research seeks to leverage the strengths of LLMs while addressing their weaknesses, ultimately paving the way for dependable and efficient AI-powered solutions in securing the ever-evolving software landscape. 

**Abstract (ZH)**: 尽管人工智能（AI）在各领域产生了革命性的影响，网络安全仍依赖于传统的静态和动态分析工具，受到高误报率和表面化的代码理解限制。虽然生成型AI为软件开发提供了有前景的自动化能力，利用大规模语言模型（LLMs）进行漏洞检测面临着独特挑战。本文探讨了LLMs在识别漏洞方面的潜力和局限性，承认其固有的弱点，如幻觉、有限的上下文长度和知识截止。以往使用机器学习模型进行漏洞检测的尝试由于实际应用有限、特征工程挑战、缺乏上下文理解以及模型训练难以跟上不断演变的威胁态势而效果不佳。因此，本文提出了一种稳健的AI驱动方法，旨在减轻这些局限性，确保基于LLMs的漏洞检测的质量和可靠性。通过结合检索增强生成（RAG）和多智能体（MoA）的研究方法，本文旨在发挥LLMs的优势并解决其弱点，最终为确保不断演变的软件环境提供可靠且高效的AI增强解决方案。 

---
# Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation 

**Title (ZH)**: 跨越领域界限：大型语言模型增强跨域序列推荐 

**Authors**: Qidong Liu, Xiangyu Zhao, Yejing Wang, Zijian Zhang, Howard Zhong, Chong Chen, Xiang Li, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.18383)  

**Abstract**: Cross-domain Sequential Recommendation (CDSR) aims to extract the preference from the user's historical interactions across various domains. Despite some progress in CDSR, two problems set the barrier for further advancements, i.e., overlap dilemma and transition complexity. The former means existing CDSR methods severely rely on users who own interactions on all domains to learn cross-domain item relationships, compromising the practicability. The latter refers to the difficulties in learning the complex transition patterns from the mixed behavior sequences. With powerful representation and reasoning abilities, Large Language Models (LLMs) are promising to address these two problems by bridging the items and capturing the user's preferences from a semantic view. Therefore, we propose an LLMs Enhanced Cross-domain Sequential Recommendation model (LLM4CDSR). To obtain the semantic item relationships, we first propose an LLM-based unified representation module to represent items. Then, a trainable adapter with contrastive regularization is designed to adapt the CDSR task. Besides, a hierarchical LLMs profiling module is designed to summarize user cross-domain preferences. Finally, these two modules are integrated into the proposed tri-thread framework to derive recommendations. We have conducted extensive experiments on three public cross-domain datasets, validating the effectiveness of LLM4CDSR. We have released the code online. 

**Abstract (ZH)**: 跨域序列推荐（CDSR）旨在从用户在不同领域的历史交互中提取偏好。尽管在CDSR方面取得了一些进展，但仍存在两个障碍，即重叠困境和转换复杂性。前者意味着现有的CDSR方法严重依赖于在所有领域都有交互记录的用户来学习跨领域的物品关系，这削弱了其实用性。后者指的是从混合行为序列中学习复杂的转换模式的困难。凭借强大的表示和推理能力，大型语言模型（LLMs）有望通过连接物品并从语义视角捕捉用户偏好来解决这两个问题。因此，我们提出了一种增强型跨域序列推荐模型（LLM4CDSR）。为了获取语义上的物品关系，我们首先提出了一种基于LLM的统一表示模块来表示物品。然后，设计了一个可训练的适配器，带有对比正则化，以适应CDSR任务。此外，设计了一个层次化的LLM用户概况模块，以总结用户的跨域偏好。最后，将这两个模块集成到提出的三线程框架中以生成推荐。我们已在三个公开的跨域数据集上进行了广泛实验，验证了LLM4CDSR的有效性，并已在线发布了代码。 

---
# Pushing the boundary on Natural Language Inference 

**Title (ZH)**: 扩展自然语言推理的边界 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2504.18376)  

**Abstract**: Natural Language Inference (NLI) is a central task in natural language understanding with applications in fact-checking, question answering, and information retrieval. Despite its importance, current NLI systems heavily rely on supervised learning with datasets that often contain annotation artifacts and biases, limiting generalization and real-world applicability. In this work, we apply a reinforcement learning-based approach using Group Relative Policy Optimization (GRPO) for Chain-of-Thought (CoT) learning in NLI, eliminating the need for labeled rationales and enabling this type of training on more challenging datasets such as ANLI. We fine-tune 7B, 14B, and 32B language models using parameter-efficient techniques (LoRA and QLoRA), demonstrating strong performance across standard and adversarial NLI benchmarks. Our 32B AWQ-quantized model surpasses state-of-the-art results on 7 out of 11 adversarial sets$\unicode{x2013}$or on all of them considering our replication$\unicode{x2013}$within a 22GB memory footprint, showing that robust reasoning can be retained under aggressive quantization. This work provides a scalable and practical framework for building robust NLI systems without sacrificing inference quality. 

**Abstract (ZH)**: 基于强化学习的Group Relative Policy Optimization (GRPO)在自然语言推理中的链式思考学习：无需标注理由且适用于ANLI等更具挑战性的数据集 

---
# Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review 

**Title (ZH)**: 大规模语言模型中的不确定性测量与缓解方法比较：一项系统性综述 

**Authors**: Toghrul Abbasli, Kentaroh Toyoda, Yuan Wang, Leon Witt, Muhammad Asif Ali, Yukai Miao, Dan Li, Qingsong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.18346)  

**Abstract**: Large Language Models (LLMs) have been transformative across many domains. However, hallucination -- confidently outputting incorrect information -- remains one of the leading challenges for LLMs. This raises the question of how to accurately assess and quantify the uncertainty of LLMs. Extensive literature on traditional models has explored Uncertainty Quantification (UQ) to measure uncertainty and employed calibration techniques to address the misalignment between uncertainty and accuracy. While some of these methods have been adapted for LLMs, the literature lacks an in-depth analysis of their effectiveness and does not offer a comprehensive benchmark to enable insightful comparison among existing solutions. In this work, we fill this gap via a systematic survey of representative prior works on UQ and calibration for LLMs and introduce a rigorous benchmark. Using two widely used reliability datasets, we empirically evaluate six related methods, which justify the significant findings of our review. Finally, we provide outlooks for key future directions and outline open challenges. To the best of our knowledge, this survey is the first dedicated study to review the calibration methods and relevant metrics for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多领域中起到了变革性的作用。然而，模型妄言——自信地输出错误信息——仍然是LLMs面临的主要挑战之一。这引发了如何准确评估和量化LLMs的不确定性的问题。传统模型领域的大量文献探讨了不确定性量化（UQ）来衡量不确定性，并采用了校准技术来解决不确定性与准确性的不一致问题。虽然其中一些方法已经被调整适用于LLMs，但文献中缺乏这些方法有效性的深入分析，也没有提供综合基准以实现现有解决方案的深入比较。在本文中，我们通过系统回顾先前关于LLMs的UQ和校准工作的代表性文献，并引入了一个严谨的基准。使用两个广泛采用的可靠性数据集，我们实证评估了六种相关方法，从而验证了我们回顾的重要发现。最后，我们提出了未来研究的关键方向，并概述了开放挑战。据我们所知，这是首个专注于回顾LLMs的校准方法及其相关指标的研究。 

---
# Efficient Single-Pass Training for Multi-Turn Reasoning 

**Title (ZH)**: 单过训练高效实现多轮推理 

**Authors**: Ritesh Goru, Shanay Mehta, Prateek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2504.18246)  

**Abstract**: Training Large Language Models ( LLMs) to generate explicit reasoning before they produce an answer has been shown to improve their performance across various tasks such as mathematics and coding. However, fine-tuning LLMs on multi-turn reasoning datasets presents a unique challenge: LLMs must generate reasoning tokens that are excluded from subsequent inputs to the LLM. This discrepancy prevents us from processing an entire conversation in a single forward pass-an optimization readily available when we fine-tune on a multi-turn non-reasoning dataset. This paper proposes a novel approach that overcomes this limitation through response token duplication and a custom attention mask that enforces appropriate visibility constraints. Our approach significantly reduces the training time and allows efficient fine-tuning on multi-turn reasoning datasets. 

**Abstract (ZH)**: 训练大型语言模型在生成答案之前生成明确的推理已被证明能提高其在数学和编码等各项任务中的性能。然而，对多轮推理数据集进行微调为LLMs带来了一个独特挑战：LLMs必须生成在后续输入中被排除的推理标记。这种不一致阻止了我们一次性处理整个对话——这是我们在使用多轮非推理数据集进行微调时可以利用的优化。本文提出了一种新颖的方法，通过响应标记复制和自定义注意力掩码克服这一限制，该掩码施加适当可见性约束。我们的方法显著缩短了训练时间，并允许在多轮推理数据集上进行高效的微调。 

---
# Evaluating Evaluation Metrics -- The Mirage of Hallucination Detection 

**Title (ZH)**: 评估评估指标——幻觉检测的幻象 

**Authors**: Atharva Kulkarni, Yuan Zhang, Joel Ruben Antony Moniz, Xiou Ge, Bo-Hsiang Tseng, Dhivya Piraviperumal, Swabha Swayamdipta, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18114)  

**Abstract**: Hallucinations pose a significant obstacle to the reliability and widespread adoption of language models, yet their accurate measurement remains a persistent challenge. While many task- and domain-specific metrics have been proposed to assess faithfulness and factuality concerns, the robustness and generalization of these metrics are still untested. In this paper, we conduct a large-scale empirical evaluation of 6 diverse sets of hallucination detection metrics across 4 datasets, 37 language models from 5 families, and 5 decoding methods. Our extensive investigation reveals concerning gaps in current hallucination evaluation: metrics often fail to align with human judgments, take an overtly myopic view of the problem, and show inconsistent gains with parameter scaling. Encouragingly, LLM-based evaluation, particularly with GPT-4, yields the best overall results, and mode-seeking decoding methods seem to reduce hallucinations, especially in knowledge-grounded settings. These findings underscore the need for more robust metrics to understand and quantify hallucinations, and better strategies to mitigate them. 

**Abstract (ZH)**: 语言模型中的幻觉是其可靠性和广泛应用的重要障碍，然而对其准确测量依然是一个持续的挑战。尽管已提出了许多特定任务和领域的评估指标来评估忠实性和事实性问题，但这些指标的鲁棒性和泛化能力尚未得到验证。本文通过在4个数据集、5大家族的37个语言模型和5种解码方法上进行大规模实证评估，6种不同的幻觉检测指标集，揭示了当前幻觉评估中存在的关键差距：指标往往无法与人类判断一致，对问题采取过于近视的观点，并且在参数缩放时显示出不一致的改进。令人鼓舞的是，基于LLM的评估，特别是使用GPT-4，获得了最好的整体结果，模式寻找解码方法似乎能够减少幻觉，尤其是在知识导向的环境中。这些发现强调了需要更多鲁棒的指标来理解和量化幻觉，并寻求更好的减少它们的策略。 

---
# Application and Optimization of Large Models Based on Prompt Tuning for Fact-Check-Worthiness Estimation 

**Title (ZH)**: 基于提示调优的大模型应用与优化以评估事实核查值 

**Authors**: Yinglong Yu, Hao Shen, Zhengyi Lyu, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2504.18104)  

**Abstract**: In response to the growing problem of misinformation in the context of globalization and informatization, this paper proposes a classification method for fact-check-worthiness estimation based on prompt tuning. We construct a model for fact-check-worthiness estimation at the methodological level using prompt tuning. By applying designed prompt templates to large language models, we establish in-context learning and leverage prompt tuning technology to improve the accuracy of determining whether claims have fact-check-worthiness, particularly when dealing with limited or unlabeled data. Through extensive experiments on public datasets, we demonstrate that the proposed method surpasses or matches multiple baseline methods in the classification task of fact-check-worthiness estimation assessment, including classical pre-trained models such as BERT, as well as recent popular large models like GPT-3.5 and GPT-4. Experiments show that the prompt tuning-based method proposed in this study exhibits certain advantages in evaluation metrics such as F1 score and accuracy, thereby effectively validating its effectiveness and advancement in the task of fact-check-worthiness estimation. 

**Abstract (ZH)**: 针对全球化和信息化背景下信息误导问题的日益严重，本文提出了一种基于提示调优的事实核验价值估计分类方法。通过使用提示调优技术构建方法论层面的事实核验价值估计模型，利用设计好的提示模板对大规模语言模型进行应用，实现基于context的学习，并借助提示调优技术提高判断声明是否有事实核验价值的准确性，特别是在处理有限或未标记数据时。通过在公共数据集上进行大量实验，表明所提出的方法在事实核验价值估计分类任务中优于或匹配多项基线方法，包括经典的预训练模型BERT，以及最近流行的大型模型如GPT-3.5和GPT-4。实验结果表明，基于提示调优的方法在F1分数和准确性等评估指标上表现出一定的优势，从而有效验证了其在事实核验价值估计任务中的有效性和先进性。 

---
# Random-Set Large Language Models 

**Title (ZH)**: 随机集大型语言模型 

**Authors**: Muhammad Mubashar, Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2504.18085)  

**Abstract**: Large Language Models (LLMs) are known to produce very high-quality tests and responses to our queries. But how much can we trust this generated text? In this paper, we study the problem of uncertainty quantification in LLMs. We propose a novel Random-Set Large Language Model (RSLLM) approach which predicts finite random sets (belief functions) over the token space, rather than probability vectors as in classical LLMs. In order to allow so efficiently, we also present a methodology based on hierarchical clustering to extract and use a budget of "focal" subsets of tokens upon which the belief prediction is defined, rather than using all possible collections of tokens, making the method scalable yet effective. RS-LLMs encode the epistemic uncertainty induced in their generation process by the size and diversity of its training set via the size of the credal sets associated with the predicted belief functions. The proposed approach is evaluated on CoQA and OBQA datasets using Llama2-7b, Mistral-7b and Phi-2 models and is shown to outperform the standard model in both datasets in terms of correctness of answer while also showing potential in estimating the second level uncertainty in its predictions and providing the capability to detect when its hallucinating. 

**Abstract (ZH)**: 大规模语言模型（LLMs）生成的测试和响应质量非常高，但生成的文本我们能信任多少呢？本文研究了LLMs中的不确定量化问题。我们提出了一种新颖的随机集大规模语言模型（RSLLM）方法，该方法预测词汇表上的有限随机集（信念函数），而不是经典的概率向量。为了实现这一点，我们还提出了一种基于层次聚类的方法，以提取并利用一组“专注”的词汇子集，这些子集上定义了信念预测，而不是使用所有可能的词汇组合，从而使方法既可扩展又有效。RS-LLMs通过预测的信念函数关联的信念集的大小，编码其生成过程中由训练集大小和多样性引起的认知不确定性。所提出的方法在CoQA和OBQA数据集上使用Llama2-7b、Mistral-7b和Phi-2模型进行评估，并在答案正确性方面优于标准模型，同时在估计预测的第二层级不确定性方面显示出潜力，并具备检测其虚构的能力。 

---
# Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization 

**Title (ZH)**: 持续预训练与推理偏好优化在医疗LLM中稳定推理 

**Authors**: Wataru Kawakami, Keita Suzuki, Junichiro Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18080)  

**Abstract**: Large Language Models (LLMs) show potential in medicine, yet clinical adoption is hindered by concerns over factual accuracy, language-specific limitations (e.g., Japanese), and critically, their reliability when required to generate reasoning explanations -- a prerequisite for trust. This paper introduces Preferred-MedLLM-Qwen-72B, a 72B-parameter model optimized for the Japanese medical domain to achieve both high accuracy and stable reasoning. We employ a two-stage fine-tuning process on the Qwen2.5-72B base model: first, Continued Pretraining (CPT) on a comprehensive Japanese medical corpus instills deep domain knowledge. Second, Reasoning Preference Optimization (RPO), a preference-based method, enhances the generation of reliable reasoning pathways while preserving high answer accuracy. Evaluations on the Japanese Medical Licensing Exam benchmark (IgakuQA) show Preferred-MedLLM-Qwen-72B achieves state-of-the-art performance (0.868 accuracy), surpassing strong proprietary models like GPT-4o (0.866). Crucially, unlike baseline or CPT-only models which exhibit significant accuracy degradation (up to 11.5\% and 3.8\% respectively on IgakuQA) when prompted for explanations, our model maintains its high accuracy (0.868) under such conditions. This highlights RPO's effectiveness in stabilizing reasoning generation. This work underscores the importance of optimizing for reliable explanations alongside accuracy. We release the Preferred-MedLLM-Qwen-72B model weights to foster research into trustworthy LLMs for specialized, high-stakes applications. 

**Abstract (ZH)**: Large Language Models (LLMs)在医学中的潜在应用及其临床采用受到事实准确性、语言特定限制（例如日语）以及在生成推理解释时的可靠性担忧的阻碍。本文介绍了Preferred-MedLLM-Qwen-72B，这是一种针对日本医学领域的720亿参数模型，旨在实现高准确性和稳定的推理。我们采用两阶段微调过程对Qwen2.5-72B基础模型进行优化：首先，通过全面的日语医学语料库进行持续预训练，灌输深厚的领域知识；其次，采用基于偏好方法的推理偏好优化（RPO），增强了可靠推理路径的生成能力，同时保持高答案准确性。在日本医学执照考试基准测试（IgakuQA）上的评估表明，Preferred-MedLLM-Qwen-72B取得最先进的性能（准确率为0.868），超越了强大的专有模型如GPT-4o（准确率为0.866）。 crucially，与基准模型或仅CPT模型相比，当被要求生成解释时，这些模型在IgakuQA上的准确率分别下降了11.5%和3.8%，而我们的模型在这些情况下仍能保持其高准确率（0.868）。这表明RPO在稳定生成推理方面非常有效。本文强调，在追求准确性的基础上优化可靠解释的重要性。我们发布了Preferred-MedLLM-Qwen-72B模型权重，以促进针对专业和高风险应用的可信大语言模型的研究。 

---
# PropRAG: Guiding Retrieval with Beam Search over Proposition Paths 

**Title (ZH)**: PropRAG: 基于命题路径的束搜索引导检索 

**Authors**: Jingjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18070)  

**Abstract**: Retrieval Augmented Generation (RAG) has become the standard non-parametric approach for equipping Large Language Models (LLMs) with up-to-date knowledge and mitigating catastrophic forgetting common in continual learning. However, standard RAG, relying on independent passage retrieval, fails to capture the interconnected nature of human memory crucial for complex reasoning (associativity) and contextual understanding (sense-making). While structured RAG methods like HippoRAG utilize knowledge graphs (KGs) built from triples, the inherent context loss limits fidelity. We introduce PropRAG, a framework leveraging contextually rich propositions and a novel beam search algorithm over proposition paths to explicitly discover multi-step reasoning chains. Crucially, PropRAG's online retrieval process operates entirely without invoking generative LLMs, relying instead on efficient graph traversal and pre-computed embeddings. This avoids online LLM inference costs and potential inconsistencies during evidence gathering. LLMs are used effectively offline for high-quality proposition extraction and post-retrieval for answer generation. PropRAG achieves state-of-the-art zero-shot Recall@5 results on PopQA (55.3%), 2Wiki (93.7%), HotpotQA (97.0%), and MuSiQue (77.3%), alongside top F1 scores (e.g., 52.4% on MuSiQue). By improving evidence retrieval through richer representation and explicit, LLM-free online path finding, PropRAG advances non-parametric continual learning. 

**Abstract (ZH)**: PropRAG：基于丰富命题的在线检索增强生成 

---
# LLM-Guided Open RAN: Empowering Hierarchical RAN Intelligent Control 

**Title (ZH)**: LLM 引导的开放RAN：赋能分层RAN智能控制 

**Authors**: Lingyan Bao, Sinwoong Yun, Jemin Lee, Tony Q.S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2504.18062)  

**Abstract**: Recent advancements in large language models (LLMs) have led to a significant interest in deploying LLMempowered algorithms for wireless communication networks. Meanwhile, open radio access network (O-RAN) techniques offer unprecedented flexibility, with the non-real-time (non-RT) radio access network (RAN) intelligent controller (RIC) (non-RT RIC) and near-real-time (near-RT) RIC (near-RT RIC) components enabling intelligent resource management across different time scales. In this paper, we propose the LLM empowered hierarchical RIC (LLM-hRIC) framework to improve the collaboration between RICs. This framework integrates LLMs with reinforcement learning (RL) for efficient network resource management. In this framework, LLMs-empowered non-RT RICs provide strategic guidance and high-level policies based on environmental context. Concurrently, RL-empowered near-RT RICs perform low-latency tasks based on strategic guidance and local near-RT observation. We evaluate the LLM-hRIC framework in an integrated access and backhaul (IAB) network setting. Simulation results demonstrate that the proposed framework achieves superior performance. Finally, we discuss the key future challenges in applying LLMs to O-RAN. 

**Abstract (ZH)**: 大语言模型赋能的无线通信网络层次化智能控制器框架 

---
# Validating Network Protocol Parsers with Traceable RFC Document Interpretation 

**Title (ZH)**: 用可追溯的RFC文档解析验证网络协议解析器 

**Authors**: Mingwei Zheng, Danning Xie, Qingkai Shi, Chengpeng Wang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18050)  

**Abstract**: Validating the correctness of network protocol implementations is highly challenging due to the oracle and traceability problems. The former determines when a protocol implementation can be considered buggy, especially when the bugs do not cause any observable symptoms. The latter allows developers to understand how an implementation violates the protocol specification, thereby facilitating bug fixes. Unlike existing works that rarely take both problems into account, this work considers both and provides an effective solution using recent advances in large language models (LLMs). Our key observation is that network protocols are often released with structured specification documents, a.k.a. RFC documents, which can be systematically translated to formal protocol message specifications via LLMs. Such specifications, which may contain errors due to the hallucination of LLMs, are used as a quasi-oracle to validate protocol parsers, while the validation results in return gradually refine the oracle. Since the oracle is derived from the document, any bugs we find in a protocol implementation can be traced back to the document, thus addressing the traceability problem. We have extensively evaluated our approach using nine network protocols and their implementations written in C, Python, and Go. The results show that our approach outperforms the state-of-the-art and has detected 69 bugs, with 36 confirmed. The project also demonstrates the potential for fully automating software validation based on natural language specifications, a process previously considered predominantly manual due to the need to understand specification documents and derive expected outputs for test inputs. 

**Abstract (ZH)**: 验证网络协议实现的正确性由于存在oracle问题和可追溯性问题而极具挑战性。oracle问题决定了何时可以认为协议实现存在bug，尤其是在这些bug不引起任何可观察症状的情况下。可追溯性问题使开发者能够理解实现如何违反协议规格，从而便于修复bug。与现有工作大多未能同时考虑这两个问题不同，本工作同时考虑了这两个问题，并利用大型语言模型（LLMs）的最新进展提供了一个有效的解决方案。我们关键的观察是，网络协议通常伴随着结构化的规格说明书，即RFC文档，这些文档可以通过LLMs系统地转换为形式化的协议消息规格。这些规格可能由于LLMs的幻觉而包含错误，但它们被用作准oracle来验证协议解析器，而验证结果反过来逐步精化了oracle。由于oracle源自文档，我们发现的协议实现中的任何bug都可以追溯到文档，从而解决了可追溯性问题。我们使用九个网络协议及其用C、Python和Go编写的实现进行了广泛评估。结果显示，我们的方法优于最先进的方法，并检测到了69个bug，其中有36个得到了确认。该项目还展示了基于自然语言规格完全自动化软件验证的潜力，而在先前，这一过程因需要理解规格说明书并为测试输入推导预期输出而被认为主要是手工操作。 

---
# RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models 

**Title (ZH)**: RAG 大型语言模型并不更安全：检索增强生成的安全性分析 

**Authors**: Bang An, Shiyue Zhang, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2504.18041)  

**Abstract**: Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs. 

**Abstract (ZH)**: 确保大型语言模型安全的努力包括安全微调、评估和红队测试。然而，尽管Retrieval-Augmented Generation (RAG)框架被广泛使用，AI安全工作主要集中在标准大型语言模型上，这意味着我们对RAG用例如何改变模型的安全特性知之甚少。我们对RAG和非RAG框架进行了详细比较分析，涉及11个大型语言模型。我们发现RAG可以使模型更不安全并改变其安全特性。我们探讨了这种变化的原因，并发现即使使用安全的模型和安全的文档也可能导致生成不安全的内容。此外，我们评估了一些现有的RAG环境中的红队测试方法，发现它们在RAG环境中的有效性低于非RAG环境。我们的研究强调了专门针对RAG大型语言模型的安全研究和红队测试方法的必要性。 

---
# Evaluating Machine Expertise: How Graduate Students Develop Frameworks for Assessing GenAI Content 

**Title (ZH)**: 评估机器专长：研究生开发评估GenAI内容框架的研究 

**Authors**: Celia Chen, Alex Leitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.17964)  

**Abstract**: This paper examines how graduate students develop frameworks for evaluating machine-generated expertise in web-based interactions with large language models (LLMs). Through a qualitative study combining surveys, LLM interaction transcripts, and in-depth interviews with 14 graduate students, we identify patterns in how these emerging professionals assess and engage with AI-generated content. Our findings reveal that students construct evaluation frameworks shaped by three main factors: professional identity, verification capabilities, and system navigation experience. Rather than uniformly accepting or rejecting LLM outputs, students protect domains central to their professional identities while delegating others--with managers preserving conceptual work, designers safeguarding creative processes, and programmers maintaining control over core technical expertise. These evaluation frameworks are further influenced by students' ability to verify different types of content and their experience navigating complex systems. This research contributes to web science by highlighting emerging human-genAI interaction patterns and suggesting how platforms might better support users in developing effective frameworks for evaluating machine-generated expertise signals in AI-mediated web environments. 

**Abstract (ZH)**: 本文探讨了研究生如何构建评估基于网页的与大规模语言模型（LLM）互动中生成的专业知识的框架。通过结合调查、LLM交互转录和对14名研究生的深入访谈的定性研究，我们识别了这些新兴专业人员评估和互动于AI生成内容的模式。研究发现，学生构建的评估框架主要由专业身份、验证能力以及系统导航经验三个因素塑造。学生在保护与专业身份密切相关的领域的同时，将其他领域委托给他人——管理者保留概念性工作，设计师保护创造性过程，程序员保持对核心技术专长的控制。这些评估框架还受到学生验证不同类型内容的能力和导航复杂系统经验的影响。本文通过强调人类-GenAI互动的新兴模式，并建议平台如何更好地支持用户在AI调解的网络环境中开发有效的评估机器生成专业知识信号的框架，为网络科学领域做出了贡献。 

---
# The Role of Open-Source LLMs in Shaping the Future of GeoAI 

**Title (ZH)**: 开源大模型在塑造GeoAI未来中的作用 

**Authors**: Xiao Huang, Zhengzhong Tu, Xinyue Ye, Michael Goodchild  

**Link**: [PDF](https://arxiv.org/pdf/2504.17833)  

**Abstract**: Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's pivotal role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. Ongoing debates on accessibility, regulation, and misuse underscore the critical need for responsible AI development strategies. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, bespoke geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a nuanced discourse on leveraging AI to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在变革地球空间人工智能（GeoAI），为数据处理、空间分析和决策支持提供了新的能力。本文探讨了开源范式在这一变革中的关键作用。尽管 proprietary LLM 提供了便捷性，但它们往往限制了为专门的地球空间任务所必需的定制化、互操作性和透明性。相反，开源替代方案显著推动了地理信息科学（GIScience）的发展，促进了更高的适应性、可重复性和社区驱动的创新。开源框架使研究人员能够量身定制解决方案、集成最新方法（例如强化学习、高级空间索引），并符合 FAIR 原则。然而，对任何 LLM 的日益依赖需要仔细考虑安全漏洞、伦理风险和 AI 生成地球空间输出的稳健治理。关于可访问性、监管和滥用的持续辩论强调了负责任的 AI 发展策略的迫切需要。本文认为，GIScience 最好不是通过单一的模型类型来推进，而是通过培养结合开源基础、定制的地球空间模型和跨学科合作的多样、互操作的生态系统来推进。通过批判性地评估开源 LLM 在更广泛的 GeoAI 地景中的机遇和挑战，本文为如何利用 AI 促进空间研究、政策和决策的进步提供了一种全面的讨论，使其更加公平、可持续且科学严谨。 

---
# VEU-Bench: Towards Comprehensive Understanding of Video Editing 

**Title (ZH)**: VEU-Bench: 朝着全面理解视频编辑的方向 

**Authors**: Bozheng Li, Yongliang Wu, Yi Lu, Jiashuo Yu, Licheng Tang, Jiawang Cao, Wenqing Zhu, Yuyang Sun, Jay Wu, Wenbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17828)  

**Abstract**: Widely shared videos on the internet are often edited. Recently, although Video Large Language Models (Vid-LLMs) have made great progress in general video understanding tasks, their capabilities in video editing understanding (VEU) tasks remain unexplored. To address this gap, in this paper, we introduce VEU-Bench (Video Editing Understanding Benchmark), a comprehensive benchmark that categorizes video editing components across various dimensions, from intra-frame features like shot size to inter-shot attributes such as cut types and transitions. Unlike previous video editing understanding benchmarks that focus mainly on editing element classification, VEU-Bench encompasses 19 fine-grained tasks across three stages: recognition, reasoning, and judging. To enhance the annotation of VEU automatically, we built an annotation pipeline integrated with an ontology-based knowledge base. Through extensive experiments with 11 state-of-the-art Vid-LLMs, our findings reveal that current Vid-LLMs face significant challenges in VEU tasks, with some performing worse than random choice. To alleviate this issue, we develop Oscars, a VEU expert model fine-tuned on the curated VEU-Bench dataset. It outperforms existing open-source Vid-LLMs on VEU-Bench by over 28.3% in accuracy and achieves performance comparable to commercial models like GPT-4o. We also demonstrate that incorporating VEU data significantly enhances the performance of Vid-LLMs on general video understanding benchmarks, with an average improvement of 8.3% across nine reasoning tasks. 

**Abstract (ZH)**: 互联网上广泛共享的视频往往会被编辑。尽管视频大语言模型（Vid-LLMs）在一般视频理解任务上取得了显著进展，但它们在视频编辑理解（VEU）任务上的能力仍待探索。为填补这一空白，本文介绍了VEU-Bench（视频编辑理解基准），这是一个全面的基准，从帧内特征（如镜头大小）到帧间属性（如剪辑类型和过渡）等多个维度对视频编辑组件进行分类。与主要集中在编辑元素分类的先前基准不同，VEU-Bench涵盖了三个阶段的19个细粒度任务：识别、推理和判断。为了增强VEU的自动注释，我们构建了一个与本体知识库集成的注释管道。通过与11个最先进的Vid-LLMs的广泛实验，我们的研究发现当前的Vid-LLMs在VEU任务中面临重大挑战，一些模型的表现甚至不如随机选择。为缓解这一问题，我们开发了Oscars，这是一个在精心筛选的VEU-Bench数据集上微调的VEU专家模型，它在VEU-Bench上的准确率上超过了现有开源Vid-LLMs超过28.3%，并在性能上可与商业模型如GPT-4o相媲美。我们还展示了在通用视频理解基准测试中融入VEU数据可以显著增强Vid-LLMs的表现，在九个推理任务中平均提升了8.3%。 

---
# EduBot -- Can LLMs Solve Personalized Learning and Programming Assignments? 

**Title (ZH)**: EduBot——大型语言模型能解决个性化学习和编程作业问题吗？ 

**Authors**: Yibin Wang, Jiaxi Xie, Lakshminarayanan Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17824)  

**Abstract**: The prevalence of Large Language Models (LLMs) is revolutionizing the process of writing code. General and code LLMs have shown impressive performance in generating standalone functions and code-completion tasks with one-shot queries. However, the ability to solve comprehensive programming tasks with recursive requests and bug fixes remains questionable. In this paper, we propose EduBot, an intelligent automated assistant system that combines conceptual knowledge teaching, end-to-end code development, personalized programming through recursive prompt-driven methods, and debugging with limited human interventions powered by LLMs. We show that EduBot can solve complicated programming tasks consisting of sub-tasks with increasing difficulties ranging from conceptual to coding questions by recursive automatic prompt-driven systems without finetuning on LLMs themselves. To further evaluate EduBot's performance, we design and conduct a benchmark suite consisting of 20 scenarios in algorithms, machine learning, and real-world problems. The result shows that EduBot can complete most scenarios in less than 20 minutes. Based on the benchmark suites, we perform a comparative study to take different LLMs as the backbone and to verify EduBot's compatibility and robustness across LLMs with varying capabilities. We believe that EduBot is an exploratory approach to explore the potential of pre-trained LLMs in multi-step reasoning and code generation for solving personalized assignments with knowledge learning and code generation. 

**Abstract (ZH)**: 大型语言模型的盛行正在革新编写代码的过程。通用和代码语言模型在一次查询中生成独立函数和代码补全任务方面展现了令人印象深刻的性能。然而，它们解决包含递归请求和错误修复的全面编程任务的能力仍有待商榷。本文提出了一种名为EduBot的智能自动化助手系统，该系统结合了概念知识教学、端到端的代码开发、递归提示驱动的个性化编程以及有限的人工干预下的调试。我们展示了EduBot可以通过递归自动提示驱动系统解决由概念到编码问题构成的复杂编程任务，而无需对语言模型进行微调。为了进一步评估EduBot的性能，我们设计并实施了一个包含20个情景的基准测试套件，涉及算法、机器学习和现实世界问题。结果显示，EduBot可以在少于20分钟内完成大多数情景。基于基准测试套件，我们进行了比较研究，使用不同的大型语言模型作为骨干，验证了EduBot在不同能力的大型语言模型之间的兼容性和鲁棒性。我们认为，EduBot探索了预训练大型语言模型在多步推理和代码生成方面解决个性化作业的潜力，结合了知识学习和代码生成。 

---
# Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的云平台网络流量监控与异常检测系统研究 

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.17807)  

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate. 

**Abstract (ZH)**: 基于大语言模型的网络流量监测与异常检测系统 

---
# Evolution of Optimization Algorithms for Global Placement via Large Language Models 

**Title (ZH)**: 大型语言模型在全局布线优化算法进化中的应用 

**Authors**: Xufeng Yao, Jiaxi Jiang, Yuxuan Zhao, Peiyu Liao, Yibo Lin, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17801)  

**Abstract**: Optimization algorithms are widely employed to tackle complex problems, but designing them manually is often labor-intensive and requires significant expertise. Global placement is a fundamental step in electronic design automation (EDA). While analytical approaches represent the state-of-the-art (SOTA) in global placement, their core optimization algorithms remain heavily dependent on heuristics and customized components, such as initialization strategies, preconditioning methods, and line search techniques. This paper presents an automated framework that leverages large language models (LLM) to evolve optimization algorithms for global placement. We first generate diverse candidate algorithms using LLM through carefully crafted prompts. Then we introduce an LLM-based genetic flow to evolve selected candidate algorithms. The discovered optimization algorithms exhibit substantial performance improvements across many benchmarks. Specifically, Our design-case-specific discovered algorithms achieve average HPWL improvements of \textbf{5.05\%}, \text{5.29\%} and \textbf{8.30\%} on MMS, ISPD2005 and ISPD2019 benchmarks, and up to \textbf{17\%} improvements on individual cases. Additionally, the discovered algorithms demonstrate good generalization ability and are complementary to existing parameter-tuning methods. 

**Abstract (ZH)**: 基于大规模语言模型的全局布局优化算法自动化框架 

---
