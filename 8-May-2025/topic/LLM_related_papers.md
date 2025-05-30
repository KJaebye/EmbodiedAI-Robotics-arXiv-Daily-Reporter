# TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution 

**Title (ZH)**: TrajEvo: 通过大语言模型驱动的进化设计轨迹预测启发式方法 

**Authors**: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.04480)  

**Abstract**: Trajectory prediction is a crucial task in modeling human behavior, especially in fields as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy, while recently proposed deep learning approaches suffer from computational cost, lack of explainability, and generalization issues that limit their practical adoption. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We introduce a Cross-Generation Elite Sampling to promote population diversity and a Statistics Feedback Loop allowing the LLM to analyze alternative predictions. Our evaluations show TrajEvo outperforms previous heuristic methods on the ETH-UCY datasets, and remarkably outperforms both heuristics and deep learning methods when generalizing to the unseen SDD dataset. TrajEvo represents a first step toward automated design of fast, explainable, and generalizable trajectory prediction heuristics. We make our source code publicly available to foster future research at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的轨迹进化预测框架 

---
# The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete 

**Title (ZH)**: 故事的力量：叙述 priming 影响大语言模型代理的合作与竞争 

**Authors**: Gerrit Großmann, Larisa Ivanova, Sai Leela Poduru, Mohaddeseh Tabrizian, Islam Mesabah, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03961)  

**Abstract**: According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment. 

**Abstract (ZH)**: 根据尤瓦尔·诺亚·哈拉里观点，大规模人类合作由共享叙事驱动，这些叙事编码了共同的信念和价值观。本研究探讨此类叙事是否能类似地促使LLM代理趋向合作。我们使用有限重复的公共品博弈实验，其中LLM代理选择合作或自私的支出策略。我们用不同程度强调团队合作的故事对代理进行预处理，并测试这如何影响谈判结果。本研究探讨四个问题：(1) 故事如何影响谈判行为？(2) 当代理共享相同的故事还是不同故事时，结果有何不同？(3) 随着代理数量的增长，会发生什么？(4) 代理是否能抵御自私的谈判者？我们发现，基于故事的预处理显著影响了谈判策略和成功率。共同的故事提高了合作水平，使每个代理受益。相反，使用不同故事的预处理逆转了这一效果，那些被引导为自私的故事的代理占上风。我们假设这些结果对多代理系统设计和AI对齐有重要意义。 

---
# EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning 

**Title (ZH)**: EchoInk-R1：通过强化学习探索多模态LLM中的音视频推理 

**Authors**: Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced perception across text, vision, and audio, yet they often struggle with structured cross-modal reasoning, particularly when integrating audio and visual signals. We introduce EchoInk-R1, a reinforcement learning framework that enhances such reasoning in MLLMs. Built upon the Qwen2.5-Omni-7B foundation and optimized with Group Relative Policy Optimization (GRPO), EchoInk-R1 tackles multiple-choice question answering over synchronized audio-image pairs. To enable this, we curate AVQA-R1-6K, a dataset pairing such audio-image inputs with multiple-choice questions derived from OmniInstruct-v1. EchoInk-R1-7B achieves 85.77% accuracy on the validation set, outperforming the base model, which scores 80.53%, using only 562 reinforcement learning steps. Beyond accuracy, EchoInk-R1 demonstrates reflective reasoning by revisiting initial interpretations and refining responses when facing ambiguous multimodal inputs. These results suggest that lightweight reinforcement learning fine-tuning enhances cross-modal reasoning in MLLMs. EchoInk-R1 is the first framework to unify audio, visual, and textual modalities for general open-world reasoning via reinforcement learning. Code and data are publicly released to facilitate further research. 

**Abstract (ZH)**: multimodal大型语言模型（MLLMs）已在文本、视觉和音频感知方面取得了进展，但在音频和视觉信号综合的结构化跨模态推理方面往往表现出色不足。我们介绍了EchoInk-R1，一种增强MLLMs此类推理的强化学习框架。基于Qwen2.5-Omni-7B基础模型并使用Group Relative Policy Optimization (GRPO)进行优化，EchoInk-R1解决了同步音频-图像对的多项选择题回答问题。为了实现这一点，我们创建了AVQA-R1-6K数据集，将此类音频-图像输入与来自OmniInstruct-v1的多项选择题配对。EchoInk-R1-7B在验证集上的准确率达到85.77%，比仅使用562步强化学习步骤的基模型80.53%的准确率更高。除了准确性外，EchoInk-R1还展示了反思性推理，能够在面对模态性输入时重新评估初始解释并完善响应。这些结果表明，轻量级的强化学习微调可以增强MLLMs的跨模态推理能力。EchoInk-R1是首个通过强化学习统一音频、视觉和文本模态进行一般开放世界推理的框架。代码和数据已公开发布，以促进进一步研究。 

---
# Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization 

**Title (ZH)**: 以火制火：通过奖励中和防御恶意RL微调 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04578)  

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models. 

**Abstract (ZH)**: 强化学习（RL）微调虽然能改造大规模语言模型，但也引入了一个我们通过实验验证的漏洞：我们的实验显示，恶意的RL微调以令人惊讶的效率拆解了安全防护措施，仅需50步和少量对抗提示，有害行为从0-2迅速升级到7-9。这种攻击途径特别威胁具有参数级访问权限的开源模型。现有的针对监督微调的防御措施无法抵抗RL的动态反馈机制。我们引入了奖励中和（Reward Neutralization），这是首个专门针对RL微调攻击的防御框架，通过建立简洁的拒绝模式使其上的恶意奖励信号无效。我们的方法训练模型产生最小信息量的拒绝，攻击者无法利用，系统地中和了向有害输出优化的尝试。实验验证显示，我们的方法在200步攻击后保持较低的有害评分（不超过2分），而标准模型迅速恶化。这项工作首次提供了构建性的证明，即针对日益可访问的RL攻击实现鲁棒防御是可行的，填补了开源权重模型的关键安全空白。 

---
# Overcoming Data Scarcity in Generative Language Modelling for Low-Resource Languages: A Systematic Review 

**Title (ZH)**: 克服生成语言模型中低资源语言数据稀缺问题：一项系统性审查 

**Authors**: Josh McGiff, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04531)  

**Abstract**: Generative language modelling has surged in popularity with the emergence of services such as ChatGPT and Google Gemini. While these models have demonstrated transformative potential in productivity and communication, they overwhelmingly cater to high-resource languages like English. This has amplified concerns over linguistic inequality in natural language processing (NLP). This paper presents the first systematic review focused specifically on strategies to address data scarcity in generative language modelling for low-resource languages (LRL). Drawing from 54 studies, we identify, categorise and evaluate technical approaches, including monolingual data augmentation, back-translation, multilingual training, and prompt engineering, across generative tasks. We also analyse trends in architecture choices, language family representation, and evaluation methods. Our findings highlight a strong reliance on transformer-based models, a concentration on a small subset of LRLs, and a lack of consistent evaluation across studies. We conclude with recommendations for extending these methods to a wider range of LRLs and outline open challenges in building equitable generative language systems. Ultimately, this review aims to support researchers and developers in building inclusive AI tools for underrepresented languages, a necessary step toward empowering LRL speakers and the preservation of linguistic diversity in a world increasingly shaped by large-scale language technologies. 

**Abstract (ZH)**: 生成语言模型因ChatGPT和Google Gemini等服务的出现而日益流行。尽管这些模型在提高生产力和沟通方面展现出变革性的潜力，但它们主要服务于如英语之类的高资源语言。这加剧了自然语言处理（NLP）领域语言不平等的问题。本文首次系统性地回顾了针对低资源语言（LRL）生成语言模型数据稀缺问题的策略。我们从54篇研究中识别、分类并评估了包括单语数据增强、回译、多语言训练和提示工程在内的技术方法，涵盖了各种生成任务。我们还分析了架构选择、语言家族表示和评估方法的趋势。研究结果突出显示了对基于转子器模型的强烈依赖、对少数几种LRL的集中关注以及研究间缺乏一致的评估。我们提出了将这些方法扩展到更广泛的LRL范围的建议，并概述了构建公平的生成语言系统所面临的开放性挑战。最终，本文旨在支持研究者和开发人员为未充分代表的语言构建包容性AI工具，这是使LRL使用者受益并保护语言多样性的重要步骤。 

---
# OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models 

**Title (ZH)**: OBLIVIATE：大型语言模型的稳健且实用的机器遗忘技术 

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04416)  

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛训练语料后存在记忆敏感、受版权保护或有毒内容的风险。为解决这一问题，我们提出OBLIVIATE，一种稳健的遗忘框架，能够在删除指定数据的同时保持模型的效果。该框架遵循结构化流程：提取目标token，构建保留集，并使用包含掩码、蒸馏和世界事实三种组件的定制损失函数进行微调。通过低秩适配器（LoRA），确保高效性而不牺牲遗忘质量。我们在哈利·波特系列、WMDP和TOFU等多个数据集上进行了实验，并使用全面的度量标准评估结果：遗忘质量（新文档级记忆得分）、模型效果和流畅性。实验结果表明，该框架在抵抗成员推断攻击、最小化保留数据影响以及在多种场景下保持鲁棒性方面表现出有效性。 

---
# The Aloe Family Recipe for Open and Specialized Healthcare LLMs 

**Title (ZH)**: Aloe 家族配方：开放与专业化医疗LLM 

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2505.04388)  

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.
Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.
Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.
Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare. 

**Abstract (ZH)**: 目的：随着大型语言模型（LLMs）在医疗领域的进步，保护公众利益的 competitive 开源模型变得尤为重要。这项工作通过优化数据预处理和训练的关键阶段，并展示如何通过直接偏好优化（DPO）提高模型安全性、通过 Retrieval-Augmented Generation（RAG）提高模型有效性，为开放医疗 LLM 领域做出了贡献。所采用的评价方法包括四种不同类型的测试，为领域内设定了新的标准。这些模型展示出与最佳私有替代产品竞争力，并在宽松的许可下发布。

方法：基于强大的基础模型如 Llama 3.1 和 Qwen 2.5，Aloe Beta 使用一个自定义的数据集，通过合成链式思考示例增强公共数据。模型经过直接偏好优化（DPO）对齐，强调在面对破解攻击时的伦理和政策对齐性能。评价包括封闭式、开放式、安全性和人类评估，以最大化结果的可靠性。

结果：在整个管道中提出了推荐方案，依托 Aloe 家族的强大性能。这些模型在医疗健康基准测试和医疗领域表现出色，并经常得到医疗专业人员的青睐。在偏见和毒性方面，Aloe Beta 模型显著提高了安全性，并显示出了对未见破解攻击的抗性。为了负责任地发布，Aloe 家族模型附带着详细的风险评估，具体针对医疗健康领域。

结论：Aloe Beta 模型及其制作方法是开源医疗 LLM 领域的重要贡献，提供了顶级性能并保持高标准的伦理要求。这项工作为在医疗健康领域开发和报告对齐的 LLM 设定了新标准。 

---
# Weaponizing Language Models for Cybersecurity Offensive Operations: Automating Vulnerability Assessment Report Validation; A Review Paper 

**Title (ZH)**: 利用语言模型进行网络安全进攻性操作：自动化漏洞评估报告验证——一篇综述论文 

**Authors**: Abdulrahman S Almuhaidib, Azlan Mohd Zain, Zalmiyah Zakaria, Izyan Izzati Kamsani, Abdulaziz S Almuhaidib  

**Link**: [PDF](https://arxiv.org/pdf/2505.04265)  

**Abstract**: This, with the ever-increasing sophistication of cyberwar, calls for novel solutions. In this regard, Large Language Models (LLMs) have emerged as a highly promising tool for defensive and offensive cybersecurity-related strategies. While existing literature has focused much on the defensive use of LLMs, when it comes to their offensive utilization, very little has been reported-namely, concerning Vulnerability Assessment (VA) report validation. Consequentially, this paper tries to fill that gap by investigating the capabilities of LLMs in automating and improving the validation process of the report of the VA. From the critical review of the related literature, this paper hereby proposes a new approach to using the LLMs in the automation of the analysis and within the validation process of the report of the VA that could potentially reduce the number of false positives and generally enhance efficiency. These results are promising for LLM automatization for improving validation on reports coming from VA in order to improve accuracy while reducing human effort and security postures. The contribution of this paper provides further evidence about the offensive and defensive LLM capabilities and therefor helps in devising more appropriate cybersecurity strategies and tools accordingly. 

**Abstract (ZH)**: 随着网络战的日益复杂，需要提出新的解决方案。在这方面，大规模语言模型（LLMs）已成为防御性和进攻性网络安全策略中极具前景的工具。尽管现有文献主要关注LLMs的防御性使用，但在其进攻性利用方面，特别是漏洞评估（VA）报告验证方面，报道相对较少。因此，本文试图通过研究LLMs在自动化和改进VA报告验证过程中的能力来填补这一空白。通过对相关文献的批判性回顾，本文提出了一种新的方法，用于自动化VA报告的分析和验证过程，该方法有望减少假阳性的数量，并提高整体效率。这些结果对使用LLM自动化改进来自VA的报告验证以提高准确性、减少人力投入和安全态势具有积极意义。本文的贡献为进一步证明了LLMs的进攻性和防御性能力，并有助于制定更合适的网络安全策略和工具。 

---
# Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering 

**Title (ZH)**: 可引导聊天机器人：基于偏好激活引导的个性化大语言模型 

**Authors**: Jessica Y. Bo, Tianyu Xu, Ishan Chatterjee, Katrina Passarella-Ward, Achin Kulshrestha, D Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04260)  

**Abstract**: As large language models (LLMs) improve in their capacity to serve as personal AI assistants, their ability to output uniquely tailored, personalized responses that align with the soft preferences of their users is essential for enhancing user satisfaction and retention. However, untrained lay users have poor prompt specification abilities and often struggle with conveying their latent preferences to AI assistants. To address this, we leverage activation steering to guide LLMs to align with interpretable preference dimensions during inference. In contrast to memory-based personalization methods that require longer user history, steering is extremely lightweight and can be easily controlled by the user via an linear strength factor. We embed steering into three different interactive chatbot interfaces and conduct a within-subjects user study (n=14) to investigate how end users prefer to personalize their conversations. The results demonstrate the effectiveness of preference-based steering for aligning real-world conversations with hidden user preferences, and highlight further insights on how diverse values around control, usability, and transparency lead users to prefer different interfaces. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的能力不断增强，使其能够作为个人AI助手并输出与用户柔和偏好相匹配的独特个性化响应，这对于提高用户满意度和留存至关重要。然而，未经训练的普通用户在提示指定能力较弱，常常难以向AI助手传达其隐性偏好。为解决这一问题，我们利用激活引导在推理过程中引导LLMs与可解释的偏好维度对齐。与需要更长用户历史的基于记忆的个性化方法相比，激活引导极为轻量级，并且可以通过一个线性强度因子轻松由用户控制。我们将激活引导嵌入到三个不同的交互式聊天机器人界面中，并进行一项针对单个用户的用户研究（n=14），以调查最终用户如何偏好个性化其对话的方式。研究结果展示了基于偏好的激活引导在将实际对话与隐藏用户偏好对齐方面的有效性，并进一步揭示了不同控制、易用性和透明度价值观如何引导用户偏好不同的界面。 

---
# Facilitating Trustworthy Human-Agent Collaboration in LLM-based Multi-Agent System oriented Software Engineering 

**Title (ZH)**: 面向软件工程的基于大语言模型的多智能体系统中可信赖的人机协作促进 

**Authors**: Krishna Ronanki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04251)  

**Abstract**: Multi-agent autonomous systems (MAS) are better at addressing challenges that spans across multiple domains than singular autonomous agents. This holds true within the field of software engineering (SE) as well. The state-of-the-art research on MAS within SE focuses on integrating LLMs at the core of autonomous agents to create LLM-based multi-agent autonomous (LMA) systems. However, the introduction of LMA systems into SE brings a plethora of challenges. One of the major challenges is the strategic allocation of tasks between humans and the LMA system in a trustworthy manner. To address this challenge, a RACI-based framework is proposed in this work in progress article, along with implementation guidelines and an example implementation of the framework. The proposed framework can facilitate efficient collaboration, ensure accountability, and mitigate potential risks associated with LLM-driven automation while aligning with the Trustworthy AI guidelines. The future steps for this work delineating the planned empirical validation method are also presented. 

**Abstract (ZH)**: 多代理自主系统在软件工程领域中比单一自主代理更擅长应对跨多个领域的挑战。本文进展文章提出了一种基于RACI的框架，并结合实施指南和框架的示例实现，以有效地分配任务，确保责任明确，并减少LLM驱动自动化可能带来的风险，同时符合可信人工智能指南。文中还提出了未来的工作计划和拟议的实证验证方法。 

---
# To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay 

**Title (ZH)**: 是否判断：使用大语言模型判断广告关键词的相关性于eBay 

**Authors**: Soumik Dey, Hansi Wu, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04209)  

**Abstract**: E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics. 

**Abstract (ZH)**: 电子商务卖家根据其库存推荐的关键短语用于广告以增加买家互动（点击/销售），广告商关键短语的相关性在防止搜索引擎被大量不相关项目淹没以及维护健康的卖家形象方面起着重要作用。本文描述了在使用点击/销售/搜索相关性信号训练广告商关键短语相关性过滤模型方面的不足，并强调了与人工判断对齐的重要性，因为卖家有权接受或拒绝这些关键短语建议。在本研究中，我们将广告商关键短语相关性视为卖家判断、广告提供可竞标的关键短语和搜索引擎进行相同关键短语拍卖之间的复杂动态交互。本文通过eBay广告案例研究探讨了使用人工判断的实践，并证明了将大规模使用的LLM作为卖家判断的可扩展代理来训练相关性模型，在确保其基于企业指标的细致评价框架的前提下，可以在三个系统之间实现更好的和谐。 

---
# On-Device LLM for Context-Aware Wi-Fi Roaming 

**Title (ZH)**: 基于设备的上下文感知Wi-Fi漫游大语言模型 

**Authors**: Ju-Hyung Lee, Yanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04174)  

**Abstract**: Wireless roaming is a critical yet challenging task for maintaining seamless connectivity in dynamic mobile environments. Conventional threshold-based or heuristic schemes often fail, leading to either sticky or excessive handovers. We introduce the first cross-layer use of an on-device large language model (LLM): high-level reasoning in the application layer that issues real-time actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i) context-aware AP selection, where structured prompts fuse environmental cues (e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold adjustment, where the model adaptively decides when to roam. To satisfy the tight latency and resource budgets of edge hardware, we apply a suite of optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and quantization. Experiments on indoor and outdoor datasets show that our approach surpasses legacy heuristics and DRL baselines, achieving a strong balance between roaming stability and signal quality. These findings underscore the promise of application-layer LLM reasoning for lower-layer wireless control in future edge systems. 

**Abstract (ZH)**: 设备上的大语言模型在跨层中的无线漫游应用：基于高阶推理的实时动作优化 

---
# Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety 

**Title (ZH)**: 揭开画布：图像生成解锁与LLM内容安全的动态基准 

**Authors**: Variath Madhupal Gautham Nair, Vishal Varma Dantuluri  

**Link**: [PDF](https://arxiv.org/pdf/2505.04146)  

**Abstract**: Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures.
We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors.
Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure. 

**Abstract (ZH)**: 揭示画布：评估大语言模型在图像生成中的漏洞基准（UTC Benchmark；UTCB） 

---
# Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model 

**Title (ZH)**: 通过构建大规模预训练语言模型法律问题库将法律知识普及于公众 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04132)  

**Abstract**: Access to legal information is fundamental to access to justice. Yet accessibility refers not only to making legal documents available to the public, but also rendering legal information comprehensible to them. A vexing problem in bringing legal information to the public is how to turn formal legal documents such as legislation and judgments, which are often highly technical, to easily navigable and comprehensible knowledge to those without legal education. In this study, we formulate a three-step approach for bringing legal knowledge to laypersons, tackling the issues of navigability and comprehensibility. First, we translate selected sections of the law into snippets (called CLIC-pages), each being a small piece of article that focuses on explaining certain technical legal concept in layperson's terms. Second, we construct a Legal Question Bank (LQB), which is a collection of legal questions whose answers can be found in the CLIC-pages. Third, we design an interactive CLIC Recommender (CRec). Given a user's verbal description of a legal situation that requires a legal solution, CRec interprets the user's input and shortlists questions from the question bank that are most likely relevant to the given legal situation and recommends their corresponding CLIC pages where relevant legal knowledge can be found. In this paper we focus on the technical aspects of creating an LQB. We show how large-scale pre-trained language models, such as GPT-3, can be used to generate legal questions. We compare machine-generated questions (MGQs) against human-composed questions (HCQs) and find that MGQs are more scalable, cost-effective, and more diversified, while HCQs are more precise. We also show a prototype of CRec and illustrate through an example how our 3-step approach effectively brings relevant legal knowledge to the public. 

**Abstract (ZH)**: 获取法律信息是获得正义的基础。然而，可访问性不仅指将法律文件提供给公众，还指使法律信息对公众易于理解和运用。将正式的法律文件，如立法和判决，转化为非法律背景人员易于导航和理解的知识是一个棘手的问题。在本研究中，我们提出了一种三步方法来将法律知识带给普通民众，解决可导航性和可理解性的问题。首先，我们将法律中的选定部分翻译成短片段（称为CLIC页面），每个片段专注于用非法律术语解释某一特定的技术法律概念。其次，我们构建了一个法律问题银行（LQB），这是一个集合了一系列法律问题的库，其答案可以在CLIC页面中找到。第三，我们设计了一个交互式的CLIC推荐系统（CRec）。根据用户对需要法律解决方案的法律情景的口头描述，CRec解释用户的输入，并从问题库中筛选出最有可能与给定法律情景相关的提问，并推荐相应的CLIC页面，以找到相关的法律知识。本文我们专注于法律问题银行的技术方面。我们展示如何使用大规模预训练语言模型（如GPT-3）生成法律问题。我们将机器生成的问题（MGQs）与人类编写的提问（HCQs）进行了比较，发现MGQs更具可扩展性、成本效益和多样性，而HCQs则更为精确。我们也展示了CRec的原型，并通过一个示例说明了我们三步方法如何有效将相关法律知识带给公众。 

---
# LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling 

**Title (ZH)**: LLMs在网络安全领域的适用性：基于STRIDE威胁建模的案例研究 

**Authors**: AbdulAziz AbdulGhaffar, Ashraf Matrawy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04101)  

**Abstract**: Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are almost nonexistent studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases. 

**Abstract (ZH)**: 人工智能（AI）有望成为下一代AI原生6G网络不可或缺的一部分。随着AI的普及，研究人员已经识别出许多AI在网络安全性方面的应用案例。然而，几乎不存在分析大型语言模型（LLMs）在网络安全性方面适用性的研究。为填补这一空白，我们检查了LLMs在网络安全性方面的适用性，特别是通过STRIDE威胁建模案例研究。我们利用四种提示技术与五种LLM对5G威胁进行STRIDE分类。从我们的评估结果中，我们指出了关键发现和详细见解，并解释了可能影响LLMs在某些威胁建模中行为的潜在因素。数值结果和见解支持调整和微调LLMs以适应网络安全性应用场景的必要性。 

---
# An Empirical Study of OpenAI API Discussions on Stack Overflow 

**Title (ZH)**: OpenAI API讨论在Stack Overflow上的实证研究 

**Authors**: Xiang Chen, Jibin Wang, Chaoyang Gao, Xiaolin Ju, Zhanqi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04084)  

**Abstract**: The rapid advancement of large language models (LLMs), represented by OpenAI's GPT series, has significantly impacted various domains such as natural language processing, software development, education, healthcare, finance, and scientific research. However, OpenAI APIs introduce unique challenges that differ from traditional APIs, such as the complexities of prompt engineering, token-based cost management, non-deterministic outputs, and operation as black boxes. To the best of our knowledge, the challenges developers encounter when using OpenAI APIs have not been explored in previous empirical studies. To fill this gap, we conduct the first comprehensive empirical study by analyzing 2,874 OpenAI API-related discussions from the popular Q&A forum Stack Overflow. We first examine the popularity and difficulty of these posts. After manually categorizing them into nine OpenAI API-related categories, we identify specific challenges associated with each category through topic modeling analysis. Based on our empirical findings, we finally propose actionable implications for developers, LLM vendors, and researchers. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进展，以OpenAI的GPT系列为代表，对自然语言处理、软件开发、教育、医疗、金融和科学研究等领域产生了显著影响。然而，OpenAI APIs引入了与传统APIs不同的独特挑战，如提示工程的复杂性、基于令牌的成本管理、非确定性输出以及以黑盒方式运行。据我们所知，之前的经验研究尚未探讨开发者在使用OpenAI APIs时遇到的挑战。为了填补这一空白，我们首次通过分析来自受欢迎的问答论坛Stack Overflow的2,874条OpenAI API相关讨论，进行全面的经验研究。我们首先分析这些帖子的流行程度和难度，然后通过主题建模分析将它们手动归类为九个OpenAI API相关类别，并识别每个类别特有的挑战。基于我们的实证发现，我们最终为开发者、LLM供应商和研究人员提出了可操作的建议。 

---
# LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress? 

**Title (ZH)**: LLM-e 猜想：在硬件进步之外，LLM 能力能否得以提升？ 

**Authors**: Teddy Foley, Spencer Guo, Henry Josephson, Anqi Qu, Jack Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04075)  

**Abstract**: This paper examines whether large language model (LLM) capabilities can continue to advance without additional compute by analyzing the development and role of algorithms used in state-of-the-art LLMs. Motivated by regulatory efforts that have largely focused on restricting access to high-performance hardware, we ask: Can LLMs progress in a compute-constrained environment, and how do algorithmic innovations perform under such conditions?
To address these questions, we introduce a novel classification framework that distinguishes between compute-dependent innovations -- which yield disproportionate benefits at high compute levels (e.g., the Transformer architecture and mixture-of-experts models) and compute-independent innovations, which improve efficiency across all compute scales (e.g., rotary positional encoding, FlashAttention, or layer normalization). We quantify these contributions using a metric called compute-equivalent gain (CEG), which estimates the additional compute that would be required to achieve similar improvements without these algorithmic advancements.
To validate this framework, we conduct small-scale training experiments with a scaled-down GPT-2 model. Our results confirm that compute-independent advancements yield meaningful performance gains even in resource-constrained settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast, compute-dependent advancements provided little benefit or even degraded performance at the small scale, reinforcing the importance of compute availability for certain algorithmic gains. 

**Abstract (ZH)**: 本文通过分析最先进的大型语言模型中使用的算法发展和作用，探讨大型语言模型的能力是否能够在无需额外计算资源的情况下继续进步。鉴于监管努力主要集中在限制高性能硬件的访问上，我们提出的问题是：在计算资源受限的环境中，大型语言模型能否进步，以及在这些条件下算法创新的表现如何？

为了回答这些问题，我们引入了一种新的分类框架，区分计算依赖型创新（在高计算水平下可获得不成比例的好处，例如Transformer架构和expert混合模型）和计算独立型创新（在所有计算规模下提高效率，例如旋转位置编码、FlashAttention或层规范化）。我们使用计算等效增益（CEG）这一度量来量化这些贡献，CEG估计了在没有这些算法进步的情况下实现相似改进所需的额外计算资源。

为了验证这一框架，我们在一个缩放后的GPT-2模型上进行了小规模训练实验。结果表明，即使在资源受限的环境中，计算独立型进步也能带来显著的性能提升，CEG最高达到3.5倍的基础模型水平。相比之下，在小规模计算环境下，计算依赖型进步几乎没有益处甚至降低了性能，这进一步强调了特定算法进步的计算资源重要性。 

---
# Advancing and Benchmarking Personalized Tool Invocation for LLMs 

**Title (ZH)**: 个性化工具调用 advancement 和基准测试 for LLMs 

**Authors**: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04072)  

**Abstract**: Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at this https URL. 

**Abstract (ZH)**: 个性化工具调用是大型语言模型（LLMs）能力扩展的关键机制，近期引发了广泛关注。它使LLMs能够通过工具调用解决复杂问题，同时访问最新的世界知识。然而，现有研究主要集中在LLMs的基本工具调用能力上，忽略了工具调用中的个性化约束。在本文中，我们引入了个性化工具调用的概念，并定义了两个关键任务：工具偏好和基于个人资料的查询。工具偏好解决在选择功能相似的工具时用户偏好问题，而基于个人资料的查询考虑了用户查询缺少某些工具参数的情况，要求模型从用户个人资料中推断这些参数。为应对这些挑战，我们提出了PTool，一种为个性化工具调用设计的数据合成框架。此外，我们构建了PTBench，这是第一个评估个性化工具调用的基准。随后，我们对各种开源模型进行了微调，展示了该框架的有效性并提供了宝贵见解。我们的基准可以在此网址访问：this https URL。 

---
# Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving 

**Title (ZH)**: 棱镜： unleashing GPU sharing for cost-efficient multi-LLM serving 

**Authors**: Shan Yu, Jiarong Xing, Yifan Qiao, Mingyuan Ma, Yangmin Li, Yang Wang, Shuo Yang, Zhiqiang Xie, Shiyi Cao, Ke Bao, Ion Stoica, Harry Xu, Ying Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04021)  

**Abstract**: Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads.
This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems. 

**Abstract (ZH)**: Prism：一种实现多大型语言模型高效服务和SLO达成的GPU共享系统 

---
# SLOT: Structuring the Output of Large Language Models 

**Title (ZH)**: SLOT：结构化大型语言模型的输出 

**Authors**: Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.04016)  

**Abstract**: Structured outputs are essential for large language models (LLMs) in critical applications like agents and information extraction. Despite their capabilities, LLMs often generate outputs that deviate from predefined schemas, significantly hampering reliable application development. We present SLOT (Structured LLM Output Transformer), a model-agnostic approach that transforms unstructured LLM outputs into precise structured formats. While existing solutions predominantly rely on constrained decoding techniques or are tightly coupled with specific models, SLOT employs a fine-tuned lightweight language model as a post-processing layer, achieving flexibility across various LLMs and schema specifications. We introduce a systematic pipeline for data curation and synthesis alongside a formal evaluation methodology that quantifies both schema accuracy and content fidelity. Our results demonstrate that fine-tuned Mistral-7B model with constrained decoding achieves near perfect schema accuracy (99.5%) and content similarity (94.0%), outperforming Claude-3.5-Sonnet by substantial margins (+25 and +20 percentage points, respectively). Notably, even compact models like Llama-3.2-1B can match or exceed the structured output capabilities of much larger proprietary models when equipped with SLOT, enabling reliable structured generation in resource-constrained environments. 

**Abstract (ZH)**: 结构化输出对于大型语言模型在智能代理和信息提取等关键应用中的应用至关重要。尽管具备强大的能力，大型语言模型（LLMs）经常生成不符合预定义模式的输出，严重影响了可靠应用的开发。我们提出了SLOT（结构化LLM输出转换器），这是一种模型无关的方法，将不结构化的LLM输出转换为精确的结构化格式。现有解决方案主要依赖受限解码技术或与特定模型紧密结合，而SLOT采用微调的轻量级语言模型作为后处理层，实现了对各种LLMs和模式规范的高度灵活性。我们介绍了一种系统化的数据采集和合成管道，以及一种正式的评估方法，该方法量化了模式准确性和内容保真度。实验结果表明，使用受限解码技术微调的Mistral-7B模型达到了近完美的模式准确率（99.5%）和内容相似度（94.0%），显著优于Claude-3.5-Sonnet（分别高出25和20个百分点）。值得注意的是，即使像Llama-3.2-1B这样的紧凑型模型，配备了SLOT后，也能达到甚至超过许多更大且专门为特定任务设计的模型的结构化输出能力，从而在资源受限的环境中实现可靠的结构化生成。 

---
# Can Large Language Models Predict Parallel Code Performance? 

**Title (ZH)**: 大型语言模型能否预测并行代码性能？ 

**Authors**: Gregory Bolet, Giorgis Georgakoudis, Harshitha Menon, Konstantinos Parasyris, Niranjan Hasabnis, Hayden Estes, Kirk W. Cameron, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03988)  

**Abstract**: Accurate determination of the performance of parallel GPU code typically requires execution-time profiling on target hardware -- an increasingly prohibitive step due to limited access to high-end GPUs. This paper explores whether Large Language Models (LLMs) can offer an alternative approach for GPU performance prediction without relying on hardware. We frame the problem as a roofline classification task: given the source code of a GPU kernel and the hardware specifications of a target GPU, can an LLM predict whether the GPU kernel is compute-bound or bandwidth-bound?
For this study, we build a balanced dataset of 340 GPU kernels, obtained from HeCBench benchmark and written in CUDA and OpenMP, along with their ground-truth labels obtained via empirical GPU profiling. We evaluate LLMs across four scenarios: (1) with access to profiling data of the kernel source, (2) zero-shot with source code only, (3) few-shot with code and label pairs, and (4) fine-tuned on a small custom dataset.
Our results show that state-of-the-art LLMs have a strong understanding of the Roofline model, achieving 100% classification accuracy when provided with explicit profiling data. We also find that reasoning-capable LLMs significantly outperform standard LLMs in zero- and few-shot settings, achieving up to 64% accuracy on GPU source codes, without profiling information. Lastly, we find that LLM fine-tuning will require much more data than what we currently have available.
This work is among the first to use LLMs for source-level roofline performance prediction via classification, and illustrates their potential to guide optimization efforts when runtime profiling is infeasible. Our findings suggest that with better datasets and prompt strategies, LLMs could become practical tools for HPC performance analysis and performance portability. 

**Abstract (ZH)**: 利用大规模语言模型进行GPU性能预测的研究 

---
# VideoLLM Benchmarks and Evaluation: A Survey 

**Title (ZH)**: VideoLLM基准与评估：一个综述 

**Authors**: Yogesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03829)  

**Abstract**: The rapid development of Large Language Models (LLMs) has catalyzed significant advancements in video understanding technologies. This survey provides a comprehensive analysis of benchmarks and evaluation methodologies specifically designed or used for Video Large Language Models (VideoLLMs). We examine the current landscape of video understanding benchmarks, discussing their characteristics, evaluation protocols, and limitations. The paper analyzes various evaluation methodologies, including closed-set, open-set, and specialized evaluations for temporal and spatiotemporal understanding tasks. We highlight the performance trends of state-of-the-art VideoLLMs across these benchmarks and identify key challenges in current evaluation frameworks. Additionally, we propose future research directions to enhance benchmark design, evaluation metrics, and protocols, including the need for more diverse, multimodal, and interpretability-focused benchmarks. This survey aims to equip researchers with a structured understanding of how to effectively evaluate VideoLLMs and identify promising avenues for advancing the field of video understanding with large language models. 

**Abstract (ZH)**: 大型语言模型的迅速发展促进了视频理解技术的重大进步。本文综述了专为视频大型语言模型（VideoLLMs）设计或使用的基准测试和评估方法。我们全面分析了当前视频理解基准测试的现状，讨论了它们的特性、评估协议及其局限性。文章分析了包括封闭集、开放集以及针对时序和空时理解任务的特殊评估方法在内的各种评估方法。我们展示了最先进的VideoLLMs在这些基准测试中的性能趋势，并指出了现有评估框架中的关键挑战。此外，我们提出了增强基准测试设计、评估指标和协议的未来研究方向，包括需要更多样化、多模态和可解释性的基准测试。本文旨在帮助研究人员系统地了解如何有效评估VideoLLMs，并识别出推进视频理解领域发展的有希望的研究方向。 

---
# Memory Assisted LLM for Personalized Recommendation System 

**Title (ZH)**: 基于内存辅助的大语言模型个性化推荐系统 

**Authors**: Jiarui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03824)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we evaluate MAP using a sequential rating prediction task under two scenarios: single domain, where memory and tasks are from the same category (e.g., movies), and cross-domain (e.g., memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决推荐任务方面的潜力显著。通过在用户偏好理解方面的 proven 能力，LLM 个性化已成为提供个性化响应的关键领域。当前研究通过提示设计和微调探索个性化，为个性化 LLM 的进一步研究铺平了道路。然而，现有方法要么在捕捉多样化用户偏好方面代价高昂且效率低下，要么无法及时更新用户历史。为解决这些不足，我们提出了一种基于记忆的个性化 LLM（MAP）。通过用户交互，我们首先为每位用户创建历史档案，记录他们对历史项目的偏好，如历史项目评分。在推荐过程中，我们基于相似性提取相关记忆，并将其融入提示中以增强个性化推荐。在我们的实验中，我们使用序列评分预测任务在两种场景下评估 MAP：单一领域场景，其中记忆和任务属于同一类别（例如，电影），以及跨领域场景（例如，电影记忆和书籍推荐任务）。结果表明，MAP 在直接通过提示设计整合用户历史的常规 LLM 推荐器中表现更优。此外，随着用户历史记录的增长，MAP 在两种场景中的优势更加显著，使其更适用于解决连续的个性化用户请求。 

---
# Program Semantic Inequivalence Game with Large Language Models 

**Title (ZH)**: 大型语言模型下的程序语义不等价性博弈 

**Authors**: Antonio Valerio Miceli-Barone, Vaishak Belle, Ali Payani  

**Link**: [PDF](https://arxiv.org/pdf/2505.03818)  

**Abstract**: Large Language Models (LLMs) can achieve strong performance on everyday coding tasks, but they can fail on complex tasks that require non-trivial reasoning about program semantics. Finding training examples to teach LLMs to solve these tasks can be challenging.
In this work, we explore a method to synthetically generate code reasoning training data based on a semantic inequivalence game SInQ: a generator agent creates program variants that are semantically distinct, derived from a dataset of real-world programming tasks, while an evaluator agent has to identify input examples that cause the original programs and the generated variants to diverge in their behaviour, with the agents training each other semi-adversarially. We prove that this setup enables theoretically unlimited improvement through self-play in the limit of infinite computational resources.
We evaluated our approach on multiple code generation and understanding benchmarks, including cross-language vulnerability detection (Lu et al., 2021), where our method improves vulnerability detection in C/C++ code despite being trained exclusively on Python code, and the challenging Python builtin identifier swap benchmark (Miceli-Barone et al., 2023), showing that whereas modern LLMs still struggle with this benchmark, our approach yields substantial improvements.
We release the code needed to replicate the experiments, as well as the generated synthetic data, which can be used to fine-tune LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以在日常编码任务中取得强大表现，但在执行需要对程序语义进行非平凡推理的复杂任务时可能会失败。找到用于训练LLMs解决这些任务的训练示例可能是具有挑战性的。

在本工作中，我们探索了一种基于语义不等价游戏SInQ的合成生成代码推理训练数据的方法：生成器代理创建与原始现实编程任务语义上不同的程序变体，而评估器代理需要识别导致原始程序和生成变体在行为上出现分歧的输入示例，代理之间以半对抗的方式进行训练。我们证明了这种设置可以通过无限计算资源下的自我对弈实现理论上无限的改进。

我们通过多个代码生成和理解基准测试评估了我们的方法，包括跨语言漏洞检测（Lu et al., 2021），在仅使用Python代码进行训练的情况下，该方法在C/C++代码中提高了漏洞检测能力；以及具有挑战性的Python内置标识符互换基准测试（Miceli-Barone et al., 2023），结果显示尽管现代LLMs在这一基准测试中仍然存在困难，但我们的方法取得了显著的改进。

我们发布了用于复现实验所需的代码，以及生成的合成数据，这些数据可以用于微调LLMs。 

---
# Cer-Eval: Certifiable and Cost-Efficient Evaluation Framework for LLMs 

**Title (ZH)**: Cer-Eval: 可认证和成本效益评估框架 for LLMs 

**Authors**: Ganghua Wang, Zhaorun Chen, Bo Li, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03814)  

**Abstract**: As foundation models continue to scale, the size of trained models grows exponentially, presenting significant challenges for their evaluation. Current evaluation practices involve curating increasingly large datasets to assess the performance of large language models (LLMs). However, there is a lack of systematic analysis and guidance on determining the sufficiency of test data or selecting informative samples for evaluation. This paper introduces a certifiable and cost-efficient evaluation framework for LLMs. Our framework adapts to different evaluation objectives and outputs confidence intervals that contain true values with high probability. We use ``test sample complexity'' to quantify the number of test points needed for a certifiable evaluation and derive tight bounds on test sample complexity. Based on the developed theory, we develop a partition-based algorithm, named Cer-Eval, that adaptively selects test points to minimize the cost of LLM evaluation. Real-world experiments demonstrate that Cer-Eval can save 20% to 40% test points across various benchmarks, while maintaining an estimation error level comparable to the current evaluation process and providing a 95% confidence guarantee. 

**Abstract (ZH)**: 基础模型继续扩大规模，训练模型的大小呈指数增长，这为模型评估带来了重大挑战。当前评估实践涉及收集越来越大的数据集来评估大型语言模型（LLMs）的性能。然而，缺乏系统分析和指导来确定测试数据的充足性或选择具有信息量的样本进行评估。本文介绍了用于LLMs的可验证且成本效益高的评估框架。我们的框架适应不同的评估目标，并输出高概率包含真实值的信任区间。我们使用“测试样本复杂性”来量化进行可验证评估所需的测试点数量，并推导出测试样本复杂性的紧致界。基于开发的理论，我们开发了一种基于分区的算法，名为Cer-Eval，该算法能够自适应地选择测试点以最小化LLM评估的成本。实验证明，Cer-Eval可以在各种基准测试中节约20%至40%的测试点，同时维持与当前评估过程相当的估计误差水平，并提供95%的信心保证。 

---
# Grouped Sequency-arranged Rotation: Optimizing Rotation Transformation for Quantization for Free 

**Title (ZH)**: 分组序排列旋转：无需代价优化旋转变换以实现量化 

**Authors**: Euntae Choi, Sumin Song, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03810)  

**Abstract**: Large Language Models (LLMs) face deployment challenges due to high computational costs, and while Post-Training Quantization (PTQ) offers a solution, existing rotation-based methods struggle at very low bit-widths like 2-bit. We introduce a novel, training-free approach to construct an improved rotation matrix, addressing the limitations of current methods. The key contributions include leveraging the Walsh-Hadamard transform with sequency ordering, which clusters similar frequency components to reduce quantization error compared to standard Hadamard matrices, significantly improving performance. Furthermore, we propose a Grouped Sequency-arranged Rotation (GSR) using block-diagonal matrices with smaller Walsh blocks, effectively isolating outlier impacts and achieving performance comparable to optimization-based methods without requiring any training. Our method demonstrates robust performance on reasoning tasks and Perplexity (PPL) score on WikiText-2. Our method also enhances results even when applied over existing learned rotation techniques. 

**Abstract (ZH)**: 一种新型无训练旋转矩阵构建方法：基于沃尔什-哈达玛变换的分组顺序排列旋转（GSR）以应对低位宽量化挑战 

---
# MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance 

**Title (ZH)**: MoEQuant: 基于专家平衡采样和亲和力引导的混合专家大型语言模型量化增强 

**Authors**: Xing Hu, Zhixuan Chen, Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, Jiangyong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03804)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLMs), which leverage dynamic routing and sparse activation to enhance efficiency and scalability, have achieved higher performance while reducing computational costs. However, these models face significant memory overheads, limiting their practical deployment and broader adoption. Post-training quantization (PTQ), a widely used method for compressing LLMs, encounters severe accuracy degradation and diminished generalization performance when applied to MoE models. This paper investigates the impact of MoE's sparse and dynamic characteristics on quantization and identifies two primary challenges: (1) Inter-expert imbalance, referring to the uneven distribution of samples across experts, which leads to insufficient and biased calibration for less frequently utilized experts; (2) Intra-expert imbalance, arising from MoE's unique aggregation mechanism, which leads to varying degrees of correlation between different samples and their assigned experts. To address these challenges, we propose MoEQuant, a novel quantization framework tailored for MoE LLMs. MoE-Quant includes two novel techniques: 1) Expert-Balanced Self-Sampling (EBSS) is an efficient sampling method that efficiently constructs a calibration set with balanced expert distributions by leveraging the cumulative probabilities of tokens and expert balance metrics as guiding factors. 2) Affinity-Guided Quantization (AGQ), which incorporates affinities between experts and samples into the quantization process, thereby accurately assessing the impact of individual samples on different experts within the MoE layer. Experiments demonstrate that MoEQuant achieves substantial performance gains (more than 10 points accuracy gain in the HumanEval for DeepSeekMoE-16B under 4-bit quantization) and boosts efficiency. 

**Abstract (ZH)**: MoE模型的高效量化框架：MoEQuant 

---
# Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth 

**Title (ZH)**: 基于自适应秩和位宽的量化模型高效微调方法 

**Authors**: Changhai Zhou, Yuhua Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03802)  

**Abstract**: QLoRA effectively combines low-bit quantization and LoRA to achieve memory-friendly fine-tuning for large language models (LLM). Recently, methods based on SVD for continuous update iterations to initialize LoRA matrices to accommodate quantization errors have generally failed to consistently improve performance. Dynamic mixed precision is a natural idea for continuously improving the fine-tuning performance of quantized models, but previous methods often optimize low-rank subspaces or quantization components separately, without considering their synergy. To address this, we propose \textbf{QR-Adaptor}, a unified, gradient-free strategy that uses partial calibration data to jointly search the quantization components and the rank of low-rank spaces for each layer, thereby continuously improving model performance. QR-Adaptor does not minimize quantization error but treats precision and rank allocation as a discrete optimization problem guided by actual downstream performance and memory usage. Compared to state-of-the-art (SOTA) quantized LoRA fine-tuning methods, our approach achieves a 4.89\% accuracy improvement on GSM8K, and in some cases even outperforms the 16-bit fine-tuned model while maintaining the memory footprint of the 4-bit setting. 

**Abstract (ZH)**: QLoRA有效地结合低比特量化和LoRA，实现大语言模型的内存友好型微调 

---
# Large Language Model Compression with Global Rank and Sparsity Optimization 

**Title (ZH)**: 全球秩和稀疏性优化的大语言模型压缩 

**Authors**: Changhai Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03801)  

**Abstract**: Low-rank and sparse composite approximation is a natural idea to compress Large Language Models (LLMs). However, such an idea faces two primary challenges that adversely affect the performance of existing methods. The first challenge relates to the interaction and cooperation between low-rank and sparse matrices, while the second involves determining weight allocation across different layers, as redundancy varies considerably among them. To address these challenges, we propose a novel two-stage LLM compression method with the capability of global rank and sparsity optimization. It is noteworthy that the overall optimization space is vast, making comprehensive optimization computationally prohibitive. Therefore, to reduce the optimization space, our first stage utilizes robust principal component analysis to decompose the weight matrices of LLMs into low-rank and sparse components, which span the low dimensional and sparse spaces containing the resultant low-rank and sparse matrices, respectively. In the second stage, we propose a probabilistic global optimization technique to jointly identify the low-rank and sparse structures within the above two spaces. The appealing feature of our approach is its ability to automatically detect the redundancy across different layers and to manage the interaction between the sparse and low-rank components. Extensive experimental results indicate that our method significantly surpasses state-of-the-art techniques for sparsification and composite approximation. 

**Abstract (ZH)**: 低秩和稀疏复合近似是压缩大型语言模型（LLMs）的自然想法。然而，这种想法面临着两个主要挑战，这些挑战会严重影响现有方法的性能。第一个挑战涉及低秩和稀疏矩阵之间的交互和协作，而第二个挑战则在于确定不同层之间的权重分配，因为这些层中的冗余程度差异很大。为了解决这些挑战，我们提出了一种具有全局秩和稀疏优化能力的新型两阶段LLM压缩方法。值得注意的是，整体优化空间非常庞大，使得全面优化在计算上是不可行的。因此，为了减少优化空间，我们第一阶段使用鲁棒主成分分析将LLM的权重矩阵分解为低秩和稀疏组件，这些组件分别占据低维和稀疏空间，其中包含相应的低秩和稀疏矩阵。在第二阶段，我们提出了一种概率全局优化技术，用于联合识别上述两个空间中的低秩和稀疏结构。我们方法的迷人之处在于其能够自动检测不同层之间的冗余，并管理稀疏和低秩组件之间的交互。广泛的研究结果表明，我们的方法在稀疏化和复合近似方面显著优于现有最先进的技术。 

---
# Scalability Matters: Overcoming Challenges in InstructGLM with Similarity-Degree-Based Sampling 

**Title (ZH)**: 可扩展性至关重要：基于相似度级别采样的InstructGLM挑战克服策略 

**Authors**: Hyun Lee, Chris Yi, Maminur Islam, B.D.S. Aritra  

**Link**: [PDF](https://arxiv.org/pdf/2505.03799)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in various natural language processing tasks; however, their application to graph-related problems remains limited, primarily due to scalability constraints and the absence of dedicated mechanisms for processing graph structures. Existing approaches predominantly integrate LLMs with Graph Neural Networks (GNNs), using GNNs as feature encoders or auxiliary components. However, directly encoding graph structures within LLMs has been underexplored, particularly in the context of large-scale graphs where token limitations hinder effective representation. To address these challenges, we propose SDM-InstructGLM, a novel instruction-tuned Graph Language Model (InstructGLM) framework that enhances scalability and efficiency without relying on GNNs. Our method introduces a similarity-degree-based biased random walk mechanism, which selectively samples and encodes graph information based on node-feature similarity and degree centrality, ensuring an adaptive and structured representation within the LLM. This approach significantly improves token efficiency, mitigates information loss due to random sampling, and enhances performance on graph-based tasks such as node classification and link prediction. Furthermore, our results demonstrate the feasibility of LLM-only graph processing, enabling scalable and interpretable Graph Language Models (GLMs) optimized through instruction-based fine-tuning. This work paves the way for GNN-free approaches to graph learning, leveraging LLMs as standalone graph reasoning models. Our source code is available on GitHub. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种自然语言处理任务中展现了强大的能力；然而，其在图相关问题中的应用受限于可扩展性限制和缺乏专门处理图结构的机制。现有方法主要将LLMs与图神经网络（GNNs）集成，使用GNNs作为特征编码器或辅助组件。然而，在大规模图中直接在LLMs中编码图结构尚未得到充分探索，尤其是当词汇量限制影响有效表示时。为解决这些问题，我们提出了一种新的指令调优图语言模型（InstructGLM）框架——SDM-InstructGLM，该框架通过不依赖于GNNs来增强可扩展性和效率。我们的方法引入了一种基于相似度-度量偏差随机游走机制，该机制根据节点特征相似性和度中心性有选择地采样和编码图信息，确保在LLM中实现适应性和结构化的表示。此方法显著提高了词汇效率，减轻了由于随机采样导致的信息损失，并提升了基于图的任务（如节点分类和链接预测）上的性能。此外，我们的结果证明了仅使用LLMs进行图处理的可行性，通过基于指令的微调实现可扩展且可解释的图语言模型（GLMs）。本工作为进一步使用LLMs作为独立图推理模型进行图学习铺平了道路。我们的源代码可在GitHub上获取。 

---
# AI-Driven IRM: Transforming insider risk management with adaptive scoring and LLM-based threat detection 

**Title (ZH)**: AI驱动的IRM：通过适应性评分和基于LLM的威胁检测转型内部风险管理工作 

**Authors**: Lokesh Koli, Shubham Kalra, Rohan Thakur, Anas Saifi, Karanpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.03796)  

**Abstract**: Insider threats pose a significant challenge to organizational security, often evading traditional rule-based detection systems due to their subtlety and contextual nature. This paper presents an AI-powered Insider Risk Management (IRM) system that integrates behavioral analytics, dynamic risk scoring, and real-time policy enforcement to detect and mitigate insider threats with high accuracy and adaptability. We introduce a hybrid scoring mechanism - transitioning from the static PRISM model to an adaptive AI-based model utilizing an autoencoder neural network trained on expert-annotated user activity data. Through iterative feedback loops and continuous learning, the system reduces false positives by 59% and improves true positive detection rates by 30%, demonstrating substantial gains in detection precision. Additionally, the platform scales efficiently, processing up to 10 million log events daily with sub-300ms query latency, and supports automated enforcement actions for policy violations, reducing manual intervention. The IRM system's deployment resulted in a 47% reduction in incident response times, highlighting its operational impact. Future enhancements include integrating explainable AI, federated learning, graph-based anomaly detection, and alignment with Zero Trust principles to further elevate its adaptability, transparency, and compliance-readiness. This work establishes a scalable and proactive framework for mitigating emerging insider risks in both on-premises and hybrid environments. 

**Abstract (ZH)**: 内部威胁对组织安全构成重大挑战，往往由于其隐蔽性和情境性而规避传统的基于规则的检测系统。本文提出了一种AI驱动的内部风险管理系统（IRM），该系统整合了行为分析、动态风险评分和实时政策执行，以实现高准确性和适应性的内部威胁检测与缓解。我们引入了一种混合评分机制——从静态的PRISM模型过渡到利用专家标注用户活动数据训练的自动编码神经网络的适应性AI模型。通过迭代反馈循环和持续学习，该系统将假阳性率降低59%，并提高了30%的真实阳性检测率，展示了在检测精度方面的显著提升。此外，该平台高效扩展，每日处理多达1000万条日志事件，并具有亚300毫秒的查询延迟，支持对政策违规行为的自动化执行措施，减少人工干预。IRM系统的部署使事件响应时间减少了47%，突显了其操作影响。未来增强包括集成可解释的AI、联邦学习、基于图的异常检测以及与零信任原则的对齐，以进一步提高其适应性、透明度和合规性。这项工作为在企业级和混合环境中缓解新兴内部风险构建了可扩展和前瞻性的框架。 

---
# LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection 

**Title (ZH)**: LENSLLM: 揭示大型语言模型选择的微调动态 

**Authors**: Xinyue Zeng, Haohui Wang, Junhong Lin, Jun Wu, Tyler Cody, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.03793)  

**Abstract**: The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a Hessian-based PAC-Bayes generalization bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel(NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at the Github link: this https URL. 

**Abstract (ZH)**: 开源大型语言模型（LLMs）的迅速增长及其多样的下游任务要求高效的选择模型，鉴于精细调整所有候选模型在计算上的不可行性。尽管在LLM选择方面取得了最近的进展，但仍有一个基本的研究问题尚未得到充分探索：如何在精细调整过程中建模LLM的动力学行为，从而提升其在多样的下游任务中表现的泛化理解？在本文中，我们提出了一种新的理论框架，以评估LLM的泛化能力，并由此实现对下游应用中高效且准确的LLM选择。特别是，我们首先推导出基于Hessian的PAC-Bayes泛化边界，揭示了LLM的精细调整动力学，然后引入了基于神经摆动核（NTK）的反向放大规模模型LENSLLM，该模型能够在保持计算效率的同时，实现对多样化任务的准确性能预测。在三个大规模基准上的广泛实验证明，我们的模型在LLM选择中达到了高达91.1%的准确率并降低了高达88.5%的计算成本，超过了5种最先进的方法。我们已在GitHub链接处开源了提出的LENSLLM模型及其结果：this https URL。 

---
# Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding 

**Title (ZH)**: 多模态大语言模型不确定性量化校准研究 

**Authors**: Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, Colin Samplawski, Brian Matejek, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03788)  

**Abstract**: We introduce a novel approach for calibrating uncertainty quantification (UQ) tailored for multi-modal large language models (LLMs). Existing state-of-the-art UQ methods rely on consistency among multiple responses generated by the LLM on an input query under diverse settings. However, these approaches often report higher confidence in scenarios where the LLM is consistently incorrect. This leads to a poorly calibrated confidence with respect to accuracy. To address this, we leverage cross-modal consistency in addition to self-consistency to improve the calibration of the multi-modal models. Specifically, we ground the textual responses to the visual inputs. The confidence from the grounding model is used to calibrate the overall confidence. Given that using a grounding model adds its own uncertainty in the pipeline, we apply temperature scaling - a widely accepted parametric calibration technique - to calibrate the grounding model's confidence in the accuracy of generated responses. We evaluate the proposed approach across multiple multi-modal tasks, such as medical question answering (Slake) and visual question answering (VQAv2), considering multi-modal models such as LLaVA-Med and LLaVA. The experiments demonstrate that the proposed framework achieves significantly improved calibration on both tasks. 

**Abstract (ZH)**: 我们提出了一种针对多模态大型语言模型的新型不确定性量化校准方法。现有的先进不确定性量化方法依赖于大型语言模型在多种设置下对输入查询生成的多个响应之间的一致性。然而，这些方法往往在大型语言模型一致错误的情况下报告更高的置信度，导致与准确度不匹配的校准置信度。为了解决这一问题，我们除了利用自我一致性之外，还利用跨模态一致性来提高多模态模型的校准。具体而言，我们将文本响应与视觉输入对接。对接模型的置信度用于校准整体置信度。由于使用对接模型会在管道中增加自身的不确定性，我们应用温度缩放——一种广泛接受的参数校准技术——来校准对接模型对未来响应准确性的置信度。我们在医疗问答（Slake）和视觉问答（VQAv2）等多个多模态任务上评估了所提出的方法，涉及多模态模型如LLaVA-Med和LLaVA。实验结果表明，所提出的框架在这两个任务上实现了显著改进的校准。 

---
# GPU Performance Portability needs Autotuning 

**Title (ZH)**: GPU 性能移植需要自动调优 

**Authors**: Burkhard Ringlein, Thomas Parnell, Radu Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2505.03780)  

**Abstract**: As LLMs grow in complexity, achieving state-of-the-art performance requires tight co-design across algorithms, software, and hardware. Today's reliance on a single dominant platform limits portability, creates vendor lock-in, and raises barriers for new AI hardware. In this work, we make the case for combining just-in-time (JIT) compilation with kernel parameter autotuning to enable portable, state-of-the-art performance LLM execution without code changes. Focusing on flash attention -- a widespread performance-critical LLM kernel -- we demonstrate that this approach explores up to 15x more kernel parameter configurations, produces significantly more diverse code across multiple dimensions, and even outperforms vendor-optimized implementations by up to 230%, all while reducing kernel code size by 70x and eliminating manual code optimizations. Our results highlight autotuning as a promising path to unlocking model portability across GPU vendors. 

**Abstract (ZH)**: 随着LLMs日益复杂，实现最优性能需要在算法、软件和硬件之间进行紧密协同设计。当前对单一主导平台的依赖限制了灵活性， creates vendor lock-in，并提高了新AI硬件的进入门槛。在本工作中，我们提出了结合即时编译（JIT）与内核参数自调优，以实现无需修改代码的便携式最优性能LLM执行。我们专注于闪注意力机制——一种广泛应用的关键性能内核——证明了这种方法探索了多达15倍更多的内核参数配置，产生了在多个维度上显著更多样化的代码，并且在某些情况下甚至比供应商优化的实现提高了230%，同时将内核代码大小减少了70倍，并消除了手动代码优化。我们的结果突显了自调优在解锁不同GPU供应商间模型便携性方面的潜力。 

---
# Splitwiser: Efficient LM inference with constrained resources 

**Title (ZH)**: Splitwiser：在受限资源下高效的LM推理 

**Authors**: Asad Aali, Adney Cardoza, Melissa Capo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03763)  

**Abstract**: Efficient inference of LLMs remains a crucial challenge, with two main phases: a compute-intensive prompt computation and a memory-intensive token generation. Despite existing batching and scheduling techniques, token generation phases fail to fully utilize compute resources, especially when compared to prompt computation phases. To address these challenges, we propose Splitwiser, a methodology that splits the two phases of an LLM inference request onto the same GPU, thereby reducing overhead and improving memory access and cache utilization. By eliminating the need to transfer data across devices, Splitwiser aims to minimize network-related overheads. In this report, we describe the basic structure of our proposed pipeline while sharing preliminary results and analysis. We implement our proposed multiprocessing design on two widely-used and independent LLM architectures: Huggingface and vLLM. We open-source our code for the respective implementations: 1) Huggingface (this https URL), and 2) vLLM (this https URL). 

**Abstract (ZH)**: 高效的大型语言模型推理仍然是一项关键挑战，主要包括两个主要阶段：密集的提示计算和密集的标记生成。尽管存在现有的批处理和调度技术，但标记生成阶段未能充分利用计算资源，尤其是在与提示计算阶段相比时。为应对这些挑战，我们提出了一种名为Splitwiser的方法，该方法将大型语言模型推理请求的两个阶段分配到同一个GPU上，从而减少开销并提高内存访问和缓存利用率。通过消除在不同设备之间传输数据的需要，Splitwiser旨在最小化与网络相关的时间开销。在本报告中，我们描述了我们提出的管道的基本结构，并分享了初步结果和分析。我们将在两个广泛使用的独立大型语言模型架构Huggingface和vLLM上实现我们提出的多处理设计。我们开源了相应的实现代码：1) Huggingface (点击此链接), 2) vLLM (点击此链接)。 

---
# Improving the Serving Performance of Multi-LoRA Large Language Models via Efficient LoRA and KV Cache Management 

**Title (ZH)**: 通过高效LoRA和KV缓存管理提高多LoRA大型语言模型的服务性能 

**Authors**: Hang Zhang, Jiuchen Shi, Yixiao Wang, Quan Chen, Yizhou Shan, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03756)  

**Abstract**: Multiple Low-Rank Adapters (Multi-LoRAs) are gaining popularity for task-specific Large Language Model (LLM) applications. For multi-LoRA serving, caching hot KV caches and LoRA adapters in high bandwidth memory of accelerations can improve inference performance. However, existing Multi-LoRA inference systems fail to optimize serving performance like Time-To-First-Toke (TTFT), neglecting usage dependencies when caching LoRAs and KVs. We therefore propose FASTLIBRA, a Multi-LoRA caching system to optimize the serving performance. FASTLIBRA comprises a dependency-aware cache manager and a performance-driven cache swapper. The cache manager maintains the usage dependencies between LoRAs and KV caches during the inference with a unified caching pool. The cache swapper determines the swap-in or out of LoRAs and KV caches based on a unified cost model, when the HBM is idle or busy, respectively. Experimental results show that ELORA reduces the TTFT by 63.4% on average, compared to state-of-the-art works. 

**Abstract (ZH)**: 多低秩适配器（Multi-LoRAs）缓存系统FASTLIBRA：优化任务特定大型语言模型推理性能 

---
# Promoting Security and Trust on Social Networks: Explainable Cyberbullying Detection Using Large Language Models in a Stream-Based Machine Learning Framework 

**Title (ZH)**: 基于流式机器学习框架的可解释网络欺凌检测：使用大型语言模型促进社交网络的安全与信任 

**Authors**: Silvia García-Méndez, Francisco De Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2505.03746)  

**Abstract**: Social media platforms enable instant and ubiquitous connectivity and are essential to social interaction and communication in our technological society. Apart from its advantages, these platforms have given rise to negative behaviors in the online community, the so-called cyberbullying. Despite the many works involving generative Artificial Intelligence (AI) in the literature lately, there remain opportunities to study its performance apart from zero/few-shot learning strategies. Accordingly, we propose an innovative and real-time solution for cyberbullying detection that leverages stream-based Machine Learning (ML) models able to process the incoming samples incrementally and Large Language Models (LLMS) for feature engineering to address the evolving nature of abusive and hate speech online. An explainability dashboard is provided to promote the system's trustworthiness, reliability, and accountability. Results on experimental data report promising performance close to 90 % in all evaluation metrics and surpassing those obtained by competing works in the literature. Ultimately, our proposal contributes to the safety of online communities by timely detecting abusive behavior to prevent long-lasting harassment and reduce the negative consequences in society. 

**Abstract (ZH)**: 社交媒体平台 enables 即时和普遍的连接性，并在我们的技术社会中对于社会互动和交流是必不可少的。除了其优势，这些平台还催生了在线社区中的负面行为，即网络欺凌。尽管近期文献中涉及生成人工智能（AI）的工作很多，但仍有机会研究其性能，不仅限于零/少-shot 学习策略。因此，我们提出了一个创新且实时的网络欺凌检测解决方案，该方案利用基于流的机器学习（ML）模型以逐增量处理传入样本，并运用大型语言模型（LLMs）进行特征工程，以应对在线恶意和仇恨言论的演变性质。提供了可解释性仪表板以促进系统的可信度、可靠性和问责制。实验数据上的结果报告显示，在所有评估指标上接近90%的性能表现，并且超过文献中竞争工作的结果。最终，我们的提案通过及时检测恶意行为来 contributeto 在线社区的安全，从而预防持久性骚扰并减少社会的负面影响。 

---
# AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design 

**Title (ZH)**: AccLLM: 通过算法-硬件协同设计加速长上下文LLM推理 

**Authors**: Yanbiao Liang, Huihong Shi, Haikuo Shao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03745)  

**Abstract**: Recently, large language models (LLMs) have achieved huge success in the natural language processing (NLP) field, driving a growing demand to extend their deployment from the cloud to edge devices. However, deploying LLMs on resource-constrained edge devices poses significant challenges, including (1) intensive computations and huge model sizes, (2) great memory and bandwidth demands introduced by the autoregressive generation process, and (3) limited scalability for handling long sequences. To address these challenges, we propose AccLLM, a comprehensive acceleration framework that enables efficient and fast long-context LLM inference through algorithm and hardware co-design. At the algorithmic level, we integrate (1) pruning, (2) {\Lambda}-shaped attention, and (3) an innovative W2A8KV4 (2-bit weights, 8-bit activations, and 4-bit KV cache) quantization scheme, thus effectively reducing memory and bandwidth requirements while facilitating LLMs' long-sequence generation. At the hardware level, we design a dedicated FPGA-based accelerator with a reconfigurable computing engine to effectively and flexibly accommodate diverse operations arising from our compression algorithm, thereby fully translating the algorithmic innovations into tangible hardware efficiency. We validate AccLLM on the Xilinx Alveo U280 FPGA, demonstrating a 4.07x energy efficiency and a 2.98x throughput compared to the state-of-the-art work FlightLLM. 

**Abstract (ZH)**: 近期，大规模语言模型（LLMs）在自然语言处理（NLP）领域取得了巨大成功，推动了它们从云端向边缘设备部署的需求增长。然而，在资源受限的边缘设备上部署LLMs带来了显著挑战，包括（1）密集的计算和庞大的模型规模，（2）自回归生成过程中引入的巨大内存和带宽需求，以及（3）处理长序列时的有限扩展性。为应对这些挑战，我们提出了一种全面的加速框架AccLLM，通过算法和硬件协同设计实现了高效且快速的长上下文LLM推理。在算法层面，我们整合了（1）剪枝，（2）Λ形注意机制，以及（3）一种创新的W2A8KV4（2位权重，8位激活和4位KV缓存）量化方案，从而有效地降低了内存和带宽需求，促进了LLMs长序列生成能力的增强。在硬件层面，我们设计了一种专用的基于FPGA的加速器，配备可重构计算引擎，能够有效且灵活地适应来自我们压缩算法的各种操作，从而全面将算法创新转化为实际的硬件效率。我们在Xilinx Alveo U280 FPGA上验证了AccLLM，相比于当前最先进的工作FlightLLM，展示了4.07倍的能量效率和2.98倍的吞吐量。 

---
