# MUVLA: Learning to Explore Object Navigation via Map Understanding 

**Title (ZH)**: MUVLA：通过地图理解学习物体导航探索 

**Authors**: Peilong Han, Fan Jia, Min Zhang, Yutao Qiu, Hongyao Tang, Yan Zheng, Tiancai Wang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25966)  

**Abstract**: In this paper, we present MUVLA, a Map Understanding Vision-Language-Action model tailored for object navigation. It leverages semantic map abstractions to unify and structure historical information, encoding spatial context in a compact and consistent form. MUVLA takes the current and history observations, as well as the semantic map, as inputs and predicts the action sequence based on the description of goal object. Furthermore, it amplifies supervision through reward-guided return modeling based on dense short-horizon progress signals, enabling the model to develop a detailed understanding of action value for reward maximization. MUVLA employs a three-stage training pipeline: learning map-level spatial understanding, imitating behaviors from mixed-quality demonstrations, and reward amplification. This strategy allows MUVLA to unify diverse demonstrations into a robust spatial representation and generate more rational exploration strategies. Experiments on HM3D and Gibson benchmarks demonstrate that MUVLA achieves great generalization and learns effective exploration behaviors even from low-quality or partially successful trajectories. 

**Abstract (ZH)**: 本文介绍了MUVLA，一种针对物体导航定制的时空图理解视觉语言行动模型。该模型利用语义地图抽象来统一和结构化历史信息，以紧凑且一致的形式编码空间上下文。MUVLA 采用当前和历史观察结果以及语义地图作为输入，并基于目标物体的描述预测行动序列。此外，通过基于密集的短期进展信号进行奖励导向的返回建模来增强监督，从而促使模型发展出详细的行动价值理解以实现奖励最大化。MUVLA 采用三阶段训练管道：学习地图级别空间理解、模仿混合质量的演示行为以及奖励放大。这一策略使 MUVLA 能够将多种多样的演示统一为 robust 的空间表示，并生成更合理的探索策略。在 HM3D 和 Gibson 基准测试中，MUVLA 展示了出色的泛化能力，并能够从低质量或部分成功的轨迹中学习有效的探索行为。 

---
# dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought 

**Title (ZH)**: dVLA：具备多模态链式思维的扩散视觉语言行动模型 

**Authors**: Junjie Wen, Minjie Zhu, Jiaming Liu, Zhiyuan Liu, Yicun Yang, Linfeng Zhang, Shanghang Zhang, Yichen Zhu, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25681)  

**Abstract**: Vision-Language-Action (VLA) models are emerging as a next-generation paradigm for robotics. We introduce dVLA, a diffusion-based VLA that leverages a multimodal chain-of-thought to unify visual perception, language reasoning, and robotic control in a single system. dVLA jointly optimizes perception, language understanding, and action under a single diffusion objective, enabling stronger cross-modal reasoning and better generalization to novel instructions and objects. For practical deployment, we mitigate inference latency by incorporating two acceleration strategies, a prefix attention mask and KV caching, yielding up to around times speedup at test-time inference. We evaluate dVLA in both simulation and the real world: on the LIBERO benchmark, it achieves state-of-the-art performance with a 96.4% average success rate, consistently surpassing both discrete and continuous action policies; on a real Franka robot, it succeeds across a diverse task suite, including a challenging bin-picking task that requires multi-step planning, demonstrating robust real-world performance. Together, these results underscore the promise of unified diffusion frameworks for practical, high-performance VLA robotics. 

**Abstract (ZH)**: 基于扩散的视觉-语言-动作模型：统一的多模态推理在机器人学中的应用 

---
# STaR-Attack: A Spatio-Temporal and Narrative Reasoning Attack Framework for Unified Multimodal Understanding and Generation Models 

**Title (ZH)**: STaR-攻击：统一多模态理解与生成模型的时空与叙述推理攻击框架 

**Authors**: Shaoxiong Guo, Tianyi Du, Lijun Li, Yuyao Wu, Jie Li, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26473)  

**Abstract**: Unified Multimodal understanding and generation Models (UMMs) have demonstrated remarkable capabilities in both understanding and generation tasks. However, we identify a vulnerability arising from the generation-understanding coupling in UMMs. The attackers can use the generative function to craft an information-rich adversarial image and then leverage the understanding function to absorb it in a single pass, which we call Cross-Modal Generative Injection (CMGI). Current attack methods on malicious instructions are often limited to a single modality while also relying on prompt rewriting with semantic drift, leaving the unique vulnerabilities of UMMs unexplored. We propose STaR-Attack, the first multi-turn jailbreak attack framework that exploits unique safety weaknesses of UMMs without semantic drift. Specifically, our method defines a malicious event that is strongly correlated with the target query within a spatio-temporal context. Using the three-act narrative theory, STaR-Attack generates the pre-event and the post-event scenes while concealing the malicious event as the hidden climax. When executing the attack strategy, the opening two rounds exploit the UMM's generative ability to produce images for these scenes. Subsequently, an image-based question guessing and answering game is introduced by exploiting the understanding capability. STaR-Attack embeds the original malicious question among benign candidates, forcing the model to select and answer the most relevant one given the narrative context. Extensive experiments show that STaR-Attack consistently surpasses prior approaches, achieving up to 93.06% ASR on Gemini-2.0-Flash and surpasses the strongest prior baseline, FlipAttack. Our work uncovers a critical yet underdeveloped vulnerability and highlights the need for safety alignments in UMMs. 

**Abstract (ZH)**: 统一多模态理解与生成模型（UMMs）在理解和生成任务中展现出了显著的能力。然而，我们发现UMMs中存在的生成-理解耦合漏洞。攻击者可以利用生成功能构建信息丰富的 adversarial 图像，然后借助理解功能在一个步骤中吸收该图像，我们将其称为跨模态生成注入（CMGI）。目前针对恶意指令的攻击方法往往局限于单一模态，并依赖语义漂移的提示重写，从而未能探索UMMs的独特漏洞。我们提出了STaR-攻击，这是首个不依赖语义漂移利用UMMs独特安全弱点的多回合越界攻击框架。具体而言，我们的方法在时空上下文中定义了一个与目标查询强相关的恶意事件。利用三幕剧叙事理论，STaR-攻击生成预事件和后事件场景，将恶意事件隐藏为隐含高潮。在实施攻击策略时，初始的两个回合利用UMMs的生成能力为这些场景生成图像。随后，通过利用理解能力引入基于图像的问题猜测与回答游戏。STaR-攻击将原始的恶意问题嵌入良性候选问题中，迫使模型在叙述背景下选择并回答最相关的问题。广泛的实验表明，STaR-攻击在Gemini-2.0-Flash上达到了93.06%的ASR，超越了最强的先前基线FlipAttack。我们的工作揭示了一个关键但尚未充分发展的漏洞，并强调了UMMs中安全性对齐的必要性。 

---
# Towards Unified Multimodal Misinformation Detection in Social Media: A Benchmark Dataset and Baseline 

**Title (ZH)**: 面向社交媒体中统一的多模态 misinformation 检测：一个基准数据集和基线方法 

**Authors**: Haiyang Li, Yaxiong Wang, Lianwei Wu, Lechao Cheng, Zhun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2509.25991)  

**Abstract**: In recent years, detecting fake multimodal content on social media has drawn increasing attention. Two major forms of deception dominate: human-crafted misinformation (e.g., rumors and misleading posts) and AI-generated content produced by image synthesis models or vision-language models (VLMs). Although both share deceptive intent, they are typically studied in isolation. NLP research focuses on human-written misinformation, while the CV community targets AI-generated artifacts. As a result, existing models are often specialized for only one type of fake content. In real-world scenarios, however, the type of a multimodal post is usually unknown, limiting the effectiveness of such specialized systems. To bridge this gap, we construct the Omnibus Dataset for Multimodal News Deception (OmniFake), a comprehensive benchmark of 127K samples that integrates human-curated misinformation from existing resources with newly synthesized AI-generated examples. Based on this dataset, we propose Unified Multimodal Fake Content Detection (UMFDet), a framework designed to handle both forms of deception. UMFDet leverages a VLM backbone augmented with a Category-aware Mixture-of-Experts (MoE) Adapter to capture category-specific cues, and an attribution chain-of-thought mechanism that provides implicit reasoning guidance for locating salient deceptive signals. Extensive experiments demonstrate that UMFDet achieves robust and consistent performance across both misinformation types, outperforming specialized baselines and offering a practical solution for real-world multimodal deception detection. 

**Abstract (ZH)**: 近年来，社交媒体上假多模态内容的检测受到了不断增加的关注。两种主要形式的欺骗占主导地位：人类设计的信息误导（如谣言和误导性帖子）和由图像合成模型或视觉语言模型生成的AI内容。尽管两者都具有误导意图，但通常分别研究。NLP研究专注于人类编写的误导性内容，而CV社区针对AI生成的伪造内容。因此，现有模型往往只能针对一种类型的虚假内容。然而，在实际场景中，多模态帖子的类型通常是未知的，限制了此类专门系统的有效性。为了弥合这一差距，我们构建了面向多模态新闻欺诈的综合基准数据集（OmniFake），该数据集包含127,000个样本，整合了现有资源中的人工编curated误Informmation信和新合成的AI生成示例。基于此数据集，我们提出了一种统一的多模态虚假内容检测框架（UMFDet），该框架旨在处理这两种欺骗形式。UMFDet利用了一个增强的视觉语言模型骨干，并结合了一个类别感知的混合专家（MoE）适配器来捕捉类别特定的线索，以及一种归属链推理机制，该机制提供了隐式推理指导，以定位显著的误导信号。 extensive实验证明，UMFDet在两种误导类别上均表现出稳健且一致的性能，优于专门基准，并为实际场景中的多模态欺骗检测提供了实用解决方案。 

---
# Automated Model Discovery via Multi-modal & Multi-step Pipeline 

**Title (ZH)**: 多模态多步管道驱动的自动化模型发现 

**Authors**: Lee Jung-Mok, Nam Hyeon-Woo, Moon Ye-Bin, Junhyun Nam, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2509.25946)  

**Abstract**: Automated model discovery is the process of automatically searching and identifying the most appropriate model for a given dataset over a large combinatorial search space. Existing approaches, however, often face challenges in balancing the capture of fine-grained details with ensuring generalizability beyond training data regimes with a reasonable model complexity. In this paper, we present a multi-modal \& multi-step pipeline for effective automated model discovery. Our approach leverages two vision-language-based modules (VLM), AnalyzerVLM and EvaluatorVLM, for effective model proposal and evaluation in an agentic way. AnalyzerVLM autonomously plans and executes multi-step analyses to propose effective candidate models. EvaluatorVLM assesses the candidate models both quantitatively and perceptually, regarding the fitness for local details and the generalibility for overall trends. Our results demonstrate that our pipeline effectively discovers models that capture fine details and ensure strong generalizability. Additionally, extensive ablation studies show that both multi-modality and multi-step reasoning play crucial roles in discovering favorable models. 

**Abstract (ZH)**: 自动化模型发现是通过在大规模组合搜索空间中自动搜索和识别最适合给定数据集的模型的过程。现有方法往往难以平衡捕捉细粒度细节与在合理模型复杂度下确保泛化能力之间的关系。本文提出了一种多模态与多步的自动化模型发现管线。我们的方法利用了两个基于视觉-语言的模块（VLM），AnalyzerVLM和EvaluatorVLM，以一种有agency的方式进行有效的模型提案和评估。AnalyzerVLM自主规划并执行多步分析以提出有效的候选模型。EvaluatorVLM从定量和感知的角度评估候选模型，考虑其对局部细节的适应性和对整体趋势的泛化能力。我们的结果表明，该管线能够有效地发现既能捕捉细节又能确保强大泛化能力的模型。此外，广泛的消融研究显示，多模态性和多步推理在发现有利模型中起着关键作用。 

---
# GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination 

**Title (ZH)**: GroundSight: 通过 Grounding 信息和反 hallucination 增强视觉-语言模型 

**Authors**: Xinxi Chen, Tianyang Chen, Lijia Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.25669)  

**Abstract**: We propose a method to improve Visual Question Answering (VQA) with Retrieval-Augmented Generation (RAG) by introducing text-grounded object localization. Rather than retrieving information based on the entire image, our approach enables the model to generate a bounding box around the object most relevant to the question, allowing for targeted image cropping and focused retrieval. This reduces background noise, improves alignment between visual and textual cues, and helps mitigate hallucinations. Our RAG method enhances context-aware VQA responses increased the accuracy from 22.19% to 25.64%, with an absolute increase of 3.45 percentage points, compared to the baseline Llama-3.2-Vision-11B agent. We also proposed a de-hallucination method based on question type which can effectively reduce the hallucination rate from 65.79% to 13.88% and improves the truthfulness score. 

**Abstract (ZH)**: 我们提出了一种通过引入文本基础的对象定位来改进视觉问答（VQA）的方法，利用检索增强生成（RAG）技术。我们的方法使模型能够生成与问题最相关的对象的边界框，从而实现目标图像裁剪和聚焦检索，减少背景噪声，提高视觉和文本线索的对齐，有助于减少幻觉现象。与基础模型Llama-3.2-Vision-11B相比，我们的RAG方法提升了上下文感知的VQA响应准确性，从22.19%提高到25.64%，绝对提高3.45个百分点。同时，我们还提出了一种基于问题类型的减幻觉方法，有效将幻觉率从65.79%降低到13.88%，提高了真实性评分。 

---
# Iterative Residual Cross-Attention Mechanism: An Integrated Approach for Audio-Visual Navigation Tasks 

**Title (ZH)**: 迭代残差跨注意力机制：一种用于音频-视觉导航任务的集成方法 

**Authors**: Hailong Zhang, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25652)  

**Abstract**: Audio-visual navigation represents a significant area of research in which intelligent agents utilize egocentric visual and auditory perceptions to identify audio targets. Conventional navigation methodologies typically adopt a staged modular design, which involves first executing feature fusion, then utilizing Gated Recurrent Unit (GRU) modules for sequence modeling, and finally making decisions through reinforcement learning. While this modular approach has demonstrated effectiveness, it may also lead to redundant information processing and inconsistencies in information transmission between the various modules during the feature fusion and GRU sequence modeling phases. This paper presents IRCAM-AVN (Iterative Residual Cross-Attention Mechanism for Audiovisual Navigation), an end-to-end framework that integrates multimodal information fusion and sequence modeling within a unified IRCAM module, thereby replacing the traditional separate components for fusion and GRU. This innovative mechanism employs a multi-level residual design that concatenates initial multimodal sequences with processed information sequences. This methodological shift progressively optimizes the feature extraction process while reducing model bias and enhancing the model's stability and generalization capabilities. Empirical results indicate that intelligent agents employing the iterative residual cross-attention mechanism exhibit superior navigation performance. 

**Abstract (ZH)**: 基于迭代残差交叉注意力机制的视听导航 

---
# Radiology's Last Exam (RadLE): Benchmarking Frontier Multimodal AI Against Human Experts and a Taxonomy of Visual Reasoning Errors in Radiology 

**Title (ZH)**: Radiology's Last Exam (RadLE): 评估前沿多模态AI与人类专家的表现及其在放射学中视觉推理错误的分类基准 

**Authors**: Suvrankar Datta, Divya Buchireddygari, Lakshmi Vennela Chowdary Kaza, Mrudula Bhalke, Kautik Singh, Ayush Pandey, Sonit Sai Vasipalli, Upasana Karnwal, Hakikat Bir Singh Bhatti, Bhavya Ratan Maroo, Sanjana Hebbar, Rahul Joseph, Gurkawal Kaur, Devyani Singh, Akhil V, Dheeksha Devasya Shama Prasad, Nishtha Mahajan, Ayinaparthi Arisha, Rajesh Vanagundi, Reet Nandy, Kartik Vuthoo, Snigdhaa Rajvanshi, Nikhileswar Kondaveeti, Suyash Gunjal, Rishabh Jain, Rajat Jain, Anurag Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.25559)  

**Abstract**: Generalist multimodal AI systems such as large language models (LLMs) and vision language models (VLMs) are increasingly accessed by clinicians and patients alike for medical image interpretation through widely available consumer-facing chatbots. Most evaluations claiming expert level performance are on public datasets containing common pathologies. Rigorous evaluation of frontier models on difficult diagnostic cases remains limited. We developed a pilot benchmark of 50 expert-level "spot diagnosis" cases across multiple imaging modalities to evaluate the performance of frontier AI models against board-certified radiologists and radiology trainees. To mirror real-world usage, the reasoning modes of five popular frontier AI models were tested through their native web interfaces, viz. OpenAI o3, OpenAI GPT-5, Gemini 2.5 Pro, Grok-4, and Claude Opus 4.1. Accuracy was scored by blinded experts, and reproducibility was assessed across three independent runs. GPT-5 was additionally evaluated across various reasoning modes. Reasoning quality errors were assessed and a taxonomy of visual reasoning errors was defined. Board-certified radiologists achieved the highest diagnostic accuracy (83%), outperforming trainees (45%) and all AI models (best performance shown by GPT-5: 30%). Reliability was substantial for GPT-5 and o3, moderate for Gemini 2.5 Pro and Grok-4, and poor for Claude Opus 4.1. These findings demonstrate that advanced frontier models fall far short of radiologists in challenging diagnostic cases. Our benchmark highlights the present limitations of generalist AI in medical imaging and cautions against unsupervised clinical use. We also provide a qualitative analysis of reasoning traces and propose a practical taxonomy of visual reasoning errors by AI models for better understanding their failure modes, informing evaluation standards and guiding more robust model development. 

**Abstract (ZH)**: 面向临床的多模态AI系统（如大规模语言模型和视觉语言模型）通过广泛使用的消费者聊天机器人被医生和患者用于医学图像解释。大多数声称专家级性能的评估是在包含常见病理的公共数据集上进行的。对前沿模型在困难诊断案例上的严格评估仍然有限。我们开发了一个包含50个专家级“即时诊断”案例的试点基准，涉及多种成像模态，用于评估前沿AI模型与认证放射科医生和放射科培训生的性能。为了反映实际使用情况，五种流行前沿AI模型的推理模式通过其原生网页界面进行了测试，分别是OpenAI o3、OpenAI GPT-5、Gemini 2.5 Pro、Grok-4和Claude Opus 4.1。准确率由盲评专家评分，并在三次独立运行中评估了可重复性。此外，GPT-5在多种推理模式下进行了评估。评估了推理质量错误，并定义了视觉推理错误的分类。认证放射科医生的诊断准确率最高（83%），优于培训生（45%）和所有AI模型（最佳表现为GPT-5：30%）。GPT-5和o3的可靠性较大，Gemini 2.5 Pro和Grok-4的可靠性中等，Claude Opus 4.1的可靠性较差。这些发现表明，在困难的诊断案例中，高级前沿模型远不及放射科医生。我们的基准指出了通用AI在医学成像领域的当前局限性，并提醒不应在未经监督的情况下临床使用。我们还提供了推理轨迹的定性分析，并提出了AI模型视觉推理错误的实用分类，以更好地理解其失败模式，指导评估标准并引导更稳健的模型开发。 

---
# Stitch: Training-Free Position Control in Multimodal Diffusion Transformers 

**Title (ZH)**: Stitch: 无需训练的多模态扩散变换器位置控制 

**Authors**: Jessica Bader, Mateusz Pach, Maria A. Bravo, Serge Belongie, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2509.26644)  

**Abstract**: Text-to-Image (T2I) generation models have advanced rapidly in recent years, but accurately capturing spatial relationships like "above" or "to the right of" poses a persistent challenge. Earlier methods improved spatial relationship following with external position control. However, as architectures evolved to enhance image quality, these techniques became incompatible with modern models. We propose Stitch, a training-free method for incorporating external position control into Multi-Modal Diffusion Transformers (MMDiT) via automatically-generated bounding boxes. Stitch produces images that are both spatially accurate and visually appealing by generating individual objects within designated bounding boxes and seamlessly stitching them together. We find that targeted attention heads capture the information necessary to isolate and cut out individual objects mid-generation, without needing to fully complete the image. We evaluate Stitch on PosEval, our benchmark for position-based T2I generation. Featuring five new tasks that extend the concept of Position beyond the basic GenEval task, PosEval demonstrates that even top models still have significant room for improvement in position-based generation. Tested on Qwen-Image, FLUX, and SD3.5, Stitch consistently enhances base models, even improving FLUX by 218% on GenEval's Position task and by 206% on PosEval. Stitch achieves state-of-the-art results with Qwen-Image on PosEval, improving over previous models by 54%, all accomplished while integrating position control into leading models training-free. Code is available at this https URL. 

**Abstract (ZH)**: 基于文本到图像生成中外部位置控制的训练-free方法：Stitch 

---
# SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP 

**Title (ZH)**: SeMoBridge: 语义模态桥梁，实现CLIP的高效少样本适应 

**Authors**: Christoph Timmermann, Hyunse Lee, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.26036)  

**Abstract**: While Contrastive Language-Image Pretraining (CLIP) excels at zero-shot tasks by aligning image and text embeddings, its performance in few-shot classification is hindered by a critical limitation: intra-modal misalignment. This issue, caused by a persistent modality gap and CLIP's exclusively inter-modal training objective, leaves the embedding spaces uncalibrated, making direct image-to-image comparisons unreliable. Existing methods attempt to address this by refining similarity logits or by computationally expensive per-sample optimization. To overcome these challenges, we introduce SeMoBridge, a lightweight yet powerful approach that directly addresses the misalignment. Our method maps images into the text modality, while keeping their semantic content intact through what we call a Semantic Modality Bridge. SeMoBridge is closed-form and can optionally be trained through multi-modal supervision, combining image and text-alignment losses to optimize the projection. Experiments show that the trained version, SeMoBridge-T, requires only a fraction of the training time while overall outperforming other methods, particularly in low-data scenarios (1, 2, and 4 shots). The code is available at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 尽管对比语言-图像预训练（CLIP）在零样本任务中通过对齐图像和文本嵌入表现出色，但在少样本分类中的性能受到一个关键限制的阻碍：模内对齐不匹配。这个问题由持续存在的模态差距和CLIP exclusively的跨模态训练目标引起，使得嵌入空间无法校准，从而使直接图像-图像比较不可靠。现有方法试图通过细化相似性逻辑或通过昂贵的逐样本优化来解决这个问题。为克服这些挑战，我们引入了SeMoBridge，这是一种轻量级但强大的方法，直接解决了对齐问题。该方法通过所谓的语义模态桥将图像映射到文本模态，同时保持其语义内容不变。SeMoBridge 是闭式形式的，并且可以通过多模态监督可选地进行训练，结合图像和文本对齐损失来优化投影。实验表明，训练版本 SeMoBridge-T 只需少量的训练时间，而在总体上优于其他方法，尤其是在低数据情景（1, 2, 和 4 次射击）中。代码可供参考：\href{this https URL}{this http URL}。 

---
# V-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs 

**Title (ZH)**: V-HUB: 以视觉为中心的视频幽默理解基准数据集 

**Authors**: Zhengpeng Shi, Hengli Li, Yanpeng Zhao, Jianqun Zhou, Yuxuan Wang, Qinrong Cui, Wei Bi, Songchun Zhu, Bo Zhao, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25773)  

**Abstract**: AI models capable of comprehending humor hold real-world promise -- for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel visual-centric video humor understanding benchmark. v-HUB comprises a curated collection of minimally verbal short videos, sourced from classic silent films and online resources, and reflecting real-world scenarios where humor can be appreciated purely through visual cues. Each video clip is paired with rich annotations, including captions, descriptions, and explanations, supporting evaluation tasks like caption matching and humor explanation. To broaden its applicability, we further construct an open-ended video QA task, making it readily integrable into existing video understanding benchmarks. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. For example, all models exhibit a marked performance drop on caption matching when moving from text-based to video-based evaluation (without audio). Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the informativeness of sound and the promise of integrating richer modalities for complex video understanding tasks. 

**Abstract (ZH)**: 具备理解幽默能力的AI模型在实际应用中具有真实潜力——例如，提升人机互动中的参与度。为了评估和诊断多模态大规模语言模型（MLLMs）在幽默理解方面的能力，我们引入了v-HUB，一个以视觉为中心的视频幽默理解基准。v-HUB包含一系列来自经典无声电影和在线资源的少量文字简短视频，反映的是通过视觉线索即可欣赏幽默的真实世界场景。每个视频片段都配有丰富的注释，包括字幕、描述和解释，支持字幕匹配和幽默解释等评估任务。为了提高其通用性，我们进一步构建了一个开放式的视频问答任务，使其能够无缝集成到现有的视频理解基准中。我们评估了一组多样的MLLMs，从专门处理视频的Video-LLMs到能够处理音频的全能型OmniLLMs，涵盖了开源和专有领域。实验结果揭示了MLLMs仅凭视觉线索理解幽默所面临的困难。例如，所有模型在从基于文本的评估转向基于视频的评估（不包含音频）时，在字幕匹配任务中表现显著下降。我们的研究还表明，引入音频有助于视频幽默的理解，突显了声音信息的价值以及集成更丰富模态的潜力，以应对复杂的视频理解任务。 

---
# Dolphin v1.0 Technical Report 

**Title (ZH)**: Dolphin v1.0 技术报告 

**Authors**: Taohan Weng, Chi zhang, Chaoran Yan, Siya Liu, Xiaoyang Liu, Yalun Wu, Boyang Wang, Boyan Wang, Jiren Ren, Kaiwen Yan, Jinze Yu, Kaibing Hu, Henan Liu, Haoyun zheng, Anjie Le, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25748)  

**Abstract**: Ultrasound is crucial in modern medicine but faces challenges like operator dependence, image noise, and real-time scanning, hindering AI integration. While large multimodal models excel in other medical imaging areas, they struggle with ultrasound's complexities. To address this, we introduce Dolphin v1.0 (V1) and its reasoning-augmented version, Dolphin R1-the first large-scale multimodal ultrasound foundation models unifying diverse clinical tasks in a single vision-language this http URL tackle ultrasound variability and noise, we curated a 2-million-scale multimodal dataset, combining textbook knowledge, public data, synthetic samples, and general corpora. This ensures robust perception, generalization, and clinical this http URL Dolphin series employs a three-stage training strategy: domain-specialized pretraining, instruction-driven alignment, and reinforcement-based refinement. Dolphin v1.0 delivers reliable performance in classification, detection, regression, and report generation. Dolphin R1 enhances diagnostic inference, reasoning transparency, and interpretability through reinforcement learning with ultrasound-specific this http URL on U2-Bench across eight ultrasound tasks, Dolphin R1 achieves a U2-score of 0.5835-over twice the second-best model (0.2968) setting a new state of the art. Dolphin v1.0 also performs competitively, validating the unified framework. Comparisons show reasoning-enhanced training significantly improves diagnostic accuracy, consistency, and interpretability, highlighting its importance for high-stakes medical AI. 

**Abstract (ZH)**: 超声在现代医学中至关重要，但面临着操作者依赖、图像噪声和实时扫描等挑战，阻碍了人工智能的集成。虽然大型多模态模型在其他医学影像领域表现出色，但在处理超声的复杂性方面却力不从心。为了解决这一问题，我们介绍了Dolphin v1.0（V1）及其增强推理版本Dolphin R1——首个统一多种临床任务的大规模多模态超声基础模型。为应对超声变异性和噪声，我们构建了一个规模达200万的多模态数据集，整合了教科书知识、公共数据、合成样本和通用语料。这一数据集确保了模型的稳健感知、泛化能力和临床应用能力。Dolphin系列采用三阶段训练策略：领域特异化的预训练、指令驱动的对齐以及基于强化学习的精炼。Dolphin v1.0在分类、检测、回归和报告生成中提供可靠的性能。Dolphin R1通过针对超声的具体强化学习增强诊断推断、推理透明性和可解释性。在U2-Bench上进行的八项超声任务测试中，Dolphin R1实现了0.5835的U2分数，超过了第二名模型（0.2968）的两倍，创下了新的性能纪录。Dolphin v1.0也表现出色，验证了统一框架的有效性。比较结果显示，增强推理的训练显著提高了诊断准确度、一致性和可解释性，突显了其在高风险医疗AI中的重要性。 

---
# Probing the Limits of Stylistic Alignment in Vision-Language Models 

**Title (ZH)**: 探究视觉语言模型风格对齐的极限 

**Authors**: Asma Farajidizaji, Akash Gupta, Vatsal Raina  

**Link**: [PDF](https://arxiv.org/pdf/2509.25568)  

**Abstract**: Vision-language models are increasingly used to generate image captions in specific styles, such as humor or romantic. However, these transformer-based models often struggle with this subjective task in a zero-shot setting. While preference data can be used to align them toward a desired style, such data is expensive to acquire, limiting the ability to explore the models' full capabilities. This work addresses this by studying the data efficiency of aligning small vision-language models to humor and romantic styles. This approach helps to define the performance limits of these models and determine how little preference data is needed to achieve stylistic saturation, benchmarking their capabilities and limitations. 

**Abstract (ZH)**: 基于视觉-语言模型的小样本数据效率研究：幽默与浪漫风格的对齐及其性能边界确定 

---
# InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions 

**Title (ZH)**: InfMasking: 利用对比多模态交互释放协同信息 

**Authors**: Liangjian Wen, Qun Dai, Jianzhuang Liu, Jiangtao Zheng, Yong Dai, Dongkai Wang, Zhao Kang, Jun Wang, Zenglin Xu, Jiang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2509.25270)  

**Abstract**: In multimodal representation learning, synergistic interactions between modalities not only provide complementary information but also create unique outcomes through specific interaction patterns that no single modality could achieve alone. Existing methods may struggle to effectively capture the full spectrum of synergistic information, leading to suboptimal performance in tasks where such interactions are critical. This is particularly problematic because synergistic information constitutes the fundamental value proposition of multimodal representation. To address this challenge, we introduce InfMasking, a contrastive synergistic information extraction method designed to enhance synergistic information through an \textbf{Inf}inite \textbf{Masking} strategy. InfMasking stochastically occludes most features from each modality during fusion, preserving only partial information to create representations with varied synergistic patterns. Unmasked fused representations are then aligned with masked ones through mutual information maximization to encode comprehensive synergistic information. This infinite masking strategy enables capturing richer interactions by exposing the model to diverse partial modality combinations during training. As computing mutual information estimates with infinite masking is computationally prohibitive, we derive an InfMasking loss to approximate this calculation. Through controlled experiments, we demonstrate that InfMasking effectively enhances synergistic information between modalities. In evaluations on large-scale real-world datasets, InfMasking achieves state-of-the-art performance across seven benchmarks. Code is released at this https URL. 

**Abstract (ZH)**: 在多模态表示学习中，模态之间的协同交互不仅提供互补信息，还能通过特定的交互模式创造出单一模态无法实现的独特成果。现有方法可能难以有效捕捉协同信息的整个谱系，导致在需要此类交互的任务中表现不佳。由于协同信息构成了多模态表示的基本价值主张，因此这一挑战尤为关键。为解决这一问题，我们 introduced InfMasking，一种通过无限掩蔽策略增强协同信息的对比性协同信息提取方法。InfMasking 在融合过程中随机遮蔽每个模态的大部分特征，仅保留部分信息以生成具有多种协同模式的表示。未遮蔽的融合表示通过最大化互信息与遮蔽表示对齐，从而编码全面的协同信息。无限掩蔽策略在训练过程中使模型接触到多种不同的部分模态组合，从而捕获更丰富的交互。由于使用无限掩蔽计算互信息估计在计算上是不可行的，我们推导出一种 InfMasking 损失来近似此计算。通过受控实验，我们证明 InfMasking 有效地增强了模态之间的协同信息。在大规模现实世界数据集的评估中，InfMasking 在七个基准测试中实现了最先进的性能。代码托管在 this https URL。 

---
